"""增强会话编排：策略、语义理解、原文取证、知识补充与焦点原子提交。"""
import json
import time
import uuid
from app.rag.contracts import EvidenceContext
from app.rag.context_builder import ContextBuilder
from app.rag.dialogue_focus import context as focus_context, update_in_transaction
from app.rag.enhanced_resolver import EnhancedResolver, HistoricalReferenceError
from app.rag.query_planner import QueryPlanner
from app.rag.enhanced_runtime import EnhancedCalls
from app.rag.enhanced_answer import EnhancedAnswerService
from app.rag.memory import answer_units
from app.rag.reference_evidence import ReferenceEvidenceLoader
from app.rag.llm_client import LLMCallError
from app.rag.aspect_evidence import retrieval_plans, select_evidence
from app.rag.comparison_task import display_aliases, make_plan
from app.rag.comparison_task_repository import ComparisonTaskRepository
from app.store import now_iso

class EnhancedConversationFlow:
    def __init__(self, parent):
        self.parent=parent
        self.flow,self.repo,self.config=parent.flow,parent.repo,parent.config
        self.resolver=EnhancedResolver(self.config)
        self.planner=QueryPlanner()
        self.solver=EnhancedAnswerService(self.config)
        self.comparison_tasks=ComparisonTaskRepository(self.repo)

    def units(self, sections, citations):
        """条目保留来源类型；模型知识可继续追问，但没有虚构的文档chunk。"""
        result=[]
        for section in sections:
            if section["source_type"]=="limitation":continue
            # 只把本区块实际引用的来源交给条目解析，避免同回答其他文档串入焦点。
            section_citations=[c for c in citations if c["citation_number"] in section["citation_numbers"]]
            units=answer_units(section["content"],section_citations)
            if not units:
                text=section["content"]
                units=[dict(unit_id="1",ordinal=1,display_number=1,title=text.split("。")[0][:80],
                            text=text,document_ids=list(dict.fromkeys(c["document_id"] for c in section_citations))
                            if section["source_type"]=="document" else [],cited_chunk_ids=[
                                c["chunk_id"] for c in section_citations])]
            for unit in units:
                unit.update(unit_id=str(len(result)+1),ordinal=len(result)+1,source_type=section["source_type"])
                if section["source_type"]=="model_knowledge":
                    unit.update(document_ids=[],cited_chunk_ids=[])
                result.append(unit)
        return result

    def bounded(self, hits):
        """必要来源已由loader约束；附加召回不能挤掉原文锚点。"""
        builder=ContextBuilder(self.config)
        ordered=sorted(hits,key=lambda h:h.metadata.get("evidence_role")!="referenced_source")
        result=[];seen=set()
        for hit in ordered:
            if hit.chunk_id in seen:continue
            candidate=result+[hit]
            count=builder.counter(json.dumps([EvidenceContext(i,h.filename,h.page,h.text).to_dict()
                                             for i,h in enumerate(candidate,1)],ensure_ascii=False))
            if count>self.config.evidence_max_context_tokens:
                if hit.metadata.get("evidence_role")=="referenced_source":
                    from app.rag.context_builder import ContextOverflow
                    raise ContextOverflow("必要原文超过本轮证据预算，请缩小问题范围",
                                          {"phase":"evidence_budget","token_count_method":builder.method,
                                           "input_count":count,"input_limit":self.config.evidence_max_context_tokens,
                                           "component_counts":{"referenced_evidence":count}})
                continue
            result=candidate;seen.add(hit.chunk_id)
        return result

    async def retrieve(self, query, resolution, memory, events, previous=(), *,
                       missing_document_ids=(), supplemental=False):
        """按方面分别召回后轮流选证据；补检索返回实际新增块数。"""
        scope=resolution["document_ids"]
        plans=retrieval_plans(resolution,query,supplemental=supplemental or bool(previous),
                              missing_document_ids=missing_document_ids)
        batches=[]
        if scope:
            for aspect_id, planned_query, target_document in plans:
                local_scope=[target_document] if target_document else scope
                # 明确选中单份文档时局部放宽上限；多文档仍遵循原有均衡规则。
                cap=(6 if previous else 4) if len(local_scope)==1 and (
                    target_document or len(plans)>1 or previous) else None
                batch=await self.flow._retrieve_knowledge(type("Args",(),dict(
                    query=planned_query,top_k=6 if previous else 4,
                    document_ids=local_scope,per_document_cap=cap))())
                batches.append((aspect_id,batch))
                events.extend([{"node":"reranker","status":batch.trace["status"],
                                "output":{**batch.trace,"aspect_id":aspect_id,
                                          "document_id":target_document}},
                               {"node":"evidence_expansion","status":batch.expansion_trace["status"],
                                "output":{**batch.expansion_trace,"aspect_id":aspect_id,
                                          "document_id":target_document}}])
        hits=select_evidence(previous,batches,6 if previous else 4)
        if len(plans)>1 or resolution.get("comparison_task"):
            events.append({"node":"aspect_retrieval","status":"ok","output":{
                "aspect_ids":[a for a,_,_ in plans],"queries":[q for _,q,_ in plans],
                "document_ids":[d for _,_,d in plans],
                "candidate_counts":[len(batch) for _,batch in batches],
                "selected_chunk_ids":[h.chunk_id for h in hits]}})
        for unit in resolution.get("referenced_unit_ids",[]) or [resolution.get("referenced_unit_id")]:
            if not unit:continue
            hits,trace=ReferenceEvidenceLoader(self.flow.store,self.config).merge(
                {**resolution,"referenced_unit_id":unit},memory,hits)
            events.append({"node":"reference_evidence","status":trace["status"],"output":trace})
        hits=self.bounded(hits)
        if resolution.get("comparison_task"):
            covered={h.document_id for h in hits}
            events.append({"node":"comparison_evidence","status":"complete" if all(
                d in covered for d in resolution["document_ids"]) else "missing_document",
                "output":{"document_ids":resolution["document_ids"],
                          "selected_counts":{d:sum(h.document_id==d for h in hits)
                                             for d in resolution["document_ids"]},
                          "selected_chunk_ids":{d:[h.chunk_id for h in hits if h.document_id==d]
                                                for d in resolution["document_ids"]},
                          "missing_document_ids":[d for d in resolution["document_ids"] if d not in covered]}})
        new_count=len({h.chunk_id for h in hits}-{h.chunk_id for h in previous})
        events.append({"node":"tool","tool":"retrieve_knowledge","status":"ok",
                       "query":query,"result_count":len(hits),"new_chunk_count":new_count,
                       "citations":[h.citation() for h in hits]})
        return hits,new_count

    async def chat(self, session, question, document_ids=None, request_id=None, scope_revision=None):
        turn_id,cached=self.repo.begin_turn(session,request_id or str(uuid.uuid4()),question,
                                            request_context={"answer_policy":"document_enhanced"})
        if cached:return cached
        events=[];run_id=str(uuid.uuid4());started=time.perf_counter()
        calls=EnhancedCalls(self.config,self.flow.llm.client,events)
        snapshot=None;resolution=None
        try:
            snapshot=self.repo.snapshot(session,document_ids,scope_revision)
            snapshot.update(answer_policy="document_enhanced")
            memory=self.parent.memory.snapshot(session)
            focus=focus_context(self.repo,session,memory)
            pending=focus.get("pending_comparison_task") or {}
            if pending and (not set(pending.get("document_ids",[])) <= set(snapshot["document_ids"]) or
                            any(snapshot["versions"].get(doc)!=version
                                for doc,version in pending.get("versions",{}).items())):
                # 范围增加第三份资料不使原两篇失效；移除来源或重建索引才清除待完成焦点。
                focus["pending_comparison_task"]=None
                events.append({"node":"comparison_focus","status":"invalidated_source_change",
                               "output":{"previous_document_ids":pending.get("document_ids",[]),
                                         "current_document_ids":snapshot["document_ids"]}})
            events.extend([{"node":"scope_snapshot","status":"ok","output":snapshot},
                           {"node":"dialogue_focus","status":"ok","output":focus},
                           {"node":"memory","status":"ok","output":memory["trace"]}])
            # 明确的继续概述或跨轮重号可直接规划，减少长历史进入解析模型。
            documents=self.repo.documents(snapshot["document_ids"])
            available_documents=self.repo.documents()
            historical_tasks=self.comparison_tasks.list(session)
            resolution=self.planner.preplan(question,snapshot,memory,focus,enhanced=True,
                                            documents=documents,comparison_tasks=historical_tasks,
                                            available_documents=available_documents,lookup_events=events)
            if resolution is None:
                try:
                    resolution=await self.resolver.resolve(question,snapshot,memory,focus,
                                                           documents,calls)
                except HistoricalReferenceError:
                    # 拒绝模型编造的 turn/unit ID，只用本会话任务索引重新定位一次。
                    resolution=self.planner.recover_historical_reference(
                        question,snapshot,historical_tasks,documents,available_documents,events)
            if (resolution.get("intent")=="comparison" and not resolution.get("comparison_task")
                    and len(resolution.get("document_ids",[]))==2):
                # 模型虽识别出两篇比较，也必须转成带稳定任务 ID 的原文专属计划。
                pair=resolution["document_ids"]
                resolution=make_plan(question,pair,snapshot["versions"],documents,
                                     method="validated_model_pair_comparison") or resolution
            resolution=self.planner.enforce(question,resolution,snapshot,focus,enhanced=True,memory=memory)
            events.append({"node":"query_resolution","status":"ok","output":resolution})
            citations=[];coverage=None;tools=[];effective="document_enhanced"
            if resolution["intent"]=="clarification":
                result=dict(answer=display_aliases(resolution["clarification_question"],
                                                   snapshot["document_ids"],documents),answer_status="clarification",
                            answer_basis="none",answer_sections=[],unresolved_aspects=[],cited_numbers=[])
            elif resolution["intent"]=="overview":
                # 概述始终走已有的纯文档链路，不能因默认增强而向论文添加通用背景。
                value,citations,coverage=await self.parent.overview.summarize(
                    resolution["resolved_query"],resolution["document_ids"],snapshot)
                effective="document_strict"
                sections=[dict(source_type="document",content=value.answer,
                               citation_numbers=list(value.cited_numbers),aspect_ids=["a1"])]
                result=dict(answer=value.answer,answer_status=value.status,answer_basis="document_only",
                            answer_sections=sections,unresolved_aspects=[],cited_numbers=list(value.cited_numbers))
                events.extend([{"node":"overview","status":"ok","output":coverage},
                               {"node":"llm","status":value.trace.get("status"),"output":value.trace}])
            else:
                # 检索改写只决定候选原文，显示给用户的resolved_query仍是其实际问题。
                hits,_=await self.retrieve(resolution.get("retrieval_query") or resolution["resolved_query"],
                                           resolution,memory,events)
                task=resolution.get("comparison_task")
                # 比较对象已由固定计划确定；生成阶段不再携带旧打卡回答等无关轮次。
                answer_memory=memory if not task else {"rounds":[],"state":{"summary":""}}
                if task:
                    missing=[d for d in task["document_ids"] if d not in {h.document_id for h in hits}]
                    if missing:
                        # 零命中侧只补一次；不重新搜索已有原文的一侧，也不消耗模型生成次数。
                        hits,new_count=await self.retrieve(resolution["resolved_query"],resolution,memory,events,
                            hits,missing_document_ids=missing,supplemental=True)
                        events.append({"node":"comparison_supplemental_retrieval",
                                       "status":"expanded" if new_count else "no_new_evidence",
                                       "output":{"document_ids":missing,"new_chunk_count":new_count}})
                def evidence():
                    return [EvidenceContext(i,h.filename,h.page,h.text,h.document_id)
                            for i,h in enumerate(hits,1)]
                if self.config.llm_mode=="mock":
                    text="[Mock] 仅展示流程证据，不提供模型通用知识。"
                    sections=[]
                    if hits:
                        text+="\n"+"\n".join(h.text+f" [{i}]" for i,h in enumerate(hits,1))
                        sections=[dict(source_type="document",content=text,citation_numbers=list(range(1,len(hits)+1)),
                                       aspect_ids=[a["id"] for a in resolution["requested_aspects"]])]
                    result=dict(answer=text,answer_status="mock",answer_basis="document_only" if hits else "none",
                                answer_sections=sections,unresolved_aspects=[],cited_numbers=list(range(1,len(hits)+1)))
                elif not hits and all(a["requires_document"] for a in resolution["requested_aspects"]):
                    result=self.solver.no_document_evidence(resolution)
                else:
                    decision,result,trace=await self.solver.generate(question,resolution,evidence(),answer_memory,calls)
                    if decision.action=="retrieve_more":
                        missing=([d for d in task["document_ids"] if d not in {h.document_id for h in hits}]
                                 if task else [])
                        hits,new_count=await self.retrieve(decision.supplemental_query,resolution,memory,events,hits,
                            missing_document_ids=missing,supplemental=True)
                        events.append({"node":"supplemental_retrieval",
                                       "status":"expanded" if new_count else "no_new_evidence",
                                       "query":decision.supplemental_query,"new_chunk_count":new_count})
                        if new_count or (task and hits):
                            # 比较题可在最终轮保留已有一侧的原文，给出部分回答。
                            decision,result,trace=await self.solver.generate(
                                question,resolution,evidence(),answer_memory,calls,True)
                        else:
                            result=self.solver.no_new_evidence(resolution)
                    events.append({"node":"llm","status":"ok","output":trace})
                citations=[dict(citation_number=i,**h.citation()) for i,h in enumerate(hits,1)
                           if i in result["cited_numbers"]]
                tools=[dict(name="retrieve_knowledge",input={"query":e["query"],"document_ids":resolution["document_ids"]},
                            result_count=e["result_count"]) for e in events if e["node"]=="tool"]
            self.repo.validate(snapshot)
            sections=result["answer_sections"]
            # 概述直接采用结构化条目，避免从多批文本重复提取“第二点”。
            units=coverage["units"] if coverage and coverage.get("units") else self.units(sections,citations)
            response=dict(run_id=run_id,turn_id=turn_id,answer=result["answer"],answer_status=result["answer_status"],
                citations=citations,tool_calls=tools,intent=resolution["intent"],resolved_query=resolution["resolved_query"],
                scope_snapshot=snapshot,coverage=coverage,answer_units=units,answer_policy="document_enhanced",
                effective_policy=effective,answer_basis=result["answer_basis"],answer_sections=sections,
                unresolved_aspects=result["unresolved_aspects"])
            payload=dict(response=response,resolution=resolution,answer_units=units,snapshot=snapshot,
                         request_context={"answer_policy":"document_enhanced"})
            events.extend([{"node":"answer_policy","status":"ok","output":{
                "selected":"document_enhanced","effective":effective,"answer_basis":result["answer_basis"],
                "logical_calls":calls.logical_calls}},
                {"node":"critic","status":"ok","output":{
                    "grounded":None,"grounding_status":"source_structure_validated_semantics_requires_review"}},
                {"node":"workflow","status":"ok","latency_ms":round((time.perf_counter()-started)*1000,3)}])
            with self.repo.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                self.repo.validate(snapshot)
                update_in_transaction(self.repo,db,session,dict(id=turn_id,response=response),resolution,focus)
                comparison=resolution.get("comparison_task")
                if comparison:
                    task_status={"answered":"answered","partial_answer":"partial",
                                 "insufficient_evidence":"insufficient","mock":"retryable"}.get(
                                     response["answer_status"],"retryable")
                    # 任务索引与完成轮次同一事务发布；成功任务也继续供历史定位。
                    self.comparison_tasks.save(db,session,turn_id,comparison,task_status)
                db.execute("UPDATE conversation_turns SET status='completed',payload=? WHERE id=?",
                           (json.dumps(payload,ensure_ascii=False),turn_id))
                db.executemany("INSERT INTO messages(session_id,role,content,created_at) VALUES(?,?,?,?)",[
                    (session,"user",question,now_iso()),(session,"assistant",result["answer"],now_iso())])
                db.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)",
                           (run_id,session,"completed",json.dumps(events,ensure_ascii=False),None,now_iso()))
            self.parent.memory.schedule(session)
            return response
        except Exception as exc:
            if isinstance(exc,LLMCallError):
                exc.run_id=run_id
                events.append({"node":"llm","status":"error","output":exc.trace})
            from app.rag.context_builder import ContextOverflow
            if isinstance(exc,ContextOverflow):
                exc.run_id=run_id
                exc.trace.setdefault("phase","enhanced_workflow")
                if not any(event["node"]=="context_budget" and event["status"]=="error" for event in events):
                    events.append({"node":"context_budget","status":"error","output":exc.trace})
            events.append({"node":"workflow","status":"error","error_type":type(exc).__name__,
                           "latency_ms":round((time.perf_counter()-started)*1000,3)})
            self.repo.finish_turn(turn_id,{"run_id":run_id,"error_type":type(exc).__name__,
                                         "request_context":{"answer_policy":"document_enhanced"}},"failed")
            if snapshot and resolution and resolution.get("comparison_task"):
                try:
                    # 已固定两篇但生成失败时保留可重试任务；来源变化则拒绝保存旧版本。
                    self.repo.validate(snapshot)
                    with self.repo.connect() as db:
                        db.execute("BEGIN IMMEDIATE")
                        self.comparison_tasks.save(db,session,turn_id,
                                                   resolution["comparison_task"],"retryable")
                    events.append({"node":"comparison_task","status":"retryable",
                                   "output":{"task_id":resolution["comparison_task"]["task_id"]}})
                except Exception:
                    events.append({"node":"comparison_task","status":"not_saved_after_failure"})
            self.flow.store.save_run(run_id,session,"failed",events,type(exc).__name__)
            raise

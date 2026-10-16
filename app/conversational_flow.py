"""新会话编排：快照、记忆、问题解析、取证与生成，各模块保持独立职责。"""
import json
import time
import uuid
from app.conversation import ConversationError
from app.rag.contracts import EvidenceContext
from app.rag.memory import MemoryManager, answer_units
from app.rag.query_resolution import QueryResolver
from app.rag.query_planner import QueryPlanner
from app.rag.dialogue_focus import context as focus_context, update_in_transaction
from app.rag.overview import OverviewService
from app.rag.solver import AnswerResult
from app.rag.reference_evidence import ReferenceEvidenceLoader
from app.rag.context_builder import ContextOverflow
from app.rag.llm_client import LLMCallError
from app.store import now_iso

class ConversationalFlow:
    def __init__(self, flow):
        self.flow,self.repo,self.config=flow,flow.conversations,flow.settings
        self.memory=MemoryManager(self.repo,self.config,flow.llm.client)
        self.resolver=QueryResolver(self.config,flow.llm.client)
        self.planner=QueryPlanner()
        self.overview=OverviewService(self.repo,self.config,flow.llm)

    async def chat(self,session,question,document_ids=None,request_id=None,scope_revision=None,answer_policy="document_strict"):
        # 增强路径独立实现和校验，旧严格流程保持兼容。
        if answer_policy=="document_enhanced":
            from app.enhanced_flow import EnhancedConversationFlow
            return await EnhancedConversationFlow(self).chat(session,question,document_ids,request_id,scope_revision)
        turn_id,cached=self.repo.begin_turn(session,request_id or str(uuid.uuid4()),question,request_context={"answer_policy":"document_strict"})
        if cached:return cached
        run_id=str(uuid.uuid4());events=[];started=time.perf_counter()
        try:
            snapshot=self.repo.snapshot(session,document_ids,scope_revision)
            events.append({"node":"scope_snapshot","status":"ok","output":snapshot})
            memory=self.memory.snapshot(session)
            focus=focus_context(self.repo,session,memory)
            events.append({"node":"memory","status":"ok","output":memory["trace"]})
            # 续跑和跨轮重号先由确定性规划器判定，其余语言理解仍复用旧解析器。
            resolution=self.planner.preplan(question,snapshot,memory,focus,enhanced=False)
            if resolution is None:
                resolution=await self.resolver.resolve(question,snapshot,memory,self.repo.documents(snapshot["document_ids"]))
            resolution=self.planner.enforce(question,resolution,snapshot,focus,enhanced=False,memory=memory)
            events.append({"node":"query_resolution","status":"ok","output":resolution})
            citations=[];tools=[];coverage=None
            if resolution["intent"]=="clarification":
                result=AnswerResult(resolution["clarification_question"],(),"clarification",
                                    {"status":"skipped_clarification","attempts":0,"usage":None})
            elif resolution["intent"]=="overview":
                result,citations,coverage=await self.overview.summarize(
                    resolution["resolved_query"],resolution["document_ids"],snapshot)
                events.append({"node":"overview","status":"ok","output":coverage})
            else:
                batch=await self.flow._retrieve_knowledge(type("Args",(),{
                    "query":resolution.get("retrieval_query") or resolution["resolved_query"],"top_k":4,"document_ids":resolution["document_ids"]})())
                events.append({"node":"reranker","status":batch.trace["status"],"output":batch.trace})
                events.append({"node":"evidence_expansion","status":batch.expansion_trace["status"],"output":batch.expansion_trace})
                # 定位到历史条目后重读当前规范来源，不能只依赖改写后的中文BM25。
                # 历史答案仅说明用户指什么，最终证据仍是本次读取并核验的原文。
                merged,reference_trace=ReferenceEvidenceLoader(self.flow.store,self.config).merge(
                    resolution,memory,batch)
                batch[:]=merged
                events.append({"node":"reference_evidence","status":reference_trace["status"],
                               "output":reference_trace})
                events.append({"node":"tool","tool":"retrieve_knowledge","status":"ok",
                               "result_count":len(batch),"citations":[h.citation() for h in batch]})
                tools=[{"name":"retrieve_knowledge","input":{"query":resolution["resolved_query"],
                    "top_k":4,"document_ids":resolution["document_ids"]},"result_count":len(batch)}]
                if not batch:
                    result=await self.flow.llm.answer(resolution["resolved_query"],[],[])
                else:
                    evidence=[EvidenceContext(i,h.filename,h.page,h.text) for i,h in enumerate(batch,1)]
                    # 历史条目只用于改写问题和定位原文；旧回答正文不作为 pinned 输入。
                    pinned=[]
                    # 条目含义追问先回答原文已支持的部分；缺少进一步细节要说明边界，
                    # 不能因原文未给出完整推导就把明确写出的目标/分类也整体拒答。
                    followup_instruction=(
                        "\n本轮是解释已定位条目。先用用户语言说明证据已支持的部分及该条目的含义。"
                        "若原文只给出目标或分类，可忠实转述这些内容；未提供的定义、推导或细节明确说明，"
                        "不要补外部知识，也不要因为缺少进一步细节而对已有明确依据的内容整体拒答。"
                        "说明缺失时必须限定为当前引用片段未展开，不能断言整篇文档没有说明；其他章节可能提供细节。"
                    ) if resolution.get("referenced_unit_id") else ""
                    result=await self.flow.llm.generate_context(resolution["resolved_query"],evidence,memory["rounds"],
                        instructions=followup_instruction,pinned=pinned,payload={"question":resolution["resolved_query"],"original_question":question,
                                              "evidence":[e.to_dict() for e in evidence],
                                              "conversation_summary":memory["state"].get("summary","")})
                    citations=[{"citation_number":i,**h.citation()} for i,h in enumerate(batch,1) if i in result.cited_numbers]
            self.repo.validate(snapshot)
            events.append({"node":"llm","status":result.trace.get("status","unknown"),"output":result.trace})
            if result.trace.get("context_budget"):
                events.append({"node":"context_budget","status":"ok","output":result.trace["context_budget"]})
            events.append({"node":"critic","status":"ok","output":{
                "grounded":None,"grounding_status":"citation_numbers_validated_semantics_not_evaluated",
                "answer_status":result.status,"citation_count":len(citations)}})
            events.append({"node":"workflow","status":"ok","latency_ms":round((time.perf_counter()-started)*1000,3)})
            # 新概述已有通过原文编号核验的条目；其他回答仍走旧文本兼容解析。
            units=coverage["units"] if coverage and coverage.get("units") else answer_units(result.answer,citations)
            response=dict(run_id=run_id,turn_id=turn_id,answer=result.answer,answer_status=result.status,
                citations=citations,tool_calls=tools,intent=resolution["intent"],resolved_query=resolution["resolved_query"],
                scope_snapshot=snapshot,coverage=coverage,answer_units=units,
                answer_policy="document_strict",effective_policy="document_strict",
                answer_basis="document_only" if citations else "none",answer_sections=[],unresolved_aspects=[])
            payload={"response":response,"resolution":resolution,"answer_units":units,"snapshot":snapshot,"request_context":{"answer_policy":"document_strict"}}
            # 完整轮次、会话消息与成功运行记录原子发布，重复请求直接返回同一结果。
            with self.repo.connect() as db:
                # 发布回答时持写锁再次核验来源，堵住校验后到提交前的更新竞争窗口。
                db.execute("BEGIN IMMEDIATE")
                self.repo.validate(snapshot)
                # 与回答同事务保存当前焦点、候选和用户纠正，失败轮次不覆盖实质焦点。
                update_in_transaction(self.repo,db,session,dict(id=turn_id,response=response),resolution,focus)
                db.execute("UPDATE conversation_turns SET status='completed',payload=? WHERE id=?",
                           (json.dumps(payload,ensure_ascii=False),turn_id))
                db.executemany("INSERT INTO messages(session_id,role,content,created_at) VALUES(?,?,?,?)",
                    [(session,"user",question,now_iso()),(session,"assistant",result.answer,now_iso())])
                db.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)",
                    (run_id,session,"completed",json.dumps(events,ensure_ascii=False),None,now_iso()))
            self.memory.schedule(session)
            return response
        except Exception as exc:
            if isinstance(exc,ContextOverflow):
                # 严格模式也提供运行编号与阶段，失败仍保存为可追查的运行。
                exc.run_id=run_id
                exc.trace.setdefault("phase","document_strict")
                events.append({"node":"context_budget","status":"error","output":exc.trace})
            if isinstance(exc,LLMCallError):
                exc.run_id=run_id;events.append({"node":"llm","status":"error","output":exc.trace})
            # 失败运行也记录总耗时，避免界面只展示最后一次模型调用。
            events.append({"node":"workflow","status":"error","error_type":type(exc).__name__,
                           "latency_ms":round((time.perf_counter()-started)*1000,3)})
            self.repo.finish_turn(turn_id,{"run_id":run_id,"error_type":type(exc).__name__},"failed")
            self.flow.store.save_run(run_id,session,"failed",events,type(exc).__name__)
            raise

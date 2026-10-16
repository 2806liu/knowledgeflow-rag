"""指定文档概述：全篇规范来源、分批归纳、分层汇总、版本缓存与部分覆盖。"""
from dataclasses import asdict, replace
import asyncio
import hashlib
import json
import time
from app.rag.contracts import EvidenceContext
from app.rag.solver import AnswerResult, _JSON_PROMPT, _SYSTEM_PROMPT
from app.rag.context_builder import ContextBuilder, ContextOverflow
from app.rag.llm_client import LLMCallError, LLMClient
from app.rag.structured_overview import StructuredOverview

_INSTRUCTION=(
    "\n任务是概述指定文档的主题、主要内容和结论，必须综合给定的全部章节，特别保留末尾独有要点。"
    "使用清晰编号列出内容，避免只概括开头。不同文件分别说明。"
    "只输出带原文引用的编号要点，不写未带引用的前言；文档主题如单列也必须标明支持原文编号。"
    "每个事实结论引用给定原文编号；中间摘要不是新的来源编号。"
    "只能转述原文明确给出的事实，不增添适用群体、目的、替代主体或额外要求。"
)

class OverviewService:
    def __init__(self, conversations, config, solver):
        from app.rag.solver import EvidenceSolver
        # 概述输出独立预算，同时收紧可用输入；不以普通短答的上限截断长文汇总。
        overview_config=replace(config,llm_max_output_tokens=config.overview_max_output_tokens)
        self.repo,self.config=conversations,overview_config
        self.solver=EvidenceSolver(overview_config)
        if not isinstance(solver.client,LLMClient):
            self.solver.client=solver.client  # 隔离测试注入不触发真实模型。
        self.builder=ContextBuilder(overview_config)
        # 新概述统一走结构化批次与断点；旧实现保留为历史兼容阅读层，不改旧回答。
        self.structured=StructuredOverview(conversations,config,self.solver.client)

    def sources(self, ids):
        rows=self.repo.store.chunks(tuple(ids))
        rows=sorted(rows,key=lambda r:(ids.index(r["document_id"]),r["position"]))
        evidence=[];citations=[]
        for number,row in enumerate(rows,1):
            meta=json.loads(row["metadata"] or "{}")
            evidence.append(EvidenceContext(number,row["filename"],meta.get("page"),row["text"]))
            citations.append(dict(citation_number=number,chunk_id=row["id"],document_id=row["document_id"],
                filename=row["filename"],page=meta.get("page"),position=row["position"],score=None,snippet=row["text"],
                extraction_method=meta.get("extraction_method"),ocr_confidence=meta.get("ocr_mean_confidence"),
                evidence_role="overview",section_id=meta.get("section_id")))
        return evidence,citations

    def fits(self,payload):
        system=(_JSON_PROMPT if self.config.llm_answer_format=="json" else _SYSTEM_PROMPT)+_INSTRUCTION
        try:
            self.builder.build(system,payload)
            return True
        except ContextOverflow:
            return False

    def batches(self,items,payload):
        """按最终封装预算贪心装批，不能仅按字符猜测模型输入大小。"""
        groups=[];current=[];skipped=[]
        for item in items:
            if self.fits(payload(current+[item])):
                current.append(item)
            else:
                if current: groups.append(current)
                current=[item] if self.fits(payload([item])) else []
                if not current: skipped.append(item)
        if current: groups.append(current)
        return groups,skipped

    async def summarize(self,question,ids,snapshot):
        """新请求使用结构化概述；完整缓存和部分断点都由同一存储管理。"""
        return await self.structured.summarize(question,ids,snapshot)

    async def summarize_legacy(self,question,ids,snapshot):
        self.repo.validate(snapshot)
        evidence,citations=self.sources(ids)
        if not evidence:
            return AnswerResult("所选资料没有可用于概述的正文。",(),"no_evidence"),[],{"complete":False}
        key=hashlib.sha256(json.dumps([ids,snapshot["versions"],question,self.config.llm_model,
              self.config.llm_answer_format,self.config.llm_max_output_tokens,"overview-v2"],ensure_ascii=False).encode()).hexdigest()
        with self.repo.connect() as db:
            row=db.execute("SELECT payload FROM overview_cache WHERE cache_key=?",(key,)).fetchone()
        if row:
            data=json.loads(row[0]);self.repo.validate(snapshot)
            saved=data["result"]
            result=AnswerResult(saved["answer"],tuple(saved["cited_numbers"]),saved["status"],
                               {"status":"cache_hit","attempts":0,"usage":None,"model":self.config.llm_model})
            return result,data["citations"],{**data["coverage"],"cache_hit":True}
        def raw_payload(group):
            return {"question":question,"evidence":[e.to_dict() for e in group]}
        groups,skipped=self.batches(evidence,raw_payload)
        calls=[];attempts=0;processed=[];summaries=[];failed=[];last_error=None
        started=time.monotonic()
        async def generate(payload,available):
            nonlocal attempts
            remaining=self.config.overview_timeout_seconds-(time.monotonic()-started)
            # 为一次调用的全部潜在重试预留额度，真实尝试数记入硬上限。
            if remaining<=0 or attempts+self.config.llm_max_retries+1>self.config.overview_max_calls:
                raise TimeoutError("overview budget exhausted")
            try:
                value=await asyncio.wait_for(self.solver.generate_context(question,available,
                         instructions=_INSTRUCTION,payload=payload),timeout=remaining)
                attempts+=value.trace.get("attempts",0)
                calls.append(value.trace)
                return value
            except LLMCallError as exc:
                attempts+=exc.trace.get("attempts",0);calls.append(exc.trace)
                raise
        for group in groups:
            try:
                value=await generate(raw_payload(group),group)
                if value.status not in {"answered","mock"}:
                    raise ValueError("overview output invalid")
                processed.extend(e.citation_number for e in group)
                summaries.append({"text":value.answer,"source_numbers":list(value.cited_numbers)})
            except (LLMCallError,TimeoutError,ValueError) as exc:
                last_error=exc;failed.extend(e.citation_number for e in group)
        failed.extend(e.citation_number for e in skipped)
        if not summaries:
            if isinstance(last_error,LLMCallError): raise last_error
            raise ContextOverflow("概述未完成任何章节，请缩小范围或检查模型及输入预算")
        result=None
        # 短文只生成一次；长文中间要点携带原文编号，最终仍验证原始编号集合。
        if len(summaries)==1:
            text=summaries[0]["text"];numbers=tuple(summaries[0]["source_numbers"])
            result=AnswerResult(text,numbers,"mock" if self.config.llm_mode=="mock" else "answered")
        else:
            level=0
            while len(summaries)>1 and level<5:
                level+=1
                payload=lambda group:{"question":question,"section_summaries":group,
                    "instruction":"汇总各节要点。沿用source_numbers中的原始引用编号，保留所有章节独有信息。"}
                packs,oversized=self.batches(summaries,payload)
                if oversized or not packs:
                    failed.extend(processed);break
                next_level=[]
                for pack in packs:
                    try:
                        value=await generate(payload(pack),evidence)
                        if value.status not in {"answered","mock"}: raise ValueError("invalid reduction")
                        next_level.append({"text":value.answer,"source_numbers":list(value.cited_numbers)})
                    except (LLMCallError,TimeoutError,ValueError):
                        next_level.extend(pack);failed.extend(n for item in pack for n in item["source_numbers"])
                if len(next_level)>=len(summaries):
                    summaries=next_level;break
                summaries=next_level
            if len(summaries)==1:
                result=AnswerResult(summaries[0]["text"],tuple(summaries[0]["source_numbers"]),
                                   "mock" if self.config.llm_mode=="mock" else "answered")
        if result is None:
            result=AnswerResult("\n\n".join(s["text"] for s in summaries),
                    tuple(dict.fromkeys(n for s in summaries for n in s["source_numbers"])),"partial_answer")
        complete=not failed and len(processed)==len(evidence) and result.status!="partial_answer"
        if not complete:
            result=AnswerResult(result.answer+"\n\n本次仅完成部分概述，未覆盖范围请查看运行详情。",
                                result.cited_numbers,"partial_answer")
        self.repo.validate(snapshot)
        selected=[c for c in citations if c["citation_number"] in result.cited_numbers]
        coverage={"complete":complete,"total_chunks":len(evidence),"processed_chunks":len(set(processed)),
                  "missing_citation_numbers":sorted(set(failed)),"source_versions":snapshot["versions"],
                  "processing_coverage":len(set(processed))/len(evidence),"semantic_keypoint_coverage":"requires_review"}
        result=AnswerResult(result.answer,result.cited_numbers,result.status,
                            {"status":"ok" if complete else "partial","mode":self.config.llm_mode,
                             "model":self.config.llm_model,"attempts":attempts,"calls":calls,
                             "latency_ms":round((time.monotonic()-started)*1000,3)})
        if complete and result.status=="answered":
            with self.repo.connect() as db:
                db.execute("INSERT OR REPLACE INTO overview_cache VALUES(?,?)",
                    (key,json.dumps({"result":asdict(result),"citations":selected,"coverage":coverage},ensure_ascii=False)))
        return result,selected,coverage

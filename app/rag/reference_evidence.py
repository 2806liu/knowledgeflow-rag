"""追问来源取证：按历史条目的来源ID重读当前原文，弥补跨语言词法检索遗漏。"""
import json
from app.rag.contracts import EvidenceContext, RetrievalHit
from app.rag.context_builder import ContextOverflow
from app.rag.evidence_expansion import token_counter


class ReferenceEvidenceLoader:
    """历史只提供定位，不复用历史答案或旧snippet充当当前证据。"""
    def __init__(self, store, config):
        self.store,self.config=store,config
        self.method,self.counter=token_counter(config.llm_model)

    def merge(self, resolution, memory, hits):
        """必要锚点优先；限制片段数与统一证据预算，附加召回可以裁剪。"""
        trace={"status":"not_referenced","anchors":[],"skipped":[]}
        turn_id=resolution.get("referenced_turn_id")
        unit_id=resolution.get("referenced_unit_id")
        if not turn_id or not unit_id:
            return list(hits),trace
        turn=next((t for t in memory["turns"] if t["id"]==turn_id),None)
        unit=next((u for u in (turn or {}).get("answer_units",[]) if u["unit_id"]==unit_id),None)
        if unit is None:
            return list(hits),{**trace,"status":"missing_unit"}
        allowed=set(resolution["document_ids"])
        old={c["chunk_id"]:c for c in turn["response"].get("citations",[])}
        requested=list(dict.fromkeys(unit.get("cited_chunk_ids",[])))
        canonical={str(r["id"]):r for r in self.store.chunks_by_ids(tuple(requested))}
        anchors=[];seen=set();current_rows=None
        def normalized(text):
            return " ".join(text.split())
        for cid in requested:
            citation=old.get(cid)
            # 只接受该轮真实引用中、且仍属于当前范围的来源，拒绝历史伪造或越界ID。
            if not citation or citation["document_id"] not in allowed:
                trace["skipped"].append({"chunk_id":cid,"reason":"missing_citation_or_out_of_scope"})
                continue
            row=canonical.get(cid);match="chunk_id"
            if row is None:
                # 重新索引会更换ID；仅对同文档、同位置且正文一致的当前块重新绑定。
                # 正文已改动时绝不拿旧snippet作事实，保留普通当前检索并记录失效。
                if current_rows is None:current_rows=self.store.chunks(tuple(allowed))
                same=[r for r in current_rows if str(r["document_id"])==citation["document_id"]
                      and r["position"]==citation.get("position")
                      and normalized(str(r["text"]))==normalized(citation["snippet"])]
                row=same[0] if len(same)==1 else None
                match="reindexed_identical_content"
            if (row is None or str(row["document_id"]) not in allowed
                    or str(row["document_id"])!=citation["document_id"]
                    or normalized(str(row["text"]))!=normalized(citation["snippet"])):
                trace["skipped"].append({"chunk_id":cid,"reason":"stale_missing_or_changed_source"})
                continue
            current_id=str(row["id"])
            if current_id in seen:continue
            meta=json.loads(row["metadata"] or "{}")
            anchors.append(RetrievalHit(current_id,str(row["document_id"]),str(row["text"]),0.0,
                int(row["position"]),str(row["filename"]),meta.get("page"),
                {**meta,"evidence_role":"referenced_source","retrieval_score":None,
                 "rerank_score":None,"reference_turn_id":turn_id,"reference_unit_id":unit_id}))
            seen.add(current_id)
            trace["anchors"].append({"chunk_id":current_id,"document_id":str(row["document_id"]),
                                     "previous_chunk_id":cid,"match":match})
        def cost(values):
            return self.counter(json.dumps([EvidenceContext(i,h.filename,h.page,h.text).to_dict()
                                            for i,h in enumerate(values,1)],ensure_ascii=False))
        limit=self.config.evidence_max_context_tokens
        # 复用已有6000计数单位证据预算，并限制必要来源最多16块；装不下必须明确失败。
        anchor_count=cost(anchors)
        if len(anchors)>16 or anchor_count>limit:
            raise ContextOverflow("所指条目的必要原文超过证据预算，请明确要展开的子主题",
                                  {"phase":"reference_evidence","token_count_method":self.method,
                                   "input_count":anchor_count,"input_limit":limit,
                                   "component_counts":{"anchor_count":len(anchors),
                                                       "referenced_evidence":anchor_count}})
        merged=list(anchors)
        for hit in hits:
            if hit.document_id not in allowed or hit.chunk_id in seen:continue
            if cost(merged+[hit])>limit:
                trace["skipped"].append({"chunk_id":hit.chunk_id,"reason":"optional_evidence_budget"})
                continue
            merged.append(hit);seen.add(hit.chunk_id)
        trace.update(status="anchored" if anchors else "no_current_anchor",anchor_count=len(anchors),
                     token_count_method=self.method,context_count=cost(merged),context_limit=limit)
        return merged,trace

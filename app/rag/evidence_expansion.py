"""Bounded, canonical neighbor evidence without rewriting citation passages."""
from functools import lru_cache
import json
import re
import time

from app.rag.contracts import EvidenceContext, RetrievalHit, source_dedupe_key


@lru_cache(maxsize=8)
def token_counter(model):
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return encoding.name, lambda text: len(encoding.encode(text, disallowed_special=()))
    except Exception:
        # Proxy/unknown model: a conservative byte budget, never a claimed
        # exact model token count. No remote tokenizer code is executed.
        return "utf8_bytes_conservative_budget", lambda text: len(text.encode("utf-8"))


class EvidenceExpander:
    def __init__(self, store, config, *, counter=None):
        if (not 1 <= config.evidence_neighbor_radius <= 8
                or not 0 <= config.evidence_max_extra_chunks <= 16
                or config.evidence_max_extra_tokens < 0 or config.evidence_max_context_tokens <= 0):
            raise ValueError("invalid evidence expansion limits")
        self.store, self.config = store, config
        self.counter_override = counter

    def expand(self, query, hits, document_ids=None):
        started = time.perf_counter()
        result = list(hits)
        trace = {"enabled": self.config.evidence_expansion_enabled, "status": "disabled",
                 "seed_count": len(hits), "added_count": 0, "additions": [], "skipped": [],
                 "limits": {"radius": self.config.evidence_neighbor_radius,
                    "extra_chunks": self.config.evidence_max_extra_chunks,
                    "extra_tokens": self.config.evidence_max_extra_tokens,
                    "context_tokens": self.config.evidence_max_context_tokens}}

        def finish(status):
            trace.update(status=status, added_count=len(result) - len(hits),
                         latency_ms=round((time.perf_counter() - started) * 1000, 3))
            return result, trace

        if not self.config.evidence_expansion_enabled:
            return finish("disabled")
        if not hits or document_ids == ():
            return finish("skipped_no_seeds")
        method, counter = self.counter_override or token_counter(self.config.llm_model)

        def cost(values):
            payload = [EvidenceContext(i, hit.filename, hit.page, hit.text).to_dict()
                       for i, hit in enumerate(values, 1)]
            return counter(json.dumps(payload, ensure_ascii=False))

        initial = cost(result)
        trace.update(token_count_method=method, initial_context_tokens=initial,
                     final_context_tokens=initial, extra_tokens=0)
        if initial >= self.config.evidence_max_context_tokens:
            return finish("skipped_seed_budget_exceeded")
        if not self.config.evidence_max_extra_chunks or not self.config.evidence_max_extra_tokens:
            return finish("skipped_zero_budget")
        canonical = {str(row["id"]): row for row in self.store.chunks_by_ids(tuple(h.chunk_id for h in hits))}
        seen = {source_dedupe_key(h) for h in hits}
        seen_ids = {h.chunk_id for h in hits}
        enumeration = bool(re.search(r"包括|哪些|哪[一二三四五六七八九十\d]+[个项种类]|分别|列举|步骤|原则|层次|完整|全部|如何|怎么", query))
        for seed in hits:
            if len(result) - len(hits) >= self.config.evidence_max_extra_chunks:
                break
            row = canonical.get(seed.chunk_id)
            if row is None or row["document_id"] != seed.document_id or row["text"] != seed.text:
                trace["skipped"].append({"chunk_id": seed.chunk_id, "reason": "stale_or_missing_seed"})
                continue
            metadata = json.loads(row["metadata"] or "{}")
            section_id = metadata.get("section_id")
            if not section_id:
                trace["skipped"].append({"chunk_id": seed.chunk_id, "reason": "legacy_missing_section_id"})
                continue
            if not (metadata.get("list_part_count", 0) > 1
                    or (enumeration and metadata.get("section_chunk_count", 0) > 1)):
                continue
            neighbors = self.store.section_neighbors(seed.document_id, section_id, int(row["position"]),
                                                     self.config.evidence_neighbor_radius)
            neighbors.sort(key=lambda n: (abs(int(n["position"]) - int(row["position"])),
                                           int(n["position"]) < int(row["position"]), int(n["position"])))
            for neighbor in neighbors:
                if len(result) - len(hits) >= self.config.evidence_max_extra_chunks:
                    break
                doc_id = str(neighbor["document_id"])
                origin = json.loads(neighbor["metadata"] or "{}")
                if (doc_id != seed.document_id or origin.get("section_id") != section_id
                        or (document_ids is not None and doc_id not in document_ids)):
                    continue
                cid = str(neighbor["id"])
                hit = RetrievalHit(cid, doc_id, str(neighbor["text"]), 0.0,
                    int(neighbor["position"]), str(neighbor["filename"]), origin.get("page"),
                    {**origin, "evidence_role": "neighbor", "expanded_from_chunk_id": seed.chunk_id,
                     "retrieval_score": None, "rerank_score": None, "reranker_model": None})
                if cid in seen_ids or source_dedupe_key(hit) in seen:
                    continue
                if metadata.get("list_part_count", 0) > 1 and origin.get("list_group_id") != metadata.get("list_group_id"):
                    continue
                candidate_cost = cost(result + [hit])
                if (candidate_cost > self.config.evidence_max_context_tokens
                        or candidate_cost - initial > self.config.evidence_max_extra_tokens):
                    trace["skipped"].append({"chunk_id": cid, "reason": "token_budget"})
                    continue
                result.append(hit)
                seen_ids.add(cid)
                seen.add(source_dedupe_key(hit))
                trace.update(final_context_tokens=candidate_cost, extra_tokens=candidate_cost - initial)
                trace["additions"].append({"chunk_id": cid, "document_id": doc_id, "page": hit.page,
                    "section_id": section_id, "expanded_from_chunk_id": seed.chunk_id})
        return finish("expanded" if len(result) > len(hits) else "not_needed_or_no_eligible_neighbor")

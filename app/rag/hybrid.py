"""BM25 + dense candidate union with reciprocal-rank fusion."""

from __future__ import annotations

from dataclasses import replace
from copy import copy

from app.rag.contracts import RetrievalHit, source_dedupe_key
from app.rag.retrieval import BM25Retriever
from app.rag.vector_index import QdrantVectorIndex


class HybridRetriever:
    """Apply diversity and final top-K only after the two raw candidate pools fuse."""

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: QdrantVectorIndex,
        *,
        candidate_k: int = 32,
        rrf_k: int = 60,
        max_hits_per_document: int = 2,
    ) -> None:
        if candidate_k <= 0 or rrf_k <= 0 or max_hits_per_document <= 0:
            raise ValueError("candidate_k, rrf_k and max_hits_per_document must be positive")
        self.bm25 = bm25
        self.dense = dense
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.max_hits_per_document = max_hits_per_document

    def search(
        self, query: str, top_k: int = 4, document_ids: tuple[str, ...] | None = None
    ) -> list[RetrievalHit]:
        hits, _trace = self.search_with_trace(query, top_k, document_ids)
        return hits

    def search_candidates(self, query: str, candidate_k: int = 32, document_ids=None):
        """Fuse raw pools before the final cap; do not mutate shared settings."""
        candidate_retriever = copy(self)
        candidate_retriever.candidate_k = candidate_k
        candidate_retriever.max_hits_per_document = max(1, candidate_k * 2)
        hits, _ = candidate_retriever.search_with_trace(query, candidate_k, document_ids)
        return hits

    def search_with_trace(
        self, query: str, top_k: int = 4, document_ids: tuple[str, ...] | None = None
    ) -> tuple[list[RetrievalHit], dict]:
        if top_k <= 0 or document_ids == ():
            return [], {"bm25": [], "dense": [], "fused": []}
        sparse = self.bm25.search_candidates(query, self.candidate_k, document_ids)
        dense = self.dense.search_candidates(query, self.candidate_k, document_ids)
        by_id: dict[str, RetrievalHit] = {}
        ranks: dict[str, dict[str, int]] = {}
        for source, pool in (("bm25", sparse), ("dense", dense)):
            for rank, hit in enumerate(pool, 1):
                by_id.setdefault(hit.chunk_id, hit)
                ranks.setdefault(hit.chunk_id, {})[source] = rank

        fused: list[RetrievalHit] = []
        for chunk_id, hit in by_id.items():
            source_ranks = ranks[chunk_id]
            score = sum(1 / (self.rrf_k + rank) for rank in source_ranks.values())
            fused.append(replace(
                hit,
                score=round(score, 6),
                metadata={
                    **hit.metadata,
                    "retriever_version": "hybrid_rrf_v1",
                    "bm25_rank": source_ranks.get("bm25"),
                    "dense_rank": source_ranks.get("dense"),
                    "rrf_k": self.rrf_k,
                },
            ))
        fused.sort(key=lambda hit: (
            -hit.score,
            min(ranks[hit.chunk_id].values()),
            hit.chunk_id,
        ))

        selected: list[RetrievalHit] = []
        seen_sources: set[tuple[str, int | None, str]] = set()
        document_counts: dict[str, int] = {}
        fused_trace: list[dict] = []
        eligible_rank = 0
        for fusion_rank, hit in enumerate(fused, 1):
            source_key = source_dedupe_key(hit)
            reason = "selected"
            if source_key in seen_sources:
                reason = "duplicate_source_text"
            elif document_counts.get(hit.document_id, 0) >= self.max_hits_per_document:
                reason = "document_cap"
            else:
                seen_sources.add(source_key)
                document_counts[hit.document_id] = document_counts.get(hit.document_id, 0) + 1
                eligible_rank += 1
                if eligible_rank <= top_k:
                    selected.append(hit)
                else:
                    reason = "top_k"
            fused_trace.append({
                "chunk_id": hit.chunk_id,
                "document_id": hit.document_id,
                "page": hit.page,
                "fusion_rank": fusion_rank,
                "eligible_rank": eligible_rank if reason in {"selected", "top_k"} else None,
                "bm25_rank": hit.metadata["bm25_rank"],
                "dense_rank": hit.metadata["dense_rank"],
                "rrf_score": hit.score,
                "selection": reason,
            })
        return selected, {
            "bm25": [hit.chunk_id for hit in sparse],
            "dense": [hit.chunk_id for hit in dense],
            "fused": fused_trace,
        }

"""Explain why a labeled source did not survive lexical candidate selection."""

from __future__ import annotations

import json

from app.rag.retrieval import BM25Retriever, BalancedLexicalRetriever, LexicalRetriever, tokens
from evaluation.corpus import QueryCase, comparable_text
from evaluation.metrics import matches


def _relevant(record: dict, case: QueryCase) -> bool:
    metadata = json.loads(record.get("metadata") or "{}")
    return any(
        record["document_id"] == evidence.document_id
        and metadata.get("page") == evidence.page
        and comparable_text(evidence.anchor) in comparable_text(record["text"])
        for evidence in case.evidence
    )


def diagnose_case(
    case: QueryCase,
    records: list[dict],
    retriever: LexicalRetriever,
    hits: list,
    top_k: int,
) -> dict:
    """Diagnose a miss without changing the production search result."""

    relevant = [record for record in records if _relevant(record, case)]
    if not relevant:
        return {"reason": "no_chunk_contains_anchor", "relevant_chunk_count": 0}

    query_terms = set(tokens(case.query))
    overlap_by_id = {
        record["id"]: len(query_terms & set(tokens(record["text"]))) for record in records
    }
    relevant_coverages = [
        overlap_by_id[record["id"]] / len(query_terms) if query_terms else 0.0
        for record in relevant
    ]
    base = {
        "relevant_chunk_count": len(relevant),
        "best_relevant_query_coverage": round(max(relevant_coverages), 4),
        "relevant_positions": [record["position"] for record in relevant],
    }
    if any(any(matches(hit, evidence) for evidence in case.evidence) for hit in hits):
        return {"reason": "hit", **base}

    if isinstance(retriever, BM25Retriever):
        eligible = [record for record in relevant if overlap_by_id[record["id"]] > 0]
        if not eligible:
            return {"reason": "insufficient_lexical_overlap", **base}
    elif isinstance(retriever, BalancedLexicalRetriever):
        eligible = [record for record in relevant if overlap_by_id[record["id"]] >= (1 if len(query_terms) == 1 else 2)]
        if not eligible:
            return {"reason": "insufficient_lexical_overlap", **base}
    else:
        strict_exists = any(
            overlap > 0
            and (len(query_terms) == 1 or overlap / len(query_terms) >= retriever.min_query_coverage)
            for overlap in overlap_by_id.values()
        ) if query_terms else False
        eligible = [
            record
            for record in relevant
            if overlap_by_id[record["id"]] > 0
            and (
                (len(query_terms) == 1 or overlap_by_id[record["id"]] / len(query_terms) >= retriever.min_query_coverage)
                if strict_exists
                else overlap_by_id[record["id"]] >= 2
            )
        ]
        if not eligible:
            reason = (
                "filtered_by_query_coverage"
                if strict_exists and any(overlap_by_id[record["id"]] >= 2 for record in relevant)
                else "insufficient_lexical_overlap"
            )
            return {"reason": reason, **base}

    # Remove only the per-document cap to locate the correct candidate's raw
    # rank. This distinguishes an early cap from the final top-K cutoff.
    if isinstance(retriever, BM25Retriever):
        uncapped_retriever = BM25Retriever(
            retriever.store, max_hits_per_document=max(len(records), 1), k1=retriever.k1, b=retriever.b,
            idf_exponent=retriever.idf_exponent,
        )
    else:
        uncapped_retriever = type(retriever)(
            retriever.store,
            min_query_coverage=retriever.min_query_coverage,
            max_hits_per_document=max(len(records), 1),
        )
    uncapped = uncapped_retriever.search(case.query, top_k=max(len(records), 1))
    relevant_rank = next(
        (rank for rank, hit in enumerate(uncapped, 1) if any(matches(hit, evidence) for evidence in case.evidence)),
        None,
    )
    if relevant_rank is None:
        return {"reason": "deduplicated_or_unavailable", **base}
    relevant_hit = uncapped[relevant_rank - 1]
    prior_from_document = sum(
        hit.document_id == relevant_hit.document_id for hit in uncapped[: relevant_rank - 1]
    )
    beyond_cap = prior_from_document >= retriever.max_hits_per_document
    beyond_top_k = relevant_rank > top_k
    if beyond_cap and beyond_top_k:
        reason = "ranked_beyond_document_cap_and_top_k"
    elif beyond_cap:
        reason = "ranked_beyond_document_cap"
    elif beyond_top_k:
        reason = "ranked_beyond_top_k"
    else:
        reason = "deduplicated_or_unavailable"
    return {
        "reason": reason,
        "uncapped_rank": relevant_rank,
        "same_document_candidates_before": prior_from_document,
        **base,
    }


def diagnose_hybrid_case(case: QueryCase, records: list[dict], trace: dict) -> dict:
    """Separate candidate recall from fusion and final-selection failures."""

    relevant_ids = {record["id"] for record in records if _relevant(record, case)}
    if not relevant_ids:
        return {"reason": "no_chunk_contains_anchor", "relevant_chunk_count": 0}
    bm25 = {chunk_id: rank for rank, chunk_id in enumerate(trace["bm25"], 1)}
    dense = {chunk_id: rank for rank, chunk_id in enumerate(trace["dense"], 1)}
    fused = {item["chunk_id"]: item for item in trace["fused"]}
    relevant_fused = sorted(
        (fused[chunk_id] for chunk_id in relevant_ids if chunk_id in fused),
        key=lambda item: item["fusion_rank"],
    )
    best = next((item for item in relevant_fused if item["selection"] == "selected"), None)
    if best is None and relevant_fused:
        best = relevant_fused[0]
    if best is None:
        reason = "not_in_candidate_pools"
    else:
        reason = {
            "selected": "hit",
            "document_cap": "excluded_after_fusion_by_document_cap",
            "duplicate_text": "excluded_after_fusion_as_duplicate",
            "top_k": "ranked_beyond_final_top_k",
        }[best["selection"]]
    return {
        "reason": reason,
        "relevant_chunk_count": len(relevant_ids),
        "candidate_in_bm25": any(chunk_id in bm25 for chunk_id in relevant_ids),
        "candidate_in_dense": any(chunk_id in dense for chunk_id in relevant_ids),
        "best_bm25_rank": min((bm25[chunk_id] for chunk_id in relevant_ids if chunk_id in bm25), default=None),
        "best_dense_rank": min((dense[chunk_id] for chunk_id in relevant_ids if chunk_id in dense), default=None),
        "best_fusion_rank": best["fusion_rank"] if best else None,
        "best_eligible_rank": best["eligible_rank"] if best else None,
        "selection": best["selection"] if best else None,
    }

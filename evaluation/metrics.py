"""Evidence-level ranking metrics for a frozen retrieval query set."""

from __future__ import annotations

import math
from statistics import mean

from app.rag.contracts import RetrievalHit
from evaluation.corpus import Evidence, QueryCase, comparable_text


def matches(hit: RetrievalHit, evidence: Evidence) -> bool:
    return (
        hit.document_id == evidence.document_id
        and hit.page == evidence.page
        and comparable_text(evidence.anchor) in comparable_text(hit.text)
    )


def score_case(case: QueryCase, hits: list[RetrievalHit], ks: tuple[int, ...]) -> dict:
    seen: set[int] = set()
    grades: list[int] = []
    coverage_by_rank: list[int] = []
    first_relevant_rank: int | None = None
    ranked_hits: list[dict] = []
    for rank, hit in enumerate(hits, 1):
        matched = [index for index, evidence in enumerate(case.evidence) if matches(hit, evidence)]
        new_matches = [index for index in matched if index not in seen]
        if new_matches and first_relevant_rank is None:
            first_relevant_rank = rank
        grade = max((case.evidence[index].relevance for index in new_matches), default=0)
        grades.append(grade)
        seen.update(new_matches)
        coverage_by_rank.append(len(seen))
        ranked_hits.append(
            {
                "rank": rank,
                "document_id": hit.document_id,
                "page": hit.page,
                "position": hit.position,
                "score": hit.score,
                "matched_evidence": matched,
                "snippet": hit.text,
            }
        )

    metrics: dict[str, float] = {}
    ideal = sorted((item.relevance for item in case.evidence), reverse=True)
    for k in ks:
        matched_count = coverage_by_rank[min(k, len(hits)) - 1] if hits else 0
        metrics[f"Recall@{k}"] = round(matched_count / len(case.evidence), 4)
        metrics[f"MRR@{k}"] = round(
            1 / first_relevant_rank if first_relevant_rank is not None and first_relevant_rank <= k else 0.0,
            4,
        )
        dcg = sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades[:k]))
        ideal_dcg = sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(ideal[:k]))
        metrics[f"nDCG@{k}"] = round(dcg / ideal_dcg if ideal_dcg else 0.0, 4)
    return {
        "id": case.case_id,
        "query": case.query,
        "first_relevant_rank": first_relevant_rank,
        "metrics": metrics,
        "hits": ranked_hits,
    }


def macro_average(results: list[dict]) -> dict[str, float]:
    if not results:
        return {}
    return {
        name: round(mean(result["metrics"][name] for result in results), 4)
        for name in results[0]["metrics"]
    }

"""Evaluate the live /chat retrieval and citation path against frozen labels.

This command sends real chat requests. It adds uniquely named evaluation
sessions and runs, but never modifies documents or indexes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import httpx

from evaluation.corpus import Evidence, comparable_text, load_cases, load_documents, load_manifest, validate_evidence
from evaluation.stored_index import load_active_chunks


@dataclass(frozen=True)
class AnswerCase:
    case_id: str
    category: str
    query: str
    scope_document_ids: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    source_case_id: str | None = None
    reference_answer: str = ""
    required_points: tuple[str, ...] = ()
    label_review_status: str = "unknown"


def load_suite(path: Path, db_path: Path) -> tuple[list[AnswerCase], dict[str, str]]:
    """Resolve positive labels from existing sets; validate pages and hashes."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    if raw.get("freeze"):
        freeze_path = base / raw["freeze"]
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        for filename, expected_hash in frozen["files_sha256"].items():
            if hashlib.sha256((base / filename).read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"frozen evaluation source changed: {filename}")
    cases: list[AnswerCase] = []
    source_hashes: dict[str, str] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}
    known: dict[str, Any] = {}
    seen_ids: set[str] = set()
    for group in raw["groups"]:
        manifest_path = base / group["manifest"]
        query_path = base / group["queries"]
        manifest = load_manifest(manifest_path)
        labeled = load_cases(query_path)
        metadata = {item["id"]: item for item in
                    (json.loads(line) for line in query_path.read_text(encoding="utf-8").splitlines() if line.strip())}
        validate_evidence(labeled, load_documents(db_path, manifest))
        for source in (manifest_path, query_path):
            source_hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        by_id = {case.case_id: case for case in labeled}
        ids = list(by_id) if group["case_ids"] == "all" else group["case_ids"]
        for case_id in ids:
            if case_id not in by_id:
                raise ValueError(f"unknown labeled case: {case_id}")
            labeled_case = by_id[case_id]
            if case_id in known or case_id in seen_ids:
                raise ValueError(f"duplicate case ID: {case_id}")
            known[case_id] = labeled_case
            seen_ids.add(case_id)
            cases.append(AnswerCase(case_id, group["category"], labeled_case.query,
                                    labeled_case.scope_document_ids, labeled_case.evidence, case_id,
                                    metadata[case_id].get("reference_answer", ""),
                                    tuple(metadata[case_id].get("required_points", [])),
                                    metadata[case_id].get("label_review_status", "unknown")))
    for extra in raw["additional_cases"]:
        case_id = extra["id"]
        if case_id in seen_ids:
            raise ValueError(f"duplicate case ID: {case_id}")
        seen_ids.add(case_id)
        source_id = extra.get("source_case_id")
        if source_id:
            if source_id not in known:
                raise ValueError(f"unknown source case: {source_id}")
            source = known[source_id]
            evidence = source.evidence
            scope = source.scope_document_ids
        else:
            if extra["category"] != "unanswerable":
                raise ValueError(f"case {case_id} needs an existing labeled source")
            evidence = ()
            scope = tuple(extra["scope_document_ids"])
            if not scope:
                raise ValueError(f"case {case_id} needs a document scope")
            # A negative label cannot be proven by an anchor. At least ensure
            # its stated scope still consists of active, frozen documents.
            manifest_path = base / extra["manifest"]
            manifest = load_manifest(manifest_path)
            load_documents(db_path, manifest)
            manifest_ids = {item["document_id"] for item in manifest["documents"]}
            if not set(scope) <= manifest_ids:
                raise ValueError(f"case {case_id} has a document outside its manifest")
            source_hashes[str(manifest_path)] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        cases.append(AnswerCase(case_id, extra["category"], extra["query"], scope, evidence, source_id,
                                extra.get("reference_answer", ""), tuple(extra.get("required_points", [])),
                                extra.get("label_review_status", "unknown")))
    if raw.get("freeze"):
        source_hashes[str(freeze_path)] = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    return cases, source_hashes


def matching_labels(item: dict[str, Any], evidence: tuple[Evidence, ...]) -> list[int]:
    return [index for index, label in enumerate(evidence) if
            item.get("document_id") == label.document_id
            and item.get("page") == label.page
            and comparable_text(label.anchor) in comparable_text(item.get("snippet") or "")]


def assess(case: AnswerCase, response: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    tool_event = next((event for event in run["events"] if event.get("tool") == "retrieve_knowledge"), {})
    solver_event = next((event for event in run["events"] if event.get("node") == "solver"), {})
    hits = tool_event.get("citations", [])
    citations = response.get("citations", [])
    first_rank = next((rank for rank, hit in enumerate(hits, 1) if matching_labels(hit, case.evidence)), None)
    retrieval_coverage = {
        f"Recall@{k}": round(len({index for hit in hits[:k] for index in matching_labels(hit, case.evidence)})
                              / len(case.evidence), 4)
        for k in (1, 2, 4)
    } if case.evidence else None
    matched_labels = {index for citation in citations for index in matching_labels(citation, case.evidence)}
    available = {index + 1: hit["chunk_id"] for index, hit in enumerate(hits)}
    answer_markers = {int(number) for number in re.findall(r"\[(\d+)\]", response.get("answer") or "")}
    citation_numbers_valid = all(
        citation.get("citation_number") in available
        and citation.get("chunk_id") == available[citation["citation_number"]]
        for citation in citations
    ) and answer_markers == {citation.get("citation_number") for citation in citations}
    cited_anchor_count = sum(bool(matching_labels(citation, case.evidence)) for citation in citations)
    review_flags: list[str] = []
    if case.evidence:
        if first_rank is None:
            failure = "retrieval_miss"
        elif response.get("answer_status") != "answered":
            failure = "solver_did_not_answer"
        elif not citations or not citation_numbers_valid:
            failure = "missing_or_invalid_citation"
        elif not matched_labels:
            failure = "answer_cites_no_labeled_anchor"
        else:
            failure = None
            if cited_anchor_count != len(citations):
                review_flags.append("additional_citations_not_in_gold")
    else:
        failure = None if response.get("answer_status") in {"no_evidence", "insufficient_evidence"} and not citations else "failed_to_refuse"
    return {
        "id": case.case_id,
        "category": case.category,
        "source_case_id": case.source_case_id,
        "query": case.query,
        "scope_document_ids": list(case.scope_document_ids),
        "expected": "answer_with_labeled_evidence" if case.evidence else "refuse",
        "reference_answer": case.reference_answer,
        "required_points": list(case.required_points),
        "label_review_status": case.label_review_status,
        "gold_evidence": [vars(label) for label in case.evidence],
        "run_id": response.get("run_id"),
        "solver_mode": solver_event.get("mode"),
        "answer_status": response.get("answer_status"),
        "answer": response.get("answer"),
        "retrieved_count": len(hits),
        "retrieved_hits": [
            {"rank": rank, **hit, "matches_labeled_anchor": bool(matching_labels(hit, case.evidence))}
            for rank, hit in enumerate(hits, 1)
        ],
        "first_relevant_rank": first_rank,
        "retrieval": retrieval_coverage,
        "reciprocal_rank_at_4": round(1 / first_rank, 4) if first_rank and first_rank <= 4 else 0.0 if case.evidence else None,
        "citations": citations,
        "citation_numbers_valid": citation_numbers_valid,
        "cited_labeled_anchor_count": cited_anchor_count,
        "cited_label_coverage": round(len(matched_labels) / len(case.evidence), 4) if case.evidence else None,
        "failure": failure,
        "review_flags": review_flags,
        "requires_semantic_review": bool(case.evidence and response.get("answer_status") == "answered"),
    }


def aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [case for case in cases if case["expected"] == "answer_with_labeled_evidence" and "error" not in case]
    negatives = [case for case in cases if case["expected"] == "refuse" and "error" not in case]
    return {
        "total": len(cases),
        "completed": sum("error" not in case for case in cases),
        "request_errors": sum("error" in case for case in cases),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "retrieval": {name: round(mean(case["retrieval"][name] for case in positives), 4) if positives else None
                      for name in ("Recall@1", "Recall@2", "Recall@4")},
        "MRR@4": round(mean(case["reciprocal_rank_at_4"] for case in positives), 4) if positives else None,
        "positive_with_labeled_citation_rate": round(mean(case["cited_labeled_anchor_count"] > 0 for case in positives), 4) if positives else None,
        "negative_refusal_rate": round(mean(case["failure"] is None for case in negatives), 4) if negatives else None,
        "failure_counts": {reason: sum(case.get("failure") == reason for case in cases) for reason in
                           sorted({case.get("failure") for case in cases if case.get("failure")})},
        "review_flag_counts": {flag: sum(flag in case.get("review_flags", []) for case in cases) for flag in
                               sorted({flag for case in cases for flag in case.get("review_flags", [])})},
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    cases, source_hashes = load_suite(args.suite, args.data_dir / "studyflow.db")
    index_rows, active_documents, index_hash = load_active_chunks(args.data_dir / "studyflow.db")
    snapshot = {row["id"]: row for row in index_rows}
    results: list[dict[str, Any]] = []
    batch_id = uuid.uuid4().hex[:12]
    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=args.timeout) as client:
        health = client.get("/health")
        health.raise_for_status()
        for case in cases:
            session_id = f"answer-eval-{batch_id}-{case.case_id}"
            try:
                response_http = client.post("/chat", json={
                    "session_id": session_id,
                    "message": case.query,
                    "document_ids": list(case.scope_document_ids),
                    "stream": False,
                })
                response_http.raise_for_status()
                response = response_http.json()
                run_http = client.get(f"/runs/{response['run_id']}")
                run_http.raise_for_status()
                result = assess(case, response, run_http.json())
                for hit in result["retrieved_hits"]:
                    row = snapshot.get(hit["chunk_id"])
                    if row is None or row["document_id"] != hit["document_id"] or comparable_text(row["text"]) != comparable_text(hit["snippet"]):
                        raise ValueError("live service evidence differs from the local index snapshot")
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                result = {"id": case.case_id, "category": case.category, "query": case.query,
                          "expected": "answer_with_labeled_evidence" if case.evidence else "refuse",
                          "reference_answer": case.reference_answer, "required_points": list(case.required_points),
                          "gold_evidence": [vars(label) for label in case.evidence],
                          "error": f"{type(exc).__name__}: {exc}"}
            if result.get("solver_mode") == "mock":
                raise RuntimeError("answer evaluation requires a real LLM mode; no report was written")
            results.append(result)
            print(f"{case.case_id}: {result.get('failure') or result.get('error') or 'ok'}", flush=True)
            # Save partial output independently so a dropped API connection does
            # not erase already-paid model outputs. It is never a final report.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_suffix(".partial.json").write_text(
                json.dumps({"status": "partial", "batch_id": batch_id, "source_sha256": source_hashes,
                            "index_sha256": index_hash, "cases": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    if load_active_chunks(args.data_dir / "studyflow.db")[2] != index_hash:
        raise RuntimeError("index changed during evaluation; partial results saved, final report rejected")
    if load_suite(args.suite, args.data_dir / "studyflow.db")[1] != source_hashes:
        raise RuntimeError("labels changed during evaluation; partial results saved, final report rejected")
    categories = sorted({case.category for case in cases})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": suite["name"],
        "suite_status": suite.get("status", "unknown"),
        "source_sha256": source_hashes,
        "index_sha256": index_hash,
        "active_document_count": active_documents,
        "base_url": args.base_url,
        "batch_id": batch_id,
        "runtime": _public_runtime(),
        "runtime_provenance": "Expected parameters read from local configuration; solver mode is also checked per live run. Other server parameters require independent verification.",
        "measurement_limits": [
            "Anchor matching checks source text, page, and document; it does not prove semantic correctness of the answer.",
            "Unanswerable cases are draft human labels; review the complete scoped documents before treating refusal rate as final.",
        ],
        "aggregate": aggregate(results),
        "by_category": {category: aggregate([item for item in results if item["category"] == category]) for category in categories},
        "bad_cases": [item["id"] for item in results if item.get("failure") or item.get("error")],
        "review_cases": [item["id"] for item in results if item.get("review_flags")],
        "cases": results,
    }


def _public_runtime() -> dict[str, Any]:
    """Only public evaluation parameters; keys and endpoint credentials stay private."""
    from app.config import settings
    return {name: getattr(settings, name) for name in
            ("retrieval_strategy", "llm_mode", "llm_model", "embedding_model", "qdrant_collection")}


def regrade_report(path: Path) -> dict[str, Any]:
    """Reapply citation classification to saved model outputs without new API calls."""

    report = json.loads(path.read_text(encoding="utf-8"))
    for case in report["cases"]:
        if case.get("failure") == "answer_also_cites_unlabeled_evidence":
            case["failure"] = None
            case["review_flags"] = ["additional_citations_not_in_gold"]
        else:
            case.setdefault("review_flags", [])
    report["aggregate"] = aggregate(report["cases"])
    report["by_category"] = {
        category: aggregate([item for item in report["cases"] if item["category"] == category])
        for category in sorted({item["category"] for item in report["cases"]})
    }
    report["bad_cases"] = [item["id"] for item in report["cases"] if item.get("failure") or item.get("error")]
    report["review_cases"] = [item["id"] for item in report["cases"] if item.get("review_flags")]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate live /chat retrieval, refusal, and citations")
    parser.add_argument("--data-dir", type=Path, default=Path(".studyflow"))
    parser.add_argument("--suite", type=Path, default=Path("evaluation/answer-validation-suite.json"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/answer-validation.json"))
    parser.add_argument("--regrade", type=Path, help="Reclassify a saved report without calling the LLM again")
    args = parser.parse_args()
    report = regrade_report(args.regrade) if args.regrade else evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    print(f"Detailed report: {args.output}")


if __name__ == "__main__":
    main()

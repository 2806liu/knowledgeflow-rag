"""Reproduce list/section fixes on a real indexed document and fixed dev set.

--reindex explicitly rebuilds the selected document and syncs its vectors.
Both modes create real evaluation sessions/runs through the running API.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid

import httpx
from app.config import settings
from evaluation.corpus import comparable_text, load_cases
from evaluation.stored_index import evaluate, load_active_chunks

DOCUMENT = "8b85fa42-bfda-4f0b-b5af-647cfa866829"
POINTS = ("底线责任", "预防责任", "恢复责任", "超越责任")


def retrieval_report():
    return evaluate(argparse.Namespace(data_dir=settings.data_dir,
        manifest=Path("evaluation/independent-v1/corpus.json"),
        queries=Path("evaluation/independent-v1/queries.jsonl"), ks="1,2,4",
        min_query_coverage=settings.retrieval_min_query_coverage,
        max_hits_per_document=settings.retrieval_bm25_max_hits_per_document,
        hybrid_candidate_k=settings.retrieval_hybrid_candidate_k,
        hybrid_rrf_k=settings.retrieval_hybrid_rrf_k, retriever="bm25_idf_v2", scope_mode="label"))


def write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/section-v3-validation.json"))
    args = parser.parse_args()
    result_stem = args.output.stem.removesuffix("-validation")
    db = settings.data_dir / "studyflow.db"
    rows, count, before_hash = load_active_chunks(db)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "dataset_status": "development_regression",
              "reindexed": args.reindex, "document_id": DOCUMENT, "active_documents": count,
              "chunks_before": len(rows), "index_sha256_before": before_hash,
              "document_chunks_before": [r for r in rows if r["document_id"] == DOCUMENT], "bad_cases": []}
    frozen_other = {r["id"]: r for r in rows if r["document_id"] != DOCUMENT}
    before = retrieval_report()
    write(args.output.with_name(f"{result_stem}-before-retrieval.json") if args.reindex
          else args.output.with_name(f"{result_stem}-current-retrieval.json"), before)
    report["retrieval_before"] = before["retrievers"]["bm25_idf_v2"]["aggregate"]
    with httpx.Client(base_url=args.api_url, timeout=180) as client:
        client.get("/health").raise_for_status()
        schema = client.get("/openapi.json").json()
        if "section_id" not in schema["components"]["schemas"]["Citation"]["properties"]:
            raise ValueError("running service is stale; restart before validation")
        response = client.get("/documents/" + DOCUMENT)
        response.raise_for_status()
        report["document_before"] = response.json()
        if args.reindex:
            backup = settings.data_dir / "backups" / ("before-section-v3-" + uuid.uuid4().hex + ".db")
            backup.parent.mkdir(parents=True, exist_ok=True)
            source = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
            target = sqlite3.connect(backup)
            try:
                source.backup(target)
            finally:
                target.close()
                source.close()
            report["sqlite_backup"] = str(backup.resolve())
            response = client.post("/documents/" + DOCUMENT + "/reindex")
            response.raise_for_status()
            report["reindex_response"] = response.json()
        response = client.get("/documents/" + DOCUMENT)
        response.raise_for_status()
        report["document_after"] = response.json()
        current, _, after_hash = load_active_chunks(db)
        report.update(chunks_after=len(current), index_sha256_after=after_hash,
                      document_chunks_after=[r for r in current if r["document_id"] == DOCUMENT])
        report["other_chunks_unchanged"] = frozen_other == {r["id"]: r for r in current if r["document_id"] != DOCUMENT}
        report["source_hash_unchanged"] = report["document_before"]["content_hash"] == report["document_after"]["content_hash"]
        relevant = [r for r in current if r["document_id"] == DOCUMENT and all(p in r["text"] for p in POINTS)]
        report["four_points_in_one_chunk"] = bool(relevant)
        if not relevant or not report["other_chunks_unchanged"] or not report["source_hash_unchanged"]:
            raise AssertionError("list completeness or document isolation check failed")
        after = retrieval_report()
        write(args.output.with_name(f"{result_stem}-after-retrieval.json"), after)
        report["retrieval_after"] = after["retrievers"]["bm25_idf_v2"]["aggregate"]
        before_by_id = {c["id"]: c for c in before["retrievers"]["bm25_idf_v2"]["cases"]}
        for case in after["retrievers"]["bm25_idf_v2"]["cases"]:
            deltas = {key: round(value - before_by_id[case["id"]]["metrics"][key], 4) for key, value in case["metrics"].items()}
            if any(value < 0 for value in deltas.values()):
                report["bad_cases"].append({"id": case["id"], "kind": "retrieval_regression", "metric_delta": deltas})
        cases = load_cases(Path("evaluation/independent-v1/queries.jsonl"))
        live_cases = [(next(c for c in cases if c.case_id == name), name == "iv-r09") for name in ("iv-r09", "iv-x404")]
        report["live_checks"] = []
        for case, list_case in live_cases:
            response = client.post("/chat", json={"session_id": "section-v3-" + uuid.uuid4().hex,
                "message": case.query, "document_ids": list(case.scope_document_ids), "stream": False})
            response.raise_for_status()
            body = response.json()
            run_response = client.get("/runs/" + body["run_id"])
            run_response.raise_for_status()
            run = run_response.json()
            expander = next(e["output"] for e in run["events"] if e["node"] == "evidence_expansion")
            canonical = {r["id"]: r for r in current}
            citation_ok = all(c["snippet"] == canonical[c["chunk_id"]]["text"]
                and c["document_id"] == canonical[c["chunk_id"]]["document_id"]
                and c["filename"] == canonical[c["chunk_id"]]["filename"]
                and c["page"] == json.loads(canonical[c["chunk_id"]]["metadata"])["page"] for c in body["citations"])
            if list_case:
                points_ok = all(p in body["answer"] for p in POINTS)
                source_ok = any(c["document_id"] == DOCUMENT and c["page"] == 3
                                and all(p in c["snippet"] for p in POINTS) for c in body["citations"])
            else:
                points_ok = "学号-姓名-报告/程序" in comparable_text(body["answer"])
                source_ok = any(c["document_id"] == "363c87f1-195d-4cb8-b9a8-54ca1a2e1153" and c["page"] == 6
                                for c in body["citations"])
            passed = bool(points_ok and source_ok and citation_ok and body["answer_status"] == "answered")
            report["live_checks"].append({"id": case.case_id, "passed_mechanical_checks": passed,
                "points_check": points_ok, "supporting_source_check": source_ok,
                "canonical_citations_check": citation_ok, "response": body, "run": run,
                "expansion": expander, "semantic_review": "pending_assistant_review"})
            if not passed:
                report["bad_cases"].append({"id": case.case_id, "kind": "live_check_failed"})
            write(args.output, report)
            print(json.dumps({"id": case.case_id, "passed": passed, "answer_status": body["answer_status"],
                              "expansion_status": expander["status"]}), flush=True)
        report["index_unchanged_during_queries"] = load_active_chunks(db)[2] == after_hash
        report["passed"] = not report["bad_cases"] and report["index_unchanged_during_queries"]
        write(args.output, report)
        print(json.dumps({"passed": report["passed"], "before": report["retrieval_before"],
                          "after": report["retrieval_after"], "bad_cases": report["bad_cases"]}), flush=True)


if __name__ == "__main__":
    main()

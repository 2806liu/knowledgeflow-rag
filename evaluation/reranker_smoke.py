"""Actual FastAPI + local cross-encoder + configured LLM development smoke.

Writes evaluation sessions/runs, but never reparses or reindexes documents.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import uuid

from fastapi.testclient import TestClient
from app import main as api
from app.config import settings
from app.services import KnowledgeFlow
from evaluation.corpus import comparable_text, load_cases
from evaluation.stored_index import load_active_chunks


def evaluate(output):
    if settings.llm_mode == "mock":
        raise ValueError("this smoke requires a configured real LLM")
    fixed = next(c for c in load_cases(Path("evaluation/independent-v1/queries.jsonl")) if c.case_id == "iv-x404")
    cases = [
        {"id": fixed.case_id, "query": fixed.query, "scope": fixed.scope_document_ids,
         "points": ["学号-姓名-报告/程序"], "evidence_doc": "363c87f1-195d-4cb8-b9a8-54ca1a2e1153", "page": 6},
        {"id": "mixed-principles", "query": "可持续发展战略的原则是什么？",
         "scope": ("f92cb1f3-668e-48e6-b639-b05de6e70e36",),
         "points": ["公平性", "持续性", "共同性"], "page": 4},
        {"id": "image-workflow", "query": "Plan-and-Solve 的计划由谁来编排？",
         "scope": ("9a5ec049-3832-4990-bea3-05f946ef22ac",), "points": ["框架", "代码"], "page": 1},
        {"id": "no-match", "query": "ZXQ987654321_UNSEEN_TOKEN", "scope": fixed.scope_document_ids, "negative": True},
    ]
    rows, _, before = load_active_chunks(api.store.db_path)
    canonical = {r["id"]: r for r in rows}
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "status": "development_smoke",
              "llm_mode": settings.llm_mode, "llm_model": settings.llm_model,
              "retrieval_strategy": settings.retrieval_strategy, "index_sha256_before": before,
              "checks_are": "mechanical source/point checks, assistant semantic review required",
              "cases": [], "bad_cases": []}
    original = api.flow
    try:
        with TestClient(api.app) as client:
            for enabled in (False, True):
                config = replace(settings, reranker_enabled=enabled, reranker_failure_policy="error")
                api.flow = KnowledgeFlow(api.store, config)
                for case in cases:
                    started = time.perf_counter()
                    response = client.post("/chat", json={"session_id": f"reranker-smoke-{uuid.uuid4()}",
                        "message": case["query"], "document_ids": list(case["scope"]), "stream": False})
                    response.raise_for_status()
                    body = response.json()
                    run = client.get("/runs/" + body["run_id"]).json()
                    trace = next(e["output"] for e in run["events"] if e["node"] == "reranker")
                    source_ok = True
                    for citation in body["citations"]:
                        row = canonical[citation["chunk_id"]]
                        meta = json.loads(row["metadata"])
                        source_ok &= (citation["document_id"] == row["document_id"] and citation["filename"] == row["filename"]
                                      and citation["page"] == meta.get("page") and citation["snippet"] == row["text"])
                    numbered = {int(x) for x in re.findall(r"\[(\d+)\]", body["answer"])}
                    citations_ok = numbered == {c["citation_number"] for c in body["citations"]}
                    if case.get("negative"):
                        passed = body["answer_status"] == "no_evidence" and not body["citations"] and trace["status"] in {"disabled", "skipped_no_candidates"}
                    else:
                        normalized = comparable_text(body["answer"])
                        points_ok = all(comparable_text(p) in normalized for p in case["points"])
                        supporting_source = any(c["page"] == case["page"] and c["document_id"] == case.get("evidence_doc", case["scope"][0])
                                                for c in body["citations"])
                        passed = body["answer_status"] == "answered" and points_ok and supporting_source
                        if enabled:
                            passed &= trace["status"] == "ok" and all(c["rerank_score"] is not None for c in body["citations"])
                    passed &= source_ok and citations_ok
                    record = {"id": case["id"], "query": case["query"], "enabled": enabled,
                        "passed_mechanical_checks": bool(passed), "source_check": bool(source_ok),
                        "citation_numbers_check": citations_ok, "response": body, "run": run,
                        "end_to_end_ms": round((time.perf_counter() - started) * 1000, 3)}
                    report["cases"].append(record)
                    if not passed:
                        report["bad_cases"].append({"id": case["id"], "enabled": enabled})
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    print(json.dumps({"id": case["id"], "enabled": enabled, "passed": bool(passed),
                                      "answer_status": body["answer_status"], "reranker_status": trace["status"]}), flush=True)
    finally:
        api.flow = original
    report["index_sha256_after"] = load_active_chunks(api.store.db_path)[2]
    report["index_unchanged"] = before == report["index_sha256_after"]
    if not report["index_unchanged"]:
        raise AssertionError("smoke changed the document index")
    report["passed"] = not report["bad_cases"]
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/reranker-v1-live-smoke.json"))
    evaluate(parser.parse_args().output)

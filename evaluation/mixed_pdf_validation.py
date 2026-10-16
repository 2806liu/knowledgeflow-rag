"""Explicit live reparse validation for one existing mixed PDF.

This command changes only the supplied document's parsed text/index and calls
the configured live LLM once. It saves before/after evidence and Bad Cases.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import uuid

import httpx

from app.config import settings


def snapshot(document_id):
    connection = sqlite3.connect(settings.data_dir.resolve().joinpath("studyflow.db").as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        others = {table: [dict(row) for row in connection.execute(
            f"SELECT * FROM {table} WHERE {'id' if table == 'documents' else 'document_id'} <> ? ORDER BY 1,2", (document_id,))]
            for table in ("documents", "document_pages", "chunks")}
        target = {table: [dict(row) for row in connection.execute(
            f"SELECT * FROM {table} WHERE {'id' if table == 'documents' else 'document_id'} = ? ORDER BY 1,2", (document_id,))]
            for table in ("documents", "document_pages", "chunks")}
        signature = hashlib.sha256(json.dumps(others, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        return target, signature
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/mixed-pdf-reparse-validation.json"))
    args = parser.parse_args()
    before, others_before = snapshot(args.document_id)
    report = {"document_id": args.document_id, "before": before, "other_documents_before_sha256": others_before,
              "checks": {}, "bad_cases": []}
    with httpx.Client(base_url=args.base_url, timeout=240) as client:
        start = time.monotonic()
        response = client.post(f"/documents/{args.document_id}/reparse")
        report["reparse"] = {"http_status": response.status_code, "response": response.json(),
                             "seconds": round(time.monotonic() - start, 3)}
        if response.is_success:
            response = client.post("/chat", json={"session_id": "mixed-ocr-" + str(uuid.uuid4()),
                "message": "可持续发展战略的原则是什么", "document_ids": [args.document_id]})
            report["chat"] = {"http_status": response.status_code, "response": response.json()}
            if response.is_success:
                report["run"] = client.get("/runs/" + response.json()["run_id"]).json()
        after, others_after = snapshot(args.document_id)
        report["after"] = after
        report["other_documents_after_sha256"] = others_after
        before_doc = before["documents"][0]
        after_doc = after["documents"][0]
        page4 = next((p for p in after["document_pages"] if p["page_number"] == 4), None)
        principles = ("公平性", "持续性", "共同性")
        answer = report.get("chat", {}).get("response", {})
        citations = answer.get("citations", [])
        report["checks"] = {
            "reparse_success": report["reparse"]["http_status"] == 200,
            "document_id_and_created_at_retained": all(before_doc[k] == after_doc[k] for k in ("id", "created_at", "filename")),
            "raw_source_fingerprint_retained": before_doc["content_hash"] == after_doc["content_hash"],
            "other_documents_pages_chunks_unchanged": others_before == others_after,
            "physical_page_count_retained": len(before["document_pages"]) == len(after["document_pages"]),
            "page4_contains_all_principles": bool(page4) and all(p in page4["text"] for p in principles),
            "page4_mixed_provenance": bool(page4) and json.loads(page4["metadata"]).get("extraction_method") == "mixed",
            "vectors_indexed": after_doc["vector_status"] == "indexed",
            "answered_all_principles": answer.get("answer_status") == "answered" and all(p in answer.get("answer", "") for p in principles),
            "citation_page4_covers_principles": any(c["document_id"] == args.document_id and c["page"] == 4
                and c.get("extraction_method") == "mixed" and all(p in c["snippet"] for p in principles) for c in citations),
            "citation_chunks_match_saved_source": bool(citations) and all(any(
                c["chunk_id"] == row["id"] and c["snippet"] == row["text"] and c["page"] == json.loads(row["metadata"]).get("page")
                for row in after["chunks"]) for c in citations),
        }
    report["bad_cases"] = [{"check": key, "failure": "validation_failed"} for key, value in report["checks"].items() if not value]
    report["limitations"] = "One live development case; name/quote checks are not automatic semantic judgments or full OCR character accuracy."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": report["checks"], "bad_case_count": len(report["bad_cases"])}))
    if report["bad_cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

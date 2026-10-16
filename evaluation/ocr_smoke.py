"""Actual local OCR ingestion and fixed development query smoke test.

Uses a separate temporary database and mock answers: no remote embeddings or
LLM, no changes to the canonical .studyflow documents or historical reports.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import time
import uuid

from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter

from app.config import settings
from app.rag.parsers import render_pdf_page
from app.services import KnowledgeFlow
from app.store import Store


def normalized(text: str) -> str:
    return "".join(text.split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("Test_RagFile/专业知识问答.pdf"))
    parser.add_argument("--native-source", type=Path, default=Path("Test_RagFile/工程硕士工程伦理复习资料（2023春）(3).pdf"))
    parser.add_argument("--queries", type=Path, default=Path("evaluation/ocr-v1/queries.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/ocr-v1-smoke.json"))
    args = parser.parse_args()
    fixture_root = Path("tmp/ocr-smoke") / str(uuid.uuid4())
    fixture_root.mkdir(parents=True)
    config = replace(settings, data_dir=fixture_root / "data", vector_enabled=False,
                     retrieval_strategy="bm25_idf_v2", llm_mode="mock", ocr_enabled=True)
    scan = args.source.read_bytes()
    image_path = fixture_root / "scan-page-1.png"
    with render_pdf_page(scan, 0, config) as image:
        image.save(image_path)
    # Native choice page, physical blank page, actual raster scan: prove routing
    # and physical page numbers without combining native and OCR text.
    writer = PdfWriter()
    native_reader = PdfReader(args.native_source)
    writer.add_page(native_reader.pages[2])
    writer.add_blank_page(width=595, height=842)
    writer.add_page(PdfReader(io.BytesIO(scan)).pages[0])
    mixed_stream = io.BytesIO()
    writer.write(mixed_stream)
    mixed_path = fixture_root / "mixed.pdf"
    mixed_path.write_bytes(mixed_stream.getvalue())
    sources = {"scan": (args.source.name, scan, "application/pdf"),
               "image": ("scan-page-1.png", image_path.read_bytes(), "image/png"),
               "mixed": ("mixed.pdf", mixed_stream.getvalue(), "application/pdf")}
    from app import main as api

    previous = api.store, api.flow, api.settings
    store = Store(config.data_dir)
    api.store, api.settings = store, config
    api.flow = KnowledgeFlow(store, config)
    output = {"suite": json.loads(args.queries.read_text(encoding="utf-8"))["name"],
              "queries_sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest(),
              "limitations": ["Development smoke only, not an unseen acceptance set.",
                              "Actual local OCR; answers are mock, not real LLM semantic evaluation.",
                              "Page/anchor recall does not measure complete OCR accuracy or CER.",
                              "Only first-page query anchors were visually labeled; no full-document OCR quality acceptance."],
              "model_manifest": json.loads((config.ocr_model_dir / "models.json").read_text(encoding="utf-8")),
              "fixtures_dir": str(fixture_root.resolve()), "documents": {}, "checks": {}, "cases": [], "bad_cases": []}
    try:
        with TestClient(api.app) as client:
            for key, (filename, data, mime) in sources.items():
                started = time.perf_counter()
                response = client.post("/documents/upload", files={"file": (filename, data, mime)})
                if response.status_code != 200:
                    raise RuntimeError(f"{key} upload failed: {response.status_code} {response.text}")
                result = response.json()
                doc_id = result["document_id"]
                output["documents"][key] = {**result, "elapsed_seconds": round(time.perf_counter() - started, 3),
                                           "pages": client.get(f"/documents/{doc_id}/pages").json()}
                output["checks"][f"{key}_original_source_matches"] = client.get(f"/documents/{doc_id}/source").content == data
                output["checks"][f"{key}_indexed"] = result["indexed"]
                print(f"{key}: {result['page_count']} pages, {result['chunks']} chunks, {result['metadata']['ocr_page_count']} OCR pages", flush=True)
            mixed_pages = output["documents"]["mixed"]["pages"]
            output["checks"]["mixed_page_routes"] = [p["metadata"]["extraction_method"] for p in mixed_pages] == ["native_text", "native_text", "ocr"]
            output["checks"]["blank_page_preserved"] = mixed_pages[1]["page_number"] == 2 and not mixed_pages[1]["text"].strip()
            for case in json.loads(args.queries.read_text(encoding="utf-8"))["cases"]:
                doc_id = output["documents"][case["document"]]["document_id"]
                hits = api.flow.retriever.search(case["query"], top_k=4, document_ids=(doc_id,))
                chat = client.post("/chat", json={"session_id": case["id"], "message": case["query"], "document_ids": [doc_id]}).json()
                if case.get("expected") == "no_evidence":
                    passed = not hits and chat["answer_status"] == "no_evidence" and not chat["citations"]
                    rank = None
                else:
                    relevant = [index + 1 for index, hit in enumerate(hits)
                                if hit.page == case["page"] and normalized(case["anchor"]) in normalized(hit.text)]
                    rank = min(relevant) if relevant else None
                    passed = rank is not None and chat["answer_status"] == "mock" and any(
                        c["page"] == case["page"] and normalized(case["anchor"]) in normalized(c["snippet"])
                        for c in chat["citations"])
                citation_valid = all(c["document_id"] == doc_id and c["chunk_id"] in {hit.chunk_id for hit in hits}
                                     and any(hit.chunk_id == c["chunk_id"] and hit.page == c["page"] and hit.text == c["snippet"] for hit in hits)
                                     for c in chat["citations"])
                item = {**case, "relevant_rank": rank, "passed": passed and citation_valid,
                        "citation_source_valid": citation_valid, "retrieved": [hit.citation() for hit in hits],
                        "response": chat}
                output["cases"].append(item)
                if not item["passed"]:
                    output["bad_cases"].append(item)
            # Re-upload is cheap, reindex keeps OCR provenance, delete hides all evidence.
            image_doc = output["documents"]["image"]["document_id"]
            duplicate = client.post("/documents/upload", files={"file": sources["image"]}).json()
            output["checks"]["duplicate_reuses_document"] = duplicate["duplicate"] and duplicate["document_id"] == image_doc
            output["checks"]["reindex_succeeds"] = client.post(f"/documents/{image_doc}/reindex").status_code == 200
            output["checks"]["reindex_keeps_ocr_metadata"] = all(json.loads(row["metadata"])["extraction_method"] == "ocr" for row in store.chunks((image_doc,)))
            output["checks"]["delete_succeeds"] = client.delete(f"/documents/{image_doc}").status_code == 200
            output["checks"]["deleted_source_unavailable"] = client.get(f"/documents/{image_doc}/source").status_code == 404 and not store.source_path(image_doc).exists()
            output["checks"]["deleted_not_retrievable"] = not api.flow.retriever.search("开环控制", document_ids=(image_doc,))
            for name, passed in output["checks"].items():
                if not passed:
                    output["bad_cases"].append({"id": name, "failure": "lifecycle_check"})
    finally:
        api.store, api.flow, api.settings = previous
        store.conn.close()
    positives = [case for case in output["cases"] if case.get("expected") != "no_evidence"]
    output["metrics"] = {"positive_cases": len(positives), "total_cases": len(output["cases"]),
                         "Recall@1": sum(case["relevant_rank"] == 1 for case in positives) / len(positives),
                         "Recall@4": sum(case["relevant_rank"] is not None for case in positives) / len(positives),
                         "MRR@4": sum(1 / case["relevant_rank"] if case["relevant_rank"] else 0 for case in positives) / len(positives),
                         "bad_case_count": len(output["bad_cases"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["metrics"]))
    print("Saved report: " + str(args.output))
    if output["bad_cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

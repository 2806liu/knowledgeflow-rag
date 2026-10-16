"""Paired reranker development evaluation on one immutable active-index snapshot."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
import time

from app.config import settings
from app.rag.embedding import OpenAICompatibleEmbeddingProvider
from app.rag.hybrid import HybridRetriever
from app.rag.retrieval import BM25Retriever, BM25RareTermRetriever
from app.rag.reranker import RerankingRetriever, get_reranker, file_hash
from app.rag.vector_index import QdrantVectorIndex
from evaluation.corpus import load_cases, load_documents, load_manifest, validate_evidence
from evaluation.metrics import macro_average, matches, score_case
from evaluation.stored_index import SnapshotStore, CachedEmbeddingProvider, load_active_chunks


def timing_summary(values):
    ordered = sorted(values)
    def percentile(p):
        x = (len(ordered) - 1) * p
        lo, hi = int(x), min(int(x) + 1, len(ordered) - 1)
        return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (x - lo), 3)
    return {"mean_ms": round(mean(values), 3), "p50_ms": percentile(.5),
            "p95_ms": percentile(.95), "max_ms": round(max(values), 3)}


def evaluate(args):
    db = settings.data_dir / "studyflow.db"
    manifest, cases = load_manifest(args.manifest), load_cases(args.queries)
    validate_evidence(cases, load_documents(db, manifest))
    rows, doc_count, index_hash = load_active_chunks(db)
    names = args.retrievers.split(",")
    if not names or set(names) - {"bm25_idf_v2", "bm25_v1", "dense_v1", "hybrid_rrf_v1"}:
        raise ValueError("unsupported retrieval comparison")
    ks = (1, 2, 4)
    store = SnapshotStore(rows)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"name": manifest["name"], "status": "development_set",
                    "query_count": len(cases), "manifest_sha256": file_hash(args.manifest),
                    "queries_sha256": file_hash(args.queries)},
        "index": {"documents": doc_count, "chunks": len(rows), "sha256_before": index_hash},
        "parameters": {"scope": args.scope_mode, "candidate_k": args.candidate_k,
                       "top_k": 4, "batch_size": settings.reranker_batch_size,
                       "max_length": settings.reranker_max_length,
                       "failure_policy": "error", "timing": "warm model/index; query embeddings preloaded and reused; one pass per mode"},
        "retrievers": {}, "bad_cases": []}
    started = time.perf_counter()
    engine = get_reranker(str(settings.reranker_model_dir.resolve()),
                          settings.reranker_batch_size, settings.reranker_max_length)
    report["model"] = {**engine.manifest, "initialization_ms": round((time.perf_counter() - started) * 1000, 3)}
    smoke = engine.score("可持续发展原则", ["可持续发展的原则包括公平性、持续性和共同性。", "太阳系有八大行星。",
                                         "可持续发展的原则包括公平性、持续性和共同性。"])
    if smoke[0] != smoke[2] or smoke[0] <= smoke[1]:
        raise AssertionError("real-model pair mapping or relevance smoke failed")
    report["model"]["smoke_scores"] = smoke
    vector = None
    if any(name in {"dense_v1", "hybrid_rrf_v1"} for name in names):
        for case in cases:
            scope = case.scope_document_ids if args.scope_mode == "label" else None
            missing = {r["document_id"] for r in store.chunks(scope)
                       if r["vector_status"] != "indexed" or r["embedding_model"] != settings.embedding_model}
            if missing:
                raise ValueError(f"vectors missing or incompatible in comparison scope: {sorted(missing)}")
        embedding = CachedEmbeddingProvider(OpenAICompatibleEmbeddingProvider(
            settings.embedding_base_url, settings.embedding_model, settings.embedding_api_key))
        started = time.perf_counter()
        for case in cases:
            embedding.embed([case.query])
        report["embedding_preload"] = {"queries": len(embedding.cache),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3), "model": embedding.model}
        vector = QdrantVectorIndex(store, embedding, settings.qdrant_collection, url=settings.qdrant_url)
    for name in names:
        if name == "dense_v1":
            base = vector
        elif name == "hybrid_rrf_v1":
            base = HybridRetriever(BM25Retriever(store, max_hits_per_document=settings.retrieval_bm25_max_hits_per_document),
                vector, candidate_k=settings.retrieval_hybrid_candidate_k, rrf_k=settings.retrieval_hybrid_rrf_k,
                max_hits_per_document=settings.retrieval_hybrid_max_hits_per_document)
        else:
            cls = BM25RareTermRetriever if name == "bm25_idf_v2" else BM25Retriever
            base = cls(store, max_hits_per_document=settings.retrieval_bm25_max_hits_per_document)
        configs = [replace(settings, reranker_enabled=flag, reranker_candidate_k=args.candidate_k,
                           reranker_failure_policy="error") for flag in (False, True)]
        pipelines = [RerankingRetriever(base, config, reranker=engine) for config in configs]
        first = cases[0]
        scope = first.scope_document_ids if args.scope_mode == "label" else None
        for pipeline in pipelines:
            pipeline.search(first.query, 4, scope)
        mode_results = {"off": [], "on": []}
        oracle = []
        for number, case in enumerate(cases, 1):
            scope = case.scope_document_ids if args.scope_mode == "label" else None
            # Alternate order to reduce systematic off/on timing order bias.
            for i in ((0, 1) if number % 2 else (1, 0)):
                hits, trace = pipelines[i].search_with_trace(case.query, 4, scope)
                result = score_case(case, hits, ks)
                result["citations"] = [hit.citation() for hit in hits]
                result["trace"] = trace
                for hit in hits:
                    canonical = next(r for r in rows if r["id"] == hit.chunk_id)
                    if canonical["document_id"] != hit.document_id or canonical["text"] != hit.text or canonical["filename"] != hit.filename:
                        raise AssertionError("citation source changed after ranking")
                mode_results["on" if i else "off"].append(result)
            candidates = base.search_candidates(case.query, max(4, args.candidate_k), scope)
            coverage = sum(any(matches(hit, evidence) for hit in candidates) for evidence in case.evidence) / len(case.evidence)
            oracle.append(coverage)
            off, on = mode_results["off"][-1], mode_results["on"][-1]
            deltas = {key: round(on["metrics"][key] - off["metrics"][key], 4) for key in off["metrics"]}
            if any(deltas[key] < 0 for key in ("Recall@1", "Recall@4", "MRR@4")) or on["metrics"]["Recall@4"] < 1:
                report["bad_cases"].append({"retriever": name, "id": case.case_id, "query": case.query,
                    "metric_delta": deltas, "candidate_evidence_recall": coverage,
                    "reason": "regression" if any(v < 0 for v in deltas.values()) else "remaining_miss",
                    "off_chunk_ids": [c["chunk_id"] for c in off["citations"]],
                    "on_chunk_ids": [c["chunk_id"] for c in on["citations"]]})
            if number % 5 == 0:
                print(json.dumps({"retriever": name, "completed": number, "total": len(cases)}), flush=True)
        result = {}
        for mode, records in mode_results.items():
            result[mode] = {"aggregate": macro_average(records), "cases": records,
                "timing": {field: timing_summary([r["trace"][field] for r in records])
                           for field in ("retrieval_ms", "rerank_ms", "total_ms")}}
        result["metric_delta"] = {k: round(result["on"]["aggregate"][k] - result["off"]["aggregate"][k], 4)
                                  for k in result["off"]["aggregate"]}
        result["candidate_evidence_recall"] = round(mean(oracle), 4)
        result["rerank_ok_count"] = sum(r["trace"]["status"] == "ok" for r in mode_results["on"])
        report["retrievers"][name] = result
        write_report(args.output.with_suffix(".partial.json"), report)
        print(json.dumps({"retriever": name, "off": result["off"]["aggregate"],
                          "on": result["on"]["aggregate"], "on_timing": result["on"]["timing"]}), flush=True)
    report["index"]["sha256_after"] = load_active_chunks(db)[2]
    report["index"]["unchanged"] = report["index"]["sha256_after"] == index_hash
    if not report["index"]["unchanged"]:
        raise AssertionError("index changed during paired evaluation")
    write_report(args.output, report)
    args.output.with_suffix(".partial.json").unlink(missing_ok=True)
    return report


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/independent-v1/corpus.json"))
    parser.add_argument("--queries", type=Path, default=Path("evaluation/independent-v1/queries.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/reranker-v1-dev-scoped.json"))
    parser.add_argument("--scope-mode", choices=("label", "all"), default="label")
    parser.add_argument("--candidate-k", type=int, default=32)
    parser.add_argument("--retrievers", default="bm25_idf_v2,bm25_v1,dense_v1,hybrid_rrf_v1")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()

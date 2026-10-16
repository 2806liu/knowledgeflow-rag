"""Evaluate active SQLite chunks without changing them.

The all mode mirrors unscoped /chat; label mode passes each query's document
scope through the same retriever contract as scoped /chat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.rag.embedding import OpenAICompatibleEmbeddingProvider
from app.rag.hybrid import HybridRetriever
from app.rag.retrieval import BM25RareTermRetriever, BM25Retriever, BalancedLexicalRetriever, LexicalRetriever
from app.rag.vector_index import QdrantVectorIndex
from evaluation.corpus import comparable_text, load_cases, load_documents, load_manifest, validate_evidence
from evaluation.diagnostics import diagnose_case, diagnose_hybrid_case
from evaluation.metrics import macro_average, score_case
from evaluation.run import MemoryStore


class SnapshotStore(MemoryStore):
    def chunks_by_ids(self, chunk_ids: tuple[str, ...]) -> list[dict]:
        wanted = set(chunk_ids)
        return [row for row in self.records if row["id"] in wanted]


class CachedEmbeddingProvider:
    """Avoid billing twice for the same query in one comparison run."""

    def __init__(self, provider: OpenAICompatibleEmbeddingProvider):
        self.provider = provider
        self.model = provider.model
        self.cache: dict[tuple[str, ...], list[list[float]]] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        key = tuple(texts)
        if key not in self.cache:
            self.cache[key] = self.provider.embed(texts)
        return self.cache[key]


def load_active_chunks(db_path: Path) -> tuple[list[dict], int, str]:
    if not db_path.is_file():
        raise FileNotFoundError(f"database not found: {db_path}")
    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        document_count = int(
            connection.execute("SELECT COUNT(*) FROM documents WHERE status <> 'deleted'").fetchone()[0]
        )
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT chunks.*, documents.filename AS filename,
                       documents.vector_status AS vector_status,
                       documents.embedding_model AS embedding_model
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                WHERE documents.status <> 'deleted'
                ORDER BY chunks.document_id, chunks.position
                """
            )
        ]
    finally:
        connection.close()
    digest = hashlib.sha256()
    for row in rows:
        for value in (row["id"], row["document_id"], row["text"], row["metadata"]):
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return rows, document_count, digest.hexdigest()


def evaluate(args: argparse.Namespace) -> dict:
    manifest = load_manifest(args.manifest)
    cases = load_cases(args.queries)
    db_path = args.data_dir / "studyflow.db"
    # Labels remain tied to the original sources even though retrieval runs
    # against every active document in the database.
    validate_evidence(cases, load_documents(db_path, manifest))
    rows, document_count, index_hash = load_active_chunks(db_path)
    ks = tuple(sorted({int(value) for value in args.ks.split(",")}))
    if (not ks or ks[0] <= 0 or not 0 <= args.min_query_coverage <= 1
            or args.max_hits_per_document <= 0 or args.hybrid_candidate_k <= 0 or args.hybrid_rrf_k <= 0):
        raise ValueError("K and max hits must be positive; query coverage must be 0..1")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": manifest["name"],
            "status": manifest.get("status", "unknown"),
            "query_count": len(cases),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "queries_sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest(),
        },
        "index": {
            "source": "active_sqlite_chunks",
            "scope_document_ids_ignored": args.scope_mode == "all",
            "active_documents": document_count,
            "active_chunks": len(rows),
            "index_sha256": index_hash,
        },
        "parameters": {
            "ks": ks,
            "min_query_coverage": args.min_query_coverage,
            "max_hits_per_document": args.max_hits_per_document,
            "retriever": args.retriever,
            "scope_mode": args.scope_mode,
            "hybrid_candidate_k": args.hybrid_candidate_k,
            "hybrid_rrf_k": args.hybrid_rrf_k,
            "v2_strict_bonus": BalancedLexicalRetriever.STRICT_BONUS,
            "v2_formula_bonus": BalancedLexicalRetriever.FORMULA_BONUS,
        },
        "retrievers": {},
    }
    names = ("lexical_v2", "bm25_v1", "bm25_idf_v2", "dense_v1", "hybrid_rrf_v1") if args.retriever == "independent_comparison" else ("bm25_v1", "dense_v1", "hybrid_rrf_v1") if args.retriever == "compare_hybrid" else ("lexical_v1", "lexical_v2", "bm25_v1", "dense_v1") if args.retriever == "all_with_dense" else ("lexical_v1", "lexical_v2", "bm25_v1") if args.retriever == "all" else ("lexical_v1", "lexical_v2") if args.retriever == "both" else (args.retriever,)
    store = SnapshotStore(rows)
    embedding = CachedEmbeddingProvider(OpenAICompatibleEmbeddingProvider(
        settings.embedding_base_url, settings.embedding_model, settings.embedding_api_key
    )) if any(name in {"dense_v1", "hybrid_rrf_v1"} for name in names) else None
    vector_index = QdrantVectorIndex(
        store, embedding, settings.qdrant_collection, url=settings.qdrant_url
    ) if embedding is not None else None
    if "hybrid_rrf_v1" in names:
        scopes = {case.scope_document_ids if args.scope_mode == "label" else None for case in cases}
        for scope in scopes:
            missing = {
                row["document_id"] for row in store.chunks(scope)
                if row["vector_status"] != "indexed" or row["embedding_model"] != settings.embedding_model
            }
            if missing:
                raise ValueError(f"hybrid comparison requires vectors for every scoped document: {sorted(missing)}")
    for name in names:
        if name == "dense_v1":
            retriever = vector_index
        elif name == "hybrid_rrf_v1":
            retriever = HybridRetriever(
                BM25Retriever(store, max_hits_per_document=args.max_hits_per_document),
                vector_index,
                candidate_k=args.hybrid_candidate_k,
                rrf_k=args.hybrid_rrf_k,
                max_hits_per_document=args.max_hits_per_document,
            )
        elif name == "bm25_v1":
            retriever = BM25Retriever(store, max_hits_per_document=args.max_hits_per_document)
        elif name == "bm25_idf_v2":
            retriever = BM25RareTermRetriever(store, max_hits_per_document=args.max_hits_per_document)
        else:
            retriever_type = LexicalRetriever if name == "lexical_v1" else BalancedLexicalRetriever
            retriever = retriever_type(
                store,
                min_query_coverage=args.min_query_coverage,
                max_hits_per_document=args.max_hits_per_document,
            )
        results = []
        for case in cases:
            scope = case.scope_document_ids if args.scope_mode == "label" else None
            if name == "hybrid_rrf_v1":
                hits, trace = retriever.search_with_trace(case.query, top_k=max(ks), document_ids=scope)
            else:
                hits = retriever.search(case.query, top_k=max(ks), document_ids=scope)
            result = score_case(case, hits, ks)
            if name == "hybrid_rrf_v1":
                result["diagnosis"] = diagnose_hybrid_case(case, store.chunks(scope), trace)
                result["candidate_counts"] = {
                    "bm25": len(trace["bm25"]),
                    "dense": len(trace["dense"]),
                    "union": len(trace["fused"]),
                }
            elif result["first_relevant_rank"] is not None:
                result["diagnosis"] = {"reason": "hit"}
            elif name == "dense_v1":
                relevant = [
                    row for row in store.chunks(scope)
                    if any(
                        row["document_id"] == evidence.document_id
                        and comparable_text(evidence.anchor) in comparable_text(row["text"])
                        for evidence in case.evidence
                    )
                ]
                reason = "no_chunk_contains_anchor" if not relevant else (
                    "vector_not_indexed" if any(row["vector_status"] != "indexed" for row in relevant)
                    else "missing_from_vector_top_k"
                )
                result["diagnosis"] = {"reason": reason}
            else:
                result["diagnosis"] = diagnose_case(case, store.chunks(scope), retriever, hits, max(ks))
            results.append(result)
        report["retrievers"][name] = {"aggregate": macro_average(results), "cases": results}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate active stored chunks read-only, as /chat searches them")
    parser.add_argument("--data-dir", type=Path, default=Path(".studyflow"))
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/chongyou-corpus.json"))
    parser.add_argument("--queries", type=Path, default=Path("evaluation/chongyou-queries.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/chongyou-stored-index.json"))
    parser.add_argument("--retriever", choices=("lexical_v1", "lexical_v2", "bm25_v1", "bm25_idf_v2", "dense_v1", "hybrid_rrf_v1", "compare_hybrid", "independent_comparison", "both", "all", "all_with_dense"), default="all")
    parser.add_argument("--scope-mode", choices=("all", "label"), default="all")
    parser.add_argument("--ks", default="1,2,4")
    parser.add_argument("--min-query-coverage", type=float, default=0.5)
    parser.add_argument("--max-hits-per-document", type=int, default=2)
    parser.add_argument("--hybrid-candidate-k", type=int, default=32)
    parser.add_argument("--hybrid-rrf-k", type=int, default=60)
    args = parser.parse_args()
    report = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Active index: {report['index']['active_documents']} documents, {report['index']['active_chunks']} chunks")
    for name, result in report["retrievers"].items():
        print(f"{name}: {result['aggregate']}")
    print(f"Detailed report: {args.output}")


if __name__ == "__main__":
    main()

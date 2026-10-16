"""Compare fixed_v1 and the current recursive splitter on one labeled corpus.

Run from the repository root:
    python -m evaluation.run --data-dir .studyflow
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from app.rag.chunking import create_chunker
from app.rag.retrieval import BM25Retriever, BalancedLexicalRetriever, LexicalRetriever
from evaluation.corpus import build_records, load_cases, load_documents, load_manifest, validate_evidence
from evaluation.diagnostics import diagnose_case
from evaluation.metrics import macro_average, score_case


class MemoryStore:
    def __init__(self, records: list[dict]):
        self.records = records
        self.index_generation = 0

    def chunks(self, document_ids: tuple[str, ...] | None = None) -> list[dict]:
        if document_ids is None:
            return self.records
        scope = set(document_ids)
        return [record for record in self.records if record["document_id"] in scope]


def evaluate(args: argparse.Namespace) -> dict:
    manifest = load_manifest(args.manifest)
    cases = load_cases(args.queries)
    documents = load_documents(args.data_dir / "studyflow.db", manifest)
    validate_evidence(cases, documents)

    ks = tuple(sorted({int(value) for value in args.ks.split(",")}))
    if not ks or ks[0] <= 0 or not 0 <= args.min_query_coverage <= 1 or args.max_hits_per_document <= 0 or args.v2_max_hits_per_document <= 0:
        raise ValueError("K and max hits must be positive; query coverage must be 0..1")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": manifest["name"],
            "status": manifest.get("status", "unknown"),
            "query_count": len(cases),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "queries_sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest(),
            "documents": [
                {"document_id": doc.document_id, "filename": doc.filename, "content_hash": doc.content_hash}
                for doc in documents.values()
            ],
        },
        "parameters": {
            "ks": ks,
            "min_query_coverage": args.min_query_coverage,
            "max_hits_per_document": args.max_hits_per_document,
            "retriever": args.retriever,
            "v2_max_hits_per_document": args.v2_max_hits_per_document,
            "v2_strict_bonus": BalancedLexicalRetriever.STRICT_BONUS,
            "v2_formula_bonus": BalancedLexicalRetriever.FORMULA_BONUS,
            "v2_min_formula_label_match": BalancedLexicalRetriever.MIN_FORMULA_LABEL_MATCH,
            "fixed_size": 800,
            "fixed_overlap": 120,
            "recursive_target_size": args.target_size,
            "recursive_overlap": args.overlap,
            "recursive_min_size": args.min_size,
            "recursive_max_size": args.max_size,
            "quality_filter": "app.rag.quality.is_low_quality_chunk",
        },
        "strategies": {},
    }

    for strategy in ("fixed", "recursive"):
        chunker = create_chunker(
            strategy,
            target_size=args.target_size,
            overlap=args.overlap,
            min_size=args.min_size,
            max_size=args.max_size,
        )
        records, chunk_stats = build_records(documents, chunker)
        retriever_names = ("lexical_v1", "lexical_v2", "bm25_v1") if args.retriever == "all" else ("lexical_v1", "lexical_v2") if args.retriever == "both" else (args.retriever,)
        for retriever_name in retriever_names:
            case_results: list[dict] = []
            for case in cases:
                scope = set(case.scope_document_ids)
                scoped_records = [record for record in records if record["document_id"] in scope]
                max_hits = args.max_hits_per_document if retriever_name == "lexical_v1" else args.v2_max_hits_per_document
                if retriever_name == "bm25_v1":
                    retriever = BM25Retriever(MemoryStore(scoped_records), max_hits_per_document=max_hits)
                else:
                    retriever_type = LexicalRetriever if retriever_name == "lexical_v1" else BalancedLexicalRetriever
                    retriever = retriever_type(
                        MemoryStore(scoped_records),
                        min_query_coverage=args.min_query_coverage,
                        max_hits_per_document=max_hits,
                    )
                hits = retriever.search(case.query, top_k=max(ks))
                result = score_case(case, hits, ks)
                result["diagnosis"] = diagnose_case(case, scoped_records, retriever, hits, max(ks))
                case_results.append(result)
            name = f"{chunker.name}_v{chunker.version}"
            if args.retriever != "lexical_v1":
                name += f"__{retriever_name}"
            report["strategies"][name] = {
                "chunk_stats": chunk_stats,
                "retriever": retriever_name,
                "max_hits_per_document": max_hits,
                "aggregate": macro_average(case_results),
                "cases": case_results,
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate chunking strategies without changing the application index")
    parser.add_argument("--data-dir", type=Path, default=Path(".studyflow"))
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/corpus.json"))
    parser.add_argument("--queries", type=Path, default=Path("evaluation/queries.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/attendance-baseline.json"))
    parser.add_argument("--ks", default="1,2,4")
    parser.add_argument("--min-query-coverage", type=float, default=0.5)
    parser.add_argument("--max-hits-per-document", type=int, default=2)
    parser.add_argument("--retriever", choices=("lexical_v1", "lexical_v2", "bm25_v1", "both", "all"), default="lexical_v1")
    parser.add_argument("--v2-max-hits-per-document", type=int, default=2)
    parser.add_argument("--target-size", type=int, default=450)
    parser.add_argument("--overlap", type=int, default=60)
    parser.add_argument("--min-size", type=int, default=40)
    parser.add_argument("--max-size", type=int, default=800)
    args = parser.parse_args()

    report = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Dataset: {report['dataset']['name']} ({report['dataset']['query_count']} queries)")
    for strategy, result in report["strategies"].items():
        print(f"{strategy}: chunks={result['chunk_stats']['retrievable_chunks']} metrics={result['aggregate']}")
    print(f"Detailed report: {args.output}")


if __name__ == "__main__":
    main()

"""Read a frozen document corpus and its human-reviewable relevance labels."""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from app.rag.chunking import ChunkStrategy
from app.rag.contracts import DocumentPage
from app.rag.quality import is_low_quality_chunk


def comparable_text(value: str) -> str:
    """Ignore PDF layout whitespace while preserving words and punctuation."""

    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()


@dataclass(frozen=True)
class Evidence:
    document_id: str
    page: int | None
    anchor: str
    relevance: int


@dataclass(frozen=True)
class QueryCase:
    case_id: str
    query: str
    scope_document_ids: tuple[str, ...]
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    filename: str
    content_hash: str
    content: str
    pages: tuple[DocumentPage, ...]


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("documents"):
        raise ValueError("corpus manifest contains no documents")
    return manifest


def load_cases(path: Path) -> list[QueryCase]:
    cases: list[QueryCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        case_id = str(raw["id"])
        if case_id in seen_ids:
            raise ValueError(f"duplicate query ID: {case_id}")
        seen_ids.add(case_id)
        evidence = tuple(
            Evidence(
                document_id=str(item["document_id"]),
                page=int(item["page"]) if item.get("page") is not None else None,
                anchor=str(item["anchor"]),
                relevance=int(item["relevance"]),
            )
            for item in raw["evidence"]
        )
        scope = tuple(str(doc_id) for doc_id in raw["scope_document_ids"])
        if not raw["query"].strip() or not scope or not evidence:
            raise ValueError(f"query line {line_number} needs a query, scope and evidence")
        if any(not item.anchor.strip() or item.relevance not in {1, 2, 3} for item in evidence):
            raise ValueError(f"query {case_id} has an empty anchor or invalid relevance")
        if any(item.document_id not in scope for item in evidence):
            raise ValueError(f"query {case_id} cites a document outside its scope")
        cases.append(QueryCase(case_id, str(raw["query"]), scope, evidence))
    if not cases:
        raise ValueError("query set is empty")
    return cases


def load_documents(db_path: Path, manifest: dict) -> dict[str, SourceDocument]:
    """Open SQLite read-only; refuse stale IDs and hashes before evaluation."""

    if not db_path.is_file():
        raise FileNotFoundError(f"database not found: {db_path}")
    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    documents: dict[str, SourceDocument] = {}
    try:
        for expected in manifest["documents"]:
            document_id = str(expected["document_id"])
            row = connection.execute(
                "SELECT id, filename, content, content_hash, status FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
            if row is None or row["status"] == "deleted":
                raise ValueError(f"corpus document is missing or deleted: {document_id}")
            if row["content_hash"] != expected["content_hash"]:
                raise ValueError(f"content hash changed for document {document_id}")
            pages = tuple(
                DocumentPage(page_number=int(page["page_number"]), text=str(page["text"] or ""))
                for page in connection.execute(
                    "SELECT page_number, text FROM document_pages WHERE document_id = ? ORDER BY page_number",
                    (document_id,),
                )
            )
            documents[document_id] = SourceDocument(
                document_id=document_id,
                filename=str(row["filename"]),
                content_hash=str(row["content_hash"]),
                content=str(row["content"] or ""),
                pages=pages,
            )
    finally:
        connection.close()
    return documents


def validate_evidence(cases: list[QueryCase], documents: dict[str, SourceDocument]) -> None:
    """Catch labels that do not actually occur on their claimed source page."""

    for case in cases:
        if any(document_id not in documents for document_id in case.scope_document_ids):
            raise ValueError(f"query {case.case_id} references a document outside the corpus")
        for evidence in case.evidence:
            document = documents[evidence.document_id]
            sources = (
                [page.text for page in document.pages if page.page_number == evidence.page]
                if evidence.page is not None
                else [document.content]
            )
            if not sources or not any(comparable_text(evidence.anchor) in comparable_text(source) for source in sources):
                raise ValueError(
                    f"query {case.case_id} anchor was not found in document {evidence.document_id}, "
                    f"page {evidence.page}: {evidence.anchor}"
                )


def build_records(
    documents: dict[str, SourceDocument], chunker: ChunkStrategy
) -> tuple[list[dict], dict[str, float | int]]:
    """Recreate production chunks in memory without touching stored indexes."""

    records: list[dict] = []
    generated = 0
    dropped = 0
    lengths: list[int] = []
    for document in documents.values():
        parts = chunker.split_document(document.content, document.pages, document_id=document.document_id)
        generated += len(parts)
        for part in parts:
            if is_low_quality_chunk(part.text):
                dropped += 1
                continue
            lengths.append(len(part.text))
            records.append(
                {
                    "id": f"{chunker.name}:{chunker.version}:{document.document_id}:{part.position}",
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "text": part.text,
                    "position": part.position,
                    "metadata": json.dumps(part.metadata, ensure_ascii=False),
                }
            )
    stats: dict[str, float | int] = {
        "generated_chunks": generated,
        "retrievable_chunks": len(records),
        "filtered_chunks": dropped,
        "mean_chunk_characters": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "max_chunk_characters": max(lengths, default=0),
    }
    return records, stats

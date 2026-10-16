import json
import hashlib
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.rag.contracts import DocumentPage, ParsedDocument

#SQLite 持久化层，负责存取文档、分块、会话消息和工作流运行记录
def now_iso() -> str:
    """返回当前 UTC 时间的 ISO 8601 字符串。"""

    return datetime.now(timezone.utc).isoformat()


class Store:
    """使用 SQLite 持久化文档、分块、会话消息和工作流运行记录。"""

    def __init__(self, data_dir: Path):
        """初始化数据库连接，并创建所需的数据表。"""

        self.data_dir = data_dir
        self.index_generation = 0
        self.data_dir.mkdir(parents=True, exist_ok=True)#创建数据目录，已存在不报错
        self.db_path = data_dir / "studyflow.db"        #数据库文件路径
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)#连接数据库，允许多线程共用
                                                                            #check_same_thread=False：SQLite 默认不允许跨线程用同一连接。
                                                                            #FastAPI 是多线程的，关掉这个检查让多线程能共用连接。
        self.conn.row_factory = sqlite3.Row     #让查询结果能像字典一样按键访问
        self.conn.executescript(                #执行建表 SQL
            # 使用 IF NOT EXISTS 保证应用重复启动时不会覆盖已有数据。
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, filename TEXT NOT NULL, content TEXT NOT NULL,
                content_type TEXT NOT NULL DEFAULT 'text/plain',
                metadata TEXT NOT NULL DEFAULT '{}',
                content_hash TEXT,
                size_bytes INTEGER,
                status TEXT NOT NULL DEFAULT 'uploaded',
                indexed_at TEXT,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                chunk_strategy TEXT NOT NULL DEFAULT 'fixed',
                chunk_version TEXT NOT NULL DEFAULT '1',
                vector_status TEXT NOT NULL DEFAULT 'not_indexed',
                vector_indexed_at TEXT,
                embedding_model TEXT,
                deleted_at TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS document_pages (
                document_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                text TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY(document_id, page_number),
                FOREIGN KEY(document_id) REFERENCES documents(id)
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY, document_id TEXT NOT NULL, text TEXT NOT NULL,
                position INTEGER NOT NULL, metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY(document_id) REFERENCES documents(id)
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL, status TEXT NOT NULL,
                events TEXT NOT NULL, error TEXT, created_at TEXT NOT NULL
            );
            """
        )
        # Existing local databases were created before chunk metadata existed.
        # Add the column in place instead of forcing the user to delete data.
        self._ensure_column("documents", "content_type", "TEXT NOT NULL DEFAULT 'text/plain'")
        self._ensure_column("documents", "metadata", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column("documents", "content_hash", "TEXT")
        self._ensure_column("documents", "size_bytes", "INTEGER")
        self._ensure_column("documents", "status", "TEXT NOT NULL DEFAULT 'uploaded'")
        self._ensure_column("documents", "indexed_at", "TEXT")
        self._ensure_column("documents", "chunk_count", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("documents", "chunk_strategy", "TEXT NOT NULL DEFAULT 'fixed'")
        self._ensure_column("documents", "chunk_version", "TEXT NOT NULL DEFAULT '1'")
        self._ensure_column("documents", "vector_status", "TEXT NOT NULL DEFAULT 'not_indexed'")
        self._ensure_column("documents", "vector_indexed_at", "TEXT")
        self._ensure_column("documents", "embedding_model", "TEXT")
        self._ensure_column("documents", "deleted_at", "TEXT")
        self._ensure_column("chunks", "metadata", "TEXT NOT NULL DEFAULT '{}'")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_content_hash "
            "ON documents(content_hash)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_status "
            "ON documents(status)"
        )
        # Existing databases have chunks but no lifecycle metadata. Mark those
        # rows as legacy-indexed so they can be reindexed explicitly later.
        self.conn.execute(
            """
            UPDATE documents
            SET chunk_count = (
                    SELECT COUNT(*) FROM chunks WHERE chunks.document_id = documents.id
                ),
                status = CASE
                    WHEN EXISTS (
                        SELECT 1 FROM chunks WHERE chunks.document_id = documents.id
                    ) THEN 'indexed'
                    ELSE status
                END,
                indexed_at = CASE
                    WHEN EXISTS (
                        SELECT 1 FROM chunks WHERE chunks.document_id = documents.id
                    ) THEN COALESCE(indexed_at, created_at)
                    ELSE indexed_at
                END,
                chunk_strategy = CASE
                    WHEN EXISTS (
                        SELECT 1 FROM chunks WHERE chunks.document_id = documents.id
                    ) THEN 'legacy'
                    ELSE chunk_strategy
                END,
                chunk_version = CASE
                    WHEN EXISTS (
                        SELECT 1 FROM chunks WHERE chunks.document_id = documents.id
                    ) THEN 'legacy'
                    ELSE chunk_version
                END
            WHERE status <> 'deleted'
              AND indexed_at IS NULL
              AND chunk_count = 0
            """
        )
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def add_document(
        self,
        doc_id: str,
        filename: str,
        content: str,
        *,
        content_type: str = "text/plain",
        metadata: dict[str, Any] | None = None,
        pages: tuple[DocumentPage, ...] = (),
        content_hash: str | None = None,
        size_bytes: int | None = None,
    ) -> None:
        """新增或替换一篇文档，并记录写入时间。"""

        self.conn.execute("DELETE FROM document_pages WHERE document_id = ?", (doc_id,))
        self.conn.execute(
            """
            INSERT OR REPLACE INTO documents
                (id, filename, content, content_type, metadata, content_hash, size_bytes,
                 status, indexed_at, chunk_count, chunk_strategy, chunk_version,
                 vector_status, vector_indexed_at, embedding_model, deleted_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'uploaded', NULL, 0, 'fixed', '1',
                    'not_indexed', NULL, NULL, NULL, ?)
            """,
            (
                doc_id,
                filename,
                content,
                content_type,
                json.dumps(metadata or {}, ensure_ascii=False),
                content_hash,
                size_bytes,
                now_iso(),
            ),
        )
        self.conn.executemany(
            "INSERT INTO document_pages(document_id, page_number, text, metadata) VALUES (?, ?, ?, ?)",
            [
                (doc_id, page.page_number, page.text, json.dumps(page.metadata, ensure_ascii=False))
                for page in pages
            ],
        )
        self.conn.commit()

    def documents(self, *, include_deleted: bool = False) -> list[sqlite3.Row]:
        """按创建时间返回全部文档。"""

        if include_deleted:
            return self.conn.execute("SELECT * FROM documents ORDER BY created_at").fetchall()
        return self.conn.execute(
            "SELECT * FROM documents WHERE status <> 'deleted' ORDER BY created_at"
        ).fetchall()

    def document(self, document_id: str) -> sqlite3.Row | None:
        """Return one document without exposing storage details to routes."""

        return self.conn.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()

    def find_document_by_hash(self, content_hash: str) -> sqlite3.Row | None:
        """Find a document uploaded from the same raw bytes."""

        return self.conn.execute(
            "SELECT * FROM documents WHERE content_hash = ? AND status <> 'deleted' "
            "ORDER BY created_at LIMIT 1",
            (content_hash,),
        ).fetchone()

    def find_document_by_content(self, content: str) -> sqlite3.Row | None:
        """Fallback for documents created before raw content hashes were stored."""

        return self.conn.execute(
            "SELECT * FROM documents WHERE content = ? AND status <> 'deleted' "
            "ORDER BY created_at LIMIT 1",
            (content,),
        ).fetchone()

    def set_document_fingerprint(self, document_id: str, content_hash: str, size_bytes: int) -> None:
        """Backfill the raw upload fingerprint for a legacy document row."""

        self.conn.execute(
            "UPDATE documents SET content_hash = ?, size_bytes = ? WHERE id = ?",
            (content_hash, size_bytes, document_id),
        )
        self.conn.commit()

    def document_pages(self, document_id: str) -> list[sqlite3.Row]:
        """Return page-level source units for a document, if available."""

        return self.conn.execute(
            "SELECT * FROM document_pages WHERE document_id = ? ORDER BY page_number",
            (document_id,),
        ).fetchall()

    def source_path(self, document_id: str) -> Path:
        """Source names never contain uploaded filenames or caller path segments."""
        root = (self.data_dir / "sources").resolve()
        if not root.is_relative_to(self.data_dir.resolve()):
            raise ValueError("source directory is outside the data directory")
        return root / (hashlib.sha256(document_id.encode("utf-8")).hexdigest() + ".bin")

    def save_source(self, document_id: str, data: bytes) -> None:
        target = self.source_path(document_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        try:
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def delete_source(self, document_id: str) -> None:
        self.source_path(document_id).unlink(missing_ok=True)

    def replace_document_chunks(
        self,
        document_id: str,
        chunks: list[tuple[str, str, str, int, dict[str, Any]]],
        *,
        chunk_strategy: str = "fixed",
        chunk_version: str = "1",
    ) -> None:
        """Replace only one document's chunks for incremental indexing."""

        self.conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        rows = [
            (chunk_id, document_id, text, position, json.dumps(metadata, ensure_ascii=False))
            for chunk_id, _chunk_document_id, text, position, metadata in chunks
        ]
        self.conn.executemany(
            "INSERT INTO chunks(id, document_id, text, position, metadata) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self.conn.execute(
            """
            UPDATE documents
            SET status = ?, indexed_at = ?, chunk_count = ?,
                chunk_strategy = ?, chunk_version = ?, deleted_at = NULL,
                vector_status = 'not_indexed', vector_indexed_at = NULL,
                embedding_model = NULL
            WHERE id = ?
            """,
            (
                "indexed" if rows else "failed",
                now_iso() if rows else None,
                len(rows),
                chunk_strategy,
                chunk_version,
                document_id,
            ),
        )
        self.conn.commit()
        self.index_generation += 1

    def set_vector_status(self, document_id: str, status: str, model: str | None = None) -> None:
        if status not in {"not_indexed", "indexed", "failed"}:
            raise ValueError("invalid vector index status")
        self.conn.execute(
            "UPDATE documents SET vector_status = ?, vector_indexed_at = ?, embedding_model = ? WHERE id = ?",
            (status, now_iso() if status == "indexed" else None, model, document_id),
        )
        self.conn.commit()

    def replace_parsed_document(self, document_id: str, parsed: ParsedDocument, chunks,
                                *, fingerprint: str, size_bytes: int,
                                chunk_strategy: str, chunk_version: str) -> None:
        """Atomically publish text/pages/chunks, preserving identity and creation time.

        Use a separate connection so other request commits on the shared legacy
        connection cannot accidentally commit this multi-step transaction.
        Qdrant is synced after commit; failed vectors remain explicitly failed
        and stale points cannot resolve to the new canonical chunk IDs.
        """
        metadata = {**parsed.metadata, "reparsed_at": now_iso()}
        connection = sqlite3.connect(self.db_path)
        try:
            with connection:
                updated = connection.execute(
                    "UPDATE documents SET content=?, content_type=?, metadata=?, content_hash=?, size_bytes=?, "
                    "status='indexed', indexed_at=?, chunk_count=?, chunk_strategy=?, chunk_version=?, "
                    "vector_status='not_indexed', vector_indexed_at=NULL, embedding_model=NULL "
                    "WHERE id=? AND status <> 'deleted'",
                    (parsed.text, parsed.content_type, json.dumps(metadata, ensure_ascii=False), fingerprint,
                     size_bytes, now_iso(), len(chunks), chunk_strategy, chunk_version, document_id),
                )
                if updated.rowcount != 1:
                    raise ValueError("document no longer active; reparse not saved")
                connection.execute("DELETE FROM document_pages WHERE document_id=?", (document_id,))
                connection.executemany(
                    "INSERT INTO document_pages(document_id,page_number,text,metadata) VALUES (?,?,?,?)",
                    [(document_id, p.page_number, p.text, json.dumps(p.metadata, ensure_ascii=False)) for p in parsed.pages],
                )
                connection.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
                connection.executemany(
                    "INSERT INTO chunks(id,document_id,text,position,metadata) VALUES (?,?,?,?,?)",
                    [(cid, document_id, text, pos, json.dumps(meta, ensure_ascii=False))
                     for cid, _, text, pos, meta in chunks],
                )
        finally:
            connection.close()
        self.index_generation += 1

    def document_chunk_count(self, document_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS count FROM chunks WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        return int(row["count"] if row else 0)

    def soft_delete_document(self, document_id: str) -> sqlite3.Row | None:
        """Remove searchable data while retaining the document audit row."""

        row = self.document(document_id)
        if row is None:
            return None
        deleted_at = now_iso()
        with self.conn:
            self.conn.execute("DELETE FROM document_pages WHERE document_id = ?", (document_id,))
            self.conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            self.conn.execute(
                """
                UPDATE documents
                SET status = 'deleted', deleted_at = ?, indexed_at = NULL,
                    chunk_count = 0, vector_status = 'deleted',
                    vector_indexed_at = NULL
                WHERE id = ?
                """,
                (deleted_at, document_id),
            )
        self.index_generation += 1
        return self.document(document_id)

    def chunks(self, document_ids: tuple[str, ...] | None = None) -> list[sqlite3.Row]:
        """按文档和原文位置返回全部文本分块。"""
        #JOIN 的作用：chunks 表里只有 document_id，文件名在 documents 表里。
        # JOIN 后每行既有 chunk 字段，又有 filename。
        # RetrievalHit 需要 filename，检索时能一起返回。
        if document_ids is not None and not document_ids:
            return []
        scope_clause = ""
        params: tuple[str, ...] = ()
        if document_ids is not None:
            scope_clause = f" AND chunks.document_id IN ({','.join('?' for _ in document_ids)})"
            params = document_ids
        return self.conn.execute(
            f"""
            SELECT chunks.*, documents.filename AS filename
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            WHERE documents.status <> 'deleted'
            {scope_clause}
            ORDER BY chunks.document_id, chunks.position
            """,
            params,
        ).fetchall()

    def chunks_by_ids(self, chunk_ids: tuple[str, ...]) -> list[sqlite3.Row]:
        """Load canonical active rows for Qdrant hits; omit deleted or stale points."""

        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        return self.conn.execute(
            f"""
            SELECT chunks.*, documents.filename AS filename,
                   documents.vector_status AS vector_status
            FROM chunks JOIN documents ON documents.id = chunks.document_id
            WHERE documents.status <> 'deleted' AND chunks.id IN ({placeholders})
            """,
            chunk_ids,
        ).fetchall()

    def section_neighbors(self, document_id: str, section_id: str, position: int, radius: int) -> list[dict]:
        """Read bounded canonical neighbors; never cross a document or section.

        Metadata is checked in Python so old SQLite installations need no
        JSON extension. Missing section IDs cannot authorize expansion.
        """
        if not section_id or radius <= 0:
            return []
        rows = self.conn.execute(
            """SELECT chunks.*, documents.filename AS filename
               FROM chunks JOIN documents ON documents.id=chunks.document_id
               WHERE documents.status <> 'deleted' AND chunks.document_id=?
                 AND chunks.position BETWEEN ? AND ? AND chunks.position <> ?
               ORDER BY chunks.position""",
            (document_id, position - radius, position + radius, position),
        ).fetchall()
        return [dict(row) for row in rows if json.loads(row["metadata"] or "{}").get("section_id") == section_id]

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """保存一条会话消息，role 通常为 user 或 assistant。"""

        self.conn.execute(
            "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now_iso()),
        )
        self.conn.commit()

    def history(self, session_id: str, limit: int = 10) -> list[dict[str, str]]:
        """返回会话最近的消息，并恢复为时间正序。"""

        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def save_run(self, run_id: str, session_id: str, status: str, events: list[dict[str, Any]], error: str | None = None) -> None:
        """保存一次工作流运行状态及其事件列表。"""

        self.conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, session_id, status, json.dumps(events, ensure_ascii=False), error, now_iso()),
        )
        self.conn.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """按运行 ID 查询记录，并将事件 JSON 还原为列表。"""

        row = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["events"] = json.loads(result["events"])
        return result

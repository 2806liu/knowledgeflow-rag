import hashlib
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager  #定义 lifespan 钩子

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.web import attach_workspace
from app.rag.quality import assess_chunk_quality
from app.schemas import DocumentPageResponse, DocumentResponse, UploadResponse
from app.rag.chunking import create_chunker
from app.rag.contracts import DocumentPage
from app.rag.parsers import IMAGE_TYPES, DocumentLimitError, document_suffix, parse_document
from app.rag.ocr import OCRExecutionError, OCRUnavailable
from app.rag.reranker import RerankerUnavailable, RerankerExecutionError
from app.schemas import ChatRequest, ChatResponse, IndexResponse, RunResponse   #Pydantic 请求/响应模型
from app.rag.llm_client import LLMCallError
from app.services import KnowledgeFlow      #服务层，编排检索和生成
from app.store import Store             #上一节的 SQLite 持久化层


# 创建全局存储和知识流编排器，供所有 API 请求复用。
store = Store(settings.data_dir)        #全局存储，所有请求共用。
flow = KnowledgeFlow(store, settings)   #知识流编排器，接收 store 和配置


@asynccontextmanager
async def lifespan(_: FastAPI):
    """恢复中断轮次，退出时协作停止摄取和记忆任务。"""

    # 启动时恢复中断轮次；退出时协作停止本项目后台任务。
    flow.conversations.recover()
    try:
        yield
    finally:
        if getattr(flow,"_conversation_flow",None) is not None:
            await flow._conversation_flow.memory.close()
        if hasattr(flow,"_ingestion_service"):
            flow._ingestion_service.close()


app = FastAPI(title="KnowledgeFlow RAG", version="0.1.0", lifespan=lifespan)
# 工作台复用当前服务配置和业务接口，不连接其他项目。
attach_workspace(app, lambda: settings)


def _validate_upload(filename: str, data: bytes, reported_content_type: str | None) -> str:
    """Validate an upload before parsing it and return its normalized suffix."""

    if not data:
        raise HTTPException(status_code=400, detail="document is empty")
    if len(data) > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes / (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"document exceeds the {limit_mb:.1f} MB upload limit")

    suffix = document_suffix(filename, data)
    if suffix not in {".pdf", ".txt", ".md", *IMAGE_TYPES}:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported document type for {filename!r}; use .pdf, .txt, .md, .png, .jpg, .jpeg, or .webp; files without an extension must contain a recognizable PDF or image signature",
        )

    mime = (reported_content_type or "").split(";", 1)[0].strip().lower()
    if suffix == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="file is not a valid PDF")
        if mime and mime not in {"application/pdf", "application/octet-stream"}:
            raise HTTPException(status_code=415, detail="PDF content type does not match the file")
    elif suffix in IMAGE_TYPES:
        signatures = {
            ".png": data.startswith(b"\x89PNG\r\n\x1a\n"),
            ".jpg": data.startswith(b"\xff\xd8\xff"),
            ".jpeg": data.startswith(b"\xff\xd8\xff"),
            ".webp": data.startswith(b"RIFF") and data[8:12] == b"WEBP",
        }
        if not signatures[suffix]:
            raise HTTPException(status_code=400, detail="image signature does not match its filename")
        if mime and mime not in {IMAGE_TYPES[suffix][1], "application/octet-stream"}:
            raise HTTPException(status_code=415, detail="image content type does not match the file")
    elif mime and not (mime.startswith("text/") or mime == "application/octet-stream"):
        raise HTTPException(status_code=415, detail="text document content type does not match the file")
    return suffix


def _build_document_chunks(document_id, text, pages, document_metadata):
    """Prepare all chunks before publishing a reparsed document."""
    try:
        chunker = create_chunker(
            settings.chunk_strategy,
            target_size=settings.chunk_target_size,
            overlap=settings.chunk_overlap,
            min_size=settings.chunk_min_size,
            max_size=settings.chunk_max_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    chunks: list[tuple[str, str, str, int, dict[str, object]]] = []
    parts = chunker.split_document(text, pages, document_id=document_id)
    page_metadata = {page.page_number: page.metadata for page in pages}
    for part in parts:
        quality = assess_chunk_quality(part.text)
        metadata = dict(part.metadata)
        if document_metadata.get("extraction_method"):
            metadata["extraction_method"] = document_metadata["extraction_method"]
        origin = page_metadata.get(metadata.get("page"), {})
        for key in ("extraction_method", "ocr_engine", "ocr_engine_version", "ocr_mean_confidence",
                    "ocr_dropped_lines", "ocr_regions", "coordinate_system", "warnings"):
            if key in origin:
                metadata[key] = origin[key]
        metadata["quality_flags"] = list(quality.flags)
        metadata["is_retrievable"] = quality.exclude_reason is None
        if quality.exclude_reason is not None:
            continue
        chunks.append((str(uuid.uuid4()), document_id, part.text, part.position, metadata))
    return chunks, chunker


def _sync_document_vectors(document_id: str) -> None:
    if flow.vector_index is not None:
        try:
            vector_count = flow.vector_index.replace_document(document_id)
            store.set_vector_status(
                document_id,
                "indexed" if vector_count else "not_indexed",
                settings.embedding_model if vector_count else None,
            )
        except Exception as exc:
            store.set_vector_status(document_id, "failed", settings.embedding_model)
            raise HTTPException(status_code=502, detail=f"vector indexing failed: {exc}; retry /documents/{document_id}/reindex") from exc


def _index_document(document_id: str) -> int:
    """Chunk and index one document without replacing other documents' chunks."""
    doc = store.document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    pages = [DocumentPage(page_number=int(page["page_number"]), text=str(page["text"] or ""),
                          metadata=json.loads(page["metadata"] or "{}"))
             for page in store.document_pages(document_id)]
    chunks, chunker = _build_document_chunks(document_id, str(doc["content"] or ""), pages,
                                            json.loads(doc["metadata"] or "{}"))

    store.replace_document_chunks(
        document_id,
        chunks,
        chunk_strategy=chunker.name,
        chunk_version=chunker.version,
    )
    _sync_document_vectors(document_id)
    return len(chunks)


def _document_response(row: object) -> DocumentResponse:
    """Convert a SQLite document row into the public lifecycle response."""

    document_id = str(row["id"])
    pages = store.document_pages(document_id)
    return DocumentResponse(
        document_id=document_id,
        filename=str(row["filename"]),
        content_type=str(row["content_type"]),
        content_hash=row["content_hash"],
        size_bytes=row["size_bytes"],
        characters=len(str(row["content"])),
        page_count=len(pages),
        status=str(row["status"]),
        indexed_at=row["indexed_at"],
        chunk_count=int(row["chunk_count"] or 0),
        chunk_strategy=str(row["chunk_strategy"]),
        chunk_version=str(row["chunk_version"]),
        vector_status=str(row["vector_status"]),
        vector_indexed_at=row["vector_indexed_at"],
        embedding_model=row["embedding_model"],
        deleted_at=row["deleted_at"],
        created_at=str(row["created_at"]),
        metadata=json.loads(row["metadata"] or "{}"),
        source_available=str(row["status"]) != "deleted" and store.source_path(document_id).is_file(),
    )


@app.get("/health")
def health() -> dict[str, str]:
    """健康检查接口，用于确认服务已经正常启动。"""

    return {"status": "ok"}


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Validate, deduplicate, parse, persist, and index one document."""

    filename = file.filename or "document.txt"
    data = await file.read(settings.max_upload_bytes + 1)
    _validate_upload(filename, data, file.content_type)
    content_hash = hashlib.sha256(data).hexdigest()

    # Check the raw file hash before parsing to make identical uploads cheap.
    existing = store.find_document_by_hash(content_hash)
    if existing is not None:
        document_id = str(existing["id"])
        if not store.source_path(document_id).is_file():
            store.save_source(document_id, data)
        chunks = store.document_chunk_count(document_id)
        if chunks == 0:
            chunks = _index_document(document_id)
        elif flow.vector_index is not None and (
            existing["vector_status"] != "indexed"
            or existing["embedding_model"] != settings.embedding_model
        ):
            try:
                flow.vector_index.replace_document(document_id)
                store.set_vector_status(document_id, "indexed", settings.embedding_model)
            except Exception as exc:
                store.set_vector_status(document_id, "failed", settings.embedding_model)
                raise HTTPException(status_code=502, detail=f"vector indexing failed: {exc}") from exc
        pages = store.document_pages(document_id)
        return UploadResponse(
            document_id=document_id,
            filename=str(existing["filename"]),
            content_hash=content_hash,
            characters=len(str(existing["content"])),
            page_count=len(pages),
            chunks=chunks,
            indexed=chunks > 0,
            duplicate=True,
            metadata=json.loads(existing["metadata"] or "{}"),
        )

    try:
        parsed = await run_in_threadpool(parse_document, filename, data, config=settings)
    except OCRUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OCRExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except DocumentLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"document parsing failed: {exc}") from exc

    extracted_characters = len(re.sub(r"\s+", "", parsed.text))
    if extracted_characters < settings.min_extracted_characters:
        if parsed.metadata.get("ocr_page_count", 0):
            detail = "OCR returned too little usable text; review image clarity and recognition confidence"
        elif parsed.content_type == "application/pdf":
            detail = (
                "PDF text extraction returned too little text; "
                "the file may be a scanned PDF and require OCR"
            )
        else:
            detail = "document is empty or contains too little extractable text"
        raise HTTPException(status_code=400, detail=detail)

    # Migration fallback for old rows created before content_hash existed.
    legacy_duplicate = store.find_document_by_content(parsed.text)
    if (legacy_duplicate is not None and legacy_duplicate["content_hash"] is None
            and legacy_duplicate["filename"] == parsed.filename
            and legacy_duplicate["content_type"] == parsed.content_type
            and not parsed.metadata.get("ocr_page_count", 0)):
        document_id = str(legacy_duplicate["id"])
        store.set_document_fingerprint(document_id, content_hash, len(data))
        store.save_source(document_id, data)
        chunks = store.document_chunk_count(document_id)
        if chunks == 0:
            chunks = _index_document(document_id)
        pages = store.document_pages(document_id)
        return UploadResponse(
            document_id=document_id,
            filename=str(legacy_duplicate["filename"]),
            content_hash=content_hash,
            characters=len(parsed.text),
            page_count=len(pages),
            chunks=chunks,
            indexed=chunks > 0,
            duplicate=True,
            metadata=json.loads(legacy_duplicate["metadata"] or "{}"),
        )

    doc_id = str(uuid.uuid4())
    store.add_document(
        doc_id,
        parsed.filename,
        parsed.text,
        content_type=parsed.content_type,
        metadata=parsed.metadata,
        pages=parsed.pages,
        content_hash=content_hash,
        size_bytes=len(data),
    )
    store.save_source(doc_id, data)
    chunks = _index_document(doc_id)
    return UploadResponse(
        document_id=doc_id,
        filename=parsed.filename,
        content_hash=content_hash,
        characters=len(parsed.text),
        page_count=len(parsed.pages),
        chunks=chunks,
        indexed=chunks > 0,
        duplicate=False,
        metadata=parsed.metadata,
    )


@app.post("/documents/index", response_model=IndexResponse)
def index_documents() -> IndexResponse:
    """Rebuild the index for all non-deleted documents."""

    indexed = 0
    total_chunks = 0
    for doc in store.documents():
        total_chunks += _index_document(str(doc["id"]))
        indexed += 1
    return IndexResponse(indexed_documents=indexed, chunks=total_chunks)


@app.get("/documents", response_model=list[DocumentResponse])
def list_documents(include_deleted: bool = False) -> list[DocumentResponse]:
    """List document lifecycle and indexing status."""

    return [
        _document_response(row)
        for row in store.documents(include_deleted=include_deleted)
    ]


@app.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str) -> DocumentResponse:
    row = store.document(document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="document not found")
    return _document_response(row)


@app.get("/documents/{document_id}/pages", response_model=list[DocumentPageResponse])
def get_document_pages(document_id: str) -> list[DocumentPageResponse]:
    row = store.document(document_id)
    if row is None or row["status"] == "deleted":
        raise HTTPException(status_code=404, detail="active document not found")
    return [DocumentPageResponse(document_id=document_id, page_number=page["page_number"],
                                 text=page["text"], metadata=json.loads(page["metadata"] or "{}"))
            for page in store.document_pages(document_id)]


@app.get("/documents/{document_id}/source")
def get_document_source(document_id: str) -> FileResponse:
    row = store.document(document_id)
    if row is None or row["status"] == "deleted" or not store.source_path(document_id).is_file():
        raise HTTPException(status_code=404, detail="original source is unavailable")
    return FileResponse(store.source_path(document_id), media_type=row["content_type"],
                        filename=str(row["filename"]).replace("\\", "/").rsplit("/", 1)[-1])


@app.delete("/documents/{document_id}", response_model=DocumentResponse)
def delete_document(document_id: str) -> DocumentResponse:
    row = store.document(document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="document not found")
    if flow.vector_index is not None:
        try:
            flow.vector_index.delete_document(document_id)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"vector deletion failed: {exc}") from exc
    row = store.soft_delete_document(document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="document not found")
    store.delete_source(document_id)
    return _document_response(row)


@app.post("/documents/{document_id}/reindex", response_model=DocumentResponse)
def reindex_document(document_id: str) -> DocumentResponse:
    row = store.document(document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="document not found")
    if row["status"] == "deleted":
        raise HTTPException(status_code=409, detail="deleted document cannot be reindexed")
    _index_document(document_id)
    updated = store.document(document_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="document not found")
    return _document_response(updated)


@app.post("/documents/{document_id}/reparse", response_model=DocumentResponse)
def reparse_document(document_id: str) -> DocumentResponse:
    """Re-read the saved source and replace this document's text and indexes."""
    row = store.document(document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="document not found")
    if row["status"] == "deleted":
        raise HTTPException(status_code=409, detail="deleted document cannot be reparsed")
    source = store.source_path(document_id)
    if not source.is_file():
        raise HTTPException(status_code=404, detail="original source is unavailable; upload the original file to restore it")
    with source.open("rb") as stream:
        data = stream.read(settings.max_upload_bytes + 1)
    _validate_upload(str(row["filename"]), data, str(row["content_type"]))
    fingerprint = hashlib.sha256(data).hexdigest()
    if row["content_hash"] and row["content_hash"] != fingerprint:
        raise HTTPException(status_code=409, detail="saved source fingerprint does not match the document")
    try:
        parsed = parse_document(str(row["filename"]), data, config=settings)
    except OCRUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OCRExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except DocumentLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"document parsing failed: {exc}") from exc
    if len(re.sub(r"\s+", "", parsed.text)) < settings.min_extracted_characters:
        raise HTTPException(status_code=400, detail="reparse returned too little usable text; existing index retained")
    chunks, chunker = _build_document_chunks(document_id, parsed.text, parsed.pages, parsed.metadata)
    if not chunks:
        raise HTTPException(status_code=400, detail="reparse produced no retrievable chunks; existing index retained")
    try:
        store.replace_parsed_document(document_id, parsed, chunks, fingerprint=fingerprint,
                                      size_bytes=len(data), chunk_strategy=chunker.name, chunk_version=chunker.version)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _sync_document_vectors(document_id)
    logger_data = {"event": "document_reparsed", "document_id": document_id,
                   "pages": len(parsed.pages), "chunks": len(chunks), "parser_version": parsed.metadata.get("parser_version")}
    logging.getLogger(__name__).info(json.dumps(logger_data))
    return _document_response(store.document(document_id))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """执行一次基于知识库的会话问答。"""

    # 服务层负责检索、调用模型、记录事件以及保存会话消息。
    # 当前只提供完整 JSON，显式拒绝尚未实现的流式参数，避免误导调用方。
    if request.stream:
        raise HTTPException(status_code=422, detail="streaming is not supported; use stream=false")
    document_ids = tuple(dict.fromkeys(request.document_ids)) if request.document_ids is not None else None
    if document_ids is not None:
        for document_id in document_ids:
            row = store.document(document_id)
            if row is None or row["status"] == "deleted":
                raise HTTPException(status_code=404, detail=f"active document not found: {document_id}")
    try:
        result = await flow.chat(request.session_id, request.message, document_ids=document_ids,
                                 request_id=request.request_id, scope_revision=request.scope_revision, answer_policy=request.answer_policy)
    except LLMCallError as exc:
        # 错误不伪装成证据不足，不向客户端暴露上游正文或密钥。
        raise HTTPException(status_code=exc.http_status, detail={"code": exc.category, "message": str(exc), "run_id": getattr(exc, "run_id", None)}) from exc
    except RerankerUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RerankerExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ChatResponse(**result)   #**result：把字典展开成关键字参数，构造 ChatResponse
                                        #async def + await，因为调 LLM 是 IO 密集操作，异步不阻塞其他请求

@app.get("/sessions/{session_id}")
def session_history(session_id: str) -> dict[str, object]:
    """返回指定会话最近的消息记录。"""

    return {"session_id": session_id, "messages": store.history(session_id, limit=50)}


@app.get("/runs/{run_id}", response_model=RunResponse)
def run_details(run_id: str) -> RunResponse:
    """查询工作流运行状态、节点事件和错误信息。"""

    run = store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return RunResponse(run_id=run["id"], session_id=run["session_id"], status=run["status"], events=run["events"], error=run["error"])

# 新接口使用运行时provider，保持项目内隔离并便于测试替换store/flow。
from app.conversation_api import attach_conversation_routes
attach_conversation_routes(app, lambda: flow, _validate_upload, _build_document_chunks)

@app.get("/documents/{document_id}/chunks/{chunk_id}")
def get_source_chunk(document_id: str, chunk_id: str):
    """按独立chunk ID回溯当前规范原文，供引用核验，不拼接或生成新证据。"""
    rows=store.chunks_by_ids((chunk_id,))
    if not rows or rows[0]["document_id"]!=document_id:
        raise HTTPException(status_code=404,detail="source chunk not found or stale")
    row=rows[0];meta=json.loads(row["metadata"] or "{}")
    return {"chunk_id":row["id"],"document_id":row["document_id"],"filename":row["filename"],
            "text":row["text"],"page":meta.get("page"),"position":row["position"],"metadata":meta}

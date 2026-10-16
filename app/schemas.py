from typing import Any, Literal
from pydantic import BaseModel, Field   #BaseModel：Pydantic 的基类，所有模型都继承它.Field：给字段加约束（如最小长度）。

#定义了 API 的请求/响应数据模型（Pydantic 模型），
# 用来规定“接口收什么、返回什么”。它是接口层和内部逻辑之间的数据契约。

class ChatRequest(BaseModel):
    """聊天接口的请求参数。"""

    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    # Omit to search every active document. Explicit IDs constrain retrieval.
    document_ids: list[str] | None = Field(default=None, min_length=1)
    # 当前返回完整 JSON；传 true 会明确拒绝，避免冒充流式实现。
    stream: bool = False
    # 省略保留旧API严格模式；工作台明确提交增强策略。
    answer_policy: Literal["document_strict","document_enhanced"] | None = None
    # 同一会话的幂等键和范围版本，旧客户端可省略。
    request_id: str | None = Field(default=None, max_length=128)
    scope_revision: int | None = None


class Citation(BaseModel):
    """用户可追溯的知识库引用。"""

    citation_number: int
    chunk_id: str
    document_id: str
    filename: str | None = None
    page: int | None = None
    position: int | None = None
    score: float | None = None  # 概述来源未参与检索，不伪造相关性分数。
    snippet: str
    extraction_method: str | None = None
    ocr_confidence: float | None = None
    retrieval_score: float | None = None
    rerank_score: float | None = None
    reranker_model: str | None = None
    section_id: str | None = None
    evidence_role: str = "retrieved"
    expanded_from_chunk_id: str | None = None


class ChatResponse(BaseModel):
    """聊天接口返回的答案、引用和工具调用信息。"""

    run_id: str
    answer: str
    answer_status: str
    citations: list[Citation]
    tool_calls: list[dict[str, Any]]
    # 新字段保留旧接口兼容；来源版本和覆盖信息用于页面与审计。
    turn_id: str | None = None
    intent: str | None = None
    resolved_query: str | None = None
    scope_snapshot: dict[str, Any] | None = None
    coverage: dict[str, Any] | None = None
    answer_units: list[dict[str, Any]] = Field(default_factory=list)
    answer_policy: str | None = None
    effective_policy: str | None = None
    answer_basis: str | None = None
    answer_sections: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_aspects: list[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    """Document ingestion result, including deduplication and indexing state."""

    document_id: str
    filename: str
    content_hash: str
    characters: int
    page_count: int
    chunks: int
    indexed: bool
    duplicate: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    """Document lifecycle and indexing status."""

    document_id: str
    filename: str
    content_type: str
    content_hash: str | None = None
    size_bytes: int | None = None
    characters: int
    page_count: int
    status: str
    indexed_at: str | None = None
    chunk_count: int
    chunk_strategy: str
    chunk_version: str
    vector_status: str = "not_indexed"
    vector_indexed_at: str | None = None
    embedding_model: str | None = None
    deleted_at: str | None = None
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_available: bool = False


class DocumentPageResponse(BaseModel):
    document_id: str
    page_number: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexResponse(BaseModel):
    """文档索引接口返回的统计信息。"""

    indexed_documents: int
    chunks: int


class RunResponse(BaseModel):
    """工作流运行记录的查询结果。"""

    run_id: str
    session_id: str
    status: str
    events: list[dict[str, Any]]
    error: str | None = None

"""RAG 模块共用的稳定数据契约。

这些数据结构不依赖 SQLite、Qdrant、LangChain 或特定向量模型，
因此可以替换存储和检索实现，而无需修改 HTTP 层。
"""

#解析文件，文本chunk,检索结果

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentPage:
    """One physical page emitted by a page-aware document parser."""

    page_number: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedDocument:
    """解析器输出的规范化文档内容。"""

    filename: str           #原始文件名
    text: str               #解析出的纯文本
    content_type: str          #文件类型（pdf、txt、md），如 text/plain、text/markdown、application/pdf
    metadata: dict[str, Any] = field(default_factory=dict)  #额外信息（页数、上传时间等）
    pages: tuple[DocumentPage, ...] = ()


@dataclass(frozen=True)
class TextChunk:        #切分后的文本块
    """带有来源元数据、可被检索的文本单元。"""

    document_id: str        #属于哪个文档
    text: str               #这一块的文本
    position: int              #在文档里的第几块
    metadata: dict[str, Any] = field(default_factory=dict)  #额外信息（页码等）


@dataclass(frozen=True)
class RetrievalHit:
    """排序后的检索结果，以及生成引用所需的信息。"""

    chunk_id: str               #命中的块 ID（存储层生成）
    document_id: str
    text: str                   #块的文本
    score: float                    #相关性得分
    position: int | None = None
    filename: str | None = None
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def citation(self) -> dict[str, Any]:#RAG 系统通常要在回答里标注“这句话来自哪个文档、哪一块”。citation() 就是给前端/用户看的引用信息。
        """返回可直接序列化为 JSON 的引用摘要。"""

        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "page": self.page,
            "position": self.position,
            "score": self.score,
            "snippet": self.text,       #引用要显示原文片段
            "extraction_method": self.metadata.get("extraction_method"),
            "ocr_confidence": self.metadata.get("ocr_mean_confidence"),
            "retrieval_score": self.metadata.get("retrieval_score", self.score),
            "rerank_score": self.metadata.get("rerank_score"),
            "reranker_model": self.metadata.get("reranker_model"),
            "section_id": self.metadata.get("section_id"),
            "evidence_role": self.metadata.get("evidence_role", "retrieved"),
            "expanded_from_chunk_id": self.metadata.get("expanded_from_chunk_id"),
        }


def source_dedupe_key(hit: RetrievalHit) -> tuple[str, int | None, str]:
    """Collapse layout duplicates only when document and physical page agree."""

    return hit.document_id, hit.page, " ".join(hit.text.split())


@dataclass(frozen=True)
class EvidenceContext:
    """A retrieved passage passed to a solver, with an explicit citation label."""

    citation_number: int
    filename: str | None
    page: int | None
    text: str
    document_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Stable solver input shape for prompts, logs, or future model adapters."""

        value = {
            "citation_number": self.citation_number,
            "filename": self.filename,
            "page": self.page,
            "text": self.text,
        }
        # 增强比较问答需要校验引用归属；旧调用点未传时仍维持原有输入形状。
        if self.document_id is not None:
            value["document_id"] = self.document_id
        return value

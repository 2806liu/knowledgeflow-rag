"""KnowledgeFlow RAG 组件。

本包提供可替换的文档解析、文本分块、检索和数据契约模块。
API 与工作流代码应依赖这些接口，避免把检索细节直接写入 Agent 服务。
"""

from app.rag.chunking import ChunkStrategy, RecursiveChunker, TextChunker, create_chunker, split_chunks
from app.rag.contracts import DocumentPage, ParsedDocument, RetrievalHit, TextChunk
from app.rag.retrieval import LexicalRetriever


def __getattr__(name: str):
    # Keep parser exports compatible without pre-importing the OCR CLI module.
    if name in {"extract_text", "parse_document"}:
        from app.rag import parsers
        return getattr(parsers, name)
    raise AttributeError(name)

__all__ = [
    "LexicalRetriever",
    "DocumentPage",
    "ParsedDocument",
    "RetrievalHit",
    "TextChunk",
    "TextChunker",
    "RecursiveChunker",
    "ChunkStrategy",
    "create_chunker",
    "extract_text",
    "parse_document",
    "split_chunks",
]

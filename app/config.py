from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


# Resolve from this file so startup works from either the project root or app/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)


def project_path(value: str) -> Path:
    """Resolve relative runtime paths against this project's directory."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    """应用运行配置，优先从环境变量读取，便于在不同环境中部署。"""

    # 本地 SQLite 数据库及其他运行数据的保存目录。
    data_dir: Path = project_path(os.getenv("STUDYFLOW_DATA_DIR", ".studyflow"))
    # LLM 调用模式；默认使用内置的模拟回答逻辑。
    llm_mode: str = os.getenv("STUDYFLOW_LLM_MODE", "mock")
    # OpenAI 兼容接口的服务地址、鉴权密钥和模型名称。
    llm_base_url: str = os.getenv("STUDYFLOW_LLM_BASE_URL", "")
    llm_api_key: str = os.getenv("STUDYFLOW_LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    llm_model: str = os.getenv("STUDYFLOW_LLM_MODEL", "")
    # LLM 单次限时、有限重试及输出上限；两个调用适配器共用这些配置。
    llm_timeout_seconds: float = float(os.getenv("STUDYFLOW_LLM_TIMEOUT_SECONDS", "60"))
    llm_max_retries: int = int(os.getenv("STUDYFLOW_LLM_MAX_RETRIES", "2"))
    llm_retry_backoff_seconds: float = float(os.getenv("STUDYFLOW_LLM_RETRY_BACKOFF_SECONDS", "0.5"))
    llm_max_output_tokens: int = int(os.getenv("STUDYFLOW_LLM_MAX_OUTPUT_TOKENS", "1024"))
    # 留空时不传 temperature，兼容不支持此参数的模型；JSON 模式需上游支持。
    llm_temperature: float | None = float(os.environ["STUDYFLOW_LLM_TEMPERATURE"]) if os.getenv("STUDYFLOW_LLM_TEMPERATURE") else None
    # 兼容支持 enable_thinking 的服务；留空保持服务商默认，不对其他模型强加参数。
    llm_enable_thinking: bool | None = (os.environ["STUDYFLOW_LLM_ENABLE_THINKING"].lower() in {"1", "true", "yes"}) if os.getenv("STUDYFLOW_LLM_ENABLE_THINKING") else None
    llm_answer_format: str = os.getenv("STUDYFLOW_LLM_ANSWER_FORMAT", "text")
    # 会话、概述和上传任务都有明确预算，所有路径复用同一模型客户端。
    # 增强问答与语义解析单独限时；不改变长文概述期限。
    enhanced_timeout_seconds: float = float(os.getenv("STUDYFLOW_ENHANCED_TIMEOUT_SECONDS", "120"))
    resolution_timeout_seconds: float = float(os.getenv("STUDYFLOW_RESOLUTION_TIMEOUT_SECONDS", "20"))
    context_window: int = int(os.getenv("STUDYFLOW_CONTEXT_WINDOW", "24000"))
    context_safety_margin: int = int(os.getenv("STUDYFLOW_CONTEXT_SAFETY_MARGIN", "512"))
    history_budget: int = int(os.getenv("STUDYFLOW_HISTORY_BUDGET", "2000"))
    memory_summary_budget: int = int(os.getenv("STUDYFLOW_MEMORY_SUMMARY_BUDGET", "600"))
    memory_recent_rounds: int = int(os.getenv("STUDYFLOW_MEMORY_RECENT_ROUNDS", "3"))
    overview_max_output_tokens: int = int(os.getenv("STUDYFLOW_OVERVIEW_MAX_OUTPUT_TOKENS", "4096"))
    overview_max_calls: int = int(os.getenv("STUDYFLOW_OVERVIEW_MAX_CALLS", "24"))
    overview_timeout_seconds: float = float(os.getenv("STUDYFLOW_OVERVIEW_TIMEOUT_SECONDS", "240"))
    ingestion_timeout_seconds: float = float(os.getenv("STUDYFLOW_INGESTION_TIMEOUT_SECONDS", "600"))
    ingestion_max_retries: int = int(os.getenv("STUDYFLOW_INGESTION_MAX_RETRIES", "2"))
    ingestion_queue_limit: int = int(os.getenv("STUDYFLOW_INGESTION_QUEUE_LIMIT", "32"))
    # Lexical baseline guard; this is provisional and will be calibrated with
    # the labeled RAG evaluation set in the next project stage.
    retrieval_min_query_coverage: float = float(os.getenv("STUDYFLOW_RETRIEVAL_MIN_QUERY_COVERAGE", "0.5"))
    retrieval_max_hits_per_document: int = int(os.getenv("STUDYFLOW_RETRIEVAL_MAX_HITS_PER_DOCUMENT", "2"))
    retrieval_strategy: str = os.getenv("STUDYFLOW_RETRIEVAL_STRATEGY", "bm25_idf_v2")
    retrieval_v2_max_hits_per_document: int = int(os.getenv("STUDYFLOW_RETRIEVAL_V2_MAX_HITS_PER_DOCUMENT", "2"))
    retrieval_bm25_max_hits_per_document: int = int(os.getenv("STUDYFLOW_RETRIEVAL_BM25_MAX_HITS_PER_DOCUMENT", "2"))
    retrieval_hybrid_candidate_k: int = int(os.getenv("STUDYFLOW_RETRIEVAL_HYBRID_CANDIDATE_K", "32"))
    retrieval_hybrid_rrf_k: int = int(os.getenv("STUDYFLOW_RETRIEVAL_HYBRID_RRF_K", "60"))
    retrieval_hybrid_max_hits_per_document: int = int(os.getenv("STUDYFLOW_RETRIEVAL_HYBRID_MAX_HITS_PER_DOCUMENT", "2"))
    vector_enabled: bool = os.getenv("STUDYFLOW_VECTOR_ENABLED", "false").lower() in {"1", "true", "yes"}
    reranker_enabled: bool = os.getenv("STUDYFLOW_RERANKER_ENABLED", "false").lower() in {"1", "true", "yes"}
    reranker_model_dir: Path = project_path(os.getenv("STUDYFLOW_RERANKER_MODEL_DIR", str(data_dir / "reranker_models")))
    reranker_candidate_k: int = int(os.getenv("STUDYFLOW_RERANKER_CANDIDATE_K", "32"))
    reranker_batch_size: int = int(os.getenv("STUDYFLOW_RERANKER_BATCH_SIZE", "8"))
    reranker_max_length: int = int(os.getenv("STUDYFLOW_RERANKER_MAX_LENGTH", "512"))
    reranker_failure_policy: str = os.getenv("STUDYFLOW_RERANKER_FAILURE_POLICY", "fallback")
    evidence_expansion_enabled: bool = os.getenv("STUDYFLOW_EVIDENCE_EXPANSION_ENABLED", "true").lower() in {"1", "true", "yes"}
    evidence_neighbor_radius: int = int(os.getenv("STUDYFLOW_EVIDENCE_NEIGHBOR_RADIUS", "2"))
    evidence_max_extra_chunks: int = int(os.getenv("STUDYFLOW_EVIDENCE_MAX_EXTRA_CHUNKS", "2"))
    evidence_max_extra_tokens: int = int(os.getenv("STUDYFLOW_EVIDENCE_MAX_EXTRA_TOKENS", "1200"))
    evidence_max_context_tokens: int = int(os.getenv("STUDYFLOW_EVIDENCE_MAX_CONTEXT_TOKENS", "6000"))
    embedding_base_url: str = os.getenv("STUDYFLOW_EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    embedding_api_key: str = os.getenv("STUDYFLOW_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("STUDYFLOW_EMBEDDING_MODEL", "text-embedding-3-small")
    qdrant_url: str = os.getenv("STUDYFLOW_QDRANT_URL", "http://127.0.0.1:6333")
    qdrant_collection: str = os.getenv("STUDYFLOW_QDRANT_COLLECTION", "knowledgeflow_chunks_v1")
    # Upload guardrails.  Keep these configurable so local experiments can
    # use a smaller limit while deployment can enforce a stricter one.
    max_upload_bytes: int = int(os.getenv("STUDYFLOW_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
    min_extracted_characters: int = int(os.getenv("STUDYFLOW_MIN_EXTRACTED_CHARACTERS", "20"))
    # OCR is lazy: normal text uploads do not load the optional runtime.
    ocr_enabled: bool = os.getenv("STUDYFLOW_OCR_ENABLED", "true").lower() in {"1", "true", "yes"}
    ocr_model_dir: Path = project_path(os.getenv("STUDYFLOW_OCR_MODEL_DIR", str(data_dir / "ocr_models")))
    ocr_pdf_min_text_characters: int = int(os.getenv("STUDYFLOW_OCR_PDF_MIN_TEXT_CHARACTERS", "40"))
    ocr_max_pages: int = int(os.getenv("STUDYFLOW_OCR_MAX_PAGES", "30"))
    ocr_max_regions_per_page: int = int(os.getenv("STUDYFLOW_OCR_MAX_REGIONS_PER_PAGE", "64"))
    ocr_dpi: int = int(os.getenv("STUDYFLOW_OCR_DPI", "180"))
    ocr_max_render_pixels: int = int(os.getenv("STUDYFLOW_OCR_MAX_RENDER_PIXELS", "12000000"))
    ocr_max_image_pixels: int = int(os.getenv("STUDYFLOW_OCR_MAX_IMAGE_PIXELS", "20000000"))
    ocr_min_confidence: float = float(os.getenv("STUDYFLOW_OCR_MIN_CONFIDENCE", "0.6"))
    # Chunking is strategy based so the legacy fixed splitter remains
    # available while recursive_v2 becomes the default production baseline.
    chunk_strategy: str = os.getenv("STUDYFLOW_CHUNK_STRATEGY", "recursive")
    chunk_target_size: int = int(os.getenv("STUDYFLOW_CHUNK_TARGET_SIZE", "450"))
    chunk_overlap: int = int(os.getenv("STUDYFLOW_CHUNK_OVERLAP", "60"))
    chunk_min_size: int = int(os.getenv("STUDYFLOW_CHUNK_MIN_SIZE", "40"))
    chunk_max_size: int = int(os.getenv("STUDYFLOW_CHUNK_MAX_SIZE", "800"))


# 创建全局配置实例，供各模块共享。
settings = Settings()
# 启动时确保数据目录存在，避免首次运行时数据库创建失败。
settings.data_dir.mkdir(parents=True, exist_ok=True)

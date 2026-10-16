"""可替换的 RAG 检索实现，当前提供离线词法检索基线。"""
 # 词法检索基线, 后续改进可以用向量检索

import json
import math
import re           #分词
from copy import copy
from collections import Counter
from collections.abc import Mapping     #表示“像字典一样按键访问的对象”。
from typing import Any

from app.rag.contracts import RetrievalHit, source_dedupe_key  #：上一节的数据契约，检索的产出。


def _row_value(record: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """读取新旧存储行，兼容 metadata 列迁移前的 SQLite 数据库。"""

    try:
        return record[key]
    except (KeyError, IndexError):
        return default


def _row_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    raw = _row_value(record, "metadata", "{}") or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def tokens(text: str) -> list[str]:
    """提取英文单词和中文二元片段，供词法检索使用。

    ``\\w+`` 会把一整段连续中文当成一个 token，导致查询“实验室”
    无法命中“实验室考勤管理制度”。英文/数字按单词处理，中文按二元
    片段处理；单字查询保留单字 token 作为回退。
    """

    normalized = text.lower()
    latin_tokens = re.findall(r"[a-z0-9_]+", normalized)
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", normalized)
    cjk_tokens: list[str] = []
    for run in cjk_runs:
        if len(run) == 1:
            cjk_tokens.append(run)
        else:
            cjk_tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return latin_tokens + cjk_tokens


def _scoped_rows(store: Any, document_ids: tuple[str, ...] | None) -> list[Mapping[str, Any]]:
    """Ask the store to filter before ranking, so out-of-scope chunks cannot win."""

    return store.chunks(document_ids)


def _unique_hits(candidates: list[RetrievalHit], top_k: int, max_hits_per_document: int) -> list[RetrievalHit]:
    results: list[RetrievalHit] = []
    seen_sources: set[tuple[str, int | None, str]] = set()
    document_counts: dict[str, int] = {}
    for item in sorted(candidates, key=lambda hit: (-hit.score, hit.chunk_id)):
        source_key = source_dedupe_key(item)
        if source_key in seen_sources:
            continue
        if document_counts.get(item.document_id, 0) >= max_hits_per_document:
            continue
        seen_sources.add(source_key)
        document_counts[item.document_id] = document_counts.get(item.document_id, 0) + 1
        results.append(item)
        if len(results) >= top_k:
            break
    return results


class LexicalRetriever:
    """基于词项重叠度的确定性离线检索器。"""

    def __init__(self, store: Any, min_query_coverage: float = 0.5, max_hits_per_document: int = 2):
        # store 只需要提供 chunks()，因此可以替换 SQLite 或向量存储实现。
        if not 0 <= min_query_coverage <= 1:
            raise ValueError("minimum query coverage must be between 0 and 1")
        if max_hits_per_document <= 0:
            raise ValueError("maximum hits per document must be positive")
        self.store = store
        self.min_query_coverage = min_query_coverage
        self.max_hits_per_document = max_hits_per_document

    def search_candidates(self, query: str, candidate_k: int = 32, document_ids=None):
        """Use the same lexical ranking without the final document cap."""
        candidate_retriever = copy(self)
        candidate_retriever.max_hits_per_document = candidate_k
        return candidate_retriever.search(query, candidate_k, document_ids)

    def search(self, query: str, top_k: int = 4, document_ids: tuple[str, ...] | None = None) -> list[RetrievalHit]:
        """计算查询与各文本分块的相关性，并返回分数最高的结果。"""

        if top_k <= 0:
            return []

        # 查询词去重后参与计算，避免重复词放大相关性分数。
        query_terms = set(tokens(query))    #查询词去重。
        strict_hits: list[RetrievalHit] = []
        fallback_hits: list[RetrievalHit] = []
        for row in _scoped_rows(self.store, document_ids):
            # 将 SQLite 行或其他映射对象统一按键访问。
            record: Mapping[str, Any] = row
            chunk_terms = tokens(str(record["text"]))    #record 是 SQLite 行或字典，统一用 record["text"] 访问
            if not chunk_terms:
                continue
            # 用查询词和分块词的交集计算重叠度，再按词数进行归一化。
            overlap = len(query_terms.intersection(chunk_terms))    #记录重叠词的数量。查询词和块词的交集，即重叠了几个词
            coverage = overlap / len(query_terms) if query_terms else 0.0
            score = overlap / math.sqrt(max(len(query_terms), 1) * len(set(chunk_terms)))
            if score <= 0:
                continue
            strict_match = len(query_terms) == 1 or coverage >= self.min_query_coverage
            # Chinese bigrams can include query-only joins such as “卡几”
            # in “打卡几次”. When strict coverage finds nothing, keep chunks
            # matching at least two distinct terms as a generic fallback.
            if not strict_match and overlap < 2:
                continue
            metadata = {
                **_row_metadata(record),
                "query_coverage": round(coverage, 4),
                "retrieval_match": "strict" if strict_match else "fallback",
            }
            page = metadata.get("page")
            hit = RetrievalHit(
                chunk_id=str(_row_value(record, "id", "")),
                document_id=str(_row_value(record, "document_id", "")),
                text=str(_row_value(record, "text", "")),
                score=round(score, 4),
                position=int(_row_value(record, "position")) if _row_value(record, "position") is not None else None,
                filename=_row_value(record, "filename"),
                page=int(page) if page is not None else None,
                metadata=metadata,
            )
            (strict_hits if strict_match else fallback_hits).append(hit)
        # Keep distinct document/page sources even when their text is equal.
        # Collapse only repeated layout chunks from the same source page.
        results: list[RetrievalHit] = []
        seen_sources: set[tuple[str, int | None, str]] = set()
        document_counts: dict[str, int] = {}
        candidates = strict_hits if strict_hits else fallback_hits
        for item in sorted(candidates, key=lambda hit: hit.score, reverse=True):
            source_key = source_dedupe_key(item)
            if source_key in seen_sources:
                continue
            if document_counts.get(item.document_id, 0) >= self.max_hits_per_document:
                continue
            seen_sources.add(source_key)
            document_counts[item.document_id] = document_counts.get(item.document_id, 0) + 1
            results.append(item)
            if len(results) >= top_k:
                break
        return results


class BalancedLexicalRetriever(LexicalRetriever):
    """Lexical v2: retain weak candidates and favor an explicitly requested formula.

    This is a deterministic retrieval experiment, not semantic understanding. The
    formula signal uses the text around an equals sign and is domain independent.
    """

    _FORMULA_LABEL = re.compile(r"([\u4e00-\u9fff]{2,16})\s*[=＝]")
    _FORMULA_QUESTION = ("如何计算", "怎样计算", "怎么计算", "怎么算", "由哪些部分组成", "分别占多少")
    STRICT_BONUS = 0.18
    FORMULA_BONUS = 0.45
    MIN_FORMULA_LABEL_MATCH = 0.5

    def __init__(self, store: Any, min_query_coverage: float = 0.5, max_hits_per_document: int = 2):
        super().__init__(store, min_query_coverage, max_hits_per_document)

    def search(self, query: str, top_k: int = 4, document_ids: tuple[str, ...] | None = None) -> list[RetrievalHit]:
        if top_k <= 0:
            return []
        query_terms = set(tokens(query))
        if not query_terms:
            return []
        wants_formula = any(phrase in query for phrase in self._FORMULA_QUESTION)
        candidates: list[RetrievalHit] = []
        for row in _scoped_rows(self.store, document_ids):
            record: Mapping[str, Any] = row
            body = str(record["text"])
            chunk_terms = set(tokens(body))
            if not chunk_terms:
                continue
            overlap = len(query_terms & chunk_terms)
            # Candidate union: a weak match may contain the answer even when
            # unrelated title terms cause another chunk to pass the old gate.
            if overlap < (1 if len(query_terms) == 1 else 2):
                continue
            coverage = overlap / len(query_terms)
            score = overlap / math.sqrt(len(query_terms) * len(chunk_terms))
            if coverage >= self.min_query_coverage:
                score += self.STRICT_BONUS
            formula_match = 0.0
            if wants_formula:
                for label in self._FORMULA_LABEL.findall(body):
                    label_terms = set(tokens(label))
                    if label_terms:
                        formula_match = max(formula_match, len(query_terms & label_terms) / len(label_terms))
                if formula_match >= self.MIN_FORMULA_LABEL_MATCH:
                    score += self.FORMULA_BONUS * formula_match
            metadata = {
                **_row_metadata(record),
                "query_coverage": round(coverage, 4),
                "retrieval_match": "strict" if coverage >= self.min_query_coverage else "fallback",
                "formula_label_match": round(formula_match, 4),
                "retriever_version": "lexical_v2",
            }
            page = metadata.get("page")
            candidates.append(
                RetrievalHit(
                    chunk_id=str(_row_value(record, "id", "")),
                    document_id=str(_row_value(record, "document_id", "")),
                    text=body,
                    score=round(score, 4),
                    position=int(_row_value(record, "position")) if _row_value(record, "position") is not None else None,
                    filename=_row_value(record, "filename"),
                    page=int(page) if page is not None else None,
                    metadata=metadata,
                )
            )

        results: list[RetrievalHit] = []
        seen_sources: set[tuple[str, int | None, str]] = set()
        document_counts: dict[str, int] = {}
        for item in sorted(candidates, key=lambda hit: hit.score, reverse=True):
            source_key = source_dedupe_key(item)
            if source_key in seen_sources:
                continue
            if document_counts.get(item.document_id, 0) >= self.max_hits_per_document:
                continue
            seen_sources.add(source_key)
            document_counts[item.document_id] = document_counts.get(item.document_id, 0) + 1
            results.append(item)
            if len(results) >= top_k:
                break
        return results


class BM25Retriever(LexicalRetriever):
    """Chinese-bigram/Latin-word BM25 baseline over the current active chunks."""

    def __init__(self, store: Any, max_hits_per_document: int = 2, k1: float = 1.5, b: float = 0.75,
                 idf_exponent: float = 1.0):
        super().__init__(store, min_query_coverage=0.0, max_hits_per_document=max_hits_per_document)
        if k1 <= 0 or not 0 <= b <= 1 or idf_exponent < 1:
            raise ValueError("BM25 k1 must be positive, b must be between 0 and 1, and IDF exponent >= 1")
        self.k1 = k1
        self.b = b
        self.idf_exponent = idf_exponent
        self._cache: dict[tuple[tuple[str, ...] | None, int], tuple[list[Mapping[str, Any]], list[Counter[str]], Counter[str], float]] = {}

    def _corpus(self, document_ids: tuple[str, ...] | None):
        generation = int(getattr(self.store, "index_generation", 0))
        key = (document_ids, generation)
        if key not in self._cache:
            rows = _scoped_rows(self.store, document_ids)
            frequencies = [Counter(tokens(str(row["text"]))) for row in rows]
            document_frequency: Counter[str] = Counter()
            for frequency in frequencies:
                document_frequency.update(frequency.keys())
            average_length = sum(frequency.total() for frequency in frequencies) / len(frequencies) if frequencies else 0.0
            # Old generations are discarded after an upload, reindex, or delete.
            self._cache = {key: (rows, frequencies, document_frequency, average_length)}
        return self._cache[key]

    def search_candidates(
        self, query: str, candidate_k: int = 32, document_ids: tuple[str, ...] | None = None
    ) -> list[RetrievalHit]:
        """Return raw ranked candidates before text dedupe or per-document caps."""

        if candidate_k <= 0 or document_ids == ():
            return []
        query_terms = set(tokens(query))
        if not query_terms:
            return []
        rows, frequencies, df, average_length = self._corpus(document_ids)
        n = len(rows)
        candidates: list[RetrievalHit] = []
        for row, frequency in zip(rows, frequencies):
            matching_terms = query_terms & frequency.keys()
            if not matching_terms:
                continue
            length = frequency.total()
            score = 0.0
            for term in matching_terms:
                idf = math.log1p((n - df[term] + 0.5) / (df[term] + 0.5))
                tf = frequency[term]
                denominator = tf + self.k1 * (1 - self.b + self.b * length / average_length)
                # The default exponent is standard BM25. A higher exponent is
                # a separate experiment that prioritizes distinctive terms
                # over repeated document-title words in long questions.
                score += idf**self.idf_exponent * tf * (self.k1 + 1) / denominator
            metadata = {
                **_row_metadata(row),
                "query_coverage": round(len(matching_terms) / len(query_terms), 4),
                "retrieval_match": "bm25",
                "retriever_version": "bm25_idf_v2" if self.idf_exponent > 1 else "bm25_v1",
            }
            page = metadata.get("page")
            candidates.append(
                RetrievalHit(
                    chunk_id=str(_row_value(row, "id", "")),
                    document_id=str(_row_value(row, "document_id", "")),
                    text=str(row["text"]),
                    score=round(score, 4),
                    position=int(_row_value(row, "position")) if _row_value(row, "position") is not None else None,
                    filename=_row_value(row, "filename"),
                    page=int(page) if page is not None else None,
                    metadata=metadata,
                )
            )
        return sorted(candidates, key=lambda hit: (-hit.score, hit.chunk_id))[:candidate_k]

    def search(self, query: str, top_k: int = 4, document_ids: tuple[str, ...] | None = None) -> list[RetrievalHit]:
        if top_k <= 0:
            return []
        # Existing BM25 behavior remains capped; Hybrid calls search_candidates.
        candidates = self.search_candidates(
            query, candidate_k=max(len(self._corpus(document_ids)[0]), top_k), document_ids=document_ids
        )
        return _unique_hits(candidates, top_k, self.max_hits_per_document)


class BM25RareTermRetriever(BM25Retriever):    #BM25 + IDF²
    """BM25 experiment with squared IDF; the v1 baseline remains unchanged."""

    def __init__(self, store: Any, max_hits_per_document: int = 2, k1: float = 1.5, b: float = 0.75):
        super().__init__(store, max_hits_per_document, k1, b, idf_exponent=2.0)

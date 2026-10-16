"""Conservative heuristics for excluding navigation noise from the index."""

import re
from dataclasses import dataclass


# A contents entry uses a *run* of leader marks followed by a page number.
# Single periods in question/answer labels ("20.", "A.", "B.") are not leaders.
_PAGE_LEADER = re.compile(r"(?:\.{3,}|·{3,}|…{2,})\s*\d{1,4}\s*(?:页)?(?=\s|$)")
_OPTION_MARKER = re.compile(r"(?<![A-Za-z0-9])[A-H][.．、](?=\s|[\u4e00-\u9fff])")


@dataclass(frozen=True)
class ChunkQuality:
    exclude_reason: str | None = None
    flags: tuple[str, ...] = ()


def normalize_text(text: str) -> str:
    """Collapse layout whitespace so quality checks are stable across parsers."""

    return re.sub(r"\s+", " ", text).strip()


def assess_chunk_quality(text: str) -> ChunkQuality:
    """Exclude clear noise; retain ambiguous dot-heavy text for retrieval."""

    # 仅含标题与HTML注释的块没有正文；保留代码围栏中的注释讲解。
    if re.fullmatch(r"(?:\s*#{1,6}[^\n]*\n\s*)*<!--.*?-->\s*", text.strip(), re.S):
        return ChunkQuality(exclude_reason="comment_only")
    normalized = normalize_text(text)
    if len(normalized) < 40:
        # 短文的独立章节可能只有一句完整规则，不能因小于40字丢掉有效证据。
        # 仍排除页码、孤立标题和极短噪声；目录特征继续由下方规则判定。
        body = re.sub(r"(?m)^\s*#{1,6}[^\n]*\n?", "", text).strip()
        if len(normalized) < 20 or len(normalize_text(body)) < 12 or not re.search(r"[。！？；;]", body):
            return ChunkQuality(exclude_reason="too_short")

    leaders = len(_PAGE_LEADER.findall(normalized))
    if leaders >= 2 or (normalized.startswith("目录") and leaders >= 1):
        return ChunkQuality(exclude_reason="table_of_contents")

    # A high proportion of isolated periods is common in multiple-choice
    # material. Record it for inspection without deleting the evidence.
    dot_like = sum(char in ".·…" for char in normalized)
    dot_ratio = dot_like / len(normalized)
    flags = ("dot_heavy_unconfirmed",) if dot_ratio >= 0.04 and sum(char.isdigit() for char in normalized) >= 3 else ()

    # A page header/footer repeated across many pages is generally too weak to
    # stand alone as evidence. Keep normal prose that contains punctuation.
    punctuation = sum(char in "。！？；;，," for char in normalized)
    # 英文论文常用句号而没有逗号；已有目录引导符检查在前，不能把完整英文句子当标题删掉。
    western_sentence = bool(re.search(r"[.!?](?:\s|$)", normalized))
    if len(normalized) >= 80 and punctuation == 0 and not western_sentence and not _OPTION_MARKER.search(normalized):
        return ChunkQuality(exclude_reason="heading_only", flags=flags)
    return ChunkQuality(flags=flags)


def low_quality_reason(text: str) -> str | None:
    """Compatibility helper for offline evaluators and existing callers."""

    return assess_chunk_quality(text).exclude_reason


def is_low_quality_chunk(text: str) -> bool:
    return low_quality_reason(text) is not None

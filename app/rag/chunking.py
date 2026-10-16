"""RAG 导入流程中的可替换文本分块策略。"""

import re       #re：正则表达式，用于清洗文本。
import hashlib
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Protocol, Sequence

from app.rag.contracts import DocumentPage, TextChunk


class ChunkStrategy(Protocol):#Protocol 是Python 的“鸭子类型”接口。任何类只要有这些属性和方法，就“符合”这个协议，不需要显式继承。
    """Stable contract used by document indexing regardless of split strategy."""

    name: str
    version: str

    def split_document(
        self,
        text: str,
        pages: Sequence[DocumentPage],
        *,
        document_id: str,
    ) -> list[TextChunk]:
        ...


class TextChunker:
    """按固定窗口切分文本，并保留相邻分块之间的重叠内容。"""

    name = "fixed"
    version = "1"

    def __init__(self, size: int = 800, overlap: int = 120):#size：每块最大字符数，默认 800。overlap：相邻块的重叠字符数，默认 120。
        # 提前校验参数，避免切分过程中出现无限循环或空分块。
        if size <= 0:
            raise ValueError("chunk size must be positive")
        if overlap < 0 or overlap >= size:
            raise ValueError("chunk overlap must be >= 0 and smaller than size")
        self.size = size
        self.overlap = overlap

    @staticmethod
    def normalize(text: str) -> str:   #文本清洗
        """合并连续空白字符，并去除文本首尾空白。"""

        return re.sub(r"\s+", " ", text).strip() #\s+ 匹配一个或多个空白字符（空格、换行、制表符等）。替换成一个空格。.strip() 去掉首尾空白。

    def split(
        self,
        text: str,
        *,
        document_id: str = "",
        metadata: dict[str, object] | None = None,
    ) -> list[TextChunk]:
        """将文本切分为带来源位置和元数据的可检索分块。"""

        normalized = self.normalize(text)
        if not normalized:
            return []

        chunks: list[TextChunk] = []
        base_metadata = dict(metadata or {})
        start = 0
        position = 0
        while start < len(normalized):
            # 当前窗口的结束位置不会超过规范化文本长度。
            end = min(len(normalized), start + self.size)
            chunks.append(
                TextChunk(      #创建 TextChunk，记录文本、位置、起止字符。
                    document_id=document_id,
                    text=normalized[start:end],
                    position=position,
                    metadata={**base_metadata, "start_char": start, "end_char": end},
                )
            )
            position += 1
            if end == len(normalized):
                break
            # 下一块向前回退 overlap 个字符，以保留上下文连续性。
            start = max(end - self.overlap, start + 1)#max(..., start + 1) 是防死循环兜底：保证 start 每次都至少前进 1 个字符，避免卡住。
        return chunks

    def split_document(
        self,
        text: str,
        pages: Sequence[DocumentPage],
        *,
        document_id: str,
    ) -> list[TextChunk]:
        """Split page by page when page data exists, otherwise split the document."""

        if not pages:
            return self.split(text, document_id=document_id)

        chunks: list[TextChunk] = []
        for page in pages:
            chunks.extend(
                self.split(
                    page.text,
                    document_id=document_id,
                    metadata={"page": page.page_number},
                )
            )
        return [
            TextChunk(
                document_id=chunk.document_id,
                text=chunk.text,
                position=index,
                metadata=chunk.metadata,
            )
            for index, chunk in enumerate(chunks)
        ]


@dataclass(frozen=True) #递归切分器内部的结构单元。
class _Block:   #递归切分要先识别结构（标题、列表、段落），再决定怎么切。
    """A structural unit inside one page before recursive splitting."""

    text: str
    page: int | None
    block_type: str
    list_group: int | None = None


class RecursiveChunker:#递归切分器,先按结构分块，再按句子合并，最后按长度兜底。
    """Structure-aware splitter with sentence and length fallbacks.

    The splitter intentionally works page by page. This preserves citation
    locations and prevents a section at the end of one PDF page from being
    silently merged with unrelated text on the next page.
    """

    name = "recursive"
    version = "3.0"

    def __init__(
        self,
        target_size: int = 450, #目标块大小（软限制）
        overlap: int = 60,      #重叠字符数
        min_size: int = 40,     #最小块大小（避免太碎）
        max_size: int = 800,     #最大块大小（硬限制，避免太大）
    ) -> None:
        if target_size <= 0:
            raise ValueError("chunk target size must be positive")
        if overlap < 0 or overlap >= target_size:
            raise ValueError("chunk overlap must be >= 0 and smaller than target size")
        if min_size < 0 or max_size < target_size:
            raise ValueError("chunk size bounds are invalid")
        self.target_size = target_size
        self.overlap = overlap
        self.min_size = min_size
        self.max_size = max_size

    @staticmethod
    def normalize_line(text: str) -> str:   #按行清洗，保留换行符，因为换行是结构信号（段落、标题）。
        """Normalize layout noise without discarding structural line breaks."""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t\f\v]+", " ", text)
        # PDF extraction often inserts spaces between Chinese characters.
        text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
        text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[，。！？；：、（）])", "", text)
        return text.strip()

    @classmethod
    def _line_type(cls, line: str) -> str:
        if not line:
            return "paragraph"
        if len(line) > 80:
            return "list" if cls._is_list_item(line) else "paragraph"
        if re.match(r"^第[一二三四五六七八九十百千万0-9]+[章节篇]", line):
            return "heading"
        article = re.match(r"^第[一二三四五六七八九十百千万0-9]+条\s*(.*)$", line)
        if article:
            # “第六条” alone is a heading; “第六条 ……正文” is an
            # article block and must remain searchable evidence.
            return "heading" if not article.group(1).strip() else "article"
        if re.match(r"^[一二三四五六七八九十百千万]+、", line):
            return "heading"
        if re.match(r"^#{1,6}\s+\S", line):
            return "heading"
        if (
            len(line) <= 40
            and re.match(r"^\d+[.．、]\s*[\u4e00-\u9fffA-Za-z]", line)
            and not re.search(r"[。！？；;，,]", line)
        ):
            return "heading"
        if cls._is_list_item(line):
            return "list"
        # A short PDF extraction line is often only a wrapped sentence. Treat
        # it as prose unless it has an explicit heading marker above.
        return "paragraph"

    @staticmethod
    def _is_list_item(line: str) -> bool:
        return bool(
            re.match(r"^(?:[（(][一二三四五六七八九十百千万0-9]+[）)]|\d+[.．)）、](?!\d)|[-*•])\s*", line)
        )

    @staticmethod
    def _numbered_marker(line: str):
        match = re.match(r"^(\d+)([.．、)）])\s*(?!\d)(\S.*)$", line)
        if match:
            return int(match[1]), match[2]
        match = re.match(r"^[（(](\d+)[）)]\s*(\S.*)$", line)
        if match:
            return int(match[1]), "parenthesized"
        match = re.match(r"^([一二三四五六七八九十]+)、\s*(\S.*)$", line)
        if match:
            digits = {c: i for i, c in enumerate("一二三四五六七八九", 1)}
            value = match[1]
            if "十" in value and value.count("十") == 1:
                before, after = value.split("十")
                if len(before) <= 1 and len(after) <= 1:
                    return digits.get(before, 1) * 10 + digits.get(after, 0), "chinese"
            elif value in digits:
                return digits[value], "chinese"
        return None

    @classmethod
    def _classify_lines(cls, lines):
        """Resolve consecutive numbered items using surrounding document lines.

        Pure numbered chapter headings remain headings. A sequential run with
        prose items, a list introduction, or compact items beneath a heading
        is a list. Classifying across page boundaries preserves continuations.
        """
        kinds = [cls._line_type(line) for line in lines]
        groups = [None] * len(lines)
        run = []

        def finish():
            if not run:
                return
            first = run[0]
            previous = next((i for i in range(first - 1, -1, -1) if lines[i]), None)
            intro = previous is not None and bool(re.search(r"[:：]$|如下|包括|分为|列举", lines[previous]))
            indices = set(run)
            compact = len(run) > 1 and all(not lines[i] or i in indices for i in range(first, run[-1] + 1))
            beneath_heading = previous is not None and kinds[previous] == "heading"
            prose = any(kinds[i] == "list" or len(lines[i]) > 80 or bool(re.search(r"[。！？；;]", lines[i])) for i in run)
            if prose or (len(run) > 1 and (intro or (compact and beneath_heading))):
                for i in run:
                    kinds[i], groups[i] = "list", first
            run.clear()

        previous_marker = None
        for i, line in enumerate(lines):
            marker = cls._numbered_marker(line)
            if marker:
                if previous_marker is None or marker != (previous_marker[0] + 1, previous_marker[1]):
                    finish()
                run.append(i)
                previous_marker = marker
            elif kinds[i] == "heading":
                finish()
                previous_marker = None
        finish()
        # Consecutive bullet entries also form one atomic list when it fits.
        bullets = []
        def finish_bullets():
            if bullets:
                for i in bullets:
                    groups[i] = bullets[0]
                bullets.clear()
        for i, line in enumerate(lines):
            if re.match(r"^[-*•]\s*\S", line) and kinds[i] == "list":
                bullets.append(i)
            elif kinds[i] == "heading" or cls._numbered_marker(line):
                finish_bullets()
        finish_bullets()
        return kinds, groups

    @classmethod
    def _blocks_for_page(cls, text: str, page: int | None, *, line_types=None, list_groups=None) -> list[_Block]:
        lines = [cls.normalize_line(line) for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
        if line_types is None:
            line_types, list_groups = cls._classify_lines(lines)
        blocks: list[_Block] = []
        current: list[str] = []
        current_type = "paragraph"
        current_group = None

        def flush() -> None:
            nonlocal current
            if current:
                value = cls.normalize_line(" ".join(current))
                if value:
                    blocks.append(_Block(value, page, current_type, current_group))
            current = []

        for index, line in enumerate(lines):
            if not line:
                flush()
                current_type = "paragraph"
                continue
            line_type = line_types[index]
            if line_type == "heading":
                flush()
                blocks.append(_Block(line, page, "heading"))
                current_type = "paragraph"
                continue
            # PDF extraction wraps a list item across physical lines. Keep a
            # continuation with the item when the prior line is unfinished.
            continues_list = (
                current
                and current_type == "list"
                and line_type == "paragraph"
                and not current[-1].endswith(("。", "！", "？", "；", ".", "!", "?", ";"))
            )
            if current and current_type == "list" and line_type == "list":
                # Each numbered item is a separate semantic unit. Packing may
                # combine adjacent short items, but a long item stays intact.
                flush()
            if current and line_type != current_type and not continues_list:
                flush()
            if not continues_list:
                current_type = line_type
                current_group = list_groups[index]
            current.append(line)
        flush()
        packed = []
        for block in blocks:
            if (packed and block.list_group is not None and packed[-1].list_group == block.list_group
                    and packed[-1].block_type == block.block_type == "list"):
                previous = packed[-1]
                packed[-1] = replace(previous, text=previous.text + "\n" + block.text)
            else:
                packed.append(block)
        return packed

    @staticmethod
    def _sentence_units(text: str) -> list[tuple[int, int]]:    # 句子切分
        punctuation = set("。！？；.!?;")
        units: list[tuple[int, int]] = []
        start = 0
        for index, char in enumerate(text):
            if char in punctuation:
                units.append((start, index + 1))
                start = index + 1
        if start < len(text):
            units.append((start, len(text)))
        return [(start, end) for start, end in units if text[start:end].strip()]

    def _spans(self, text: str) -> list[tuple[int, int]]:   #合并句子成块
        """Pack sentence units and use hard windows only for oversized units."""

        if len(text) <= self.target_size:
            return [(0, len(text))]

        units = self._sentence_units(text) or [(0, len(text))]
        spans: list[tuple[int, int]] = []
        current_start: int | None = None
        current_end: int | None = None

        for unit_start, unit_end in units:
            if unit_end - unit_start > self.max_size:
                if current_start is not None and current_end is not None:
                    spans.append((current_start, current_end))
                    current_start = current_end = None
                piece_start = unit_start
                while piece_start < unit_end:
                    piece_end = min(unit_end, piece_start + self.target_size)
                    spans.append((piece_start, piece_end))
                    if piece_end == unit_end:
                        break
                    # Keep raw spans non-overlapping; the common overlap pass
                    # below applies the same boundary rule to every span.
                    piece_start = piece_end
                continue

            if current_start is None:
                current_start, current_end = unit_start, unit_end
            elif unit_end - current_start <= self.target_size:
                current_end = unit_end
            else:
                spans.append((current_start, current_end))
                current_start, current_end = unit_start, unit_end

        if current_start is not None and current_end is not None:
            spans.append((current_start, current_end))

        # Avoid leaving a tiny tail when it can be merged without exceeding
        # the hard maximum. Oversized hard-window spans are already overlapped
        # and are intentionally left unchanged.
        merged: list[tuple[int, int]] = []
        for span_start, span_end in spans:
            if (
                merged
                and span_start >= merged[-1][1]
                and span_end - span_start < self.min_size
                and span_end - merged[-1][0] <= self.max_size
            ):
                merged[-1] = (merged[-1][0], span_end)
            else:
                merged.append((span_start, span_end))
        if (
            len(merged) > 1
            and merged[-1][0] >= merged[-2][1]
            and merged[-1][1] - merged[-1][0] < self.min_size
            and merged[-1][1] - merged[-2][0] <= self.max_size
        ):
            merged[-2] = (merged[-2][0], merged[-1][1])
            merged.pop()
        spans = merged

        # Add the same character overlap to every adjacent span. Raw spans
        # remain non-overlapping so oversized units follow the same rule.
        result: list[tuple[int, int]] = []
        for index, (start, end) in enumerate(spans):
            if index and end - start <= self.target_size:
                desired = max(0, start - self.overlap)
                boundary_start = max(0, start - self.overlap * 2)
                boundaries = [
                    position + 1
                    for position in range(boundary_start, start)
                    if text[position].isspace() or text[position] in "。！？；.!?;"
                ]
                # Prefer a word/sentence boundary to a mid-token overlap.
                start = max(boundaries) if boundaries else desired
            start = max(start, end - self.max_size)
            result.append((start, end))
        return result

    def split_document(
    #逐页处理。每页拆成结构块。遇到标题 → flush 当前，记录 section_title。
    # 非标题 → 累积到 pending，超过 target_size 就 flush。
    # flush 时：合并 pending 文本 → _spans 切 → 每块带上 section_title、page、block_type 等元数据。
        self,
        text: str,
        pages: Sequence[DocumentPage],
        *,
        document_id: str,
    ) -> list[TextChunk]:
        source_pages: list[tuple[int | None, str]]
        if pages:
            source_pages = [(page.page_number, page.text) for page in pages]
        else:
            source_pages = [(None, text)]

        chunks: list[TextChunk] = []
        section_title: str | None = None
        section_number = 0

        def section_identity():
            value = f"{document_id}\0{section_number}\0{section_title or ''}"
            return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]

        page_lines = [[self.normalize_line(line) for line in page_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
                      for _, page_text in source_pages]
        types, groups = self._classify_lines([line for lines in page_lines for line in lines])
        cursor = 0
        for (page_number, page_text), lines in zip(source_pages, page_lines):
            blocks = self._blocks_for_page(page_text, page_number,
                line_types=types[cursor:cursor + len(lines)], list_groups=groups[cursor:cursor + len(lines)])
            cursor += len(lines)
            pending: list[_Block] = []

            def flush() -> None:
                nonlocal pending
                if not pending:
                    return
                body = "\n".join(block.text for block in pending)
                block_types = {block.block_type for block in pending}
                # The hard limit includes the inherited title and overlap.
                budget = self.max_size - (len(section_title) + 1 if section_title else 0)
                if budget <= 0:
                    raise ValueError("section title exceeds chunk maximum")
                splitter = RecursiveChunker(target_size=min(self.target_size, budget),
                    overlap=min(self.overlap, min(self.target_size, budget) - 1),
                    min_size=min(self.min_size, budget), max_size=budget)
                list_group = pending[0].list_group if len(pending) == 1 else None
                spans = [(0, len(body))] if list_group is not None and len(body) <= budget else splitter._spans(body)
                group_id = (hashlib.sha256(f"{section_identity()}:{list_group}".encode()).hexdigest()[:24]
                            if list_group is not None else None)
                for start, end in spans:
                    passage = body[start:end].strip()
                    if not passage:
                        continue
                    value = f"{section_title}\n{passage}" if section_title else passage
                    chunks.append(
                        TextChunk(
                            document_id=document_id,
                            text=value,
                            position=len(chunks),
                            metadata={
                                "page": page_number,
                                "section_title": section_title,
                                "section_id": section_identity(),
                                "section_ordinal": section_number,
                                "list_group_id": group_id,
                                "block_type": next(iter(block_types)) if len(block_types) == 1 else "mixed",
                                "start_char": start,
                                "end_char": end,
                                "offset_basis": "normalized_section",
                                "char_count": len(value),
                                "chunk_strategy": self.name,
                                "chunk_version": self.version,
                            },
                        )
                    )
                pending = []

            for block in blocks:
                if block.block_type == "heading":
                    flush()
                    section_title = block.text
                    section_number += 1
                    continue

                candidate_length = sum(len(item.text) + 1 for item in pending) + len(block.text)
                if pending and candidate_length > self.target_size:
                    flush()
                pending.append(block)
            flush()

            if blocks and all(block.block_type == "heading" for block in blocks):
                heading = blocks[-1]
                chunks.append(
                    TextChunk(
                        document_id=document_id,
                        text=heading.text,
                        position=len(chunks),
                        metadata={
                            "page": page_number,
                            "section_title": heading.text,
                            "section_id": section_identity(),
                            "section_ordinal": section_number,
                            "block_type": "heading",
                            "start_char": 0,
                            "end_char": len(heading.text),
                            "offset_basis": "normalized_section",
                            "char_count": len(heading.text),
                            "chunk_strategy": self.name,
                            "chunk_version": self.version,
                        },
                    )
                )
        sections, lists = defaultdict(list), defaultdict(list)
        for index, chunk in enumerate(chunks):
            sections[chunk.metadata["section_id"]].append(index)
            if chunk.metadata.get("list_group_id"):
                lists[chunk.metadata["list_group_id"]].append(index)
        for indices in sections.values():
            for part, index in enumerate(indices):
                chunks[index] = replace(chunks[index], metadata={**chunks[index].metadata,
                    "section_chunk_index": part, "section_chunk_count": len(indices)})
        for indices in lists.values():
            for part, index in enumerate(indices):
                chunks[index] = replace(chunks[index], metadata={**chunks[index].metadata,
                    "list_part_index": part, "list_part_count": len(indices)})
        return chunks


def create_chunker(
    strategy: str,
    *,
    target_size: int = 450,
    overlap: int = 60,
    min_size: int = 40,
    max_size: int = 800,
) -> ChunkStrategy:
    """Build a configured chunk strategy without coupling callers to classes."""

    normalized = strategy.strip().lower()
    if normalized in {"fixed", "legacy"}:
        # Keep the historical fixed_v1 behavior reproducible for A/B
        # evaluation. Recursive settings must not silently rewrite legacy
        # chunks when the fallback strategy is selected.
        return TextChunker(size=800, overlap=120)
    if normalized == "recursive":
        return RecursiveChunker(
            target_size=target_size,
            overlap=overlap,
            min_size=min_size,
            max_size=max_size,
        )
    raise ValueError(f"unsupported chunk strategy: {strategy}")


def split_chunks(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """兼容旧调用方的辅助函数，只返回分块文本而不返回元数据。"""

    return [chunk.text for chunk in TextChunker(size=size, overlap=overlap).split(text)]

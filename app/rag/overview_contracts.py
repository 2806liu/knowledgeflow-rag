"""概述的数据契约：模型只提交要点与原文编号，展示编号由程序统一生成。"""

from __future__ import annotations

import hashlib
import json
import re


def validate_units(raw: str, allowed_numbers: set[int]) -> list[dict]:
    """拒绝无来源或越出本批原文的要点，避免中间摘要伪装成新证据。"""
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"units"} or not isinstance(value["units"], list):
        raise ValueError("overview units schema invalid")
    units = []
    for item in value["units"]:
        if not isinstance(item, dict) or set(item) != {"title", "claim", "source_numbers"}:
            raise ValueError("overview unit schema invalid")
        title, claim, numbers = item["title"], item["claim"], item["source_numbers"]
        if not isinstance(title, str) or not title.strip() or not isinstance(claim, str) or not claim.strip():
            raise ValueError("overview unit text missing")
        if not isinstance(numbers, list) or not numbers or any(type(n) is not int for n in numbers):
            raise ValueError("overview unit sources missing")
        if not set(numbers) <= allowed_numbers:
            raise ValueError("overview unit source outside input")
        # 模型不得自带可与展示编号混淆的编号；引用标记统一在最终展示时添加。
        units.append({"title": re.sub(r"^\s*\d+[.、)]\s*", "", title).strip()[:120],
                      "claim": re.sub(r"\[\d+\]", "", claim).strip(),
                      "source_numbers": list(dict.fromkeys(numbers))})
    if not units:
        raise ValueError("overview produced no units")
    return units


def unit_key(overview_id: str, unit: dict, citations: dict[int, dict]) -> str:
    """内容与来源未变的条目跨部分概述修订保持同一内部键。"""
    chunks = sorted(citations[n]["chunk_id"] for n in unit["source_numbers"])
    normalized = " ".join((unit["title"] + " " + unit["claim"]).split()).casefold()
    digest = hashlib.sha256(json.dumps([overview_id, normalized, chunks], ensure_ascii=False).encode()).hexdigest()
    return digest[:24]


def publish_units(overview_id: str, units: list[dict], citations: list[dict]) -> tuple[str, tuple[int, ...], list[dict]]:
    """一次概述从 1 编号；每条直接记录原始 chunk，供后续定位和审计。"""
    by_number = {c["citation_number"]: c for c in citations}
    lines, used, published = [], [], []
    for index, unit in enumerate(units, 1):
        numbers = [n for n in unit["source_numbers"] if n in by_number]
        if not numbers:
            continue
        refs = [{"document_id": by_number[n]["document_id"],
                 "document_version": by_number[n]["document_version"],
                 "chunk_id": by_number[n]["chunk_id"], "page": by_number[n].get("page")}
                for n in numbers]
        marker = "".join(f"[{n}]" for n in numbers)
        title = unit["title"].strip()
        line = f"{index}. {title}：{unit['claim'].strip()}{marker}"
        lines.append(line)
        used.extend(numbers)
        published.append({"unit_key": unit_key(overview_id, unit, by_number),
                          "unit_id": str(index), "ordinal": index, "display_number": index,
                          "title": title, "claim": unit["claim"], "text": line,
                          "source_refs": refs, "document_ids": list(dict.fromkeys(r["document_id"] for r in refs)),
                          "cited_chunk_ids": [r["chunk_id"] for r in refs], "source_type": "document"})
    return "\n".join(lines), tuple(dict.fromkeys(used)), published

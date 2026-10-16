"""增强问答的方面检索计划：短查询、轮流取证与原文块去重。"""

import re

from app.rag.contracts import source_dedupe_key
from app.rag.comparison_task import search_query


def retrieval_plans(resolution, fallback_query, *, supplemental=False, missing_document_ids=()):
    """只拆独立文档事实；历史条目、贡献检索和补检索保留既定查询。"""
    task = resolution.get("comparison_task")
    if task:
        # 比较任务按文档和用户实际要求的维度检索；补检索只访问缺少证据的一侧。
        targets = list(missing_document_ids) if missing_document_ids else task["document_ids"]
        return [(aspect["id"],
                 search_query(task.get("aspect_topics", {}).get(aspect["id"], task["topic"]),
                              task["filenames"][aspect["document_id"]], alternate=supplemental),
                 aspect["document_id"])
                for aspect in resolution["requested_aspects"]
                if aspect.get("document_id") in targets]
    aspects = resolution.get("requested_aspects", [])
    if (supplemental or resolution.get("intent") != "document_fact" or len(aspects) < 2
            or resolution.get("retrieval_query") or resolution.get("referenced_unit_ids")):
        return [(None, fallback_query, None)]
    plans = []
    for aspect in aspects:
        query = str(aspect.get("question", "")).strip()
        # 删去检索泛词，保留“时间”“次数”等区分不同证据的核心词。
        query = re.sub(r"^(?:请问|查询|查找)", "", query)
        query = re.sub(r"(?:具体的|规定的|相关的|要求)$", "", query)
        query = query.replace("规定的", "").replace("具体的", "").strip(" ？?，,。")
        plans.append((aspect["id"], query or fallback_query, None))
    return plans


def select_evidence(previous, batches, limit):
    """每轮给每个方面一次入选机会，保持已有证据顺序并按 chunk ID 去重。"""
    selected = list(previous)
    seen = {hit.chunk_id for hit in selected}
    seen_sources = {source_dedupe_key(hit) for hit in selected}
    depth = 0
    while len(selected) < limit and any(depth < len(hits) for _, hits in batches):
        for _, hits in batches:
            if depth >= len(hits):
                continue
            hit = hits[depth]
            source = source_dedupe_key(hit)
            if hit.chunk_id not in seen and source not in seen_sources:
                selected.append(hit)
                seen.add(hit.chunk_id)
                seen_sources.add(source)
                if len(selected) >= limit:
                    break
        depth += 1
    return selected

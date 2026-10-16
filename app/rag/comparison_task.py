"""双文档比较计划：固定来源版本、比较方面与英文原文检索词。"""

import re
import uuid


PAIR = re.compile(r"这两篇|两篇(?:论文|文章|文档|资料)|两者|二者|分别(?:列出|说明).{0,10}(?:两篇|两份)|d1\s*(?:和|与|、|/)\s*d2", re.I)
TOPIC = re.compile(r"侧重点|研究重点|研究方向|主要结论|结论|核心方法|方法|创新点|贡献|区别|差异|比较")
FOLLOWUP = re.compile(r"依据不足|补充|不知道|不清楚|当前的|这两篇|两者|二者|分别|各自|侧重点|研究重点|研究方向|结论|核心方法|区别|差异|比较|^d1\s*(?:和|与|、|/)\s*d2$", re.I)


def topic(question, previous="研究侧重点"):
    """短追问仅替换明确提出的比较维度，其他措辞沿用最近待完成任务。"""
    if re.search(r"结论|结果", question):
        return "主要结论"
    if re.search(r"核心方法|方法", question):
        return "核心方法"
    if re.search(r"创新点|贡献", question):
        return "研究贡献"
    if re.search(r"侧重点|研究重点|研究方向|区别|差异|比较", question):
        return "研究侧重点"
    return previous


def topics(question, previous="研究侧重点"):
    """复合追问同时保留研究侧重点与主要结论，不能只留下后出现的维度。"""
    result = []
    if re.search(r"侧重点|研究重点|研究方向|区别|差异|比较", question):
        result.append("研究侧重点")
    if re.search(r"结论|结果", question):
        result.append("主要结论")
    if re.search(r"核心方法|方法", question):
        result.append("核心方法")
    if re.search(r"创新点|贡献", question):
        result.append("研究贡献")
    return result or [previous]


def make_plan(question, document_ids, versions, documents, *, previous=None, method="selected_pair"):
    """把当前两份资料映射成可读方面；内部 UUID 和短编号不进入用户答案。"""
    names = {item["id"]: item["filename"] for item in documents}
    if len(document_ids) != 2 or any(doc not in names for doc in document_ids):
        return None
    dimensions = topics(question, (previous or {}).get("topic", "研究侧重点"))
    dimension = dimensions[0]
    comparison = bool(re.search(r"区别|差异|比较|异同", question))
    if previous and not TOPIC.search(question):
        comparison = previous.get("mode") == "compare"
    if re.search(r"分别|各自", question) and not re.search(r"区别|差异|比较|异同", question):
        comparison = False
    aspects = []
    aspect_topics = {}
    for item_topic in dimensions:
        for doc in document_ids:
            aspect_id = f"a{len(aspects)+1}"
            aspects.append(dict(id=aspect_id, question=f"《{names[doc]}》的{item_topic}",
                                requires_document=True, document_id=doc))
            aspect_topics[aspect_id] = item_topic
        if comparison and item_topic == dimension:
            aspect_id = f"a{len(aspects)+1}"
            aspects.append(dict(id=aspect_id, question=f"两篇文章在{item_topic}上的区别",
                                requires_document=True, comparison_document_ids=list(document_ids)))
            aspect_topics[aspect_id] = item_topic
    # 首问分配稳定任务 ID；追问改变维度时沿用该 ID，仍逐轮保存原始问题。
    task = dict(task_id=(previous or {}).get("task_id") or str(uuid.uuid4()),
                document_ids=list(document_ids), versions={doc: versions[doc] for doc in document_ids},
                filenames={doc: names[doc] for doc in document_ids}, topic=dimension,
                topics=dimensions, aspect_topics=aspect_topics,
                mode="compare" if comparison else "list", original_question=(previous or {}).get("original_question", question))
    return dict(intent="comparison", resolved_query=question, document_ids=list(document_ids),
                requested_aspects=aspects, comparison_task=task, method=method,
                referenced_turn_id=None, referenced_unit_id=None, referenced_unit_ids=[],
                candidate_keys=[], clarification_question=None)


def search_query(dimension, filename, *, alternate=False):
    """限定到单篇后先查英文章节词；仅在补检索时加入文件名定位词。"""
    title = re.sub(r"(?i)\.pdf$", "", filename).replace("_", " ").replace("-", " ")
    words = re.findall(r"[A-Za-z]{4,}", title)
    title_terms = " ".join(words[-7:])
    if dimension == "主要结论":
        keywords = ("conclusion conclusions concluding summary discussion findings result results"
                    if not alternate else "summary concluding remarks discussion contribution")
    elif dimension == "核心方法":
        keywords = "method approach algorithm control formulation" if not alternate else "proposed technique solution"
    elif dimension == "研究贡献":
        keywords = "contribution novelty proposed method" if not alternate else "abstract introduction conclusion"
    else:
        keywords = "abstract introduction objective problem approach" if not alternate else "conclusion contribution method purpose"
    # 文件名中的罕见词会让封面和标题页压过正文，已按文档限定的首轮查询不需要它。
    return (keywords + (" " + title_terms if alternate else "")).strip()


def display_aliases(text, document_ids, documents):
    """用户可见澄清只显示文件名；短编号仅在解析模型的私有输入中使用。"""
    names = {item["id"]: item["filename"] for item in documents}
    def replace(match):
        index = int(match.group(1)) - 1
        if index < 0 or index >= len(document_ids):
            return match.group(0)
        return "《" + names.get(document_ids[index], "当前文档") + "》"
    return re.sub(r"\bd([1-9][0-9]*)\b", replace, text, flags=re.I)

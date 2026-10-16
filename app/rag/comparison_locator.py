"""比较对象定位：历史任务只决定两篇资料，论文事实仍须重新读取当前原文。"""

import re

from app.rag.comparison_task import PAIR, TOPIC, FOLLOWUP, make_plan


HISTORY = re.compile(r"前面|之前|历史对话|上次|刚才.{0,10}两篇")
SHORT_FOCUS = re.compile(r"^(?:各自的?)?(?:研究重点|侧重点)(?:是什么)?[？?。\s]*$")
EACH_AND_DIFFERENCE = re.compile(r"各自.{0,24}(?:研究重点|侧重点).{0,12}(?:区别|差异)")
STOP_WORDS = {"pdf", "synthetic", "journal", "paper", "document", "documents", "ieee", "transactions"}


def _aliases(filename):
    """从文件名提取作者/缩写等可辨识词，忽略年份和格式通用词。"""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", filename.lower())
    return [word for word in tokens if word not in STOP_WORDS]


def _named_documents(question, documents):
    """只在问题确实写出独特名称时才按文件名绑定资料。"""
    words = set(re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", question.lower()))
    return [doc["id"] for doc in documents if words.intersection(_aliases(doc["filename"]))]


class ComparisonTaskLocator:
    """优先显式文件名，其次历史任务，再采用当前恰好两篇的范围。"""

    @staticmethod
    def clarification(question, scope, message, method):
        """范围外与歧义是业务澄清，不作为解析模型故障。"""
        return dict(intent="clarification", resolved_query=question,
                    document_ids=list(scope["document_ids"]),
                    requested_aspects=[dict(id="a1", question=question, requires_document=False)],
                    referenced_turn_id=None, referenced_unit_id=None,
                    referenced_unit_ids=[], candidate_keys=[], clarification_question=message,
                    method=method)

    def locate(self, question, snapshot, tasks, current_documents, available_documents,
               *, recovery=False):
        """返回计划和脱敏轨迹；绝不将范围外文档直接放入检索计划。"""
        current_ids = list(snapshot["document_ids"])
        available = {doc["id"]: doc for doc in available_documents}
        named = _named_documents(question, available_documents)
        historical = bool(HISTORY.search(question))
        wants_comparison = bool((PAIR.search(question) and TOPIC.search(question)) or
                                SHORT_FOCUS.fullmatch(question.strip()) or
                                EACH_AND_DIFFERENCE.search(question) or
                                (len(named) >= 2 and TOPIC.search(question)) or
                                (len(named) == 2 and any(
                                    set(task["document_ids"]) == set(named) for task in tasks)) or
                                (historical and TOPIC.search(question)) or
                                (recovery and (historical or len(named) >= 2)))
        if not wants_comparison:
            return None, {"status": "not_comparison"}
        selected = None
        reason = None
        if len(named) >= 2:
            # 显式写出的两个文件优先；同名别名匹配超过两篇时交给用户选择。
            if len(named) != 2:
                names = "、".join(available[d]["filename"] for d in named[:6])
                return self.clarification(question, snapshot, "文件名称可能对应多份资料，请选择两篇：" + names,
                                          "ambiguous_comparison_files"), {"status": "ambiguous_names"}
            selected = next((task for task in tasks if set(task["document_ids"]) == set(named)), None)
            reason = "explicit_filenames"
        elif historical and tasks:
            # 用户写出部分文件名时只能在含该文件的旧任务中选最近一次，不能误取另一对。
            matching = [task for task in tasks if set(named) <= set(task["document_ids"])]
            selected = matching[0] if matching else None
            reason = "latest_historical_task"
        elif len(current_ids) == 2:
            selected = next((task for task in tasks if set(task["document_ids"]) == set(current_ids)
                             and task["versions"] == snapshot["versions"]), None)
            reason = "selected_pair"
        elif (PAIR.search(question) or recovery) and tasks:
            selected = tasks[0]
            reason = "latest_historical_task"
        if selected:
            ids = selected["document_ids"]
            names = [selected["filenames"][doc] for doc in ids]
            if any(doc not in available for doc in ids):
                return self.clarification(question, snapshot,
                    "历史比较涉及的资料已删除或未就绪：" + "、".join(names) + "。请重新添加资料。",
                    "historical_comparison_unavailable"), {"status": "unavailable", "task_id": selected["task_id"],
                                                              "document_ids": ids, "filenames": selected["filenames"]}
            if not set(ids) <= set(current_ids):
                return self.clarification(question, snapshot,
                    "历史比较涉及的资料仍在知识库，但未全部选入本轮范围。请选中：" + "、".join(names) + "。",
                    "historical_comparison_out_of_scope"), {"status": "out_of_scope", "task_id": selected["task_id"],
                                                               "document_ids": ids, "filenames": selected["filenames"]}
            versions = {doc: snapshot["versions"][doc] for doc in ids}
            changed = versions != selected["versions"]
            if changed and reason != "explicit_filenames":
                return self.clarification(question, snapshot,
                    "这两篇资料的索引版本已变化，请明确写出文件名后重新提问；旧原文引用不会继续使用。",
                    "historical_comparison_version_changed"), {"status": "version_changed", "task_id": selected["task_id"],
                                                                "document_ids": ids, "filenames": selected["filenames"]}
            # 明确指定文件名且版本变化时创建新任务，从当前版本重新检索。
            previous = None if changed else selected
            plan = make_plan(question, ids, versions, [available[d] for d in ids],
                             previous=previous, method="historical_comparison_" + reason)
            return plan, {"status": "replanned_current_version" if changed else "matched",
                          "task_id": selected["task_id"], "document_ids": ids,
                          "filenames": selected["filenames"], "reason": reason}
        if len(named) == 2:
            if not set(named) <= set(current_ids):
                names = [available[doc]["filename"] for doc in named]
                return self.clarification(question, snapshot,
                    "这两篇资料未全部选入本轮范围，请选中：" + "、".join(names) + "。",
                    "named_comparison_out_of_scope"), {"status": "out_of_scope",
                                                      "document_ids": named,
                                                      "filenames": {doc: available[doc]["filename"] for doc in named}}
            plan = make_plan(question, named, snapshot["versions"], current_documents,
                             method="named_pair_comparison")
            return plan, {"status": "matched", "document_ids": named,
                          "filenames": {doc: available[doc]["filename"] for doc in named},
                          "reason": "explicit_filenames"}
        if len(current_ids) == 2 and not historical:
            plan = make_plan(question, current_ids, snapshot["versions"], current_documents,
                             method="selected_pair_comparison")
            return plan, {"status": "matched", "document_ids": current_ids,
                          "filenames": {doc["id"]: doc["filename"] for doc in current_documents},
                          "reason": "selected_pair"}
        if historical or recovery:
            return self.clarification(question, snapshot,
                "没有找到能核对来源的历史两篇比较任务。请写出两篇文件名，并在本轮选择它们。",
                "historical_comparison_not_found"), {"status": "not_found"}
        return None, {"status": "not_comparison"}

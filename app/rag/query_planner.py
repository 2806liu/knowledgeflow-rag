"""问答路由边界：全文概述、定向事实和历史条目追问使用不同来源路径。"""

import re

from app.rag.dialogue_focus import candidates
from app.rag.comparison_task import PAIR, TOPIC, FOLLOWUP, make_plan
from app.rag.comparison_locator import ComparisonTaskLocator


_CONTRIBUTION = re.compile(r"创新点|研究贡献|主要贡献|技术贡献|novelty|contribution", re.I)
_RESUME = re.compile(r"^(?:请|帮我)?(?:继续|接着)(?:完成|做)?(?:概述|总结|上次的概述|剩余部分)(?:剩余部分)?[。！!？?\s]*$")
_ORDINAL = re.compile(r"第\s*[二2]\s*[点条项个]")
_CORRECTION = re.compile(r"不是.{0,30}?[，,；;]\s*(?:我)?(?:指的是|指|说的是)([^，。；;!?？]{2,60})")


def c_title_matches(question, title):
    """只接受明确标题或“选择标题”，避免把普通追问误当成候选选择。"""
    return question.strip(" 。！!？?") in {title, "选" + title, "选择" + title}


class QueryPlanner:
    """只守住确定性边界；不明白的自然语言仍交给现有解析器处理。"""

    @staticmethod
    def correction(question):
        match = _CORRECTION.search(question)
        return match.group(1).strip() if match else None

    def preplan(self, question, snapshot, memory, focus=None, enhanced=False, documents=None,
                comparison_tasks=(), available_documents=(), lookup_events=None):
        """续跑沿用原问题与版本键；含多个历史第二点且无轮次时先请用户选标题。"""
        if enhanced:
            plan, trace = ComparisonTaskLocator().locate(
                question, snapshot, comparison_tasks, documents or [], available_documents or documents or [])
            if trace["status"] != "not_comparison" and lookup_events is not None:
                lookup_events.append({"node": "task_lookup", "status": trace["status"], "output": trace})
                lookup_events.append({"node": "scope_check",
                                      "status": "blocked" if trace["status"] in {
                                          "out_of_scope", "unavailable", "version_changed"} else "ok",
                                      "output": {"document_ids": trace.get("document_ids", []),
                                                 "reason": trace["status"]}})
            if plan is not None:
                return plan
        # 已选两篇与上轮未解决任务共同确定对象；简短追问沿用最近问题，不回退到更早的打卡资料。
        if enhanced and len(snapshot["document_ids"]) == 2:
            latest_turn = memory["turns"][-1] if memory["turns"] else None
            recent_task = ((latest_turn or {}).get("resolution") or {}).get("comparison_task") or {}
            pending_task = (focus or {}).get("pending_comparison_task") or recent_task
            same_sources = (pending_task.get("document_ids") == snapshot["document_ids"]
                            and pending_task.get("versions") == snapshot["versions"])
            if PAIR.search(question) and TOPIC.search(question):
                return make_plan(question, snapshot["document_ids"], snapshot["versions"],
                                 documents or [], method="selected_pair_comparison")
            if same_sources and FOLLOWUP.search(question):
                return make_plan(question, snapshot["document_ids"], snapshot["versions"],
                                 documents or [], previous=pending_task, method="pending_comparison_followup")
        # 严格模式的上一轮澄清也保留候选键；标题回复直接回到对应历史条目。
        latest = memory["turns"][-1] if memory["turns"] else None
        pending = (latest or {}).get("resolution", {}) if latest and latest.get("response", {}).get(
            "answer_status") == "clarification" else {}
        if pending.get("candidate_keys"):
            catalog = {c["key"]: c for c in candidates(memory, focus or {}, question)}
            selected = [catalog[k] for k in pending["candidate_keys"] if k in catalog and
                        c_title_matches(question, catalog[k]["title"])]
            if len(selected) != 1:
                # 标题相同也可按澄清列表的序号选择，不依赖模型猜测轮次。
                numbered = re.fullmatch(r"(?:选|选择)?(?:第)?([1-6])(?:个|项)?", question.strip())
                if numbered and int(numbered[1]) <= len(pending["candidate_keys"]):
                    key = pending["candidate_keys"][int(numbered[1]) - 1]
                    selected = [catalog[key]] if key in catalog else []
            if len(selected) == 1:
                choice = selected[0]
                docs = [d for d in choice["document_ids"] if d in snapshot["document_ids"]]
                if choice["document_ids"] and not docs:
                    unavailable = self._plan("clarification", question, [], enhanced, "historical_source_unavailable")
                    unavailable["clarification_question"] = "该条目的原文已不在当前范围，请重新选择资料。"
                    return unavailable
                plan = self._plan("concept_explanation" if enhanced else "qa",
                                  choice["title"] + "：" + question, docs or list(snapshot["document_ids"]),
                                  enhanced, "selected_history_unit")
                plan.update(referenced_turn_id=choice["turn_id"], referenced_unit_id=choice["unit_id"])
                if enhanced:
                    plan["referenced_unit_ids"] = [choice["unit_id"]]
                return plan
        if _RESUME.fullmatch(question.strip()):
            for turn in reversed(memory["turns"]):
                response = turn.get("response", {})
                coverage = response.get("coverage") or {}
                if response.get("intent") != "overview" or coverage.get("complete") is not False:
                    continue
                docs = (turn.get("resolution") or {}).get("document_ids") or []
                versions = coverage.get("source_versions") or {}
                if docs and all(d in snapshot["document_ids"] and versions.get(d) == snapshot["versions"].get(d)
                                for d in docs):
                    original = (turn.get("resolution") or {}).get("resolved_query", turn["question"])
                    return self._plan("overview", original, docs, enhanced, "resume_partial_overview")
        # “刚才第二点”有明确最近轮次；没有轮次提示时，多轮重号必须澄清。
        if _ORDINAL.search(question) and not re.search(r"刚才|上一|上次|第\d+轮|前一", question):
            choices = [c for c in candidates(memory, focus or {}, question) if c["display_number"] == 2]
            turns = {c["turn_id"] for c in choices}
            if len(turns) > 1:
                titles = "；".join(f"{i+1}. {c['title']}" for i, c in enumerate(choices[:6]))
                plan = self._plan("clarification", question, list(snapshot["document_ids"]), enhanced,
                                  "ambiguous_history_number")
                plan["clarification_question"] = "历史回答里有多个第二点，请选择标题：" + titles
                plan["candidate_keys"] = [c["key"] for c in choices[:6]]
                return plan
        return None

    def recover_historical_reference(self, question, snapshot, comparison_tasks, documents,
                                     available_documents, events):
        """模型编造历史条目 ID 时只从可信任务重建计划；无唯一任务返回可操作澄清。"""
        plan, trace = ComparisonTaskLocator().locate(
            question, snapshot, comparison_tasks, documents, available_documents, recovery=True)
        events.append({"node": "resolution_recovery", "status": trace["status"], "output": trace})
        events.append({"node": "scope_check",
                       "status": "blocked" if trace["status"] in {
                           "out_of_scope", "unavailable", "version_changed"} else "ok",
                       "output": {"document_ids": trace.get("document_ids", []),
                                  "reason": trace["status"]}})
        if plan is not None:
            return plan
        return ComparisonTaskLocator.clarification(
            question, snapshot, "历史条目编号无法核对。请写出两篇文件名，并在本轮选中它们。",
            "unknown_historical_reference")


    @staticmethod
    def _plan(intent, query, docs, enhanced, method):
        plan = dict(intent=intent, resolved_query=query, document_ids=docs, method=method,
                    referenced_turn_id=None, referenced_unit_id=None, clarification_question=None)
        if enhanced:
            plan.update(requested_aspects=[{"id": "a1", "question": query, "requires_document": intent == "overview"}],
                        referenced_unit_ids=[], candidate_keys=[])
        return plan

    def enforce(self, question, resolution, snapshot, focus=None, enhanced=False, memory=None):
        """模型误把贡献问题标为全文概述时，强制走原文检索。"""
        result = dict(resolution)
        if _CONTRIBUTION.search(question) and result.get("document_ids"):
            result["intent"] = "document_fact" if enhanced else "qa"
            result["resolved_query"] = question
            result["method"] = result.get("method", "") + "+fact_guard"
            if enhanced:
                for aspect in result.get("requested_aspects", []):
                    aspect["requires_document"] = True
                result.update(referenced_turn_id=None, referenced_unit_id=None, referenced_unit_ids=[])
        corrected = self.correction(question)
        if corrected:
            # 用户纠正是当前检索目标；完整原问仍单独保存在原始轮次。
            result["user_correction"] = {"target": corrected}
            result["retrieval_query"] = corrected + " " + question
            if result["intent"] == "overview":
                result["intent"] = "document_fact" if enhanced else "qa"
        elif focus and focus.get("user_correction") and re.search(r"^(?:它|这个|该方法|这项).{1,30}$", question.strip()):
            result["retrieval_query"] = focus["user_correction"]["target"] + " " + question
        # 新条目的来源带文档版本；切换或重建索引后必须停止旧条目追问。
        if memory and result.get("referenced_turn_id") and result.get("referenced_unit_id"):
            turn = next((t for t in memory["turns"] if t["id"] == result["referenced_turn_id"]), None)
            unit = next((u for u in (turn or {}).get("answer_units", [])
                         if u["unit_id"] == result["referenced_unit_id"]), None)
            refs = (unit or {}).get("source_refs", [])
            if any(ref["document_id"] not in snapshot["document_ids"] or
                   ref["document_version"] != snapshot["versions"].get(ref["document_id"])
                   for ref in refs):
                result.update(intent="clarification", clarification_question=
                              "该条目引用的文档版本已变化，请重新概述当前文档后再追问。",
                              referenced_turn_id=None, referenced_unit_id=None, referenced_unit_ids=[])
        return result

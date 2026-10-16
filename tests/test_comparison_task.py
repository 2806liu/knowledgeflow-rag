"""公开合成英文资料：双文档取证、引用归属和失败后的连续追问。"""

import json
import uuid
from unittest.mock import patch

from enhanced_test_support import EnhancedBase, answer, resolution, section
from app.rag.comparison_task import search_query
from app.rag.comparison_task_repository import ComparisonTaskRepository
from app.rag.query_planner import QueryPlanner


class ComparisonTaskTests(EnhancedBase):
    def pair(self, *, second_has_evidence=True):
        """固定两篇虚构论文的原文块；不上传用户的真实 PDF。"""
        documents = []
        sources = [
            ("Synthetic_Alpha_Kinematic_Control.pdf",
             "Abstract introduction objective problem approach. Alpha paper studies task priority kinematic control of redundant manipulators. Conclusion: it organizes motion tasks by priority."),
            ("Synthetic_Beta_Redundancy_Resolution.pdf",
             "Abstract introduction objective problem approach. Beta paper studies constrained redundancy resolution through projection. Conclusion: it resolves competing constraints by projection."
             if second_has_evidence else "Unrelated appendix with no research summary."),
        ]
        for filename, text in sources:
            document = str(uuid.uuid4())
            self.api.store.add_document(document, filename, text)
            self.api.store.replace_document_chunks(document, [
                (str(uuid.uuid4()), document, text, 1,
                 {"section_id": document, "section_chunk_count": 1})])
            documents.append(document)
        self.scope(documents, active=None)
        return documents

    @staticmethod
    def grounded(messages):
        """脚本回答只从输入 evidence 读取编号，用于核验系统路由和来源约束。"""
        payload = json.loads(messages[-1]["content"])
        docs = payload["comparison_task"]["document_ids"]
        numbers = {item["document_id"]: item["citation_number"] for item in payload["evidence"]}
        sections = []
        for aspect in payload["requested_aspects"]:
            if aspect.get("document_id"):
                doc = aspect["document_id"]
                number = numbers[doc]
                label = "Alpha 关注任务优先级" if doc == docs[0] else "Beta 关注约束投影"
                sections.append(section(f"{label}。[{number}]", "document", (aspect["id"],), (number,)))
            elif aspect.get("comparison_document_ids"):
                first, second = numbers[docs[0]], numbers[docs[1]]
                sections.append(section(f"两者侧重点不同。[{first}][{second}]", "document",
                                        (aspect["id"],), (first, second)))
        return answer(*sections)

    def test_selected_pair_retrieves_each_english_source(self):
        """中文比较问题无需再澄清两篇是谁，英文片段须分别进入最终证据。"""
        docs = self.pair()
        self.script(self.grounded)
        value = self.ask("这两篇文章的侧重点区别是什么")
        self.assertEqual(value["answer_status"], "answered")
        self.assertEqual({c["document_id"] for c in value["citations"]}, set(docs))
        self.assertEqual(len(self.model.calls), 1)
        self.assertEqual(self.traces(value, "comparison_evidence")[-1]["status"], "complete")
        plans = self.traces(value, "aspect_retrieval")[0]["output"]
        self.assertEqual(plans["document_ids"], docs)

    def test_conclusion_query_prefers_body_over_title_page(self):
        """单篇已限定时，论文标题中的罕见词不应挤掉结论正文。"""
        docs = self.pair()
        rows = [
            (str(uuid.uuid4()), docs[0],
             "Synthetic Alpha Kinematic Control of Redundant Manipulators research title page", 1,
             {"section_id": "title", "section_chunk_count": 1}),
            (str(uuid.uuid4()), docs[0],
             "Summary. The study concludes that task priority organizes multiple motion objectives.", 2,
             {"section_id": "summary", "section_chunk_count": 1}),
        ]
        self.api.store.replace_document_chunks(docs[0], rows)
        hits = self.api.flow.retriever.search(
            search_query("主要结论", "Synthetic_Alpha_Kinematic_Control.pdf"), 4, (docs[0],))
        self.assertEqual(hits[0].position, 2)

    def test_wrong_paper_citation_is_repaired(self):
        """编号合法但归属另一篇论文时，严格校验仍必须拒绝并限次修复。"""
        self.pair()
        def wrong(messages):
            payload = json.loads(messages[-1]["content"])
            docs = payload["comparison_task"]["document_ids"]
            other = next(item["citation_number"] for item in payload["evidence"]
                         if item["document_id"] == docs[1])
            return answer(section(f"Alpha 的结论。[{other}]", "document", ("a1",), (other,)),
                          unresolved=("a2", "a3"))
        self.script(wrong, self.grounded)
        value = self.ask("这两篇论文的侧重点区别是什么")
        self.assertEqual(value["answer_status"], "answered")
        self.assertEqual(self.traces(value, "answer_validation")[0]["status"], "retrying")

    def test_difference_requires_both_papers(self):
        """结论即使附了合法编号，缺少另一篇来源仍不能作为两文差异放行。"""
        self.pair()
        def one_sided_difference(messages):
            payload = json.loads(messages[-1]["content"])
            first = next(item["citation_number"] for item in payload["evidence"]
                         if item["document_id"] == payload["comparison_task"]["document_ids"][0])
            return answer(section(f"两篇的差异已确定。[{first}]", "document", ("a3",), (first,)),
                          unresolved=("a1", "a2"))
        self.script(one_sided_difference, self.grounded)
        value = self.ask("这两篇论文的侧重点区别是什么")
        self.assertEqual(value["answer_status"], "answered")
        self.assertIn("both document sources", self.traces(value, "answer_validation")[0][
            "validation_reason_detail"])

    def test_insufficient_comparison_keeps_pair_for_short_followup(self):
        """依据不足后问主要结论，沿用同两篇与新维度，不回到更早的打卡话题。"""
        docs = self.pair()
        self.seed(text="实验室每天打卡四次。", basis="model_only")
        self.script(answer(unresolved=("a1", "a2", "a3")))
        first = self.ask("这两篇文章的侧重点区别是什么")
        self.assertEqual(first["answer_status"], "insufficient_evidence")
        state = self.view()["dialogue_state"]["pending_comparison_task"]
        self.assertEqual(state["document_ids"], docs)
        self.script(self.grounded)
        followup = self.ask("分别列出两篇文章的主要结论")
        self.assertEqual(followup["answer_status"], "answered")
        self.assertEqual({c["document_id"] for c in followup["citations"]}, set(docs))
        self.assertEqual(len(self.model.calls), 1)
        self.assertIsNone(self.view()["dialogue_state"]["pending_comparison_task"])

    def test_repeated_clarification_words_keep_latest_pair(self):
        """依据不足、当前两篇等简短追问继续最近任务，不重复要求用户指定文件。"""
        docs = self.pair()
        self.script(answer(unresolved=("a1", "a2", "a3")))
        self.assertEqual(self.ask("这两篇论文的侧重点区别在哪里")["answer_status"],
                         "insufficient_evidence")
        self.script(answer(unresolved=("a1", "a2", "a3")))
        reply = self.ask("依据不足如何补充你才能知道")
        self.assertEqual(reply["answer_status"], "insufficient_evidence")
        self.assertEqual(reply["scope_snapshot"]["document_ids"], docs)
        self.script(self.grounded)
        resolved = self.ask("当前的d1和d2")
        self.assertEqual(resolved["answer_status"], "answered")
        self.assertEqual({c["document_id"] for c in resolved["citations"]}, set(docs))
        # 成功后仍可依据最近完成的比较轮次理解短问，而不回到模型澄清。
        self.script(self.grounded)
        self.assertEqual(self.ask("各自的研究重点")["answer_status"], "answered")

    def test_reindex_invalidates_pending_comparison(self):
        """来源版本变化后旧任务只作历史记录，不作为本轮比较计划。"""
        docs = self.pair()
        self.script(answer(unresolved=("a1", "a2", "a3")))
        self.ask("这两篇文章的侧重点区别是什么")
        self.api.store.replace_document_chunks(docs[0], [
            (str(uuid.uuid4()), docs[0], "Updated abstract and conclusion.", 1,
             {"section_id": "updated", "section_chunk_count": 1})])
        self.script(resolution("请重新指定问题", intent="clarification", docs=docs,
                               clarification_question="资料版本已变化，请重新提问。"))
        value = self.ask("当前的d1和d2")
        self.assertEqual(value["answer_status"], "clarification")
        self.assertIsNone(self.view()["dialogue_state"]["pending_comparison_task"])
        self.assertEqual(self.traces(value, "comparison_focus")[0]["status"],
                         "invalidated_source_change")

    def test_one_sided_evidence_yields_partial_answer(self):
        """仅一篇有命中时只补检索缺失的一篇，允许有据一侧成为部分回答。"""
        docs = self.pair(second_has_evidence=False)
        def partial(messages):
            payload = json.loads(messages[-1]["content"])
            first = next(item["citation_number"] for item in payload["evidence"]
                         if item["document_id"] == docs[0])
            return answer(section(f"Alpha 研究任务优先级。[{first}]", "document", ("a1",), (first,)),
                          unresolved=("a2", "a3"))
        self.script(partial)
        value = self.ask("这两篇论文的侧重点区别是什么")
        self.assertEqual(value["answer_status"], "partial_answer")
        self.assertEqual({c["document_id"] for c in value["citations"]}, {docs[0]})
        trace = self.traces(value, "comparison_supplemental_retrieval")[0]
        self.assertEqual(trace["output"]["document_ids"], [docs[1]])
        self.assertNotIn("文档d1", value["answer"])

    def test_clarification_displays_filename_instead_of_alias(self):
        """普通澄清也不能向用户暴露解析模型使用的 d1、d2。"""
        docs = self.pair()
        self.script(resolution("请选择对象", intent="clarification", docs=docs,
                               clarification_question="请选择 d1 或 d2。"))
        value = self.ask("请选择对象")
        self.assertEqual(value["answer_status"], "clarification")
        self.assertNotIn("d1", value["answer"])
        self.assertNotIn("d2", value["answer"])
        self.assertIn("Synthetic_Alpha", value["answer"])

    def test_answered_task_survives_unrelated_document_and_three_file_scope(self):
        """已回答比较保留任务 ID；单份范围提示选择，两篇重入三份范围后继续原任务。"""
        docs = self.pair()
        self.script(self.grounded)
        first = self.ask("这两篇文章的侧重点区别是什么")
        task_id = self.view()["turns"][-1]["resolution"]["comparison_task"]["task_id"]
        tasks = ComparisonTaskRepository(self.api.flow.conversations).list(self.session)
        self.assertEqual(tasks[0]["task_id"], task_id)
        self.assertEqual(tasks[0]["status"], "answered")
        third = str(uuid.uuid4())
        self.api.store.add_document(third, "Synthetic_Nature_Philosophy.pdf", "Nature philosophy notes.")
        self.api.store.replace_document_chunks(third, [
            (str(uuid.uuid4()), third, "Nature philosophy notes.", 1,
             {"section_id": third, "section_chunk_count": 1})])
        self.scope([third], active=third)
        self.seed(document=third, text="1. 自然观。[1]")
        outside = self.ask("前面我问两篇论文的研究重点，请说区别")
        self.assertEqual(outside["answer_status"], "clarification")
        self.assertIn("未全部选入本轮范围", outside["answer"])
        self.assertIn("Synthetic_Alpha", outside["answer"])
        self.assertEqual(self.traces(outside, "task_lookup")[0]["status"], "out_of_scope")
        self.scope([third, *docs], active=None)
        self.script(self.grounded)
        named = self.ask("指的是历史对话中的 Alpha 和 Beta 两篇")
        self.assertEqual(named["answer_status"], "answered")
        self.assertEqual({c["document_id"] for c in named["citations"]}, set(docs))
        self.script(self.grounded)
        followup = self.ask("前面我问两篇论文的各自研究重点，请说明两篇的侧重点区别")
        self.assertEqual(followup["answer_status"], "answered")
        self.assertEqual({c["document_id"] for c in followup["citations"]}, set(docs))
        self.assertEqual(self.view()["turns"][-1]["resolution"]["comparison_task"]["task_id"], task_id)
        self.assertEqual(len(self.model.calls), 1)
        self.script(self.grounded)
        combined = self.ask("前面两篇论文的侧重点区别是什么，分别列出主要结论")
        self.assertEqual(combined["answer_status"], "answered")
        self.assertEqual(len(self.view()["turns"][-1]["resolution"]["requested_aspects"]), 5)
        self.assertEqual({c["document_id"] for c in combined["citations"]}, set(docs))

    def test_invalid_historical_unit_id_recovers_by_named_papers(self):
        """模型编造条目 ID 时按同会话文件名恢复，而非放行虚构引用或返回 502。"""
        docs = self.pair()
        self.script(self.grounded)
        self.ask("这两篇文章的侧重点区别是什么")
        bad = resolution("关于 Alpha 和 Beta", intent="comparison", docs=docs,
                         referenced_turn_id="invented-turn", referenced_unit_id="invented-unit",
                         referenced_unit_ids=["invented-unit"])
        self.script(bad, self.grounded)
        # 单独强制经过模型解析，检验预规划之外的异常恢复边界。
        with patch.object(QueryPlanner, "preplan", return_value=None):
            value = self.ask("关于 Alpha 和 Beta")
        self.assertEqual(value["answer_status"], "answered")
        self.assertEqual({c["document_id"] for c in value["citations"]}, set(docs))
        self.assertEqual(self.traces(value, "resolution_recovery")[0]["status"], "matched")
        resolved = self.view()["turns"][-1]["resolution"]
        self.assertIsNone(resolved["referenced_turn_id"])
        self.assertEqual(resolved["referenced_unit_ids"], [])

    def test_invalid_historical_id_without_task_becomes_clarification(self):
        """找不到可信比较对象时返回可操作澄清，不把解析错误显示为请求失败。"""
        self.script(resolution("请继续说明它们", intent="comparison",
                               referenced_turn_id="invented-turn", referenced_unit_id="invented-unit",
                               referenced_unit_ids=["invented-unit"]))
        value = self.ask("请继续说明它们")
        self.assertEqual(value["answer_status"], "clarification")
        self.assertIn("两篇文件名", value["answer"])
        self.assertEqual(self.traces(value, "resolution_recovery")[0]["status"], "not_comparison")

    def test_legacy_comparison_backfill_keeps_original_answer(self):
        """旧双篇轮次只建立任务索引，原始问题、答案和引用不被迁移程序改写。"""
        docs = self.pair()
        repo = self.api.flow.conversations
        turn, _ = repo.begin_turn(self.session, str(uuid.uuid4()), "两篇论文的侧重点区别")
        citations = [dict(citation_number=i, document_id=doc) for i, doc in enumerate(docs, 1)]
        original = dict(intent="comparison", answer_status="answered", answer="旧回答保持原样。",
                        citations=citations, answer_basis="document_only")
        payload = dict(response=original, resolution=dict(intent="comparison", document_ids=docs),
                       snapshot=repo.snapshot(self.session))
        repo.finish_turn(turn, payload)
        tasks = ComparisonTaskRepository(repo).list(self.session)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["document_ids"], docs)
        self.assertEqual(tasks[0]["status"], "answered")
        self.assertEqual(repo.turns(self.session)[-1]["response"], original)
        self.assertEqual(ComparisonTaskRepository(repo).list(self.session)[0]["task_id"], tasks[0]["task_id"])

    def test_named_old_pair_uses_current_version_after_reindex(self):
        """明确点名旧两篇后重新索引当前版本，旧任务仍供定位但不复用旧 chunk。"""
        docs = self.pair()
        self.script(self.grounded)
        self.ask("这两篇文章的侧重点区别是什么")
        old_id = self.view()["turns"][-1]["resolution"]["comparison_task"]["task_id"]
        self.api.store.replace_document_chunks(docs[0], [
            (str(uuid.uuid4()), docs[0], "Abstract introduction objective. Updated Alpha studies task priority.", 1,
             {"section_id": "updated", "section_chunk_count": 1})])
        self.script(self.grounded)
        value = self.ask("Alpha 和 Beta 两篇的研究重点")
        self.assertEqual(value["answer_status"], "answered")
        new_id = self.view()["turns"][-1]["resolution"]["comparison_task"]["task_id"]
        self.assertNotEqual(new_id, old_id)
        self.assertEqual(self.traces(value, "task_lookup")[0]["status"], "replanned_current_version")

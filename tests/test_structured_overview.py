"""结构化概述的验收回归：断点、编号、原文来源和历史输入边界。"""

import json

from conversation_test_support import APIBase, FIXTURES
from app.rag.overview_contracts import validate_units
from app.rag.query_planner import QueryPlanner


class StructuredOverviewTests(APIBase):
    def test_ambiguous_second_point_choice_reloads_its_source(self):
        """两个概述都有第二点时先列标题，选标题后只锚定对应轮次。"""
        self.upload()
        self.fake.output=json.dumps({"units":[
            {"title":"背景","claim":"背景说明","source_numbers":[1]},
            {"title":"任务优先级","claim":"任务优先级说明","source_numbers":[2]}]},ensure_ascii=False)
        self.chat("概述全文")
        self.fake.output=json.dumps({"units":[
            {"title":"背景","claim":"背景说明","source_numbers":[1]},
            {"title":"梯度投影","claim":"梯度投影说明","source_numbers":[2]}]},ensure_ascii=False)
        second=self.chat("总结整篇文档")
        self.fake.output=None
        clarification=self.chat("第二点再解释")
        self.assertEqual(clarification["answer_status"],"clarification")
        self.assertIn("任务优先级",clarification["answer"])
        self.assertIn("梯度投影",clarification["answer"])
        answer=self.chat("选择梯度投影")
        self.assertEqual(answer["answer_status"],"answered")
        self.assertEqual(self.api.flow.conversations.turns(self.session)[-1]["resolution"]["referenced_turn_id"],
                         second["turn_id"])

    def test_user_correction_survives_short_followup(self):
        """纠正目标独立保存；后续代词追问把目标加入检索词。"""
        self.upload()
        self.chat("不是第二种方法，我指梯度投影法")
        state=self.api.flow.conversations.get(self.session)["dialogue_state"]
        self.assertEqual(state["user_correction"]["target"],"梯度投影法")
        self.chat("它有什么作用")
        resolution=self.api.flow.conversations.turns(self.session)[-1]["resolution"]
        self.assertIn("梯度投影法",resolution["retrieval_query"])

    def test_partial_overview_resumes_only_missing_map_batch(self):
        """部分概述显示进度，继续时复用成功 Map，旧轮次原样保留。"""
        self.upload((FIXTURES / "long.md").read_bytes(), "long.md")
        self.fake.fail_at = {2}
        first = self.chat("概述全文")
        self.assertEqual(first["answer_status"], "partial_answer")
        self.assertTrue(first["coverage"]["resumable"])
        self.assertIn("可继续概述", first["answer"])
        first_map_calls = sum(json.loads(c[-1]["content"]).get("task") == "overview_map"
                              for c in self.fake.calls)
        self.fake.fail_at.clear()
        before = len(self.fake.calls)
        second = self.chat("继续概述")
        resumed_map_calls = sum(json.loads(c[-1]["content"]).get("task") == "overview_map"
                                for c in self.fake.calls[before:])
        self.assertLess(resumed_map_calls, first_map_calls)
        self.assertEqual(second["coverage"]["overview_id"], first["coverage"]["overview_id"])
        self.assertGreater(second["coverage"]["revision"], first["coverage"]["revision"])
        self.assertTrue(second["coverage"]["complete"])
        self.assertEqual(self.api.flow.conversations.turns(self.session)[-2]["response"]["answer"], first["answer"])

    def test_numbering_and_sources_are_per_overview(self):
        """不同概述各自从 1 编号，内部条目保留当前原文版本和 chunk ID。"""
        document = self.upload()
        first = self.chat("概述全文")
        second = self.chat("总结整篇文档")
        for response in (first, second):
            units = response["answer_units"]
            self.assertEqual([u["display_number"] for u in units], list(range(1, len(units) + 1)))
            self.assertTrue(response["answer"].startswith("1. "))
            self.assertTrue(all(u["source_refs"] and u["unit_key"] for u in units))
            self.assertTrue(all(ref["document_id"] == document and ref["document_version"] ==
                                response["scope_snapshot"]["versions"][document]
                                for unit in units for ref in unit["source_refs"]))
        self.assertNotEqual(first["coverage"]["overview_id"], second["coverage"]["overview_id"])

    def test_map_contract_rejects_invented_source(self):
        """模型即使返回合法 JSON，也不能引用未输入的原文编号。"""
        with self.assertRaises(ValueError):
            validate_units(json.dumps({"units":[{"title":"虚构","claim":"无来源断言",
                                                 "source_numbers":[99]}]}, ensure_ascii=False), {1, 2})

    def test_multiple_historical_second_points_require_choice(self):
        """未指定轮次的“第二点”在多个回答之间不能猜。"""
        def turn(turn_id, title):
            return {"id":turn_id,"question":"概述","response":{"answer":"已回答","answer_status":"answered",
                    "answer_basis":"document_only"},"answer_units":[{"unit_id":"2","ordinal":2,
                    "display_number":2,"title":title,"text":"2. "+title,"document_ids":[]}]}
        memory={"turns":[turn("a","任务优先级"),turn("b","梯度投影")]}
        snapshot={"document_ids":[],"versions":{}}
        plan=QueryPlanner().preplan("第二点再解释",snapshot,memory,{},enhanced=True)
        self.assertEqual(plan["intent"],"clarification")
        self.assertIn("任务优先级",plan["clarification_question"])
        self.assertIn("梯度投影",plan["clarification_question"])

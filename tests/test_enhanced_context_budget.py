"""长概述后的解析预算回归：只使用公开合成文字和隔离数据库。"""
import json
from dataclasses import replace
from enhanced_test_support import EnhancedBase, resolution, section, answer


class EnhancedContextBudgetTests(EnhancedBase):
    def long_overview(self, label):
        """构造重复条目与长摘录，模拟概述目录膨胀而不使用真实论文。"""
        return "\n".join(
            f"{index}. {label}主题{index}：" + "公开合成方法说明。" * 85 + " [1]"
            for index in range(1, 19))

    def test_short_paper_question_after_long_overviews_fits(self):
        """完整论文问题只带活动文档；回答仍须走文档取证和来源校验。"""
        document = self.upload(data="研究创新点是公开合成的流程改进，需据原文描述。".encode("utf-8"),
                               name="synthetic-innovation.md")
        self.seed(document=document, text=self.long_overview("第一轮"))
        self.seed(document=document, text=self.long_overview("第二轮"))

        def inspect(messages):
            payload = json.loads(messages[-1]["content"])
            self.assertEqual(payload["answer_candidates"], [])
            self.assertEqual(payload["recent_turns"], [])
            self.assertEqual(payload["history_summary"], "")
            self.assertEqual(payload["scope_snapshot"]["active_document_id"], "d1")
            return resolution("这篇资料的创新点", intent="document_fact", docs=["d1"])

        self.script(inspect, answer(section("资料中的原文可作为依据。[1]", "document", numbers=[1])))
        value = self.ask("创新点是什么")
        self.assertEqual(value["answer_status"], "answered")
        self.assertEqual(value["answer_basis"], "document_only")
        self.assertTrue(self.traces(value, "query_resolution"))

    def test_explicit_old_unit_keeps_ids_without_long_excerpts(self):
        """明确轮次保留全部对应标题与稳定ID，旧答案正文不重复发送。"""
        document = self.upload()
        self.seed(document=document, text=self.long_overview("第一轮"))
        turn, units = self.seed(document=document, text=self.long_overview("第二轮"))

        def inspect(messages):
            payload = json.loads(messages[-1]["content"])
            choices = payload["answer_candidates"]
            self.assertEqual(len(choices), 18)
            self.assertTrue(all("excerpt" not in item for item in choices))
            self.assertTrue(all(item["turn_id"] == turn for item in choices))
            self.assertIn(units[1]["unit_id"], [item["unit_id"] for item in choices])
            self.assertEqual(payload["reference_hint"]["unit_id"], units[1]["unit_id"])
            return resolution("请说明指定条目", intent="clarification", docs=["d1"],
                              clarification_question="请确认要展开的主题")

        self.script(inspect)
        self.assertEqual(self.ask("第2轮的第二点是什么意思")["answer_status"], "clarification")

    def test_model_overview_label_cannot_retrigger_long_overview_for_contribution(self):
        """模型误标overview时，贡献追问仍定位原文并走普通文档事实回答。"""
        document = self.upload(
            data=("# Kinematic Control of Redundant Manipulators\n"
                  "This tutorial report compares online kinematic control approaches. "
                  "Its conclusions discuss their relative strengths and practical limitations. "
                  "The discussion surveys Jacobian methods, task priority and inverse kinematics. "
                  "This is synthetic text for routing and retrieval verification only.\n").encode(),
            name="Author_1990_Kinematic Control of Redundant Manipulators.md")
        self.seed(document=document, text=self.long_overview("旧概述"))

        def inspect_resolution(messages):
            payload=json.loads(messages[-1]["content"])
            self.assertEqual(payload["answer_candidates"], [])
            return resolution("概述整篇论文", intent="overview", docs=["d1"])

        def inspect_answer(messages):
            payload=json.loads(messages[-1]["content"])
            self.assertEqual(payload["requested_aspects"][0]["question"], "创新点是什么")
            self.assertTrue(payload["evidence"])
            return answer(section("这份教程比较了多种控制方法。[1]", "document", numbers=[1]))

        self.script(inspect_resolution, inspect_answer)
        value=self.ask("创新点是什么")
        self.assertEqual(value["intent"], "document_fact")
        self.assertEqual(value["answer_basis"], "document_only")
        self.assertFalse(self.traces(value, "overview"))
        tools=self.traces(value, "tool")
        self.assertIn("contribution", tools[0]["query"])
        self.assertEqual(len(self.model.calls),2)

    def test_necessary_input_overflow_reports_run_and_counts(self):
        """必要规则也放不下时仍返回413，并给出脱敏的运行与阶段计数。"""
        self.api.flow.settings = replace(self.config, context_window=600,
                                         llm_max_output_tokens=200, overview_max_output_tokens=100,
                                         context_safety_margin=100)
        self.script()
        response = self.post("请解释这个概念")
        self.assertEqual(response.status_code, 413)
        detail = response.json()["detail"]
        self.assertTrue(detail["run_id"])
        budget = detail["context_budget"]
        self.assertEqual(budget["phase"], "query_resolution")
        self.assertGreater(budget["input_count"], budget["input_limit"])
        self.assertIn("system", budget["component_counts"])
        events = self.client.get("/runs/" + detail["run_id"]).json()["events"]
        self.assertTrue(any(event["node"] == "context_budget" for event in events))
        self.assertFalse(self.model.calls)

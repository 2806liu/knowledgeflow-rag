"""增强策略与生成：兼容、事实权限、来源组合与补检索次数。"""
import json
import uuid
from enhanced_test_support import EnhancedBase,resolution,section,answer

class PolicyTests(EnhancedBase):
    def test_EP01_legacy_api_defaults_strict(self):
        session=self.client.post("/conversations").json()["id"]
        self.script()
        r=self.client.post("/chat",json={"session_id":session,"message":"什么是向量"})
        self.assertEqual(r.status_code,200);self.assertEqual(r.json()["answer_status"],"clarification")
        self.assertFalse(self.model.calls)

    def test_EP02_conversation_policy_is_inherited(self):
        self.script(resolution(intent="general_question"),answer(section()))
        self.assertEqual(self.ask()["answer_policy"],"document_enhanced")

    def test_EP03_explicit_strict_overrides_enhanced(self):
        self.script()
        value=self.ask("什么是向量",answer_policy="document_strict")
        self.assertEqual(value["answer_policy"],"document_strict");self.assertFalse(self.model.calls)

    def test_EP04_invalid_policy_is_rejected(self):
        r=self.post(answer_policy="unrestricted");self.assertEqual(r.status_code,422)

    def test_EP05_policy_persists_across_reload(self):
        self.client.patch("/conversations/"+self.session+"/policy",json={"answer_policy":"document_strict"})
        self.assertEqual(self.view()["answer_policy"],"document_strict")
        self.assertEqual(self.view()["policy_revision"],2)

    def test_EP06_policy_cannot_change_during_generation(self):
        self.api.flow.conversations.begin_turn(self.session,"busy","问题")
        r=self.client.patch("/conversations/"+self.session+"/policy",json={"answer_policy":"document_strict"})
        self.assertEqual(r.status_code,409)

    def test_EP07_idempotency_cannot_change_policy(self):
        self.script(resolution(intent="general_question"),answer(section()))
        key=str(uuid.uuid4());value=self.ask(request_id=key)
        r=self.post(request_id=key,answer_policy="document_strict")
        self.assertEqual(r.status_code,409)
        self.assertEqual(len(self.model.calls),2)

    def test_EP08_overview_remains_document_only(self):
        doc=self.upload()
        self.script(resolution("概述全文",intent="overview",docs=[doc]),
                    {"answer_status":"answered","answer":"课程报告有命名要求。[1]","cited_numbers":[1]})
        value=self.ask("概述全文")
        self.assertEqual(value["effective_policy"],"document_strict")
        self.assertEqual(value["answer_basis"],"document_only")
        self.assertFalse(any(s["source_type"]=="model_knowledge" for s in value["answer_sections"]))

class GenerationTests(EnhancedBase):
    def test_EG01_document_only_answer(self):
        doc=self.upload()
        self.script(resolution("报告命名格式",intent="document_fact",docs=[doc]),
                    answer(section("报告按学号-姓名-报告命名。[1]","document",numbers=[1])))
        value=self.ask("报告命名格式")
        self.assertEqual(value["answer_basis"],"document_only");self.assertTrue(value["citations"])

    def test_EG02_document_and_model_sections(self):
        doc=self.upload()
        aspects=[dict(id="a1",question="文档介绍",requires_document=True),dict(id="a2",question="通用解释",requires_document=False)]
        self.script(resolution("报告命名格式",docs=[doc],aspects=aspects),
                    answer(section("文档给出命名格式。[1]","document",aspects=["a1"],numbers=[1]),
                           section("一般背景解释。",aspects=["a2"])))
        value=self.ask();self.assertEqual(value["answer_basis"],"document_and_model")
        self.assertEqual(len(value["answer_sections"]),2)

    def test_EG03_named_concept_can_use_model_without_evidence(self):
        self.script(resolution("解释梯度投影法"),answer(section("这是关于该方法的一般说明。")))
        value=self.ask("解释梯度投影法")
        self.assertEqual(value["answer_status"],"answered");self.assertFalse(value["citations"])

    def test_EG04_model_cannot_supply_paper_parameters(self):
        doc=self.upload()
        self.script(resolution("报告命名格式",intent="document_fact",docs=[doc]),
                    answer(section("论文参数为99。")),answer(section("论文参数为99。")))
        self.assertEqual(self.post("论文实验参数").status_code,502)

    def test_EG05_omitted_aspect_is_explicitly_partial(self):
        aspects=[dict(id="a1",question="原理",requires_document=False),
                 dict(id="a2",question="论文实验参数",requires_document=True)]
        self.script(resolution(aspects=aspects),answer(section("一般原理。",aspects=["a1"])))
        value=self.ask("解释原理和论文实验参数")
        self.assertEqual(value["answer_status"],"partial_answer")
        self.assertEqual(value["unresolved_aspects"],["论文实验参数"])

    def test_EG06_one_supplemental_retrieval(self):
        doc=self.upload()
        self.script(resolution("解释命名规则",docs=[doc]),
                    {"action":"retrieve_more","sections":[],"unresolved_aspect_ids":[],"supplemental_query":"报告命名格式"},
                    answer(section("一般解释。")))
        value=self.ask("解释命名规则")
        self.assertEqual(value["answer_status"],"answered")
        self.assertEqual(len(self.model.calls),3)
        self.assertEqual(len(self.traces(value,"supplemental_retrieval")),1)
        self.assertEqual(len(self.traces(value,"tool")),2)

    def test_EG07_repeated_tool_request_stops(self):
        doc=self.upload()
        more={"action":"retrieve_more","sections":[],"unresolved_aspect_ids":[],"supplemental_query":"报告命名"}
        self.script(resolution(docs=[doc]),more,more)
        value=self.ask()
        self.assertEqual(value["answer_status"],"insufficient_evidence")
        self.assertIn("次数已用完",value["answer"])
        self.assertEqual(len(self.model.calls),3)

    def test_EG08_mock_never_invents_general_knowledge(self):
        from dataclasses import replace
        self.api.flow.settings=replace(self.api.flow.settings,llm_mode="mock")
        self.script()
        value=self.ask("什么是向量")
        self.assertEqual(value["answer_status"],"mock");self.assertEqual(value["answer_basis"],"none")
        self.assertIn("不提供模型通用知识",value["answer"]);self.assertFalse(self.model.calls)


class AspectBoundaryTests(EnhancedBase):
    def test_generic_aspects_override_overstrict_parser(self):
        """模型把通用原理误标为必须有原文时，按已批准的增强策略处理。"""
        aspects=[dict(id="a1",question="梯度投影法的常见局限",requires_document=True)]
        self.script(resolution(aspects=aspects),answer(section("以下为一般局限说明。")))
        self.assertEqual(self.ask("解释梯度投影法的局限")["answer_basis"],"model_only")

    def test_document_attribution_is_still_required(self):
        """明确追问原文观点与具体日期，不能因出现解释字样转为模型事实。"""
        from app.rag.answer_policy import apply_policy
        for q in ["论文明确指出的局限", "报告截止日期并解释原因"]:
            value=apply_policy(resolution(aspects=[dict(id="a1",question=q,requires_document=False)]))
            self.assertTrue(value["requested_aspects"][0]["requires_document"])

    def test_parameter_general_role_is_not_paper_value(self):
        """同一术语在一般解释与具体实验值中具有不同证据权限。"""
        from app.rag.answer_policy import document_specific
        self.assertFalse(document_specific("实验参数的一般作用是什么"))
        self.assertTrue(document_specific("这篇论文实验参数具体数值是多少"))

    def test_active_document_overview_after_append(self):
        """追加 B 后概述明确活动文档，A 仍保留，不让模型猜测对象。"""
        from conversation_test_support import ATTENDANCE
        first=self.upload();second=self.upload(ATTENDANCE,"second.md")
        self.script({"answer_status":"answered","answer":"该资料说明考勤规则。[1]","cited_numbers":[1]})
        result=self.ask("这篇资料主要讲什么")
        self.assertEqual(result["effective_policy"],"document_strict")
        self.assertTrue(all(c["document_id"]==second for c in result["citations"]))
        self.assertIn(first,self.view()["associated_document_ids"])

    def test_limitation_citation_mismatch_is_repaired_without_relaxing_gate(self):
        """边界区块引用声明没有正文标记时必须重生成，不能直接清空引用放行。"""
        bad=answer(section("范围说明。","limitation",numbers=[2]))
        def inspect(messages):
            import json
            payload=json.loads(messages[-1]["content"])
            self.assertIn("limitation",payload["validation_feedback"])
            return answer(section("以下是一般解释。"))
        self.script(resolution(),bad,inspect)
        self.assertEqual(self.ask()["answer_status"],"answered")
        self.assertEqual(len(self.model.calls),3)

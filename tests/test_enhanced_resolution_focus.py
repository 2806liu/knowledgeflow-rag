"""语义路由与焦点专项：明确案例ID，覆盖真实故障过程与边界。"""
from unittest.mock import patch
from app.rag.llm_client import LLMCallError
from enhanced_test_support import EnhancedBase,resolution,section,answer

class ResolutionTests(EnhancedBase):
    def test_direct_fact_does_not_depend_on_invented_optional_history(self):
        """完整问题重新取证，不因模型多填非必要历史关系而失败，也不读取该关系。"""
        doc=self.upload();data=resolution("报告命名格式",intent="document_fact",docs=[doc])
        data.update(referenced_turn_id="unneeded-turn",referenced_unit_id="unneeded-item",referenced_unit_ids=["unneeded-item"])
        self.script(data,answer(section("报告命名格式为学号-姓名-报告。[1]","document",numbers=[1])))
        value=self.ask("请回到报告要求，报告命名格式是什么")
        self.assertEqual(value["answer_status"],"answered")
        self.assertFalse(self.traces(value,"reference_evidence"))

    def test_ER10_short_document_aliases_are_mapped_and_scoped(self):
        """模型选择短编号，返回仍是规范UUID，伪造短编号不能逃出范围。"""
        doc=self.upload()
        def inspect(messages):
            import json
            payload=json.loads(messages[-1]["content"])
            self.assertEqual(payload["documents"][0]["id"],"d1")
            self.assertEqual(payload["scope_snapshot"]["document_ids"],["d1"])
            return resolution("报告命名格式",docs=["d1"])
        self.script(inspect,answer(section("报告要求。[1]","document",numbers=[1])))
        self.assertEqual(self.ask()["citations"][0]["document_id"],doc)
        self.script(resolution(docs=["d999"]))
        self.assertEqual(self.post().status_code,502)

    def test_ER09_attached_documents_are_preferred_for_general_question(self):
        """有资料且用户要求区分来源时，解析器漏填文档不能静默跳过取证。"""
        doc=self.upload()
        self.script(resolution("报告命名格式",intent="general_question",docs=[]),
                    answer(section("报告要求。[1]","document",numbers=[1])))
        result=self.ask("请区分资料依据和一般解释")
        self.assertEqual(result["answer_basis"],"document_only")
        self.assertEqual(result["citations"][0]["document_id"],doc)

    def test_ER01_general_without_documents(self):
        self.script(resolution("什么是向量",intent="general_question"),answer(section()))
        value=self.ask("什么是向量")
        self.assertEqual(value["answer_basis"],"model_only");self.assertFalse(value["citations"])

    def test_ER02_overview_without_object_clarifies(self):
        self.script(resolution("这篇讲什么",intent="overview"))
        value=self.ask("这篇讲什么")
        self.assertEqual(value["answer_status"],"clarification")
        self.assertEqual(len(self.model.calls),1)

    def test_ER03_specific_fact_without_documents_refuses(self):
        self.script(resolution("论文实验参数是多少",intent="document_fact"))
        value=self.ask("论文实验参数是多少")
        self.assertEqual(value["answer_status"],"insufficient_evidence")
        self.assertEqual(len(self.model.calls),1)

    def test_ER04_requested_comparison_aspects_preserved(self):
        aspects=[dict(id="a1",question="原理",requires_document=False),dict(id="a2",question="局限",requires_document=False)]
        self.script(resolution(intent="comparison",aspects=aspects),answer(
            section("一般原理。",aspects=("a1",)),section("一般局限。",aspects=("a2",))))
        value=self.ask("比较原理与局限")
        self.assertEqual(value["answer_status"],"answered");self.assertFalse(value["unresolved_aspects"])

    def test_ER05_out_of_scope_document_rejected(self):
        self.script(resolution(docs=["forged-document"]))
        value=self.post();self.assertEqual(value.status_code,502)
        self.assertEqual(value.json()["detail"]["code"],"invalid_response")

    def test_ER06_invented_history_reference_becomes_clarification(self):
        """伪造的跨会话条目仍被拒绝引用，但缺少可信任务时向用户说明如何指定对象。"""
        data=resolution();data.update(referenced_turn_id="other-session",referenced_unit_id="2",referenced_unit_ids=["2"])
        self.script(data)
        value=self.ask()
        self.assertEqual(value["answer_status"],"clarification")
        self.assertFalse(value["citations"])
        self.assertNotIn("other-session",value["answer"])

    def test_ER07_parser_failure_is_not_refusal(self):
        self.script(LLMCallError("authentication",{"status":"error","attempts":1},503))
        value=self.post();self.assertEqual(value.status_code,503);self.assertIn("run_id",value.json()["detail"])

    def test_ER08_repeated_ordinals_clarify_without_guess(self):
        self.seed(text="1. 甲。\n2. 目标与范围。\n1. 丙。\n2. 梯度方法。",basis="model_only")
        self.script()
        value=self.ask("刚才第二点具体是什么意思")
        self.assertEqual(value["answer_status"],"clarification")
        self.assertIn("目标与范围",value["answer"]);self.assertFalse(self.model.calls)

class FocusTests(EnhancedBase):
    def test_EF01_refusal_does_not_replace_substantive_focus(self):
        turn,_=self.seed(basis="model_only")
        self.seed(text="资料不足。",status="insufficient_evidence",basis="none")
        def inspect(messages):
            import json
            payload=json.loads(messages[-1]["content"])
            self.assertEqual(payload["dialogue_focus"]["last_substantive_turn_id"],turn)
            return resolution(intent="general_question")
        self.script(inspect,answer(section()));self.ask("继续解释")

    def test_EF02_clarification_reply_resumes_candidate(self):
        self.seed(text="1. 背景。\n2. 目标与范围。\n1. 方法。\n2. 梯度方法。",basis="model_only")
        self.script();first=self.ask("刚才第二点是什么意思")
        pending=self.view()["dialogue_state"]["pending_reference"]
        self.assertEqual(len(pending["candidate_keys"]),2)
        self.script(answer(section("这是目标与范围的通用解释。")))
        value=self.ask("目标与范围")
        self.assertEqual(value["answer_status"],"answered");self.assertEqual(len(self.model.calls),1)
        self.assertIsNone(self.view()["dialogue_state"]["pending_reference"])

    def test_EF03_pending_state_survives_reload(self):
        self.seed(text="1. 背景。\n2. 目标。\n1. 方法。\n2. 局限。",basis="model_only")
        self.script();self.ask("刚才第二点")
        from app.conversation import ConversationService
        repo=ConversationService(self.api.store)
        self.assertTrue(repo.get(self.session)["dialogue_state"]["pending_reference"])

    def test_EF04_cancel_pending_keeps_original_focus(self):
        turn,_=self.seed(text="1. 背景。\n2. 目标。\n1. 方法。\n2. 局限。",basis="model_only")
        self.script();self.ask("刚才第二点");value=self.ask("取消")
        self.assertEqual(value["answer_status"],"clarification")
        state=self.view()["dialogue_state"]
        self.assertIsNone(state["pending_reference"]);self.assertEqual(state["last_substantive_turn_id"],turn)

    def test_EF05_model_knowledge_units_remain_unverified(self):
        self.script(resolution(intent="general_question"),answer(section("1. 一般原理。\n2. 一般局限。")))
        value=self.ask("解释原理与局限")
        self.assertTrue(all(u["source_type"]=="model_knowledge" and not u["document_ids"] for u in value["answer_units"]))

    def test_EF06_no_cross_session_reference(self):
        self.seed(basis="model_only")
        self.session=self.client.post("/conversations",json={"answer_policy":"document_enhanced"}).json()["id"]
        def inspect(messages):
            import json
            payload=json.loads(messages[-1]["content"])
            self.assertFalse(payload["answer_candidates"])
            return resolution(intent="clarification",clarification_question="请说明主题")
        self.script(inspect)
        self.assertEqual(self.ask("刚才第二点")["answer_status"],"clarification")

    def test_EF07_deleted_pending_source_never_reused(self):
        doc=self.upload()
        self.seed(document=doc,text="1. 背景。[1]\n2. 目标。[1]\n1. 方法。[1]\n2. 局限。[1]")
        self.script();self.ask("刚才第二点")
        self.client.delete("/documents/"+doc)
        self.script()
        value=self.ask("目标");self.assertEqual(value["answer_status"],"clarification")
        self.assertFalse(value["citations"])

    def test_EF08_upload_preserves_dialogue_state(self):
        self.seed(text="1. 背景。\n2. 目标。\n1. 方法。\n2. 局限。",basis="model_only")
        self.script();self.ask("刚才第二点")
        previous=self.view()["dialogue_state"]
        document=self.upload()
        self.assertEqual(self.view()["dialogue_state"],previous)
        self.assertEqual(self.view()["active_document_id"],document)

    def test_EF09_old_history_restores_focus_without_migration(self):
        turn,_=self.seed(basis="model_only")
        self.assertFalse(self.view().get("dialogue_state"))
        def inspect(messages):
            import json
            self.assertEqual(json.loads(messages[-1]["content"])["dialogue_focus"]["last_substantive_turn_id"],turn)
            return resolution(intent="general_question")
        self.script(inspect,answer(section()));self.ask("继续解释")

    def test_EF10_generation_failure_preserves_pending(self):
        self.seed(text="1. 背景。\n2. 目标。\n1. 方法。\n2. 局限。",basis="model_only")
        self.script();self.ask("刚才第二点");previous=self.view()["dialogue_state"]
        self.script(LLMCallError("timeout",{"status":"error","attempts":1},504))
        self.assertEqual(self.post("目标").status_code,504)
        self.assertEqual(self.view()["dialogue_state"],previous)

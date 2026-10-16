"""跨语言追问回归：原文锚点、重复编号、父子列表、失效来源与预算故障。"""
import json
from dataclasses import replace
from app.rag.memory import answer_units
from app.rag.reference_evidence import ReferenceEvidenceLoader
from app.rag.context_builder import ContextOverflow
from conversation_test_support import APIBase

ENGLISH=b"""# Public synthetic tutorial
## Background
This synthetic tutorial describes kinematic control of a redundant robot arm, and the selection of control methods.
## Goal and scope
Our goal is to compare four online control methods: simple Jacobian control, null-space optimization, task-priority augmentation, and inverse kinematic functions.
## Retention
The public archive retains synthetic experiment records for seven years, and checks the archive every month.
"""

class FollowupEvidenceTests(APIBase):
    def establish_english(self,repeated=False):
        doc=self.upload(ENGLISH,"synthetic-english.md")
        self.fake.output=json.dumps({"answer_status":"answered",
            "answer":"1. 主题：讨论冗余机械臂控制。[1]\n2. 目标与范围：比较四类在线控制方法。[2]"+
                     ("\n1. 方法说明。[1]\n2. 梯度方法：另一个条目。[2]" if repeated else ""),
            "cited_numbers":[1,2]})
        first=self.chat("概括全文");self.fake.output=None
        return doc,first

    def trace(self,answer,node):
        return next(e["output"] for e in self.client.get("/runs/"+answer["run_id"]).json()["events"] if e["node"]==node)

    def test_cross_language_followup_reloads_original_source(self):
        _,first=self.establish_english()
        answer=self.chat("刚才第二点具体是什么意思")
        self.assertEqual(answer["answer_status"],"answered")
        self.assertTrue(any("Our goal" in c["snippet"] for c in answer["citations"]))
        trace=self.trace(answer,"reference_evidence")
        self.assertEqual(trace["anchor_count"],1)
        self.assertEqual(answer["citations"][0]["evidence_role"],"referenced_source")
        self.assertNotIn("2.",answer["resolved_query"])
        self.assertEqual(first["citations"][1]["chunk_id"],answer["citations"][0]["chunk_id"])

    def test_repeated_legacy_numbers_become_unique_in_new_overview(self):
        self.establish_english(True);calls=len(self.fake.calls)
        answer=self.chat("刚才第二点具体是什么意思")
        # 新概述的展示编号连续，“第二点”只对应第二个结构化条目。
        self.assertEqual(answer["answer_status"],"answered")
        self.assertEqual(self.trace(answer,"reference_evidence")["anchor_count"],1)
        self.assertGreater(len(self.fake.calls),calls)

    def test_named_unit_disambiguates_repeated_number(self):
        self.establish_english(True)
        answer=self.chat("解释目标与范围这一点")
        self.assertEqual(answer["answer_status"],"answered")
        self.assertEqual(self.trace(answer,"reference_evidence")["anchor_count"],1)
        self.assertTrue(any("Our goal" in c["snippet"] for c in answer["citations"]))

    def test_bullets_stay_with_parent(self):
        units=answer_units("1. 背景。[1]\n- 补充一。[2]\n- 补充二。[3]\n2. 目标。[4]",[
            {"citation_number":n,"chunk_id":str(n),"document_id":"doc"} for n in range(1,5)])
        self.assertEqual(len(units),2)
        self.assertEqual(set(units[0]["cited_chunk_ids"]),{"1","2","3"})
        self.assertEqual(units[1]["display_number"],2)

    def test_reindex_rebinds_only_identical_current_text(self):
        doc,first=self.establish_english()
        old=first["citations"][1]["chunk_id"]
        self.client.post("/documents/"+doc+"/reindex")
        answer=self.chat("第二点再解释一下")
        # 结构化条目的版本已变化，必须先重新概述，不能沿用旧位置重绑。
        self.assertEqual(answer["answer_status"],"clarification")
        self.assertEqual(first["citations"][1]["chunk_id"],old)

    def test_changed_source_not_reused_from_history(self):
        _,first=self.establish_english()
        flow=self.api.flow;memory=flow._conversation_flow.memory.snapshot(self.session)
        resolution={"document_ids":[first["citations"][1]["document_id"]],
                    "referenced_turn_id":first["turn_id"],"referenced_unit_id":"2"}
        with self.api.store.conn:
            self.api.store.conn.execute("UPDATE chunks SET text=? WHERE id=?",(
                "The current version now describes unrelated material and has changed entirely.",
                first["citations"][1]["chunk_id"]))
        hits,trace=ReferenceEvidenceLoader(self.api.store,self.config).merge(resolution,memory,[])
        self.assertFalse(hits);self.assertEqual(trace["status"],"no_current_anchor")

    def test_reference_scope_cannot_expand(self):
        _,first=self.establish_english()
        memory=self.api.flow._conversation_flow.memory.snapshot(self.session)
        result,trace=ReferenceEvidenceLoader(self.api.store,self.config).merge(
            {"document_ids":[],"referenced_turn_id":first["turn_id"],"referenced_unit_id":"2"},memory,[])
        self.assertFalse(result);self.assertEqual(trace["anchor_count"],0)

    def test_necessary_anchor_budget_is_not_silently_truncated(self):
        _,first=self.establish_english()
        memory=self.api.flow._conversation_flow.memory.snapshot(self.session)
        resolution={"document_ids":[first["citations"][1]["document_id"]],
                    "referenced_turn_id":first["turn_id"],"referenced_unit_id":"2"}
        with self.assertRaises(ContextOverflow):
            ReferenceEvidenceLoader(self.api.store,replace(self.config,evidence_max_context_tokens=10)).merge(resolution,memory,[])

    def test_explicit_old_turn_can_be_named_after_refusal(self):
        self.establish_english()
        self.chat("zqxvdoesnotexist")
        answer=self.chat("第1轮的目标与范围具体是什么意思")
        self.assertEqual(answer["answer_status"],"answered")
        self.assertTrue(any("Our goal" in c["snippet"] for c in answer["citations"]))

    def test_followup_keeps_original_intent_and_support_boundary(self):
        self.establish_english()
        question="刚才第二点具体是什么意思，不要补充原文外的推导"
        answer=self.chat(question)
        self.assertIn(question,answer["resolved_query"])
        self.assertIn("证据已支持的部分",self.fake.calls[-1][0]["content"])
        self.assertIn("不要补外部知识",self.fake.calls[-1][0]["content"])
        self.assertIn("不能断言整篇文档没有说明",self.fake.calls[-1][0]["content"])

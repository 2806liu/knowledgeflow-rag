"""记忆12场景与上下文预算8场景；语义摘要质量需真实模型与人工复核。"""
import asyncio
import hashlib
import json
from pathlib import Path
from dataclasses import replace
import unittest
import uuid
from conversation_test_support import FakeClient,REPORT
from app.config import settings
from app.store import Store
from app.conversation import ConversationService
from app.rag.memory import MemoryManager
from app.rag.context_builder import ContextBuilder,ContextOverflow
from app.rag.query_resolution import QueryResolver
from app.rag.solver import EvidenceSolver
from app.rag.contracts import EvidenceContext

class MemoryBase(unittest.TestCase):
    def setUp(self):
        self.root=Path("tests/.tmp/memory-"+str(uuid.uuid4()))
        self.config=replace(settings,data_dir=self.root,llm_mode="openai_compatible",llm_model="fixture",
            llm_base_url="http://invalid.local/v1",llm_api_key="fixture",llm_max_retries=0,llm_answer_format="json",
            memory_recent_rounds=2,history_budget=1000,memory_summary_budget=600,
            context_window=8000,llm_max_output_tokens=512,context_safety_margin=128)
        self.store=Store(self.root);self.repo=ConversationService(self.store)
        self.session=self.repo.create()["id"];self.fake=FakeClient()
        self.memory=MemoryManager(self.repo,self.config,self.fake)
    def tearDown(self):self.store.conn.close()
    def seed(self,count=6,answer=None,session=None):
        session=session or self.session
        ids=[]
        for i in range(count):
            question="课程报告提交规则，用户已纠正截止时间，第"+str(i+1)+"轮"
            response={"answer":answer or ("说明报告规则和待解决问题。"*30),"answer_status":"answered","citations":[],"tool_calls":[]}
            turn_id,_=self.repo.begin_turn(session,str(uuid.uuid4()),question)
            self.repo.finish_turn(turn_id,{"response":response,"resolution":{"resolved_query":"报告提交规则"},
                "answer_units":[{"unit_id":"1","ordinal":1,"title":"命名规则","text":"1.命名规则","document_ids":[]},
                                {"unit_id":"2","ordinal":2,"title":"提交时间","text":"2.提交时间","document_ids":[]}]})
            ids.append(turn_id)
        return ids
    def compact(self):return asyncio.run(self.memory.compact(self.session))

class MemoryTests(MemoryBase):
    def test_M01_recent_rounds_complete(self):
        self.seed(3)
        view=self.memory.snapshot(self.session)
        self.assertEqual(len(view["rounds"]),2)
        self.assertTrue(all([m["role"] for m in pair]==["user","assistant"] for pair in view["rounds"]))

    def test_M02_compaction_watermark_and_increment(self):
        self.seed()
        result=self.compact();self.assertEqual(result["status"],"updated")
        state=self.memory.state(self.session)
        self.assertTrue(state["source_turn_ids"])
        self.assertGreater(state["covered_until"],0)
        self.assertEqual(self.memory.snapshot(self.session)["trace"]["recent_complete_turns"],2)

    def test_M03_summary_provenance_preserves_original(self):
        ids=self.seed();self.compact()
        state=self.memory.state(self.session)
        self.assertTrue(set(state["source_turn_ids"])<=set(ids))
        self.assertEqual(len(self.repo.turns(self.session)),6)
        expected=hashlib.sha256(json.dumps(state["source_turn_ids"]).encode()).hexdigest()
        self.assertEqual(state["history_digest"],expected)

    def test_M04_past_unit_survives_compaction(self):
        ids=self.seed();self.compact()
        view=self.memory.snapshot(self.session)
        snapshot={"document_ids":["A"],"active_document_id":"A","pending_focus":False}
        resolver=QueryResolver(self.config,self.fake)
        result=asyncio.run(resolver.resolve("第1轮第二点再解释一下",snapshot,view,[{"id":"A","filename":"report.md"}]))
        self.assertEqual(result["referenced_turn_id"],ids[0])
        self.assertEqual(result["referenced_unit_id"],"2")

    def test_M05_user_correction_in_summary(self):
        self.seed();self.compact()
        self.assertIn("纠正",self.memory.state(self.session)["summary"])
        self.assertIn("旧说法不再有效",self.memory.state(self.session)["summary"])

    def test_M06_summary_is_labeled_not_evidence(self):
        self.seed();self.compact();view=self.memory.snapshot(self.session)
        self.assertTrue(view["state"]["summary"])
        self.assertTrue(all(m["role"]!="system" for pair in view["rounds"] for m in pair))
        system=self.fake.calls[0][0]["content"]
        self.assertIn("助手说法不是已核实文档事实",system)

    def test_M07_failed_summary_keeps_messages_and_old_state(self):
        self.seed();self.compact();before=self.memory.state(self.session)
        self.seed(4);self.fake.fail_at={len(self.fake.calls)+1}
        result=self.compact()
        self.assertEqual(result["status"],"failed")
        self.assertEqual(self.memory.state(self.session)["summary"],before["summary"])
        self.assertEqual(len(self.repo.turns(self.session)),10)

    def test_M08_concurrent_summary_compare_and_swap(self):
        self.seed()
        async def run():
            return await asyncio.gather(self.memory.compact(self.session),self.memory.compact(self.session))
        # 每次压缩都受revision保护；第二个任务可无需更新或基于新水位继续。
        results=asyncio.run(run())
        self.assertTrue(all(r["status"] in {"updated","not_needed","stale_discarded"} for r in results))
        state=self.memory.state(self.session)
        self.assertEqual(len(state["source_turn_ids"]),len(set(state["source_turn_ids"])))

    def test_M09_no_cross_session_summary(self):
        other=self.repo.create()["id"];self.seed(session=other)
        self.assertEqual(self.memory.snapshot(self.session)["turns"],[])
        self.assertEqual(self.memory.state(self.session)["summary"],"")
        self.assertEqual(self.compact()["status"],"not_needed")

    def test_M10_repeated_compaction_no_loss_or_duplicate(self):
        self.seed();self.compact()
        for _ in range(3):self.seed(3);self.compact()
        state=self.memory.state(self.session)
        self.assertEqual(len(self.repo.turns(self.session)),15)
        self.assertEqual(len(state["source_turn_ids"]),len(set(state["source_turn_ids"])))
        self.assertIn("纠正",state["summary"])

    def test_M11_deleted_source_not_authorized_by_memory(self):
        self.seed();self.compact()
        resolver=QueryResolver(self.config,self.fake)
        result=asyncio.run(resolver.resolve("这篇讲什么",{"document_ids":[],"active_document_id":None},
            self.memory.snapshot(self.session),[]))
        self.assertEqual(result["intent"],"clarification")

    def test_M12_short_history_zero_summary_calls_and_restart(self):
        self.seed(1,answer="简短回答")
        self.assertEqual(self.compact()["status"],"not_needed");self.assertFalse(self.fake.calls)
        self.seed(6);self.compact()
        restored=MemoryManager(ConversationService(self.store),self.config,self.fake)
        self.assertEqual(restored.state(self.session)["summary"],self.memory.state(self.session)["summary"])
        self.assertEqual(len(restored.snapshot(self.session)["turns"]),7)

class BudgetTests(MemoryBase):
    def test_B01_full_envelope_budget(self):
        builder=ContextBuilder(self.config)
        value=builder.build("系统规则",{"question":"问题","evidence":[{"text":"证据"*100}]})
        self.assertLessEqual(value.trace["input_count"]+value.trace["reserved_output"]+value.trace["safety_margin"],self.config.context_window)

    def test_B02_unknown_model_explicit_estimate(self):
        builder=ContextBuilder(self.config)
        value=builder.build("规则",{"question":"问题"})
        self.assertEqual(value.trace["token_count_method"],"utf8_bytes_conservative_budget")
        self.assertFalse(value.trace["exact"])

    def test_B03_drop_only_complete_rounds(self):
        builder=ContextBuilder(replace(self.config,history_budget=700))
        pair=[{"role":"user","content":"问"*40},{"role":"assistant","content":"答"*80}]
        value=builder.build("规则",{"question":"当前"},rounds=[pair]*5)
        history=value.messages[1:-1]
        self.assertEqual(len(history)%2,0)
        self.assertTrue(value.trace["dropped"])
        self.assertEqual(history[0]["role"],"user")

    def test_B04_required_evidence_too_large_explicit(self):
        builder=ContextBuilder(self.config)
        with self.assertRaises(ContextOverflow):
            builder.build("规则",{"question":"完整列表","evidence":"必要证据"*4000})

    def test_B05_json_envelope_rechecked(self):
        builder=ContextBuilder(replace(self.config,context_window=1200))
        with self.assertRaises(ContextOverflow):
            builder.build("系统规则",{"question":"问题","evidence":[{"text":"证据","citation_number":i} for i in range(100)]})

    def test_B06_model_stages_have_budget_trace(self):
        solver=EvidenceSolver(self.config);solver.client=self.fake
        result=asyncio.run(solver.generate_context("报告规则",[EvidenceContext(1,"a.md",1,"课程报告需要提交PDF文件。")]))
        self.assertIn("context_budget",result.trace)
        self.seed();self.compact()
        self.assertIn("context_budget",self.memory.state(self.session))

    def test_B07_source_revision_changes_rejected(self):
        self.store.add_document("A","report.md",REPORT.decode())
        self.store.replace_document_chunks("A",[("old","A",REPORT.decode(),0,{})])
        self.repo.attach(self.session,["A"])
        snapshot=self.repo.snapshot(self.session)
        self.store.replace_document_chunks("A",[("new","A",REPORT.decode(),0,{})])
        from app.conversation import ConversationError
        with self.assertRaises(ConversationError):self.repo.validate(snapshot)

    def test_B08_system_and_question_not_silently_truncated(self):
        with self.assertRaises(ValueError):ContextBuilder(replace(self.config,context_window=100))
        builder=ContextBuilder(self.config)
        with self.assertRaises(ContextOverflow):builder.build("必须保留系统规则"*2000,{"question":"当前问题"})

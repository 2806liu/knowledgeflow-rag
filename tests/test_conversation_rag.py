"""概述12场景与多轮解析20场景；受控生成仅检验编排，真实语义另评。"""
import json
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch,AsyncMock
from conversation_test_support import APIBase,REPORT,ATTENDANCE,FIXTURES
from app.rag.llm_client import LLMCallError

class OverviewTests(APIBase):
    def test_O01_short_document_bypasses_keyword_search(self):
        a=self.upload()
        answer=self.chat("这篇文章讲什么")
        self.assertEqual(answer["intent"],"overview")
        self.assertEqual(answer["answer_status"],"answered")
        self.assertIn("学号-姓名-报告",answer["answer"])
        self.assertTrue(all(c["document_id"]==a and c["score"] is None for c in answer["citations"]))
        self.assertEqual(len(self.fake.calls),1)

    def test_O02_overview_expressions(self):
        self.upload()
        for question in ["这份资料的核心要点是什么？","帮我总结这份文档","概括全文主要内容"]:
            self.assertEqual(self.chat(question)["intent"],"overview")

    def test_O03_long_document_processes_all_chunks(self):
        self.upload((FIXTURES/"long.md").read_bytes(),"long.md")
        answer=self.chat("这篇文章讲什么")
        self.assertTrue(answer["coverage"]["complete"],answer)
        self.assertEqual(answer["coverage"]["processing_coverage"],1)
        self.assertGreater(len(self.fake.calls),1)

    def test_O04_long_tail_fact_retained(self):
        self.upload((FIXTURES/"long.md").read_bytes(),"long.md")
        answer=self.chat("概括全文")
        self.assertIn("7年",answer["answer"])
        self.assertTrue(any("7年" in c["snippet"] for c in answer["citations"]))

    def test_O05_image_ocr_provenance(self):
        import io
        from PIL import Image
        from app.rag.ocr import OCRResult,OCRLine
        class Engine:
            def recognize(self,image):
                return OCRResult((OCRLine("课程报告的命名格式为学号-姓名-报告，提交PDF，截止时间6月30日18:00。",.99,((0,0),(10,0),(10,10),(0,10))),))
        data=io.BytesIO();Image.new("RGB",(600,300),"white").save(data,format="PNG")
        with patch("app.rag.parsers.get_ocr_engine",return_value=Engine()):
            self.upload(data.getvalue(),"scan.png")
        answer=self.chat("这篇文章讲什么")
        self.assertEqual(answer["citations"][0]["extraction_method"],"ocr")
        self.assertEqual(answer["citations"][0]["page"],1)

    def test_O06_same_page_mixed_sources(self):
        from test_ocr_ingestion import embedded_pdf
        from app.rag.ocr import OCRResult,OCRLine
        class Engine:
            def recognize(self,image):
                return OCRResult((OCRLine("课程报告的命名格式为学号-姓名-报告，提交前请检查文件名与内容完整性。",.99,((0,0),(10,0),(10,10),(0,10))),))
        with patch("app.rag.parsers.get_ocr_engine",return_value=Engine()):
            document=self.upload(embedded_pdf(),"mixed.pdf")
        pages=self.client.get("/documents/"+document+"/pages").json()
        self.assertEqual(pages[0]["metadata"]["extraction_method"],"mixed")
        self.assertIn("Native introduction",pages[0]["text"])
        self.assertEqual(self.chat("概括全文")["citations"][0]["page"],1)

    def test_O07_multi_document_requires_object(self):
        a=self.upload();b=self.upload(ATTENDANCE,"attendance.md")
        self.scope([a,b])
        answer=self.chat("这篇文章讲什么")
        self.assertEqual(answer["answer_status"],"clarification")
        self.assertFalse(answer["citations"])

    def test_O08_explicit_multi_document_overview(self):
        a=self.upload();b=self.upload(ATTENDANCE,"attendance.md")
        answer=self.chat("请分别概述这些资料")
        self.assertEqual(answer["intent"],"overview")
        self.assertEqual({c["document_id"] for c in answer["citations"]},{a,b})

    def test_O09_empty_or_pending_document(self):
        empty=self.create_job(b"tiny","empty.md")
        self.assertEqual(self.wait(empty.json())["state"],"failed")
        self.assertEqual(self.chat("这篇文章讲什么")["answer_status"],"clarification")

    def test_O10_deleted_document_never_uses_cache(self):
        a=self.upload();self.chat("这篇文章讲什么")
        calls=len(self.fake.calls)
        self.client.delete("/documents/"+a)
        answer=self.chat("这篇文章讲什么")
        self.assertEqual(answer["answer_status"],"clarification")
        self.assertEqual(len(self.fake.calls),calls)

    def test_O11_reindex_invalidates_cache(self):
        a=self.upload();first=self.chat("这篇文章讲什么")
        self.chat("这篇文章讲什么")
        self.assertEqual(len(self.fake.calls),1)
        self.client.post("/documents/"+a+"/reindex")
        new=self.chat("这篇文章讲什么")
        self.assertNotEqual(first["scope_snapshot"]["versions"],new["scope_snapshot"]["versions"])
        self.assertEqual(len(self.fake.calls),2)
        self.assertFalse({c["chunk_id"] for c in first["citations"]}&{c["chunk_id"] for c in new["citations"]})

    def test_O12_partial_stage_failure_is_visible(self):
        self.upload((FIXTURES/"long.md").read_bytes(),"long.md")
        self.fake.fail_at={2}
        answer=self.chat("概括全文")
        self.assertEqual(answer["answer_status"],"partial_answer")
        self.assertFalse(answer["coverage"]["complete"])
        self.assertTrue(answer["coverage"]["missing_citation_numbers"])

class DialogueTests(APIBase):
    def establish(self):
        a=self.upload();first=self.chat("这篇文章讲什么")
        return a,first

    def test_C01_unique_document_reference(self):
        a,_=self.establish()
        self.assertEqual(self.chat("这篇讲什么")["citations"][0]["document_id"],a)

    def test_C02_explicit_active_document(self):
        a=self.upload();b=self.upload(ATTENDANCE,"attendance.md")
        self.scope([a,b],a)
        self.assertTrue(all(c["document_id"]==a for c in self.chat("这篇讲什么")["citations"]))

    def test_C03_ambiguous_reference_clarifies(self):
        a=self.upload();b=self.upload(ATTENDANCE,"attendance.md");self.scope([a,b])
        self.assertEqual(self.chat("这篇讲什么")["answer_status"],"clarification")

    def test_C04_second_unit_rewrites_before_retrieval(self):
        self.establish()
        answer=self.chat("刚才第二点再详细解释一下")
        self.assertNotEqual(answer["resolved_query"],"刚才第二点再详细解释一下")
        run=self.client.get("/runs/"+answer["run_id"]).json()
        resolution=next(e["output"] for e in run["events"] if e["node"]=="query_resolution")
        self.assertEqual(resolution["referenced_unit_id"],"2")
        self.assertEqual(resolution["method"],"answer_unit")

    def test_C05_citation_number_not_item_number(self):
        self.upload()
        self.fake.output=json.dumps({"answer_status":"answered","answer":"1. 报告命名规则。[2]\n2. 报告提交时间。[1]","cited_numbers":[2,1]})
        first=self.chat("这篇文章讲什么")
        self.fake.output=None
        answer=self.chat("第二点再解释一下")
        self.assertIn("提交时间",answer["resolved_query"])
        self.assertNotIn("命名规则",answer["resolved_query"])

    def test_C06_expand_previous_topic(self):
        self.establish();answer=self.chat("再详细一点")
        self.assertIn("这篇文章讲什么",answer["resolved_query"])
        self.assertEqual(answer["intent"],"overview")
        self.assertEqual(answer["answer_status"],"answered")

    def test_C07_first_principle_unit(self):
        self.establish();answer=self.chat("解释第一个原则")
        self.assertTrue(answer["resolved_query"])
        self.assertNotEqual(answer["resolved_query"],"解释第一个原则")

    def test_C08_explicit_new_topic_overrides_history(self):
        self.upload(ATTENDANCE,"attendance.md");self.chat("每天打卡几次")
        self.upload()
        answer=self.chat("课程报告命名格式是什么")
        self.assertEqual(answer["resolved_query"],"课程报告命名格式是什么")
        self.assertIn("学号-姓名-报告",answer["answer"])

    def test_C09_explicit_filename_overrides_focus(self):
        a=self.upload();b=self.upload(ATTENDANCE,"attendance.md");self.scope([a,b],a)
        answer=self.chat("请说明attendance中的打卡要求")
        self.assertTrue(answer["citations"])
        self.assertTrue(all(c["document_id"]==b for c in answer["citations"]))

    def test_C10_no_evidence_then_expand_clarifies(self):
        self.upload();first=self.chat("zqxjkvnonexistent987")
        self.assertEqual(first["answer_status"],"no_evidence")
        self.assertEqual(self.chat("再详细一点")["answer_status"],"clarification")

    def test_C11_failed_generation_then_expand(self):
        self.upload()
        with patch.object(self.fake,"generate",new=AsyncMock(side_effect=LLMCallError("timeout",{"status":"error","attempts":1},504))):
            response=self.client.post("/chat",json={"session_id":self.session,"message":"报告命名格式"})
            self.assertEqual(response.status_code,504)
        self.assertEqual(self.chat("再详细一点")["answer_status"],"clarification")

    def test_C12_nonexistent_unit_clarifies(self):
        self.establish()
        self.assertEqual(self.chat("第九点再详细解释")["answer_status"],"clarification")

    def test_C13_deleted_history_source_unavailable(self):
        a,_=self.establish();self.client.delete("/documents/"+a)
        self.assertEqual(self.chat("第二点再解释")["answer_status"],"clarification")

    def test_C14_reparsed_sources_retrieved_again(self):
        a,first=self.establish();self.client.post("/documents/"+a+"/reindex")
        answer=self.chat("第二点再解释一下")
        self.assertNotEqual(first["scope_snapshot"]["versions"],answer["scope_snapshot"]["versions"])
        self.assertTrue(all(c["chunk_id"] not in {x["chunk_id"] for x in first["citations"]} for c in answer["citations"]))

    def test_C15_no_cross_session_reference(self):
        self.establish();self.session=self.client.post("/conversations").json()["id"]
        self.assertEqual(self.chat("第二点")["answer_status"],"clarification")

    def test_C16_long_history_stays_in_budget(self):
        self.establish()
        for _ in range(5):self.chat("报告命名格式是什么")
        answer=self.chat("刚才第二点再解释一下")
        if answer["answer_status"]!="clarification":
            run=self.client.get("/runs/"+answer["run_id"]).json()
            budgets=[e["output"] for e in run["events"] if e["node"]=="context_budget"]
            self.assertTrue(all(b["input_count"]<=b["input_limit"] for b in budgets))
        self.assertTrue(self.view()["turns"])

    def test_C17_repeated_questions_have_distinct_turns(self):
        self.upload();a=self.chat("这篇讲什么");b=self.chat("这篇讲什么")
        self.assertNotEqual(a["turn_id"],b["turn_id"])
        same=self.chat("报告命名格式",request_id="idempotent")
        again=self.chat("报告命名格式",request_id="idempotent")
        self.assertEqual(same["run_id"],again["run_id"])

    def test_C18_resolution_model_timeout_is_error(self):
        self.establish()
        with patch.object(self.fake,"generate",new=AsyncMock(side_effect=LLMCallError("timeout",{"status":"error","attempts":1},504))):
            response=self.client.post("/chat",json={"session_id":self.session,"message":"之前提到的那个是什么"})
            self.assertEqual(response.status_code,504)

    def test_C19_resolution_cannot_expand_scope(self):
        self.establish()
        self.fake.output=json.dumps({"intent":"qa","resolved_query":"report","document_ids":["outside"],"clarification_question":None})
        answer=self.chat("之前提到的那个是什么")
        self.assertEqual(answer["answer_status"],"clarification")
        self.assertFalse(answer["citations"])

    def test_C20_history_is_not_document_evidence(self):
        self.upload()
        self.fake.output=json.dumps({"answer_status":"answered","answer":"电脑品牌是Quantum。[1]","cited_numbers":[1]})
        self.chat("这篇文章讲什么");self.fake.output=None
        answer=self.chat("电脑品牌Quantum是什么")
        self.assertEqual(answer["answer_status"],"no_evidence")
        self.assertFalse(answer["citations"])

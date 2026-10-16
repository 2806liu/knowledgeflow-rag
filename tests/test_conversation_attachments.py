"""会话附件12场景与上传恢复16场景的隔离API/故障测试；浏览器交互另验。"""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from conversation_test_support import APIBase,REPORT,ATTENDANCE
from app.rag.parsers import parse_document,DocumentLimitError
from app.rag.ocr import OCRUnavailable

class AttachmentTests(APIBase):
    def test_A01_append_preserves_existing(self):
        a=self.upload();b=self.upload(ATTENDANCE,"attendance.md")
        value=self.view()
        self.assertEqual(set(value["associated_document_ids"]),{a,b})
        self.assertEqual(set(value["document_ids"]),{a,b})
        self.assertEqual(value["active_document_id"],b)

    def test_A02_batch_has_no_default_focus(self):
        batch="batch-many"
        a=self.upload(batch=batch,size=2);b=self.upload(ATTENDANCE,"attendance.md",batch=batch,size=2)
        answer=self.chat("这篇文章讲什么")
        self.assertEqual(answer["answer_status"],"clarification")
        self.assertEqual(set(self.view()["document_ids"]),{a,b})

    def test_A03_partial_batch_keeps_success(self):
        a=self.upload(batch="partial",size=2)
        bad=self.create_job(b"not a supported document","bad.docx",batch="partial",size=2)
        self.assertEqual(bad.status_code,415)
        self.assertIn(a,self.view()["associated_document_ids"])

    def test_A04_identical_file_reuses_index(self):
        a=self.upload();b=self.upload(key="duplicate")
        self.assertEqual(a,b);self.assertEqual(len(self.client.get("/documents").json()),1)
        self.assertEqual(self.view()["document_ids"],[a])

    def test_A05_same_name_different_content(self):
        a=self.upload(name="same.md");b=self.upload(ATTENDANCE,"same.md")
        self.assertNotEqual(a,b)
        self.assertEqual(len(self.view()["associated_document_ids"]),2)

    def test_A06_detach_does_not_delete(self):
        a=self.upload()
        response=self.client.delete("/conversations/"+self.session+"/documents/"+a)
        self.assertEqual(response.status_code,200)
        self.assertEqual(self.view()["document_ids"],[])
        self.assertEqual(self.client.get("/documents/"+a).json()["status"],"indexed")
        self.assertEqual(self.chat("这篇文章讲什么")["answer_status"],"clarification")

    def test_A07_restore_scope_and_turns(self):
        a=self.upload();answer=self.chat("这篇文章讲什么")
        value=self.view()
        self.assertEqual(value["active_document_id"],a)
        self.assertEqual(value["turns"][0]["response"]["run_id"],answer["run_id"])
        self.assertEqual(len(self.client.get("/sessions/"+self.session).json()["messages"]),2)

    def test_A08_new_conversation_is_empty(self):
        self.upload();self.chat("这篇文章讲什么")
        self.session=self.client.post("/conversations").json()["id"]
        answer=self.chat("这篇文章讲什么")
        self.assertEqual(answer["answer_status"],"clarification")
        self.assertEqual(answer["scope_snapshot"]["document_ids"],[])

    def test_A09_explicit_range_preserves_associations(self):
        a=self.upload();b=self.upload(ATTENDANCE,"attendance.md")
        self.scope([a],a)
        self.assertEqual(set(self.view()["associated_document_ids"]),{a,b})
        self.assertEqual(self.chat("这篇文章讲什么")["scope_snapshot"]["document_ids"],[a])
        self.scope([a],a,"all")
        self.assertEqual(set(self.api.flow.conversations.snapshot(self.session)["document_ids"]),{a,b})

    def test_A10_pending_attachment_does_not_block_explicit_old_query(self):
        a=self.upload()
        self.client.post("/conversations/"+self.session+"/batches",json={"batch_id":"pending","size":1})
        answer=self.chat("课程报告命名格式是什么")
        self.assertEqual(answer["answer_status"],"answered")
        self.assertTrue(all(c["document_id"]==a for c in answer["citations"]))
        self.assertEqual(self.chat("这篇文章讲什么")["answer_status"],"clarification")

    def test_A11_request_scope_snapshot_is_immutable(self):
        a=self.upload();self.fake.enter=threading.Event();self.fake.release=threading.Event()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(self.chat,"报告命名格式是什么")
            try:
                self.assertTrue(self.fake.enter.wait(2))
                b=self.upload(ATTENDANCE,"attendance.md")
            finally:self.fake.release.set()
            answer=future.result(3)
        self.assertEqual(answer["scope_snapshot"]["document_ids"],[a])
        self.assertEqual(self.view()["active_document_id"],b)

    def test_A12_new_file_focus_and_old_completion_guard(self):
        a=self.upload();b=self.upload(ATTENDANCE,"attendance.md")
        self.assertEqual(self.chat("这篇文章讲什么")["citations"][0]["document_id"],b)
        repo=self.api.flow.conversations
        repo.register_batch(self.session,"old-batch",1)
        self.scope([b],b)
        with repo.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            value=json.loads(db.execute("SELECT payload FROM conversations WHERE id=?",(self.session,)).fetchone()[0])
            repo.attach_in_transaction(db,value,[a],batch="old-batch",batch_size=1)
        self.assertEqual(self.view()["active_document_id"],b)
        self.assertEqual(self.view()["document_ids"],[b])

    def test_removed_active_document_requires_choice(self):
        a=self.upload();b=self.upload(ATTENDANCE,"attendance.md")
        self.client.delete("/documents/"+b)
        result=self.chat("这篇文章讲什么")
        self.assertEqual(result["answer_status"],"clarification")
        self.assertFalse(result["citations"])
        self.assertIn(a,self.view()["associated_document_ids"])

class UploadTests(APIBase):
    def controlled(self,entered,release):
        def parse(filename,data,**kwargs):
            entered.set();release.wait(3);kwargs["checkpoint"]()
            return parse_document(filename,data,**kwargs)
        return parse

    def test_U01_expired_job_is_recoverable_failure(self):
        self.api.flow.settings=replace(self.config,ingestion_timeout_seconds=600)
        entered,release=threading.Event(),threading.Event()
        with patch("app.ingestion.parse_document",side_effect=self.controlled(entered,release)):
            job=self.create_job().json();self.assertTrue(entered.wait(5))
            # 解析已开始后注入过期时间，避免用20毫秒期限依赖机器调度速度。
            with self.api.flow.conversations.connect() as db:
                row=db.execute("SELECT payload FROM ingestions WHERE id=?",(job["id"],)).fetchone()
                payload=json.loads(row[0]);payload["deadline"]=0
                db.execute("UPDATE ingestions SET payload=? WHERE id=?",(json.dumps(payload),job["id"]))
            release.set()
            value=self.wait(job)
        self.assertEqual(value["error"]["code"],"PROCESSING_TIMEOUT")
        self.assertTrue(value["error"]["retryable"])

    def test_U02_processing_timeout_never_ready(self):
        self.api.flow.settings=replace(self.config,ingestion_timeout_seconds=-1)
        job=self.wait(self.create_job().json())
        self.assertEqual(job["state"],"failed");self.assertEqual(self.client.get("/documents").json(),[])

    def test_U03_cancel_before_task_acceptance_restores_batch(self):
        self.client.post("/conversations/"+self.session+"/batches",json={"batch_id":"network-cancel","size":1})
        self.assertTrue(self.view()["pending_focus"])
        response=self.client.post("/conversations/"+self.session+"/batches/network-cancel/finish")
        self.assertFalse(response.json()["pending_focus"])
        self.assertEqual(self.client.get("/conversations/"+self.session+"/ingestions").json(),[])

    def test_U04_cancel_queued_task(self):
        entered,release=threading.Event(),threading.Event()
        with patch("app.ingestion.parse_document",side_effect=self.controlled(entered,release)):
            first=self.create_job().json();self.assertTrue(entered.wait(2))
            second=self.create_job(ATTENDANCE,"attendance.md").json()
            result=self.client.post("/ingestions/"+second["id"]+"/cancel").json()
            self.assertEqual(result["state"],"cancelled")
            release.set();self.assertEqual(self.wait(first)["state"],"ready")
        self.assertEqual(len(self.client.get("/documents").json()),1)

    def test_U05_cancel_running_at_checkpoint(self):
        entered,release=threading.Event(),threading.Event()
        with patch("app.ingestion.parse_document",side_effect=self.controlled(entered,release)):
            job=self.create_job().json();self.assertTrue(entered.wait(2))
            result=self.client.post("/ingestions/"+job["id"]+"/cancel").json()
            self.assertEqual(result["state"],"cancel_requested")
            release.set();self.assertEqual(self.wait(job)["state"],"cancelled")
        self.assertEqual(self.client.get("/documents").json(),[])
        self.assertFalse(list((self.root/"ingestion_sources").glob("*.bin")))

    def test_U06_completed_cancel_race_is_explicit(self):
        document=self.upload()
        job=self.client.get("/conversations/"+self.session+"/ingestions").json()[0]
        response=self.client.post("/ingestions/"+job["id"]+"/cancel")
        self.assertEqual(response.status_code,409)
        self.assertEqual(self.client.get("/documents/"+document).json()["status"],"indexed")

    def test_U07_response_loss_lookup_and_same_key(self):
        response=self.create_job(key="lost-response")
        rows=self.client.get("/conversations/"+self.session+"/ingestions?client_request_id=lost-response").json()
        self.assertEqual(len(rows),1)
        again=self.create_job(key="lost-response")
        self.assertEqual(again.json()["id"],response.json()["id"])
        self.assertEqual(self.wait(rows[0])["state"],"ready")
        self.assertEqual(len(self.client.get("/documents").json()),1)

    def test_U08_retry_once_and_idempotency_conflict(self):
        class BrokenIndex:
            def stage_chunks(self,*args):raise OSError("fixture indexing fault")
            def delete_document(self,*args):pass
        self.api.flow.vector_index=BrokenIndex()
        failed=self.wait(self.create_job(key="retryable").json())
        self.assertEqual(failed["error"]["code"],"INDEXING_FAILED")
        self.api.flow.vector_index=None
        job=self.client.post("/ingestions/"+failed["id"]+"/retry").json()
        self.assertEqual(self.wait(job)["state"],"ready")
        conflict=self.create_job(ATTENDANCE,"other.md",key="retryable")
        self.assertEqual(conflict.status_code,409)
        self.assertEqual(len(self.client.get("/documents").json()),1)

    def test_U09_byte_limit_error(self):
        self.api.settings=replace(self.config,max_upload_bytes=20)
        self.api.flow.settings=self.api.settings
        response=self.create_job()
        self.assertEqual(response.status_code,413)
        self.assertEqual(response.json()["detail"]["code"],"UPLOAD_TOO_LARGE")

    def test_U10_format_and_signature_errors(self):
        bad=self.create_job(b"unrecognized"*5,"bad.docx")
        self.assertEqual(bad.status_code,415)
        bad=self.create_job(b"not pdf"*10,"bad.pdf")
        self.assertEqual(bad.status_code,400)
        self.assertEqual(self.client.get("/documents").json(),[])

    def test_U11_resource_limits_are_distinct(self):
        for text,expected in [("PDF exceeds OCR_MAX_PAGES","OCR_PAGE_LIMIT"),
             ("image exceeds MAX_IMAGE_PIXELS","IMAGE_PIXEL_LIMIT"),("page exceeds MAX_REGIONS","OCR_REGION_LIMIT")]:
            with self.subTest(expected=expected),patch("app.ingestion.parse_document",side_effect=DocumentLimitError(text)):
                job=self.wait(self.create_job(key=str(time.monotonic())).json())
                self.assertEqual(job["error"]["code"],expected)
                self.assertFalse(job["error"]["retryable"])

    def test_U12_ocr_unavailable_not_no_evidence(self):
        with patch("app.ingestion.parse_document",side_effect=OCRUnavailable("fixture")):
            value=self.wait(self.create_job().json())
        self.assertEqual(value["error"]["code"],"OCR_UNAVAILABLE")
        self.assertEqual(value["state"],"failed")

    def test_U13_status_read_errors_do_not_mutate_task(self):
        self.upload()
        job=self.client.get("/conversations/"+self.session+"/ingestions").json()[0]
        self.assertEqual(self.client.get("/ingestions/not-real").status_code,404)
        for _ in range(3):
            self.assertEqual(self.client.get("/ingestions/"+job["id"]).json()["state"],"ready")

    def test_U14_restart_marks_interrupted_then_retry(self):
        entered,release=threading.Event(),threading.Event()
        with patch("app.ingestion.parse_document",side_effect=self.controlled(entered,release)):
            job=self.create_job().json();self.assertTrue(entered.wait(2))
            manager=self.api.flow._ingestion_service
            manager.close();release.set();manager.pool.shutdown(wait=True)
        del self.api.flow._ingestion_service
        value=self.client.get("/ingestions/"+job["id"]).json()
        self.assertEqual(value["state"],"interrupted")
        retried=self.client.post("/ingestions/"+job["id"]+"/retry").json()
        self.assertEqual(self.wait(retried)["state"],"ready")

    def test_U15_failure_keeps_old_source(self):
        document=self.upload()
        with patch("app.ingestion.parse_document",side_effect=ValueError("bad")):
            failed=self.wait(self.create_job(ATTENDANCE,"bad.md").json())
        self.assertEqual(failed["state"],"failed")
        source=self.client.get("/documents/"+document+"/source")
        self.assertEqual(source.content,REPORT)
        self.assertEqual(self.view()["document_ids"],[document])

    def test_U16_reselection_and_queue_remain_available(self):
        entered,release=threading.Event(),threading.Event()
        with patch("app.ingestion.parse_document",side_effect=self.controlled(entered,release)):
            job=self.create_job().json();self.assertTrue(entered.wait(2))
            later=self.create_job(ATTENDANCE,"attendance.md")
            self.assertEqual(later.status_code,202);release.set()
            self.assertEqual(self.wait(job)["state"],"ready")
            self.assertEqual(self.wait(later.json())["state"],"ready")
        a=self.upload(key="reselected")
        self.assertIn(a,self.view()["associated_document_ids"])
        self.assertEqual(len(self.client.get("/documents").json()),2)

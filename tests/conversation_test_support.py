"""新模块测试夹具：隔离SQLite、受控生成器、真实API；不触碰日常资料。"""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import asyncio
import json
import re
import threading
import time
import unittest
import uuid
from fastapi.testclient import TestClient
from app.config import settings
from app.store import Store
from app.services import KnowledgeFlow

FIXTURES=Path(__file__).resolve().parents[1]/"evaluation/conversation-v2/fixtures"
REPORT=(FIXTURES/"report.md").read_bytes()
ATTENDANCE=(FIXTURES/"attendance.md").read_bytes()

class FakeClient:
    """只验证编排和契约；该生成器的成功不能充当真实模型语义验收。"""
    def __init__(self):
        self.calls=[];self.fail_at=set();self.output=None;self.enter=None;self.release=None
    async def generate(self,messages):
        self.calls.append(messages)
        if self.enter:
            self.enter.set()
            while not self.release.is_set():await asyncio.sleep(.005)
        if len(self.calls) in self.fail_at:raise TimeoutError("injected timeout")
        payload=json.loads(messages[-1]["content"])
        if self.output is not None:
            raw=self.output(payload) if callable(self.output) else self.output
        elif "previous_summary" in payload:
            raw=json.dumps({"summary":"报告提交与考勤。用户已纠正截止时间，应保留最新要求；旧说法不再有效。"})
        elif "allowed_document_ids" in payload:
            raw=json.dumps({"intent":"qa","resolved_query":"课程报告命名格式","document_ids":payload["allowed_document_ids"],"clarification_question":None})
        elif payload.get("task") == "overview_map":
            # 测试模型按新契约返回结构化要点；每条来源仍是本批原文编号。
            units=[]
            for item in payload["evidence"]:
                text=item["text"]
                if "关键点" in text:
                    anchors=re.findall(r"关键点\d+[：:].*?[。\n]",text)
                    claim=" ".join(dict.fromkeys(anchors)) or text[:100]
                elif "学号-姓名-标题" in text:
                    claim="课程报告命名格式为学号-姓名-标题，提交PDF，截止时间6月30日18:00。"
                elif "4次" in text:
                    claim="实验室每周进行4次，需记录工时。"
                else:
                    claim=text[:100]
                units.append({"title":"要点"+str(len(units)+1),"claim":claim,
                              "source_numbers":[item["citation_number"]]})
            raw=json.dumps({"units":units},ensure_ascii=False)
        elif payload.get("task") == "overview_reduce":
            # 合成模型保留全部独有要点，供测试核对末尾章节与来源覆盖。
            raw=json.dumps({"units":payload["units"]},ensure_ascii=False)
        elif "section_summaries" in payload:
            answer="\n".join(item["text"] for item in payload["section_summaries"])
            numbers=list(dict.fromkeys(int(n) for n in re.findall(r"\[(\d+)\]",answer)))
            raw=json.dumps({"answer_status":"answered","answer":answer,"cited_numbers":numbers})
        else:
            items=payload.get("evidence",[])
            lines=[];numbers=[]
            for item in items:
                text=item["text"]
                n=item.get("citation_number",(item.get("source_numbers") or [1])[0])
                if "关键点" in text:
                    anchors=re.findall(r"关键点\d+[：:].*?[。\n]",text)
                    value=" ".join(dict.fromkeys(anchors)) or text[:100]
                elif "学号-姓名-报告" in text:value="报告命名格式为学号-姓名-报告，提交PDF，截止时间6月30日18:00。"
                elif "4次" in text:value="实验室每天打卡4次，并有请假规则。"
                else:value=text[:100]
                lines.append(str(len(lines)+1)+". "+value+"["+str(n)+"]")
                numbers.append(n)
            raw=json.dumps({"answer_status":"answered","answer":"\n".join(lines),"cited_numbers":list(dict.fromkeys(numbers))})
        if payload.get("task") == "overview_map":
            # 旧测试注入的是自由文本回答；夹具在测试边界适配为新结构化契约。
            value=json.loads(raw)
            if "answer" in value and "units" not in value:
                units=[]
                for line in value["answer"].splitlines():
                    numbers=[int(n) for n in re.findall(r"\[(\d+)\]",line)]
                    if numbers:
                        clean=re.sub(r"\[\d+\]","",line)
                        clean=re.sub(r"^\s*\d+[.、)]\s*","",clean).strip()
                        title=re.split(r"[：:]",clean,maxsplit=1)[0].strip()[:20]
                        units.append({"title":title,"claim":clean,
                                      "source_numbers":numbers})
                raw=json.dumps({"units":units},ensure_ascii=False)
        return SimpleNamespace(content=raw),{"status":"ok","attempts":1,"retries":0,"usage":None,"model":"deterministic-test-fixture"}

class APIBase(unittest.TestCase):
    def setUp(self):
        """所有任务目录位于项目A忽略目录；生成器禁止访问网络。"""
        from app import main
        self.api=main;self.original=main.store,main.flow,main.settings
        self.root=Path("tests/.tmp/conversation-"+str(uuid.uuid4()))
        self.config=replace(settings,data_dir=self.root,llm_mode="openai_compatible",
            llm_base_url="http://invalid.local/v1",llm_api_key="fixture-only",llm_model="fixture",
            llm_max_retries=0,llm_answer_format="json",vector_enabled=False,reranker_enabled=False)
        main.settings=self.config;main.store=Store(self.root);main.flow=KnowledgeFlow(main.store,self.config)
        self.fake=FakeClient();main.flow.llm.client=self.fake
        self.client=TestClient(main.app).__enter__()
        self.session=self.client.post("/conversations").json()["id"]
    def tearDown(self):
        self.client.__exit__(None,None,None)
        if hasattr(self.api.flow,"_ingestion_service"):
            self.api.flow._ingestion_service.pool.shutdown(wait=True)
        self.api.store.conn.close()
        self.api.store,self.api.flow,self.api.settings=self.original
    def create_job(self,data=REPORT,name="report.md",key=None,batch=None,size=1):
        key=key or str(uuid.uuid4());batch=batch or str(uuid.uuid4())
        mime="application/pdf" if name.endswith(".pdf") else "image/png" if name.endswith(".png") else "text/markdown"
        response=self.client.post("/ingestions",data={"session_id":self.session,"client_request_id":key,"batch_id":batch,"batch_size":size},
                                  files={"file":(name,data,mime)})
        return response
    def wait(self,job,timeout=4):
        stop=time.monotonic()+timeout
        while time.monotonic()<stop:
            value=self.client.get("/ingestions/"+job["id"]).json()
            if value["state"] in {"ready","failed","cancelled","interrupted"}:return value
            time.sleep(.01)
        self.fail("job did not reach terminal state")
    def upload(self,data=REPORT,name="report.md",**kwargs):
        response=self.create_job(data,name,**kwargs)
        self.assertEqual(response.status_code,202,response.text)
        value=self.wait(response.json())
        self.assertEqual(value["state"],"ready",value)
        return value["document_id"]
    def chat(self,question,**kwargs):
        response=self.client.post("/chat",json={"session_id":self.session,"message":question,**kwargs})
        self.assertEqual(response.status_code,200,response.text)
        return response.json()
    def view(self):
        return self.client.get("/conversations/"+self.session).json()
    def scope(self,ids,active=None,mode="selected"):
        response=self.client.patch("/conversations/"+self.session+"/scope",json={"mode":mode,"document_ids":ids,"active_document_id":active})
        self.assertEqual(response.status_code,200,response.text)
        return response.json()

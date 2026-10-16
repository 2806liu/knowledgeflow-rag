"""增强模块的脚本化模型夹具：只验证契约、状态和故障，不模拟语义验收。"""
import asyncio
import json
import re
import uuid
from types import SimpleNamespace
from conversation_test_support import APIBase
from app.rag.memory import answer_units

def resolution(question="解释这个概念", intent="concept_explanation", docs=(), aspects=None, **kwargs):
    value=dict(intent=intent,resolved_query=question,document_ids=list(docs),
                requested_aspects=aspects or [dict(id="a1",question=question,requires_document=False)],
                referenced_turn_id=None,referenced_unit_id=None,referenced_unit_ids=[],
                candidate_keys=[],clarification_question=None)
    value.update(kwargs)
    return value

def section(text="这是模型的一般解释。", kind="model_knowledge", aspects=("a1",), numbers=()):
    return dict(source_type=kind,content=text,citation_numbers=list(numbers),aspect_ids=list(aspects))

def answer(*sections, unresolved=()):
    return dict(action="answer",sections=list(sections),unresolved_aspect_ids=list(unresolved),supplemental_query=None)

class ScriptClient:
    """按测试安排返回响应/异常，拒绝未安排的额外调用以发现无界循环。"""
    def __init__(self, outputs):
        self.outputs=list(outputs);self.calls=[]
    async def generate(self,messages):
        self.calls.append(messages)
        if not self.outputs:raise AssertionError("unexpected model call")
        value=self.outputs.pop(0)
        if callable(value):value=value(messages)
        if asyncio.iscoroutine(value):value=await value
        if isinstance(value,Exception):raise value
        if json.loads(messages[-1]["content"]).get("task")=="overview_map" and isinstance(value,dict) and "answer" in value:
            # 旧脚本的回答形状只在测试夹具内转成结构化概述，生产端仍严格校验。
            units=[]
            for line in value["answer"].splitlines():
                numbers=[int(n) for n in re.findall(r"\[(\d+)\]",line)]
                if numbers:
                    clean=re.sub(r"\[\d+\]","",line).strip()
                    units.append({"title":clean[:20],"claim":clean,"source_numbers":numbers})
            value={"units":units}
        return SimpleNamespace(content=json.dumps(value,ensure_ascii=False)),dict(
            status="ok",attempts=1,retries=0,usage={"input_tokens":10,"output_tokens":10,"total_tokens":20},model="scripted-fixture")

class EnhancedBase(APIBase):
    def setUp(self):
        super().setUp()
        value=self.client.patch("/conversations/"+self.session+"/policy",json={"answer_policy":"document_enhanced"})
        self.assertEqual(value.status_code,200)
    def script(self,*outputs):
        self.model=ScriptClient(outputs);self.api.flow.llm.client=self.model
        return self.model
    def post(self,question="解释这个概念",**kwargs):
        return self.client.post("/chat",json=dict(session_id=self.session,message=question,**kwargs))
    def ask(self,question="解释这个概念",**kwargs):
        response=self.post(question,**kwargs);self.assertEqual(response.status_code,200,response.text);return response.json()
    def seed(self,document=None,text="1. 背景。[1]\n2. 目标与范围。[1]",status="answered",basis="document_only"):
        citations=[]
        if document:
            row=self.api.store.chunks((document,))[0]
            citations=[dict(citation_number=1,chunk_id=row["id"],document_id=document,
                            filename=row["filename"],position=row["position"],page=None,snippet=row["text"],score=None)]
        elif basis=="model_only":
            text=text.replace("[1]","")
        turn,_=self.api.flow.conversations.begin_turn(self.session,str(uuid.uuid4()),"旧问题")
        response=dict(answer=text,answer_status=status,answer_basis=basis,citations=citations)
        units=answer_units(text,citations)
        for unit in units:unit["source_type"]="model_knowledge" if basis=="model_only" else "document"
        self.api.flow.conversations.finish_turn(turn,dict(response=response,answer_units=units,
            resolution={"resolved_query":"旧主题","document_ids":[document] if document else []}))
        return turn,units
    def traces(self,data,node):
        return [e for e in self.client.get("/runs/"+data["run_id"]).json()["events"] if e["node"]==node]

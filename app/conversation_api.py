"""新会话和异步摄取HTTP边界；运行对象通过provider获取，兼容隔离测试。"""
from fastapi import Form,File,UploadFile,Request,HTTPException,Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field
from typing import Literal
from app.conversation import ConversationError
from app.rag.context_builder import ContextOverflow

class PolicyInput(BaseModel):
    """工作台显式选择模式；空创建请求保留旧客户端默认。"""
    answer_policy: Literal["document_strict","document_enhanced"]="document_strict"

class AttachInput(BaseModel):
    document_ids:list[str]=Field(min_length=1,max_length=32)
class BatchInput(BaseModel):
    batch_id:str=Field(min_length=1,max_length=128)
    size:int=Field(ge=1,le=32)

class ScopeInput(BaseModel):
    mode:str="attached"
    document_ids:list[str]=Field(default_factory=list,max_length=1000)
    active_document_id:str|None=None
    revision:int|None=None

class ConversationMetadataInput(BaseModel):
    """历史侧栏只允许重命名与可恢复归档，不提供删除入口。"""
    title:str|None=None
    archived:bool|None=None

def attach_conversation_routes(app,provider,validate,build_chunks):
    """路由只校验输入与映射异常，任务执行及业务状态由独立模块负责。"""
    def jobs():
        flow=provider()
        if not hasattr(flow,"_ingestion_service"):
            from app.ingestion import IngestionService
            flow._ingestion_service=IngestionService(flow,build_chunks)
        return flow._ingestion_service

    @app.exception_handler(ConversationError)
    async def conversation_error(request,exc):
        return JSONResponse(status_code=exc.status,content={"detail":{"code":exc.code,"message":str(exc)}})

    @app.exception_handler(ContextOverflow)
    async def context_error(request,exc):
        # 只返回运行编号和预算诊断；界面可据此定位失败阶段，不暴露输入原文。
        return JSONResponse(status_code=413,content={"detail":{"code":"CONTEXT_TOO_LARGE","message":str(exc),
            "run_id":getattr(exc,"run_id",None),"context_budget":exc.trace}})

    @app.post("/conversations",status_code=201)
    def create_conversation(body:PolicyInput|None=None):
        return provider().conversations.create(answer_policy=body.answer_policy if body else "document_strict")

    @app.get("/conversations")
    def list_conversations(archived:bool=False,q:str="",cursor:str|None=None,
                           limit:int=Query(default=30,ge=1,le=100)):
        """分页返回轻量会话索引，不把历史回答塞进侧栏请求。"""
        return provider().conversations.list_history(archived=archived,query=q,cursor=cursor,limit=limit)

    @app.patch("/conversations/{session_id}")
    def update_conversation_metadata(session_id:str,body:ConversationMetadataInput):
        return provider().conversations.metadata(session_id,title=body.title,archived=body.archived)

    @app.patch("/conversations/{session_id}/policy")
    def set_policy(session_id:str,body:PolicyInput):
        return provider().conversations.set_answer_policy(session_id,body.answer_policy)

    @app.get("/conversations/{session_id}")
    def get_conversation(session_id:str,turn_limit:int=Query(default=100,ge=1,le=100)):
        return provider().conversations.view(session_id,turn_limit=turn_limit)

    @app.get("/conversations/{session_id}/turns")
    def list_conversation_turns(session_id:str,before:str|None=None,
                                limit:int=Query(default=50,ge=1,le=100)):
        """向上滚动时按稳定游标读取旧轮次，原有会话读取接口保持兼容。"""
        return provider().conversations.page_turns(session_id,before=before,limit=limit)

    @app.get("/conversations/{session_id}/memory")
    def memory_state(session_id:str):
        """只读派生记忆和压缩诊断；事实原文仍从文档来源接口核验。"""
        from app.rag.memory import MemoryManager
        flow=provider()
        if not flow.conversations.get(session_id):
            raise ConversationError("CONVERSATION_NOT_FOUND","会话不存在",404)
        return {**MemoryManager(flow.conversations,flow.settings,flow.llm.client).state(session_id),
                "dialogue_state":flow.conversations.get(session_id).get("dialogue_state",{})}

    @app.post("/conversations/{session_id}/documents")
    def attach_documents(session_id:str,body:AttachInput):
        return provider().conversations.attach(session_id,list(dict.fromkeys(body.document_ids)))

    @app.patch("/conversations/{session_id}/scope")
    def set_scope(session_id:str,body:ScopeInput):
        return provider().conversations.scope(session_id,body.mode,list(dict.fromkeys(body.document_ids)),
                                              body.active_document_id,body.revision)

    @app.delete("/conversations/{session_id}/documents/{document_id}")
    def detach_document(session_id:str,document_id:str):
        return provider().conversations.detach(session_id,document_id)

    @app.post("/conversations/{session_id}/batches")
    def register_batch(session_id:str,body:BatchInput):
        """传输前声明新附件焦点，防止上传中“这篇”被解释成旧文档。"""
        repo=provider().conversations
        if not repo.get(session_id):
            raise ConversationError("CONVERSATION_NOT_FOUND","会话不存在",404)
        repo.register_batch(session_id,body.batch_id,body.size)
        return repo.get(session_id)

    @app.post("/conversations/{session_id}/batches/{batch_id}/finish")
    def finish_empty_batch(session_id:str,batch_id:str):
        """没有后台活动任务时恢复未完成传输的焦点标记，不冒充服务端取消。"""
        repo=provider().conversations
        with repo.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows=db.execute("SELECT payload FROM ingestions WHERE session_id=?",(session_id,)).fetchall()
            import json
            active=any(v.get("batch_id")==batch_id and v["state"] not in {"ready","failed","cancelled","interrupted"}
                       for v in (json.loads(r[0]) for r in rows))
            row=db.execute("SELECT payload FROM conversations WHERE id=?",(session_id,)).fetchone()
            if row and not active:
                value=json.loads(row[0])
                if value.get("focus_batch")==batch_id:
                    value["pending_focus"]=False;repo.save(db,value)
        return repo.get(session_id)

    @app.post("/ingestions",status_code=202)
    async def upload(request:Request,file:UploadFile=File(...),session_id:str=Form(...),
                     client_request_id:str=Form(...),batch_id:str=Form(...),batch_size:int=Form(1,ge=1,le=32)):
        """收到完整文件后受理任务；网络传输和后台识别分别计状态。"""
        flow=provider()
        data=await file.read(flow.settings.max_upload_bytes+1)
        try:
            validate(file.filename or "document.txt",data,file.content_type)
        except HTTPException as exc:
            code="UPLOAD_TOO_LARGE" if exc.status_code==413 else "UNSUPPORTED_TYPE" if exc.status_code==415 else "INVALID_DOCUMENT"
            return JSONResponse(status_code=exc.status_code,content={"detail":{"code":code,"message":str(exc.detail)}})
        if await request.is_disconnected():
            raise ConversationError("CLIENT_DISCONNECTED","传输已取消",499)
        return jobs().create(session_id,client_request_id,file.filename or "document.txt",data,batch_id,batch_size)

    @app.get("/ingestions/{job_id}")
    def get_ingestion(job_id:str):
        return jobs().get(job_id)

    @app.get("/conversations/{session_id}/ingestions")
    def list_ingestions(session_id:str,client_request_id:str|None=None):
        if client_request_id:
            with provider().conversations.connect() as db:
                row=db.execute("SELECT id FROM ingestions WHERE session_id=? AND request_id=?",(session_id,client_request_id)).fetchone()
            return [jobs().get(row[0])] if row else []
        return jobs().list(session_id)

    @app.post("/ingestions/{job_id}/cancel")
    def cancel_ingestion(job_id:str):
        return jobs().cancel(job_id)

    @app.post("/ingestions/{job_id}/retry")
    def retry_ingestion(job_id:str):
        return jobs().retry(job_id)

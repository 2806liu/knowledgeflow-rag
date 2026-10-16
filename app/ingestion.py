"""上传任务：独立状态、有限队列、协作取消、幂等重试与原子文档发布。"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import threading
import time
import uuid
from app.conversation import ConversationError
from app.rag.parsers import parse_document, DocumentLimitError
from app.rag.ocr import OCRUnavailable
from app.store import now_iso

_TERMINAL={"ready","failed","cancelled","interrupted"}

class TaskCancelled(Exception):
    """在阶段或页边界确认取消，不能强杀正在执行的ONNX调用。"""

class IngestionService:
    def __init__(self, flow, build_chunks):
        self.flow,self.repo,self.config=flow,flow.conversations,flow.settings
        self.build_chunks=build_chunks
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix="rag-ingestion")
        self.stop=threading.Event()
        self.root=self.config.data_dir/"ingestion_sources"
        self.root.mkdir(parents=True,exist_ok=True)
        # 服务重启后保留中断记录，必须明确重试；不把未完成任务标成功。
        with self.repo.connect() as db:
            for row in db.execute("SELECT id,payload FROM ingestions").fetchall():
                value=json.loads(row["payload"])
                # 失败/中断暂存只保留24小时供重试，过期清理严格限制在任务私有目录。
                from pathlib import Path
                source=Path(value["path"]).resolve()
                if value["state"] in {"failed","interrupted","cancelled"} and time.time()-value.get("created_at",time.time())>86400:
                    if source.is_relative_to(self.root.resolve()):source.unlink(missing_ok=True)
                    value["source_expired"]=True
                    value["error"]={"code":"FILE_EXPIRED","message":"暂存已过期，请重新选择文件","retryable":False}
                    self._save(db,value)
                if value["state"] not in _TERMINAL:
                    value.update(state="interrupted",error={"code":"PROCESS_INTERRUPTED","message":"服务重启，任务已中断","retryable":True})
                    db.execute("UPDATE ingestions SET payload=?,updated=? WHERE id=?",
                               (json.dumps(value,ensure_ascii=False),time.time(),row["id"]))

    def get(self, job_id, public=True):
        with self.repo.connect() as db:
            row=db.execute("SELECT payload FROM ingestions WHERE id=?",(job_id,)).fetchone()
        if not row: raise ConversationError("INGESTION_NOT_FOUND","上传任务不存在",404)
        value=json.loads(row[0])
        return {k:v for k,v in value.items() if k!="path"} if public else value

    def list(self, session):
        with self.repo.connect() as db:
            rows=db.execute("SELECT payload FROM ingestions WHERE session_id=? ORDER BY updated DESC LIMIT 60",(session,)).fetchall()
        return [{k:v for k,v in json.loads(r[0]).items() if k!="path"} for r in rows]

    def _save(self,db,value):
        db.execute("UPDATE ingestions SET payload=?,updated=? WHERE id=?",
                   (json.dumps(value,ensure_ascii=False),time.time(),value["id"]))

    def create(self, session, request_id, filename, data, batch_id, batch_size):
        if not self.repo.get(session): raise ConversationError("CONVERSATION_NOT_FOUND","会话不存在",404)
        if not request_id or len(request_id)>128 or batch_size<1 or batch_size>32:
            raise ConversationError("INVALID_UPLOAD_REQUEST","上传标识或批次大小无效",422)
        fingerprint=hashlib.sha256(data).hexdigest()
        with self.repo.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # 归档会话只读；在任务受理事务内检查，防止上传与归档并发穿透。
            self.repo._require_writable(db, session)
            previous=db.execute("SELECT payload FROM ingestions WHERE session_id=? AND request_id=?",(session,request_id)).fetchone()
            if previous:
                value=json.loads(previous[0])
                if value["content_hash"]!=fingerprint:
                    raise ConversationError("IDEMPOTENCY_CONFLICT","相同请求标识对应了不同文件")
                return {k:v for k,v in value.items() if k!="path"}
            jobs=[json.loads(r[0]) for r in db.execute("SELECT payload FROM ingestions")]
            if sum(j["state"] not in _TERMINAL for j in jobs)>=self.config.ingestion_queue_limit:
                raise ConversationError("UPLOAD_QUEUE_FULL","上传队列已满，请稍后重试",429)
            job_id=str(uuid.uuid4())
            path=self.root/(job_id+".bin")
            try:
                path.write_bytes(data)
            except OSError as exc:
                path.unlink(missing_ok=True)
                raise ConversationError("UPLOAD_STORAGE_FAILED","无法暂存文件，请检查磁盘空间",503) from exc
            value=dict(id=job_id,session_id=session,request_id=request_id,filename=filename,
                content_hash=fingerprint,path=str(path),batch_id=batch_id,batch_size=batch_size,
                state="queued",attempt=0,error=None,document_id=None,processed_pages=0,total_pages=None,
                created_at=time.time(),deadline=time.time()+self.config.ingestion_timeout_seconds)
            try:
                db.execute("INSERT INTO ingestions VALUES(?,?,?,?,?)",(job_id,session,request_id,json.dumps(value,ensure_ascii=False),time.time()))
            except Exception:
                path.unlink(missing_ok=True)
                raise
        self.repo.register_batch(session,batch_id,batch_size)
        self.pool.submit(self._run,job_id)
        return self.get(job_id)

    def _checkpoint(self,job_id,done=None,total=None,stage=None):
        value=self.get(job_id,False)
        if self.stop.is_set(): raise InterruptedError("service stopping")
        if value["state"] in {"cancel_requested","cancelled"}: raise TaskCancelled()
        if time.time()>value["deadline"]: raise TimeoutError("ingestion deadline")
        if stage or done is not None:
            with self.repo.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                value=json.loads(db.execute("SELECT payload FROM ingestions WHERE id=?",(job_id,)).fetchone()[0])
                if value["state"] in {"cancel_requested","cancelled"}: raise TaskCancelled()
                if stage:value["state"]=stage
                if done is not None:value.update(processed_pages=done,total_pages=total)
                self._save(db,value)

    def cancel(self,job_id):
        with self.repo.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            value=json.loads(db.execute("SELECT payload FROM ingestions WHERE id=?",(job_id,)).fetchone()[0]) if db.execute("SELECT 1 FROM ingestions WHERE id=?",(job_id,)).fetchone() else None
            if value is None: raise ConversationError("INGESTION_NOT_FOUND","任务不存在",404)
            if value["state"]=="ready": raise ConversationError("ALREADY_COMPLETED","任务已完成，不能再取消")
            if value["state"] in _TERMINAL:return self.get(job_id)
            queued=value["state"]=="queued"
            value["state"]="cancelled" if queued else "cancel_requested"
            self._save(db,value)
        if queued:
            from pathlib import Path
            Path(value["path"]).unlink(missing_ok=True)
            self._release_focus(value)
        return self.get(job_id)

    def retry(self,job_id):
        from pathlib import Path
        with self.repo.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row=db.execute("SELECT payload FROM ingestions WHERE id=?",(job_id,)).fetchone()
            if not row: raise ConversationError("INGESTION_NOT_FOUND","任务不存在",404)
            value=json.loads(row[0])
            # 重试会重新发布资料，已归档会话必须先恢复。
            self.repo._require_writable(db, value["session_id"])
            if value["state"] not in {"failed","interrupted"}:
                raise ConversationError("NOT_RETRYABLE","当前任务状态不允许重试")
            if value["error"] and not value["error"].get("retryable",False):
                raise ConversationError("NOT_RETRYABLE","该错误需更换资料或修复配置后重新上传")
            if value["attempt"]>=self.config.ingestion_max_retries:
                raise ConversationError("RETRY_LIMIT","已达到重试上限")
            if not Path(value["path"]).is_file():
                raise ConversationError("FILE_EXPIRED","暂存文件已失效，请重新选择文件",410)
            value.update(state="queued",error=None,attempt=value["attempt"]+1,
                         deadline=time.time()+self.config.ingestion_timeout_seconds)
            self._save(db,value)
        self.repo.register_batch(value["session_id"],value["batch_id"],value["batch_size"])
        self.pool.submit(self._run,job_id)
        return self.get(job_id)

    def _release_focus(self,value):
        """失败/取消后解除处理中标记；旧任务不得覆盖较新的焦点。"""
        with self.repo.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row=db.execute("SELECT payload FROM conversations WHERE id=?",(value["session_id"],)).fetchone()
            if row:
                conversation=json.loads(row[0])
                if conversation.get("focus_batch")==value["batch_id"]:
                    tasks=[json.loads(r[0]) for r in db.execute("SELECT payload FROM ingestions WHERE session_id=?",(value["session_id"],))]
                    conversation["pending_focus"]=any(t.get("batch_id")==value["batch_id"] and t["state"] not in _TERMINAL for t in tasks)
                    self.repo.save(db,conversation)

    def _run(self,job_id):
        from pathlib import Path
        candidate=None;published=False;value=self.get(job_id,False);phase="parsing"
        try:
            self._checkpoint(job_id,stage="parsing")
            data=Path(value["path"]).read_bytes()
            # 相同字节且已有就绪文档时直接复用，不重复解析和索引。
            with self.repo.connect() as db:
                old=db.execute("SELECT id FROM documents WHERE content_hash=? AND status='indexed' AND chunk_count>0",
                               (value["content_hash"],)).fetchone()
            if old:
                document_id=old[0];parsed=None;chunks=None;chunker=None
            else:
                parsed=parse_document(value["filename"],data,config=self.config,
                    checkpoint=lambda done=None,total=None:self._checkpoint(job_id,done,total))
                if len(re.sub(r"\s+","",parsed.text))<self.config.min_extracted_characters:
                    raise ValueError("too little usable text")
                phase="chunking";self._checkpoint(job_id,stage=phase)
                document_id=str(uuid.uuid4());candidate=document_id
                chunks,chunker=self.build_chunks(document_id,parsed.text,parsed.pages,parsed.metadata)
                if not chunks:raise ValueError("no retrievable chunks")
                phase="indexing";self._checkpoint(job_id,stage=phase)
                if self.flow.vector_index is not None:
                    self.flow.vector_index.stage_chunks(document_id,chunks)
                self._checkpoint(job_id)
                self.repo.store.save_source(document_id,data)
            # 发布文档、规范chunk、会话关联和任务ready在同一个SQLite事务内完成。
            with self.repo.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                current=json.loads(db.execute("SELECT payload FROM ingestions WHERE id=?",(job_id,)).fetchone()[0])
                if current["state"] in {"cancel_requested","cancelled"}:raise TaskCancelled()
                if self.stop.is_set():raise InterruptedError()
                if time.time()>current["deadline"]:raise TimeoutError()
                if parsed:
                    db.execute("INSERT INTO documents(id,filename,content,content_type,metadata,content_hash,size_bytes,"
                         "status,indexed_at,chunk_count,chunk_strategy,chunk_version,vector_status,embedding_model,created_at)"
                         " VALUES(?,?,?,?,?,?,?,'indexed',?,?,?,?,?,?,?)",
                         (document_id,value["filename"],parsed.text,parsed.content_type,json.dumps(parsed.metadata,ensure_ascii=False),
                          value["content_hash"],len(data),now_iso(),len(chunks),chunker.name,chunker.version,
                          "indexed" if self.flow.vector_index is not None else "not_indexed",
                          self.config.embedding_model if self.flow.vector_index is not None else None,now_iso()))
                    db.executemany("INSERT INTO document_pages VALUES(?,?,?,?)",
                        [(document_id,p.page_number,p.text,json.dumps(p.metadata,ensure_ascii=False)) for p in parsed.pages])
                    db.executemany("INSERT INTO chunks VALUES(?,?,?,?,?)",
                        [(cid,document_id,text,pos,json.dumps(meta,ensure_ascii=False)) for cid,_,text,pos,meta in chunks])
                elif not self.repo.documents([document_id],db):
                    raise ValueError("reused document no longer ready")
                conversation=json.loads(db.execute("SELECT payload FROM conversations WHERE id=?",(value["session_id"],)).fetchone()[0])
                self.repo.attach_in_transaction(db,conversation,[document_id],batch=value["batch_id"],batch_size=value["batch_size"])
                current.update(state="ready",document_id=document_id,duplicate=parsed is None,error=None,
                               processed_pages=len(parsed.pages) if parsed else 0)
                self._save(db,current)
            published=True;self.repo.store.index_generation+=1
            Path(value["path"]).unlink(missing_ok=True)
        except Exception as exc:
            if isinstance(exc,TaskCancelled):state,code,message,retry="cancelled","CANCELLED","上传任务已取消",False
            elif isinstance(exc,InterruptedError):state,code,message,retry="interrupted","PROCESS_INTERRUPTED","服务已停止，可重试",True
            elif isinstance(exc,TimeoutError):state,code,message,retry="failed","PROCESSING_TIMEOUT","资料处理超时",True
            elif isinstance(exc,OCRUnavailable):state,code,message,retry="failed","OCR_UNAVAILABLE","OCR模型未就绪",False
            elif isinstance(exc,DocumentLimitError):
                detail=str(exc)
                code="OCR_PAGE_LIMIT" if "PAGES" in detail else "IMAGE_PIXEL_LIMIT" if "PIXEL" in detail or "decompression" in detail else "OCR_REGION_LIMIT"
                state,message,retry="failed","资料超过"+code+"限制",False
            else:state,code,message,retry="failed","INDEXING_FAILED" if phase=="indexing" else "INVALID_DOCUMENT","索引失败，可重试" if phase=="indexing" else "文档解析失败或可用文字不足",phase=="indexing"
            with self.repo.connect() as db:
                current=json.loads(db.execute("SELECT payload FROM ingestions WHERE id=?",(job_id,)).fetchone()[0])
                if current["state"]!="ready":
                    current.update(state=state,error={"code":code,"message":message,"retryable":retry,"error_type":type(exc).__name__})
                    self._save(db,current)
            self._release_focus(value)
            if state=="cancelled":Path(value["path"]).unlink(missing_ok=True)
        finally:
            if candidate and not published:
                self.repo.store.delete_source(candidate)
                if self.flow.vector_index is not None:
                    try:self.flow.vector_index.delete_document(candidate)
                    except Exception:
                        # 未发布候选没有规范chunk可解析；保留补偿失败状态用于后续清理。
                        with self.repo.connect() as db:
                            current=json.loads(db.execute("SELECT payload FROM ingestions WHERE id=?",(job_id,)).fetchone()[0])
                            current["cleanup_pending"]=True;current["candidate_document_id"]=candidate;self._save(db,current)

    def close(self):
        """协作停止，避免重启时未完成任务被错误标成功。"""
        self.stop.set()
        self.pool.shutdown(wait=False,cancel_futures=True)

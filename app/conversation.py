"""会话持久化与资料范围：独立连接、范围快照、幂等轮次和附件焦点。"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import sqlite3
import time
import uuid

class ConversationError(ValueError):
    """向接口返回明确业务错误，区别于无证据拒答。"""
    def __init__(self, code, message, status=409):
        super().__init__(message)
        self.code, self.status = code, status

class ConversationService:
    """只管理本项目 SQLite 会话数据，不执行解析、检索或模型调用。"""
    def __init__(self, store):
        self.store, self.path = store, store.db_path
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS conversations(id TEXT PRIMARY KEY,payload TEXT NOT NULL,updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS conversation_turns(
                id TEXT PRIMARY KEY,session_id TEXT NOT NULL,request_id TEXT NOT NULL,
                question TEXT NOT NULL,status TEXT NOT NULL,payload TEXT NOT NULL,created REAL NOT NULL,
                UNIQUE(session_id,request_id));
            CREATE TABLE IF NOT EXISTS conversation_memory(
                session_id TEXT PRIMARY KEY,payload TEXT NOT NULL,revision INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS ingestions(
                id TEXT PRIMARY KEY,session_id TEXT NOT NULL,request_id TEXT NOT NULL,
                payload TEXT NOT NULL,updated REAL NOT NULL,UNIQUE(session_id,request_id));
            CREATE TABLE IF NOT EXISTS overview_cache(cache_key TEXT PRIMARY KEY,payload TEXT NOT NULL);
            -- 比较任务独立于滚动摘要；已回答任务仍可按会话回查，回填标记避免重复扫描。
            CREATE TABLE IF NOT EXISTS comparison_tasks(
                task_id TEXT PRIMARY KEY,session_id TEXT NOT NULL,origin_turn_id TEXT NOT NULL,
                document_ids TEXT NOT NULL,document_versions TEXT NOT NULL,filenames TEXT NOT NULL,
                original_question TEXT NOT NULL,topic TEXT NOT NULL,mode TEXT NOT NULL,
                status TEXT NOT NULL,created_at REAL NOT NULL,updated_at REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_comparison_tasks_session_updated
                ON comparison_tasks(session_id,updated_at DESC);
            CREATE TABLE IF NOT EXISTS comparison_task_backfill(
                session_id TEXT PRIMARY KEY,migrated_at REAL NOT NULL);
            -- 轻量索引只服务历史列表；原问题、回答、记忆和比较任务仍按 session_id 保存在原表。
            CREATE TABLE IF NOT EXISTS conversation_index(
                session_id TEXT PRIMARY KEY,title TEXT NOT NULL,title_source TEXT NOT NULL,
                created_at REAL NOT NULL,last_turn_at REAL NOT NULL,archived_at REAL);
            CREATE INDEX IF NOT EXISTS idx_conversation_index_active
                ON conversation_index(archived_at,last_turn_at DESC,session_id DESC);
            """)
            self._backfill_index(db)

    @staticmethod
    def _title(question):
        """从首个问题生成可读标题，不调用模型，也不改写原问题。"""
        compact = " ".join(str(question).split())
        return compact[:36] + ("…" if len(compact) > 36 else "") if compact else "新会话"

    def _backfill_index(self, db):
        """旧会话只补列表元数据；已有索引不覆盖用户改过的标题和归档状态。"""
        rows = db.execute("SELECT id,updated FROM conversations WHERE id NOT IN "
                          "(SELECT session_id FROM conversation_index)").fetchall()
        for row in rows:
            first = db.execute("SELECT question,created FROM conversation_turns WHERE session_id=? "
                               "ORDER BY created,id LIMIT 1", (row["id"],)).fetchone()
            last = db.execute("SELECT MAX(created) FROM conversation_turns WHERE session_id=?", (row["id"],)).fetchone()[0]
            db.execute("INSERT INTO conversation_index VALUES(?,?,?,?,?,NULL)",
                       (row["id"], self._title(first["question"]) if first else "新会话",
                        "automatic", first["created"] if first else row["updated"],
                        last if last is not None else row["updated"]))

    @staticmethod
    def _cursor(value):
        """用时间与 ID 组成稳定游标，避免同秒记录翻页重复。"""
        return f"{value['created']}:{value['id']}"

    @staticmethod
    def _parse_cursor(cursor):
        try:
            stamp, identifier = cursor.split(":", 1)
            number = float(stamp)
            if not identifier or not 0 <= number < 1e20:
                raise ValueError("cursor value")
            return number, identifier
        except (ValueError, AttributeError):
            raise ConversationError("INVALID_CURSOR", "分页位置无效，请重新打开列表", 422)

    def list_history(self, archived=False, query="", cursor=None, limit=30):
        """只读标题、时间及范围计数；列表响应绝不携带长回答或原文。"""
        if not 1 <= limit <= 100:
            raise ConversationError("INVALID_LIMIT", "每页数量须在 1 到 100 之间", 422)
        where = ["i.archived_at IS NOT NULL" if archived else "i.archived_at IS NULL"]
        args = []
        if query.strip():
            where.append("instr(lower(i.title),lower(?))>0")
            args.append(query.strip()[:100])
        if cursor:
            stamp, identifier = self._parse_cursor(cursor)
            where.append("(i.last_turn_at<? OR (i.last_turn_at=? AND i.session_id<?))")
            args.extend((stamp, stamp, identifier))
        sql = ("SELECT i.*,c.payload FROM conversation_index i JOIN conversations c ON c.id=i.session_id "
               "WHERE " + " AND ".join(where) + " ORDER BY i.last_turn_at DESC,i.session_id DESC LIMIT ?")
        with self.connect() as db:
            rows = db.execute(sql, (*args, limit + 1)).fetchall()
        more = len(rows) > limit
        items = []
        for row in rows[:limit]:
            value = json.loads(row["payload"])
            items.append(dict(session_id=row["session_id"], title=row["title"],
                              created_at=row["created_at"], last_turn_at=row["last_turn_at"],
                              archived_at=row["archived_at"], answer_policy=value.get("answer_policy"),
                              scope_mode=value.get("scope_mode"),
                              selected_document_count=len(value.get("document_ids", [])),
                              associated_document_count=len(value.get("associated_document_ids",
                                                                     value.get("document_ids", [])))))
        return {"items": items, "next_cursor": f"{rows[limit-1]['last_turn_at']}:{rows[limit-1]['session_id']}" if more else None}

    def metadata(self, session_id, *, title=None, archived=None):
        """重命名及归档可恢复；归档时拒绝中断正在生成的轮次。"""
        if title is not None:
            title = " ".join(title.split()).strip()
            if not title or len(title) > 80:
                raise ConversationError("INVALID_TITLE", "标题须为 1 到 80 个字符", 422)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM conversation_index WHERE session_id=?", (session_id,)).fetchone()
            if not row:
                raise ConversationError("CONVERSATION_NOT_FOUND", "会话不存在", 404)
            if archived and db.execute("SELECT 1 FROM conversation_turns WHERE session_id=? AND status='processing'",
                                       (session_id,)).fetchone():
                raise ConversationError("TURN_IN_PROGRESS", "回答生成中，完成后再归档")
            if archived:
                source = db.execute("SELECT payload FROM conversations WHERE id=?", (session_id,)).fetchone()
                if source and json.loads(source[0]).get("pending_focus"):
                    raise ConversationError("UPLOAD_IN_PROGRESS", "资料上传处理中，完成后再归档")
                # 上传受理与批次登记之间存在短窗口；有后台任务时同样不能归档。
                jobs = db.execute("SELECT payload FROM ingestions WHERE session_id=?", (session_id,)).fetchall()
                if any(json.loads(job[0]).get("state") not in
                       {"ready", "failed", "cancelled", "interrupted"} for job in jobs):
                    raise ConversationError("UPLOAD_IN_PROGRESS", "资料上传处理中，完成后再归档")
            if title is not None:
                db.execute("UPDATE conversation_index SET title=?,title_source='manual' WHERE session_id=?",
                           (title, session_id))
            if archived is not None:
                db.execute("UPDATE conversation_index SET archived_at=? WHERE session_id=?",
                           (time.time() if archived else None, session_id))
        return self.index_entry(session_id)

    def index_entry(self, session_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM conversation_index WHERE session_id=?", (session_id,)).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _require_writable(db, session_id):
        """服务端同样约束归档只读，避免旧客户端绕过页面继续改写会话。"""
        row = db.execute("SELECT archived_at FROM conversation_index WHERE session_id=?",
                         (session_id,)).fetchone()
        if row and row[0] is not None:
            raise ConversationError("CONVERSATION_ARCHIVED", "请先恢复已归档会话，再继续操作")

    @contextmanager
    def connect(self):
        """每次事务独占连接，避免旧共享连接的 commit 意外发布其他请求。"""
        db = sqlite3.connect(self.path, timeout=20)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, session_id=None, legacy=False, answer_policy="document_strict"):
        session_id = session_id or str(uuid.uuid4())
        value = dict(id=session_id, document_ids=[], associated_document_ids=[], scope_mode="all" if legacy else "attached",
                     active_document_id=None, revision=0, focus_batch=None, pending_focus=False, answer_policy=answer_policy, policy_revision=0, dialogue_state={})
        with self.connect() as db:
            now = time.time()
            db.execute("INSERT OR IGNORE INTO conversations VALUES(?,?,?)",
                       (session_id, json.dumps(value, ensure_ascii=False), now))
            db.execute("INSERT OR IGNORE INTO conversation_index VALUES(?,?,?,?,?,NULL)",
                       (session_id,"新会话","automatic",now,now))
        return self.get(session_id)

    def set_answer_policy(self, session_id, policy):
        """策略只影响后续轮次，单独版本化；生成期间不更换策略。"""
        if policy not in {"document_strict","document_enhanced"}:
            raise ConversationError("INVALID_POLICY","未知回答策略",422)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._require_writable(db, session_id)
            row=db.execute("SELECT payload FROM conversations WHERE id=?",(session_id,)).fetchone()
            if not row:
                raise ConversationError("CONVERSATION_NOT_FOUND","会话不存在",404)
            if db.execute("SELECT 1 FROM conversation_turns WHERE session_id=? AND status='processing'",(session_id,)).fetchone():
                raise ConversationError("TURN_IN_PROGRESS","回答生成中，请完成后切换模式")
            value=json.loads(row[0])
            if value.get("answer_policy","document_strict")!=policy:
                value["answer_policy"]=policy
                value["policy_revision"]=value.get("policy_revision",0)+1
                self.save(db,value)
        return self.get(session_id)

    def get(self, session_id):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM conversations WHERE id=?", (session_id,)).fetchone()
        return json.loads(row[0]) if row else None

    @staticmethod
    def save(db, value):
        db.execute("UPDATE conversations SET payload=?,updated=? WHERE id=?",
                   (json.dumps(value, ensure_ascii=False), time.time(), value["id"]))

    def documents(self, ids=None, db=None):
        """返回就绪资料；范围为空时保持为空，绝不回退到全库。"""
        if ids == [] or ids == ():
            return []
        if db is None:
            with self.connect() as connection:
                return self.documents(ids, connection)
        clause, args = "", ()
        if ids is not None:
            clause = " AND id IN (" + ",".join("?" for _ in ids) + ")"
            args = tuple(ids)
        return [dict(r) for r in db.execute(
            "SELECT id,filename,status,chunk_count,content_hash,indexed_at FROM documents "
            "WHERE status='indexed' AND chunk_count>0" + clause + " ORDER BY created_at", args)]

    def attach(self, session_id, ids, active=None):
        """显式追加已有资料；单份关联成为焦点，多份关联要求澄清。"""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._require_writable(db, session_id)
            row = db.execute("SELECT payload FROM conversations WHERE id=?", (session_id,)).fetchone()
            if not row:
                raise ConversationError("CONVERSATION_NOT_FOUND", "会话不存在", 404)
            value = json.loads(row[0])
            if len(self.documents(ids, db)) != len(set(ids)):
                raise ConversationError("DOCUMENT_NOT_READY", "资料不存在或尚未就绪", 404)
            self.attach_in_transaction(db, value, ids, active=active)
        return self.get(session_id)

    def attach_in_transaction(self, db, value, ids, active=None, batch=None, batch_size=1):
        """供摄取任务原子发布使用；旧批次晚完成不覆盖新焦点。"""
        value["associated_document_ids"] = list(dict.fromkeys(value.get("associated_document_ids",value["document_ids"])+list(ids)))
        # 手动选择后旧任务晚完成只追加关联，不偷偷扩展明确选中的范围。
        if batch is None or value.get("focus_batch") == batch or value["scope_mode"] != "selected":
            value["document_ids"] = list(dict.fromkeys(value["document_ids"] + list(ids)))
        if batch is None or value.get("focus_batch") == batch:
            value["active_document_id"] = (active or ids[0]) if batch_size == 1 and len(ids) == 1 else None
            value["pending_focus"] = False
        value["revision"] += 1
        self.save(db, value)

    def register_batch(self, session_id, batch, size):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._require_writable(db, session_id)
            value = json.loads(db.execute("SELECT payload FROM conversations WHERE id=?", (session_id,)).fetchone()[0])
            if value.get("focus_batch") != batch:
                value.update(focus_batch=batch, pending_focus=True, focus_batch_size=size)
                if size > 1:
                    value["active_document_id"] = None
                value["revision"] += 1
                self.save(db, value)

    def scope(self, session_id, mode, ids, active, revision=None):
        """显式范围修改使用乐观版本，避免慢请求覆盖新的用户选择。"""
        if mode not in {"attached", "selected", "all"}:
            raise ConversationError("INVALID_SCOPE", "资料范围模式无效", 422)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._require_writable(db, session_id)
            row = db.execute("SELECT payload FROM conversations WHERE id=?", (session_id,)).fetchone()
            if not row:
                raise ConversationError("CONVERSATION_NOT_FOUND", "会话不存在", 404)
            value = json.loads(row[0])
            if revision is not None and revision != value["revision"]:
                raise ConversationError("SCOPE_CHANGED", "资料范围已变化，请刷新后重试")
            if len(self.documents(ids, db)) != len(set(ids)):
                raise ConversationError("DOCUMENT_NOT_READY", "选择的资料尚未就绪", 422)
            if active and active not in ids and mode != "all":
                raise ConversationError("INVALID_FOCUS", "活动文档必须位于当前范围", 422)
            value["associated_document_ids"] = list(dict.fromkeys(value.get("associated_document_ids",value["document_ids"])+ids))
            value.update(scope_mode=mode, document_ids=list(dict.fromkeys(ids)),
                         active_document_id=active, pending_focus=False,
                         focus_batch="manual-" + str(uuid.uuid4()), focus_batch_size=len(ids) if not active else 1, revision=value["revision"]+1)
            self.save(db, value)
        return self.get(session_id)

    def detach(self, session_id, document_id):
        """只移除会话关联，不删除知识库文档；关联集合与启用集合分别维护。"""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._require_writable(db, session_id)
            row=db.execute("SELECT payload FROM conversations WHERE id=?",(session_id,)).fetchone()
            if not row:
                raise ConversationError("CONVERSATION_NOT_FOUND","会话不存在",404)
            value=json.loads(row[0])
            value["associated_document_ids"]=[i for i in value.get("associated_document_ids",value["document_ids"]) if i!=document_id]
            value["document_ids"]=[i for i in value["document_ids"] if i!=document_id]
            if value["active_document_id"]==document_id:value["active_document_id"]=None
            value["revision"]+=1
            self.save(db,value)
        return self.get(session_id)

    def snapshot(self, session_id, explicit=None, revision=None):
        value = self.get(session_id) or self.create(session_id, legacy=True)
        if revision is not None and value["revision"] != revision:
            raise ConversationError("SCOPE_CHANGED", "资料范围已变化，请刷新后重试")
        ids = explicit if explicit is not None else (None if value["scope_mode"]=="all" else value["document_ids"])
        docs = self.documents(ids)
        if explicit is not None and len(docs) != len(set(explicit)):
            raise ConversationError("DOCUMENT_NOT_READY", "指定资料已删除或未就绪", 404)
        scope_ids = [d["id"] for d in docs]
        return dict(document_ids=scope_ids, versions=self.versions(scope_ids), revision=value["revision"],
                    active_document_id=value["active_document_id"] if value["active_document_id"] in scope_ids else None,
                    pending_focus=value.get("pending_focus", False), scope_mode=value["scope_mode"],
                    unavailable_focus=bool(value.get("active_document_id")) and value["active_document_id"] not in scope_ids,
                    ambiguous_batch=value.get("focus_batch_size",1)>1 and value.get("active_document_id") is None)

    def versions(self, ids):
        """原文指纹和规范chunk ID共同构成版本，重索引也会使旧缓存失效。"""
        result = {}
        with self.connect() as db:
            for doc in self.documents(ids, db):
                chunks = [r[0] for r in db.execute("SELECT id FROM chunks WHERE document_id=? ORDER BY position,id", (doc["id"],))]
                value = [doc["content_hash"], doc["indexed_at"], chunks]
                result[doc["id"]] = hashlib.sha256(json.dumps(value).encode()).hexdigest()
        return result

    def validate(self, snapshot):
        if self.versions(snapshot["document_ids"]) != snapshot["versions"]:
            raise ConversationError("SOURCE_CHANGED", "资料已更新或删除，请重新提问")

    def begin_turn(self, session_id, request_id, question, request_context=None):
        """同会话串行生成；相同幂等键返回原结果，避免重复消息和重复模型费用。"""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            index = db.execute("SELECT title_source,title FROM conversation_index WHERE session_id=?",
                               (session_id,)).fetchone()
            if index is None:
                raise ConversationError("CONVERSATION_NOT_FOUND", "会话不存在", 404)
            if db.execute("SELECT archived_at FROM conversation_index WHERE session_id=?",
                          (session_id,)).fetchone()[0] is not None:
                raise ConversationError("CONVERSATION_ARCHIVED", "请先恢复已归档会话，再继续提问")
            row = db.execute("SELECT * FROM conversation_turns WHERE session_id=? AND request_id=?",
                             (session_id, request_id)).fetchone()
            if row:
                saved=json.loads(row["payload"])
                saved_policy=saved.get("request_context",{}).get("answer_policy",
                    saved.get("response",{}).get("answer_policy","document_strict"))
                if request_context and saved_policy!=request_context["answer_policy"]:
                    raise ConversationError("IDEMPOTENCY_CONFLICT","相同请求标识对应不同回答策略")
                if row["question"] != question:
                    raise ConversationError("IDEMPOTENCY_CONFLICT", "相同请求标识对应了不同问题")
                if row["status"] == "completed":
                    return row["id"], json.loads(row["payload"])["response"]
                raise ConversationError("TURN_IN_PROGRESS", "该请求正在处理或已中断，请查看运行记录")
            busy = db.execute("SELECT 1 FROM conversation_turns WHERE session_id=? AND status='processing'", (session_id,)).fetchone()
            if busy:
                raise ConversationError("TURN_IN_PROGRESS", "该会话仍有请求正在处理")
            turn_id = str(uuid.uuid4())
            now = time.time()
            db.execute("INSERT INTO conversation_turns VALUES(?,?,?,?,?,?,?)",
                       (turn_id, session_id, request_id, question, "processing",
                        json.dumps({"request_context":request_context} if request_context else {}), now))
            if index["title_source"] == "automatic" and index["title"] == "新会话":
                db.execute("UPDATE conversation_index SET title=?,last_turn_at=? WHERE session_id=?",
                           (self._title(question), now, session_id))
            else:
                db.execute("UPDATE conversation_index SET last_turn_at=? WHERE session_id=?", (now, session_id))
        return turn_id, None

    def finish_turn(self, turn_id, payload, status="completed"):
        with self.connect() as db:
            db.execute("UPDATE conversation_turns SET status=?,payload=? WHERE id=?",
                       (status, json.dumps(payload, ensure_ascii=False), turn_id))

    def turns(self, session_id, limit=100, include_failed=False):
        """读取完整轮次；界面额外恢复失败尝试，记忆仅使用已完成轮次。"""
        statuses = "('completed','failed','interrupted')" if include_failed else "('completed')"
        with self.connect() as db:
            rows = db.execute("SELECT * FROM conversation_turns WHERE session_id=? AND status IN " + statuses +
                              " ORDER BY created DESC LIMIT ?", (session_id, limit)).fetchall()
        result=[]
        for row in reversed(rows):
            item={**dict(row), **json.loads(row["payload"])}
            # 早期轮次可能只有概述正文；读取时补建临时条目索引，绝不回写或改动旧答案。
            if (row["status"]=="completed" and item.get("response",{}).get("intent")=="overview"
                    and not item.get("answer_units")):
                from app.rag.memory import answer_units
                item["answer_units"]=answer_units(item["response"].get("answer",""),
                                                   item["response"].get("citations",[]))
            if row["status"]!="completed":
                item["response"]={"answer":"该请求失败或中断，可查看运行记录后重试。","answer_status":"error",
                    "run_id":item.get("run_id"),"citations":[],"tool_calls":[]}
            result.append(item)
        return result

    def page_turns(self, session_id, before=None, limit=50):
        """按 created 与 id 倒序翻页，再按对话顺序返回；失败轮次也保持可审计。"""
        if not self.get(session_id):
            raise ConversationError("CONVERSATION_NOT_FOUND", "会话不存在", 404)
        if not 1 <= limit <= 100:
            raise ConversationError("INVALID_LIMIT", "每页数量须在 1 到 100 之间", 422)
        clause, args = "", [session_id]
        if before:
            stamp, identifier = self._parse_cursor(before)
            clause = " AND (created<? OR (created=? AND id<?))"
            args.extend((stamp, stamp, identifier))
        with self.connect() as db:
            rows = db.execute("SELECT * FROM conversation_turns WHERE session_id=? "
                              "AND status IN ('completed','failed','interrupted')" + clause +
                              " ORDER BY created DESC,id DESC LIMIT ?", (*args, limit + 1)).fetchall()
        more = len(rows) > limit
        selected = rows[:limit]
        items = self._decode_turn_rows(reversed(selected))
        return {"items": items,
                "next_cursor": self._cursor(selected[-1]) if more else None}

    @staticmethod
    def _decode_turn_rows(rows):
        """分页和旧读取接口共用失败展示与旧概述兼容规则。"""
        result = []
        for row in rows:
            item={**dict(row), **json.loads(row["payload"])}
            if (row["status"]=="completed" and item.get("response",{}).get("intent")=="overview"
                    and not item.get("answer_units")):
                from app.rag.memory import answer_units
                item["answer_units"]=answer_units(item["response"].get("answer",""),
                                                   item["response"].get("citations",[]))
            if row["status"]!="completed":
                item["response"]={"answer":"该请求失败或中断，可查看运行记录后重试。","answer_status":"error",
                    "run_id":item.get("run_id"),"citations":[],"tool_calls":[]}
            result.append(item)
        return result

    def recover(self):
        """只在应用启动调用；中断状态可见，不冒充成功或无限自动续跑。"""
        with self.connect() as db:
            db.execute("UPDATE conversation_turns SET status='interrupted' WHERE status='processing'")

    def view(self, session_id, turn_limit=100):
        value = self.get(session_id)
        if not value:
            raise ConversationError("CONVERSATION_NOT_FOUND", "会话不存在", 404)
        page = self.page_turns(session_id, limit=turn_limit)
        return {**value, "documents":self.documents(value.get("associated_document_ids",value["document_ids"])),
                "turns":page["items"], "next_turn_cursor":page["next_cursor"],
                "history_meta":self.index_entry(session_id)}

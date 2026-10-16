"""会话内比较任务索引：原始轮次保留全文，索引只保存定位所需的来源元数据。"""

import json
import re
import time
import uuid

from app.rag.comparison_task import PAIR, TOPIC


class ComparisonTaskRepository:
    """与回答轮次共用 SQLite 事务；列表只返回当前会话的任务。"""

    def __init__(self, conversations):
        self.conversations = conversations

    def save(self, db, session_id, origin_turn_id, task, status, *, created_at=None):
        """沿用 task_id 更新追问状态；首次创建的原问题和起始轮次保持不变。"""
        if status not in {"answered", "partial", "insufficient", "retryable"}:
            raise ValueError("unknown comparison task status")
        ids = task["document_ids"]
        if len(ids) != 2 or len(set(ids)) != 2:
            raise ValueError("comparison task requires two documents")
        now = time.time()
        task_id = task["task_id"]
        db.execute("""
            INSERT INTO comparison_tasks(
                task_id,session_id,origin_turn_id,document_ids,document_versions,filenames,
                original_question,topic,mode,status,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(task_id) DO UPDATE SET
                document_ids=excluded.document_ids,
                document_versions=excluded.document_versions,
                filenames=excluded.filenames,
                topic=excluded.topic,mode=excluded.mode,status=excluded.status,
                updated_at=excluded.updated_at
            WHERE comparison_tasks.session_id=excluded.session_id
        """, (task_id, session_id, origin_turn_id, json.dumps(ids),
              json.dumps(task["versions"]), json.dumps(task["filenames"], ensure_ascii=False),
              task["original_question"], task["topic"], task["mode"], status,
              created_at or now, now))

    def _backfill(self, db, session_id):
        """旧回答不改写；只为可确认两份来源和版本的已完成比较轮次建索引。"""
        if db.execute("SELECT 1 FROM comparison_task_backfill WHERE session_id=?", (session_id,)).fetchone():
            return
        rows = db.execute("SELECT id,question,payload,created FROM conversation_turns "
                          "WHERE session_id=? AND status='completed' ORDER BY created", (session_id,)).fetchall()
        for row in rows:
            payload = json.loads(row["payload"])
            resolution = payload.get("resolution") or {}
            response = payload.get("response") or {}
            old = resolution.get("comparison_task") or {}
            ids = old.get("document_ids") or resolution.get("document_ids") or []
            if len(ids) != 2 or len(set(ids)) != 2:
                continue
            snapshot = payload.get("snapshot") or response.get("scope_snapshot") or {}
            versions = old.get("versions") or {doc: snapshot.get("versions", {}).get(doc) for doc in ids}
            if not all(versions.get(doc) for doc in ids):
                continue
            if not old:
                # 旧普通比较只有双方都实际被引用且问题明确指向比较时才能恢复对象。
                cited = {c.get("document_id") for c in response.get("citations", [])}
                if response.get("intent") != "comparison" or not set(ids) <= cited or not (
                        PAIR.search(row["question"]) or TOPIC.search(row["question"])):
                    continue
            current = {doc["id"]: doc["filename"] for doc in self.conversations.documents(ids, db)}
            filenames = old.get("filenames") or current
            if not all(filenames.get(doc) for doc in ids):
                continue
            task = dict(task_id=old.get("task_id") or str(uuid.uuid5(
                uuid.NAMESPACE_URL, session_id + ":" + row["id"])), document_ids=ids,
                versions={doc: versions[doc] for doc in ids},
                filenames={doc: filenames[doc] for doc in ids},
                original_question=old.get("original_question") or row["question"],
                topic=old.get("topic") or "研究侧重点",
                mode=old.get("mode") or ("compare" if re.search(r"区别|差异|比较|异同", row["question"])
                                         else "list"))
            state = response.get("answer_status")
            status = {"answered": "answered", "partial_answer": "partial",
                      "insufficient_evidence": "insufficient"}.get(state)
            if status:
                self.save(db, session_id, row["id"], task, status, created_at=row["created"])
        db.execute("INSERT INTO comparison_task_backfill VALUES(?,?)", (session_id, time.time()))

    def list(self, session_id, limit=2000):
        """首次访问时迁移旧轮次；任务列表不会被滚动摘要裁剪。"""
        with self.conversations.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._backfill(db, session_id)
            rows = db.execute("SELECT * FROM comparison_tasks WHERE session_id=? "
                              "ORDER BY updated_at DESC,created_at DESC LIMIT ?", (session_id, limit)).fetchall()
        return [dict(task_id=row["task_id"], origin_turn_id=row["origin_turn_id"],
                     document_ids=json.loads(row["document_ids"]),
                     versions=json.loads(row["document_versions"]),
                     filenames=json.loads(row["filenames"]),
                     original_question=row["original_question"], topic=row["topic"],
                     mode=row["mode"], status=row["status"],
                     created_at=row["created_at"], updated_at=row["updated_at"]) for row in rows]

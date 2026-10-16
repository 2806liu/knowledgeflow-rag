"""概述断点存储：批次成功后立即落盘，旧对话轮次不被改写。"""

import json
import time


class OverviewRepository:
    def __init__(self, conversations):
        self.conversations = conversations
        with conversations.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS overview_artifacts(
                cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL, updated REAL NOT NULL)""")

    def load(self, key):
        with self.conversations.connect() as db:
            row = db.execute("SELECT payload FROM overview_artifacts WHERE cache_key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def checkpoint(self, key, base, batch_id, value):
        """同一批次按 ID 合并写入，重试不会抹去其他已完成批次。"""
        with self.conversations.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM overview_artifacts WHERE cache_key=?", (key,)).fetchone()
            artifact = json.loads(row[0]) if row else dict(base)
            artifact.setdefault("maps", {})[batch_id] = value
            db.execute("INSERT OR REPLACE INTO overview_artifacts VALUES(?,?,?)",
                       (key, json.dumps(artifact, ensure_ascii=False), time.time()))
        return artifact

    def publish(self, key, base, result, coverage, units):
        """每次部分或完整结果形成新修订；历史回答仍使用其原有快照。"""
        with self.conversations.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM overview_artifacts WHERE cache_key=?", (key,)).fetchone()
            artifact = json.loads(row[0]) if row else dict(base)
            artifact["revision"] = artifact.get("revision", 0) + 1
            artifact["published"] = {"result": result, "coverage": coverage, "units": units}
            db.execute("INSERT OR REPLACE INTO overview_artifacts VALUES(?,?,?)",
                       (key, json.dumps(artifact, ensure_ascii=False), time.time()))
        return artifact["revision"]

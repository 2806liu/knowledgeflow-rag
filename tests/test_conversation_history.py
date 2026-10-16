"""历史会话验收：使用隔离 SQLite 与公开合成轮次，不调用真实模型或资料。"""
import json
import uuid

from app.conversation import ConversationService
from conversation_test_support import APIBase


class ConversationHistoryTests(APIBase):
    def _turn(self, session, question):
        """直接保存合成问答，检验历史索引而不引入检索链路。"""
        repo = self.api.flow.conversations
        turn_id, previous = repo.begin_turn(session, str(uuid.uuid4()), question)
        self.assertIsNone(previous)
        repo.finish_turn(turn_id, {"response": {"answer": "合成回答：" + question,
                                          "answer_status": "answered", "citations": []}})
        return turn_id

    def test_auto_save_search_rename_archive_and_restore(self):
        """每个会话自动入索引；归档可读可恢复，原始轮次不丢失。"""
        second = self.client.post("/conversations").json()["id"]
        first_turn = self._turn(self.session, "第一篇论文有哪些结论？")
        self._turn(second, "考勤制度有什么规定？")
        rows = self.client.get("/conversations").json()["items"]
        self.assertEqual({item["session_id"] for item in rows}, {self.session, second})
        title = next(item["title"] for item in rows if item["session_id"] == self.session)
        self.assertEqual(title, "第一篇论文有哪些结论？")
        result = self.client.get("/conversations", params={"q": "论文"}).json()["items"]
        self.assertEqual([item["session_id"] for item in result], [self.session])

        updated = self.client.patch("/conversations/" + self.session,
                                    json={"title": "Bruno 与 CFT 比较"}).json()
        self.assertEqual(updated["title"], "Bruno 与 CFT 比较")
        archived = self.client.patch("/conversations/" + self.session,
                                     json={"archived": True}).json()
        self.assertIsNotNone(archived["archived_at"])
        self.assertNotIn(self.session, [item["session_id"] for item in
                                        self.client.get("/conversations").json()["items"]])
        self.assertIn(self.session, [item["session_id"] for item in
                                      self.client.get("/conversations?archived=true").json()["items"]])
        self.assertEqual(self.client.get("/conversations/" + self.session).json()["turns"][0]["id"], first_turn)
        blocked = self.client.post("/chat", json={"session_id": self.session, "message": "继续"})
        self.assertEqual(blocked.status_code, 409)
        self.assertEqual(blocked.json()["detail"]["code"], "CONVERSATION_ARCHIVED")
        self.client.patch("/conversations/" + self.session, json={"archived": False})
        self._turn(self.session, "恢复后继续")
        self.assertEqual(len(self.client.get("/conversations/" + self.session).json()["turns"]), 2)

    def test_history_and_turn_cursors_keep_all_fifty_five_turns(self):
        """列表和长会话按稳定游标翻页，前 50 轮与旧 5 轮没有重复。"""
        sessions = [self.session] + [self.client.post("/conversations").json()["id"] for _ in range(3)]
        for number, session in enumerate(sessions):
            self._turn(session, "会话 " + str(number))
        seen = []
        cursor = None
        while True:
            page = self.client.get("/conversations", params={"limit": 2, **({"cursor": cursor} if cursor else {})})
            self.assertEqual(page.status_code, 200, page.text)
            data = page.json()
            seen.extend(item["session_id"] for item in data["items"])
            cursor = data["next_cursor"]
            if not cursor:
                break
        self.assertEqual(set(seen), set(sessions))
        self.assertEqual(len(seen), len(sessions))

        expected = [self._turn(self.session, "问题 " + str(number)) for number in range(55)]
        first = self.client.get("/conversations/" + self.session, params={"turn_limit": 50}).json()
        self.assertEqual(len(first["turns"]), 50)
        self.assertIsNotNone(first["next_turn_cursor"])
        older = self.client.get("/conversations/" + self.session + "/turns",
                                params={"before": first["next_turn_cursor"], "limit": 50}).json()
        combined = older["items"] + first["turns"]
        self.assertEqual([item["id"] for item in combined], [combined[0]["id"]] + expected)
        self.assertIsNone(older["next_cursor"])

    def test_existing_sessions_backfill_without_changing_turns(self):
        """旧会话首次建索引只补元数据，再次启动不覆盖手工标题。"""
        self._turn(self.session, "旧会话原问题")
        repo = self.api.flow.conversations
        with repo.connect() as db:
            old = db.execute("SELECT payload FROM conversation_turns WHERE session_id=?", (self.session,)).fetchone()[0]
            db.execute("DELETE FROM conversation_index WHERE session_id=?", (self.session,))
        migrated = ConversationService(self.api.store)
        self.assertEqual(migrated.index_entry(self.session)["title"], "旧会话原问题")
        migrated.metadata(self.session, title="保留手工标题")
        again = ConversationService(self.api.store)
        self.assertEqual(again.index_entry(self.session)["title"], "保留手工标题")
        with again.connect() as db:
            current = db.execute("SELECT payload FROM conversation_turns WHERE session_id=?", (self.session,)).fetchone()[0]
        self.assertEqual(json.loads(current), json.loads(old))

    def test_archived_session_rejects_scope_and_upload(self):
        """旧客户端即使绕开禁用按钮，也不能继续改写归档会话。"""
        self.client.patch("/conversations/" + self.session, json={"archived": True})
        scope = self.client.patch("/conversations/" + self.session + "/scope",
                                  json={"mode": "selected", "document_ids": []})
        self.assertEqual(scope.status_code, 409)
        self.assertEqual(scope.json()["detail"]["code"], "CONVERSATION_ARCHIVED")
        upload = self.create_job()
        self.assertEqual(upload.status_code, 409)
        self.assertEqual(upload.json()["detail"]["code"], "CONVERSATION_ARCHIVED")


if __name__ == "__main__":
    import unittest
    unittest.main()

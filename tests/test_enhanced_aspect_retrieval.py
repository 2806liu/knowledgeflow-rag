"""公开七块样本回归：按方面召回、无新增证据停止和末轮格式修复。"""

import json
import uuid

from enhanced_test_support import EnhancedBase, answer, resolution, section


CHUNKS = [
    "示范实验室考勤制度说明。考勤制度用于记录成员到场，打卡时间与打卡次数要求另见操作章节。",
    "工作时间：上午八点半至十一点半，下午两点半至五点半。",
    "打卡方法：每日四次；早间、午间、晚间各一次，离开时再打卡一次。",
    "补充规定：考勤制度中规定的打卡时间和打卡次数要求，需要按批准流程核查。",
    "补卡说明：迟到、请假和考勤记录须由负责人核对。",
    "设备维护：打卡机每周检查，并记录维护人员。",
    "归档规则：稀有归档期限为七年，记录须可追溯。",
]


class AspectRetrievalTests(EnhancedBase):
    def public_document(self):
        """直接写入独立测试库，固定七个 chunk 的边界与排序。"""
        document_id = str(uuid.uuid4())
        self.api.store.add_document(document_id, "公开合成考勤样本.md", "\n".join(CHUNKS))
        self.api.store.replace_document_chunks(document_id, [
            (str(uuid.uuid4()), document_id, value, index,
             {"section_id": f"public-{index}", "section_chunk_count": 1})
            for index, value in enumerate(CHUNKS, 1)])
        self.api.flow.conversations.attach(self.session, [document_id])
        return document_id

    def test_two_aspects_reach_answer_chunk_beyond_legacy_cap(self):
        """合并长问句的前两块不含答案，方面查询应补入第三块。"""
        doc = self.public_document()
        question = "查询示范实验室考勤制度中规定的打卡时间和打卡次数要求"
        old = self.api.flow.retriever.search(question, 4, (doc,))
        self.assertEqual(len(old), 2)
        self.assertNotIn(3, [hit.position for hit in old])
        aspects = [dict(id="a1", question="规定的打卡时间", requires_document=True),
                   dict(id="a2", question="规定的打卡次数", requires_document=True)]

        def grounded(messages):
            payload = json.loads(messages[-1]["content"])
            number = next(item["citation_number"] for item in payload["evidence"]
                          if "每日四次" in item["text"])
            return answer(section(f"打卡分三个时段，并在离开时补一次，共四次。[{number}]",
                                  "document", aspects=["a1", "a2"], numbers=[number]))

        self.script(resolution(question, intent="document_fact", docs=[doc], aspects=aspects), grounded)
        value = self.ask(question)
        self.assertEqual(value["answer_status"], "answered")
        self.assertTrue(any(c["position"] == 3 for c in value["citations"]))
        self.assertTrue(self.traces(value, "aspect_retrieval"))
        self.assertEqual(len(self.model.calls), 2)

    def test_supplemental_with_no_new_chunk_stops(self):
        """补检索只重取旧块时不再浪费一次模型调用。"""
        doc = self.public_document()
        more = dict(action="retrieve_more", sections=[], unresolved_aspect_ids=[],
                    supplemental_query="完全无关的不存在词")
        self.script(resolution("打卡规定", intent="document_fact", docs=[doc]), more)
        value = self.ask("打卡规定")
        self.assertEqual(value["answer_status"], "insufficient_evidence")
        self.assertEqual(len(self.model.calls), 2)
        trace = self.traces(value, "supplemental_retrieval")[0]
        self.assertEqual(trace["status"], "no_new_evidence")
        self.assertEqual(trace["new_chunk_count"], 0)

    def test_validation_repair_after_supplemental(self):
        """第三次生成即使自相矛盾，仍有一次独立格式修复并保留严格引用。"""
        doc = self.public_document()
        more = dict(action="retrieve_more", sections=[], unresolved_aspect_ids=[],
                    supplemental_query="稀有归档期限")

        def numbered(messages, unresolved=False):
            payload = json.loads(messages[-1]["content"])
            number = next(item["citation_number"] for item in payload["evidence"]
                          if "稀有归档期限为七年" in item["text"])
            return answer(section(f"稀有归档期限为七年。[{number}]",
                                  "document", numbers=[number]),
                          unresolved=["a1"] if unresolved else [])

        def repaired(messages):
            payload = json.loads(messages[-1]["content"])
            self.assertIn("answered aspect is also unresolved", payload["validation_feedback"])
            return numbered(messages)

        self.script(resolution("考勤制度", intent="document_fact", docs=[doc]),
                    more, lambda messages: numbered(messages, True), repaired)
        value = self.ask("考勤制度")
        self.assertEqual(value["answer_status"], "answered")
        self.assertEqual(len(self.model.calls), 4)
        self.assertEqual(self.traces(value, "supplemental_retrieval")[0]["status"], "expanded")
        self.assertTrue(self.traces(value, "answer_validation"))
        self.assertTrue(self.traces(value, "workflow")[0].get("latency_ms") is not None)

    def test_exhausted_repair_keeps_failed_run_diagnostics(self):
        """两次矛盾答案均被拒绝；失败运行仍记录校验阶段和完整耗时。"""
        doc = self.public_document()
        more = dict(action="retrieve_more", sections=[], unresolved_aspect_ids=[],
                    supplemental_query="稀有归档期限")
        bad = answer(section("考勤制度有说明。[1]", "document", numbers=[1]), unresolved=["a1"])
        self.script(resolution("考勤制度", intent="document_fact", docs=[doc]), more, bad, bad)
        response = self.post("考勤制度")
        self.assertEqual(response.status_code, 502)
        run_id = response.json()["detail"]["run_id"]
        events = self.client.get("/runs/" + run_id).json()["events"]
        self.assertEqual(len(self.model.calls), 4)
        self.assertEqual([e["status"] for e in events if e["node"] == "answer_validation"],
                         ["retrying", "error"])
        workflow = next(e for e in events if e["node"] == "workflow")
        self.assertIsInstance(workflow["latency_ms"], float)

"""LLM 边界回归：验证有限重试、错误脱敏、结构化拒答和真实 API 契约，不访问外网。"""

import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from fastapi.testclient import TestClient

from app.config import settings
from app.rag.contracts import EvidenceContext
from app.rag.llm_client import LLMCallError, LLMClient
from app.rag.solver import EvidenceSolver, INSUFFICIENT_EVIDENCE
from app.services import KnowledgeFlow
from app.store import Store


def config(**changes):
    """固定测试配置，防止开发者的 .env 把离线测试变成真实付费请求。"""
    base = dict(llm_mode="openai_compatible", llm_base_url="https://fixture.invalid/v1",
                llm_model="fixture-model", llm_api_key="fixture-secret", llm_max_retries=2,
                llm_retry_backoff_seconds=0, llm_timeout_seconds=1, llm_answer_format="text",
                llm_temperature=None, vector_enabled=False, reranker_enabled=False,
                retrieval_strategy="bm25_v1")
    base.update(changes)
    return replace(settings, **base)


def completion(content="已支持引用 [1]", **changes):
    """构造兼容接口响应；用量缺失与返回截断由具体测试覆盖。"""
    result = {"model": "fixture-model", "choices": [
        {"message": {"content": content}, "finish_reason": "stop"}]}
    result.update(changes)
    return result


class LLMClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_retry_then_success_and_usage(self):
        calls = []
        def reply(request):
            calls.append(json.loads(request.content))
            if len(calls) == 1:
                return httpx.Response(429, json={"error": "sensitive upstream body"})
            return httpx.Response(200, json=completion(usage={
                "prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}))
        client = LLMClient(config(), transport=httpx.MockTransport(reply))
        generation, trace = await client.generate([{"role": "user", "content": "question"}])
        self.assertEqual(trace["attempts"], 2)
        self.assertEqual(trace["retries"], 1)
        self.assertEqual(generation.usage["total_tokens"], 16)
        self.assertEqual(calls[0]["max_tokens"], client.settings.llm_max_output_tokens)
        self.assertNotIn("temperature", calls[0])
        self.assertNotIn("fixture-secret", json.dumps(trace))
        self.assertNotIn("sensitive", json.dumps(trace))

    async def test_authentication_failure_is_not_retried(self):
        count = 0
        def reply(request):
            nonlocal count
            count += 1
            return httpx.Response(401, json={"error": "fixture-secret"})
        with self.assertRaises(LLMCallError) as caught:
            await LLMClient(config(), transport=httpx.MockTransport(reply)).generate([])
        self.assertEqual(count, 1)
        self.assertEqual(caught.exception.category, "authentication_error")
        self.assertNotIn("fixture-secret", str(caught.exception))
        self.assertEqual(caught.exception.trace["usage"], None)

    async def test_timeout_exhausts_bounded_retries(self):
        def reply(request):
            raise httpx.ReadTimeout("fixture-secret", request=request)
        with self.assertRaises(LLMCallError) as caught:
            await LLMClient(config(), transport=httpx.MockTransport(reply)).generate([])
        self.assertEqual(caught.exception.http_status, 504)
        self.assertEqual(caught.exception.trace["attempts"], 3)
        self.assertNotIn("fixture-secret", json.dumps(caught.exception.trace))

    async def test_per_attempt_deadline_covers_sdk_wait(self):
        """SDK 等待也受单次总限时约束，达到上限后有限重试并返回 504。"""
        async def slow(*args):
            await asyncio.sleep(1)
        model = SimpleNamespace(ainvoke=AsyncMock(side_effect=slow))
        with self.assertRaises(LLMCallError) as caught:
            await LLMClient(config(llm_mode="langchain_openai", llm_timeout_seconds=0.01,
                                   llm_max_retries=1), model=model).generate([])
        self.assertEqual(model.ainvoke.await_count, 2)
        self.assertEqual(caught.exception.category, "timeout")
        self.assertEqual(caught.exception.http_status, 504)

    async def test_invalid_and_truncated_responses_are_not_retried(self):
        payloads = [{}, completion(content=""), {"choices": [
            {"message": {"content": "partial [1]"}, "finish_reason": "length"}]}]
        for payload in payloads:
            with self.subTest(payload=payload), self.assertRaises(LLMCallError) as caught:
                await LLMClient(config(), transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json=payload))).generate([])
            self.assertEqual(caught.exception.category, "invalid_response")
            self.assertEqual(caught.exception.trace["attempts"], 1)

    async def test_json_mode_sends_format_and_leaves_missing_usage_unknown(self):
        def reply(request):
            payload = json.loads(request.content)
            self.assertEqual(payload["response_format"], {"type": "json_object"})
            self.assertEqual(payload["temperature"], 0)
            return httpx.Response(200, json=completion(content='{}'))
        _, trace = await LLMClient(config(llm_answer_format="json", llm_temperature=0),
                                  transport=httpx.MockTransport(reply)).generate([])
        self.assertIsNone(trace["usage"])

    async def test_langchain_adapter_uses_same_trace(self):
        model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(
            content="支持原文 [1]", usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
            response_metadata={"model_name": "sdk-fixture", "finish_reason": "stop"})))
        generation, trace = await LLMClient(config(llm_mode="langchain_openai"), model=model).generate(
            [{"role": "user", "content": "question"}])
        self.assertEqual(trace["model"], "sdk-fixture")
        self.assertEqual(generation.usage["total_tokens"], 10)
        self.assertEqual(trace["attempts"], 1)

    async def test_actual_langchain_sdk_two_calls_on_one_loop(self):
        """实际 SDK 配合模拟 HTTP，验证多次请求的解析、连接复用和用量映射。"""
        from langchain_openai import ChatOpenAI
        payload = completion(usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8})
        payload["choices"][0]["message"]["role"] = "assistant"
        payload["choices"][0]["index"] = 0
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=payload))) as http_client:
            model = ChatOpenAI(model="fixture-model", api_key="fixture-secret",
                               base_url="https://fixture.invalid/v1", http_async_client=http_client,
                               max_retries=0, use_responses_api=False)
            client = LLMClient(config(llm_mode="langchain_openai"), model=model)
            for _ in range(2):
                result, trace = await client.generate([{"role": "user", "content": "question"}])
                self.assertEqual(result.usage["total_tokens"], 8)
                self.assertEqual(trace["attempts"], 1)

    async def test_optional_thinking_flag_reaches_http(self):
        """仅显式设置时发送服务商扩展，False 不能被当成未配置。"""
        def reply(request):
            self.assertIs(json.loads(request.content)["enable_thinking"],False)
            return httpx.Response(200,json=completion())
        await LLMClient(config(llm_enable_thinking=False),transport=httpx.MockTransport(reply)).generate([])

    async def test_optional_thinking_flag_reaches_langchain(self):
        """通过实际 SDK 序列化检查扩展字段，避免适配器只在测试替身中有效。"""
        client=LLMClient(config(llm_mode="langchain_openai",llm_enable_thinking=False))
        self.assertEqual(client.model.extra_body,{"enable_thinking":False})

    async def test_invalid_configuration_rejected(self):
        for values in ({"llm_max_retries": 6}, {"llm_timeout_seconds": 0},
                       {"llm_temperature": float("nan")}, {"llm_answer_format": "unknown"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                LLMClient(config(**values))


class StructuredAnswerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.evidence = [EvidenceContext(1, "demo.txt", 1, "报告格式为学号-姓名。")]

    async def test_json_answer_and_no_evidence_skip(self):
        solver = EvidenceSolver(config(llm_answer_format="json"))
        raw = json.dumps({"answer_status": "answered", "answer": "报告格式为学号-姓名。[1]",
                          "cited_numbers": [1]}, ensure_ascii=False)
        solver.client.transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json=completion(raw)))
        result = await solver.answer("报告格式？", self.evidence, [])
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.cited_numbers, (1,))
        self.assertEqual(result.trace["answer_format"], "json")
        solver.client.generate = AsyncMock(side_effect=AssertionError("must skip network"))
        result = await solver.answer("未见品牌？", [], [])
        self.assertEqual(result.status, "no_evidence")
        self.assertEqual(result.trace["attempts"], 0)

    async def test_structured_refusal_normalizes_text(self):
        raw = json.dumps({"answer_status": "insufficient_evidence", "answer": "", "cited_numbers": []})
        result = EvidenceSolver._validate_json_answer(raw, self.evidence)
        self.assertEqual(result.answer, INSUFFICIENT_EVIDENCE)
        self.assertEqual(result.cited_numbers, ())

    async def test_refusal_cannot_smuggle_brand_claim_or_citation(self):
        for answer, numbers in (("全文没有任何品牌。", []), ("", [1])):
            raw = json.dumps({"answer_status": "insufficient_evidence", "answer": answer, "cited_numbers": numbers})
            self.assertEqual(EvidenceSolver._validate_json_answer(raw, self.evidence).status, "invalid_response")

    async def test_invalid_json_and_citation_contract(self):
        values = ["not json", "[]", '{"answer_status":"answered"}']
        for numbers in ([2], [], [True], [1, 1]):
            values.append(json.dumps({"answer_status": "answered", "answer": "内容 [1]", "cited_numbers": numbers}))
        for raw in values:
            with self.subTest(raw=raw):
                self.assertEqual(EvidenceSolver._validate_json_answer(raw, self.evidence).status, "invalid_response")

    async def test_text_format_stays_compatible(self):
        self.assertEqual(EvidenceSolver._validate_model_answer("报告格式。[1]", self.evidence).status, "answered")
        self.assertEqual(EvidenceSolver._validate_model_answer("报告格式。[9]", self.evidence).status, "invalid_citations")


class LLMAPITests(unittest.TestCase):
    def test_api_returns_safe_error_and_persists_failed_call(self):
        """通过实际 FastAPI 路由验证错误状态和 run_id，不启动外部模型或向量库。"""
        from app import main
        root = Path.cwd() / "tests" / ".tmp"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            store = Store(Path(directory))
            store.add_document("doc", "demo.txt", "课程报告格式为学号姓名，课程报告需要准时提交。")
            store.replace_document_chunks("doc", [
                ("chunk", "doc", "课程报告格式为学号姓名，课程报告需要准时提交。", 0, {"page": 1})])
            flow = KnowledgeFlow(store, config())
            flow.llm.client.transport = httpx.MockTransport(
                lambda request: httpx.Response(401, json={"error": "fixture-secret"}))
            old_store, old_flow = main.store, main.flow
            main.store, main.flow = store, flow
            try:
                response = TestClient(main.app).post("/chat", json={
                    "session_id": "llm-failure", "message": "课程报告格式", "document_ids": ["doc"]})
                self.assertEqual(response.status_code, 502)
                self.assertNotIn("fixture-secret", response.text)
                run_id = response.json()["detail"]["run_id"]
                run = store.get_run(run_id)
                self.assertEqual(run["status"], "failed")
                event = next(x for x in run["events"] if x["node"] == "llm")
                self.assertEqual(event["output"]["attempts"], 1)
                self.assertNotIn("fixture-secret", json.dumps(run))
            finally:
                main.store, main.flow = old_store, old_flow
                store.conn.close()


if __name__ == "__main__":
    unittest.main()

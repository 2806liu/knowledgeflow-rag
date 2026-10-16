"""统一 LLM 调用：负责限时、有限重试、用量记录和脱敏错误，不处理答案语义。"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings


class LLMCallError(RuntimeError):
    """可对外展示的调用错误；trace 只含状态与计数，不保存密钥、消息或响应正文。"""

    def __init__(self, category: str, trace: dict[str, Any], http_status: int = 502):
        super().__init__(f"LLM request failed: {category}")
        self.category = category
        self.trace = trace
        self.http_status = http_status


@dataclass(frozen=True)
class LLMGeneration:
    """一次成功调用的文本及实际用量；上游未提供的字段保持为空，不估算 token。"""

    content: str
    model: str
    usage: dict[str, int] | None
    finish_reason: str | None


def _usage(values: Any, *, langchain: bool = False) -> dict[str, int] | None:
    """兼容 HTTP 和 LangChain 的用量字段，拒绝异常类型及负数。"""
    if not isinstance(values, dict):
        return None
    names = ("input_tokens", "output_tokens", "total_tokens") if langchain else (
        "prompt_tokens", "completion_tokens", "total_tokens")
    result = {}
    for target, source in zip(("input_tokens", "output_tokens", "total_tokens"), names):
        value = values.get(source)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[target] = value
    return result or None


class LLMClient:
    """两个适配器共用同一重试策略；关闭 SDK 内部重试，避免重试次数相乘。"""

    def __init__(self, settings: Settings, *, transport=None, model=None):
        self.settings = settings
        self.transport = transport
        self.model = model
        if (not math.isfinite(settings.llm_timeout_seconds) or not 0 < settings.llm_timeout_seconds <= 300
                or not 0 <= settings.llm_max_retries <= 5
                or not math.isfinite(settings.llm_retry_backoff_seconds)
                or not 0 <= settings.llm_retry_backoff_seconds <= 10
                or not 1 <= settings.llm_max_output_tokens <= 32768):
            raise ValueError("invalid LLM timeout, retry, backoff, or output token limit")
        if settings.llm_answer_format not in {"text", "json"}:
            raise ValueError("LLM answer format must be text or json")
        if settings.llm_temperature is not None and (
                not math.isfinite(settings.llm_temperature) or not 0 <= settings.llm_temperature <= 2):
            raise ValueError("LLM temperature must be between 0 and 2")
        if settings.llm_mode not in {"mock", "openai_compatible", "langchain_openai"}:
            raise ValueError("unsupported LLM mode")
        if settings.llm_mode != "mock" and not (
                settings.llm_base_url and settings.llm_model and settings.llm_api_key):
            raise ValueError("real LLM mode requires base URL, model and API key")
        if settings.llm_mode == "langchain_openai" and self.model is None:
            from langchain_openai import ChatOpenAI
            options: dict[str, Any] = {}
            if settings.llm_temperature is not None:
                options["temperature"] = settings.llm_temperature
            if settings.llm_enable_thinking is not None:
                options["extra_body"]={"enable_thinking":settings.llm_enable_thinking}
            if settings.llm_answer_format == "json":
                options["model_kwargs"] = {"response_format": {"type": "json_object"}}
            self.model = ChatOpenAI(
                model=settings.llm_model, api_key=settings.llm_api_key,
                base_url=settings.llm_base_url, timeout=settings.llm_timeout_seconds,
                max_retries=0, max_tokens=settings.llm_max_output_tokens,
                use_responses_api=False, **options)

    @staticmethod
    def _generation(content, model, usage, finish_reason) -> LLMGeneration:
        """空内容、截断和内容过滤均明确失败，避免把不完整回答当成正常答案。"""
        if not isinstance(content, str) or not content.strip():
            raise ValueError("invalid model response")
        if finish_reason not in {None, "stop"}:
            raise ValueError("unfinished model response")
        return LLMGeneration(content, str(model), usage, finish_reason)

    async def _http(self, messages) -> LLMGeneration:
        """只发送必要字段；不把上游错误正文带到异常或日志。"""
        config = self.settings
        payload = {"model": config.llm_model, "messages": messages,
                   "max_tokens": config.llm_max_output_tokens}
        if config.llm_temperature is not None:
            payload["temperature"] = config.llm_temperature
        # 服务商特定参数仅在明确配置时发送，避免破坏原有兼容接口。
        if config.llm_enable_thinking is not None:
            payload["enable_thinking"]=config.llm_enable_thinking
        if config.llm_answer_format == "json":
            payload["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=config.llm_timeout_seconds, transport=self.transport) as client:
            response = await client.post(
                f"{config.llm_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {config.llm_api_key}"}, json=payload)
            response.raise_for_status()
            data = response.json()
        choice = data["choices"][0]
        return self._generation(choice["message"]["content"], data.get("model", config.llm_model),
                                _usage(data.get("usage")), choice.get("finish_reason"))

    async def _langchain(self, messages) -> LLMGeneration:
        """读取 SDK 返回的真实 token 统计，缺失时保持 null。"""
        response = await self.model.ainvoke([(x["role"], x["content"]) for x in messages])
        metadata = response.response_metadata or {}
        usage = _usage(getattr(response, "usage_metadata", None), langchain=True)
        if usage is None:
            usage = _usage(metadata.get("token_usage"))
        return self._generation(response.content, metadata.get("model_name", self.settings.llm_model),
                                usage, metadata.get("finish_reason"))

    @staticmethod
    def _classify(exc):
        """仅超时、网络异常、408、429 和 5xx 可重试；鉴权及格式错误立即返回。"""
        status = getattr(exc, "status_code", None)
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
        name = type(exc).__name__
        # SDK在JSON解析前报告输出截断，不能归为无法解释的供应商故障。
        if name == "LengthFinishReasonError":
            return "response_length", False, None, 502
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException)) or name in {"APITimeoutError", "OpenAITimeoutError"}:
            return "timeout", True, None, 504
        # 兼容上层 SDK 的异常别名，让短暂连接故障进入已有的受限重试。
        if isinstance(exc, httpx.TransportError) or name in {"APIConnectionError", "OpenAIConnectionError"}:
            return "connection_error", True, None, 502
        if isinstance(status, int):
            retryable = status in {408, 429} or 500 <= status < 600
            category = "authentication_error" if status in {401, 403} else (
                "rate_limited" if status == 429 else "upstream_http_error")
            return category, retryable, status, 503 if status == 429 else 502
        return "invalid_response" if isinstance(exc, (ValueError, KeyError, IndexError, TypeError)) else (
            "provider_error"), False, None, 502

    async def generate(self, messages) -> tuple[LLMGeneration, dict[str, Any]]:
        """执行一次逻辑调用；attempts 是实际尝试次数，retries 不含初次请求。"""
        started = time.perf_counter()
        trace: dict[str, Any] = {"mode": self.settings.llm_mode, "model": self.settings.llm_model,
                                "attempts": 0, "retries": 0, "usage": None, "errors": []}
        for attempt in range(self.settings.llm_max_retries + 1):
            trace.update(attempts=attempt + 1, retries=attempt)
            try:
                # 单次总限时覆盖连接、读取和 SDK 等待；取消信号不被转换为重试。
                result = await asyncio.wait_for(
                    self._langchain(messages) if self.settings.llm_mode == "langchain_openai"
                    else self._http(messages), timeout=self.settings.llm_timeout_seconds)
                trace.update(status="ok", model=result.model, usage=result.usage,
                             finish_reason=result.finish_reason,
                             latency_ms=round((time.perf_counter() - started) * 1000, 2))
                return result, trace
            except Exception as exc:
                category, retryable, status, http_status = self._classify(exc)
                trace["errors"].append({"attempt": attempt + 1, "category": category,
                                        "upstream_status": status, "retryable": retryable,
                                        "error_type": type(exc).__name__})
                if retryable and attempt < self.settings.llm_max_retries:
                    await asyncio.sleep(min(self.settings.llm_retry_backoff_seconds * 2 ** attempt, 10))
                    continue
                trace.update(status="error", category=category,
                             latency_ms=round((time.perf_counter() - started) * 1000, 2))
                raise LLMCallError(category, trace, http_status) from None
        raise AssertionError("unreachable")

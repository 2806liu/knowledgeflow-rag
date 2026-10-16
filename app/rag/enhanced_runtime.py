"""增强问答的统一调用预算：有限逻辑调用、子阶段期限、完整输入计数和脱敏诊断。"""
import asyncio
import json
import time
from app.rag.context_builder import ContextBuilder, ContextOverflow
from app.rag.llm_client import LLMCallError

class EnhancedCalls:
    def __init__(self, config, client, events):
        self.config, self.client, self.events = config, client, events
        self.started = time.monotonic()
        self.logical_calls = 0
        self.repair_calls = 0

    async def generate(self, system, payload, phase, rounds=()):
        """每次重试均受同一剩余期限控制；调用异常也要留下预算和尝试记录。"""
        remaining = self.config.enhanced_timeout_seconds - (time.monotonic()-self.started)
        is_repair = phase == "enhanced_validation_repair"
        # 解析和普通生成最多三次；仅校验失败可额外修复一次，仍受同一总期限约束。
        exhausted = (self.logical_calls >= 4 or self.repair_calls >= 1) if is_repair else self.logical_calls >= 3
        if exhausted or remaining <= 0:
            raise LLMCallError("timeout", {"status":"error","attempts":0,"phase":phase,
                                         "reason":"enhanced_budget_exhausted"}, 504)
        self.logical_calls += 1
        if is_repair:
            self.repair_calls += 1
        timeout = min(remaining, self.config.resolution_timeout_seconds) if phase == "query_resolution" else remaining
        builder=ContextBuilder(self.config)
        try:
            built = builder.build(system, payload, rounds)
        except ContextOverflow as exc:
            exc.trace["phase"] = phase
            exc.trace["component_counts"] = {
                **exc.trace.get("component_counts", {}),
                **{key:builder.counter(json.dumps(value,ensure_ascii=False))
                   for key,value in payload.items()}}
            self.events.append({"node":"context_budget","status":"error","output":exc.trace})
            raise
        try:
            generation, trace = await asyncio.wait_for(self.client.generate(built.messages), timeout=timeout)
        except asyncio.TimeoutError as exc:
            trace = {"status":"error","phase":phase,"attempts":None,
                     "error_type":"TimeoutError","context_budget":built.trace,
                     "attempts_note":"deadline cancellation: provider attempt count unavailable"}
            self.events.append({"node":"model_call","status":"error","output":trace})
            raise LLMCallError("timeout", trace, 504) from exc
        except LLMCallError as exc:
            exc.trace.update(phase=phase, context_budget=built.trace)
            self.events.append({"node":"model_call","status":"error","output":exc.trace})
            raise
        trace = {**trace, "phase":phase,"logical_call":self.logical_calls,"context_budget":built.trace}
        self.events.append({"node":"model_call","status":trace.get("status","ok"),"output":trace})
        return generation.content, trace

def invalid_response(trace, reason):
    """只记录校验原因代码，不把上游原始正文暴露到异常响应。"""
    return LLMCallError("invalid_response", {**trace,"status":"error","validation_reason":reason}, 502)

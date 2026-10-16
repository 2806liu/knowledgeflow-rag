"""统一上下文预算：只组装输入，不访问数据库、不执行模型调用。"""
from dataclasses import dataclass
import json
from app.rag.evidence_expansion import token_counter

class ContextOverflow(ValueError):
    """必要输入放不下；调用方必须分批或明确拒绝，不能静默截断。"""

    def __init__(self, message, trace=None):
        super().__init__(message)
        self.trace = trace or {}

@dataclass
class BuiltContext:
    messages: list
    trace: dict

class ContextBuilder:
    def __init__(self, config):
        self.config = config
        self.method, self.counter = token_counter(config.llm_model)
        self.window = config.context_window
        self.output = config.llm_max_output_tokens
        self.margin = config.context_safety_margin
        if self.window <= self.output + self.margin or self.margin < 0:
            raise ValueError("invalid context window/output reserve")

    def count(self, messages):
        """对完整JSON封装计数并预留每消息开销；未知模型明确为保守估算。"""
        return self.counter(json.dumps(messages,ensure_ascii=False)) + 32*len(messages)

    def build(self, system, payload, rounds=(), pinned=()):
        base = [{"role":"system","content":system},
                {"role":"user","content":json.dumps(payload,ensure_ascii=False)}]
        limit = self.window-self.output-self.margin
        history_limit = self.config.history_budget
        fixed = list(pinned)
        def compose(selected):
            history = [m for pair in selected for m in pair]
            return [base[0], *history, base[1]]
        fixed_count = self.count(compose(fixed))
        if fixed_count > limit:
            # 诊断仅包含计数与阶段，由调用方补充阶段；不记录问题、证据或密钥原文。
            raise ContextOverflow("必要问题、规则或证据超过上下文预算",
                                  dict(token_count_method=self.method,input_count=fixed_count,
                                       input_limit=limit,reserved_output=self.output,
                                       safety_margin=self.margin,component_counts={
                                           "system":self.counter(system),
                                           "payload":self.counter(json.dumps(payload,ensure_ascii=False)),
                                           "pinned":self.count([m for pair in fixed for m in pair]) if fixed else 0}))
        kept, dropped = list(fixed), []
        # 从最近完整轮次向前加入，整体超限即记录移除，不截断某条消息。
        selected = []
        for pair in reversed(list(rounds)):
            candidate = [pair] + selected
            history_cost = self.count([m for p in candidate+fixed for m in p])
            if history_cost <= history_limit and self.count(compose(candidate+fixed)) <= limit:
                selected = candidate
            else:
                dropped.append({"reason":"history_budget","messages":len(pair)})
        kept = selected+fixed
        messages = compose(kept)
        trace = dict(token_count_method=self.method, input_count=self.count(messages),
                     input_limit=limit,reserved_output=self.output,safety_margin=self.margin,
                     kept_rounds=len(kept),dropped=dropped,exact=self.method!="utf8_bytes_conservative_budget")
        return BuiltContext(messages, trace)

"""任务完成率与记忆输入成本补测；固定公开合成材料、真实应用流程及配对上下文。

运行 python -m evaluation.task_memory_benchmark --output evaluation/results/task-memory-v1。
不挂载日常数据库；测量值来自供应商 usage，保守预算计数不得当成实际 Token。
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
import uuid

from evaluation.task_memory_suite import HISTORY, PROBES, SOURCES, SUITE_VERSION, TASKS


def digest(value):
    """冻结数据与审阅对象，避免修改答案后沿用旧审阅结果。"""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def usage_total(records):
    """逐调用加总；缺失、重试或失败时保持不完整，不用0冒充未知成本。"""
    totals = dict(input_tokens=0, output_tokens=0, total_tokens=0)
    complete = bool(records)
    for item in records:
        trace = item.get("trace", item)
        usage = trace.get("usage") or {}
        valid = trace.get("status") == "ok" and trace.get("attempts") == 1
        for key in totals:
            value = usage.get(key)
            if type(value) is int and value >= 0:
                totals[key] += value
            else:
                valid = False
        complete = complete and valid
    return {**totals, "complete": complete, "logical_calls": len(records),
            "scope": "provider_returned_successful_call_usage; retry_usage_unknown_if_any"}


def reduction(baseline, managed):
    """百分比按总量相除；不平均各题百分比，也不把缺失值作为节省。"""
    if baseline is None or managed is None or baseline <= 0:
        return None
    return round(100 * (baseline - managed) / baseline, 4)


def check_response(response, spec, documents, chunks):
    """保守规则核验状态、逐方面事实、当前原文和引用归属；语义仍需单独审阅。"""
    reasons = []
    if not response:
        return dict(passed=False, reasons=["missing_response"])
    sections = response.get("answer_sections", [])
    citations = {c["citation_number"]: c for c in response.get("citations", [])}
    allowed = {documents[key] for key in spec["scope"]}
    for citation in citations.values():
        row = chunks.get(citation.get("chunk_id"))
        if not row or row["document_id"] not in allowed or (
            row["document_id"] != citation.get("document_id") or row["text"] != citation.get("snippet")
        ):
            reasons.append("citation_not_current_source")
    if spec.get("refusal"):
        if response.get("answer_status") not in {"insufficient_evidence", "no_evidence"}:
            reasons.append("expected_refusal")
        if citations or any(s["source_type"] != "limitation" for s in sections):
            reasons.append("refusal_contains_claims")
        return dict(passed=not reasons, reasons=reasons)
    if response.get("answer_status") != "answered" or response.get("unresolved_aspects"):
        reasons.append("task_not_fully_answered")
    if any(s["source_type"] == "model_knowledge" for s in sections):
        reasons.append("document_task_uses_model_knowledge")
    for index, expected in enumerate(spec["facts"], 1):
        document = documents[expected["document"]]
        # 每个要点须在同文档的依据段落及其实际引用中同时出现；允许分段回答多个方面。
        supported_patterns = set()
        for section in sections:
            if section["source_type"] != "document":
                continue
            refs = [citations[n] for n in section.get("citation_numbers", []) if n in citations]
            correct = [c for c in refs if c["document_id"] == document]
            text = section["content"]
            source = "\n".join(c["snippet"] for c in correct)
            if correct:
                supported_patterns.update(p for p in expected["patterns"]
                    if re.search(p, text, re.I | re.S) and re.search(p, source, re.I | re.S))
        if not set(expected["patterns"]) <= supported_patterns:
            reasons.append("fact_" + str(index) + "_missing_or_unsupported")
    if spec.get("both_sources"):
        pair = {documents["alpha"], documents["beta"]}
        difference = any(s["source_type"] == "document" and
                         pair <= {citations[n]["document_id"] for n in s["citation_numbers"] if n in citations}
                         for s in sections)
        if not difference:
            reasons.append("difference_without_two_sources")
    for section in sections:
        numbers = set(section.get("citation_numbers", []))
        markers = {int(n) for n in re.findall(r"\[(\d+)\]", section["content"])}
        if numbers != markers or not numbers <= set(citations):
            reasons.append("invalid_citation_markers")
    return dict(passed=not reasons, reasons=sorted(set(reasons)))


def completion_metrics(cases):
    """按任务计数，任何一步失败或未审阅都不能算解决；负例单列。"""
    answerable = [c for c in cases if not c["negative_control"]]
    negatives = [c for c in cases if c["negative_control"]]
    rule_passed = sum(c["automatic_passed"] for c in answerable)
    reviewed = all(c.get("review", {}).get("verdict") in {"pass", "fail"} for c in answerable)
    solved = sum(c["automatic_passed"] and c.get("review", {}).get("verdict") == "pass"
                 for c in answerable) if reviewed else None
    return dict(task_count=len(answerable), rule_passed=rule_passed,
                rule_pass_rate_pct=round(100 * rule_passed / len(answerable), 4) if answerable else None,
                reviewed_solved=solved,
                reviewed_completion_rate_pct=round(100 * solved / len(answerable), 4) if reviewed and answerable else None,
                semantic_review_complete=reviewed, negative_control_count=len(negatives),
                correct_refusals=sum(c["automatic_passed"] for c in negatives),
                unit="task; all predefined steps must pass; no user correction or extra hints")


def token_metrics(pairs, compaction):
    """回答阶段与摊入一次摘要成本的重放实验分列，不宣称全链路或计费金额下降。"""
    base = usage_total([p["baseline"] for p in pairs])
    managed = usage_total([p["managed"] for p in pairs])
    compact = usage_total(compaction)
    complete = bool(pairs) and base["complete"] and managed["complete"] and compact["complete"]
    same = all(p["same_question_evidence"] and p["baseline"]["all_history_retained"] for p in pairs)
    eligible = complete and same
    baseline_input = base["input_tokens"]
    managed_input = managed["input_tokens"]
    net = managed_input + compact["input_tokens"]
    return dict(pair_count=len(pairs), unit="provider input_tokens", usage_complete=complete,
                same_question_evidence_and_full_baseline=same,
                baseline=base, managed=managed, compaction=compact,
                answer_input_reduction_pct=reduction(baseline_input, managed_input) if eligible else None,
                managed_plus_compaction_input_tokens=net,
                amortized_input_reduction_pct=reduction(baseline_input, net) if eligible else None,
                baseline_total_tokens=base["total_tokens"],
                managed_plus_compaction_total_tokens=managed["total_tokens"] + compact["total_tokens"],
                total_token_reduction_pct=reduction(base["total_tokens"], managed["total_tokens"] + compact["total_tokens"]) if eligible else None,
                baseline_rule_passed=sum(p["baseline"]["checks"]["passed"] for p in pairs),
                managed_rule_passed=sum(p["managed"]["checks"]["passed"] for p in pairs),
                scope="8 independent answer replays sharing the same frozen history; one compaction amortized; excludes planner/retrieval/HTTP latency")


class RecordingClient:
    """只包裹本评测的隔离客户端；保留公开合成请求，错误仅记类别。"""
    def __init__(self, client):
        self.client, self.records = client, []

    async def generate(self, messages):
        record = dict(messages=deepcopy(messages))
        self.records.append(record)
        try:
            generation, trace = await self.client.generate(messages)
            record.update(trace=deepcopy(trace), content=generation.content)
            return generation, trace
        except Exception as exc:
            record["trace"] = deepcopy(getattr(exc, "trace", dict(status="error", error_type=type(exc).__name__)))
            raise


def seed_sources(store):
    """从本模块的合成原文建立隔离索引，日常知识库不参与。"""
    documents = {}
    for key, (filename, paragraphs) in SOURCES.items():
        document = str(uuid.uuid4())
        store.add_document(document, filename, "\n\n".join(paragraphs))
        store.replace_document_chunks(document, [
            (str(uuid.uuid4()), document, text, i, dict(page=1, section_id=f"{key}-{i}", extraction_method="native_text"))
            for i, text in enumerate(paragraphs, 1)])
        documents[key] = document
    return documents


def seed_history(repo, session):
    """仅用于Token实验的固定历史，保留完整问答；不向真实任务加入答案提示。"""
    for question, answer in HISTORY:
        turn, _ = repo.begin_turn(session, str(uuid.uuid4()), question)
        repo.finish_turn(turn, dict(response=dict(answer=answer, intent="general_question",
            answer_status="answered", answer_basis="model_only", citations=[], answer_sections=[]),
            resolution=dict(resolved_query=question), answer_units=[]))


def canonical_response(result, evidence, chunks):
    """为配对回答恢复与生产回答相同的原文引用字段，随后走相同核验。"""
    citations = []
    for item in evidence:
        if item["citation_number"] not in result["cited_numbers"]:
            continue
        row = next(r for r in chunks.values() if r["document_id"] == item.get("document_id") and r["text"] == item["text"])
        citations.append(dict(citation_number=item["citation_number"], chunk_id=row["id"],
                              document_id=row["document_id"], snippet=row["text"], page=item["page"]))
    return {**result, "citations": citations}


def probe_prompt(case):
    """概述路线没有普通生成输入时，仅回取引用原文；旧概述正文不充当证据。"""
    from app.rag.enhanced_answer import SYSTEM
    step = case["steps"][-1]
    response = step.get("response") or {}
    citations = response.get("citations", [])
    if not citations:
        raise ValueError("固定探针缺少可定位的原始证据，不能删题")
    question = step["spec"]["question"]
    payload = dict(task="enhanced_answer", question=question, resolved_query=question,
        requested_aspects=[dict(id="a1", question=question, requires_document=True)], comparison_task=None,
        evidence=[dict(citation_number=c["citation_number"], filename=c["filename"], page=c.get("page"),
                       text=c["snippet"], document_id=c["document_id"]) for c in citations],
        final_attempt=True, validation_feedback=None, conversation_summary="", summary_is_document_evidence=False)
    return [dict(role="system", content=SYSTEM), dict(role="user", content=json.dumps(payload, ensure_ascii=False))]


async def run(output):
    """一次冻结运行，逐任务保存失败；不因失败重跑或挑选成功记录。"""
    from app.config import settings
    from app.conversational_flow import ConversationalFlow
    from app.rag.answer_policy import validate_decision
    from app.rag.context_builder import ContextBuilder
    from app.rag.contracts import EvidenceContext
    from app.rag.llm_client import LLMClient
    from app.services import KnowledgeFlow
    from app.store import Store

    if settings.llm_mode == "mock":
        raise RuntimeError("真实补测需要已配置的LLM；Mock不能生成成果指标")
    output = output.resolve()
    if output.exists():
        raise FileExistsError("结果目录已存在；使用新名称保留原运行")
    output.mkdir(parents=True)
    source_hash = hashlib.sha256()
    for path in sorted([*Path("app").rglob("*.py"), *Path("evaluation").glob("task_memory_*.py")]):
        source_hash.update(str(path).encode()); source_hash.update(path.read_bytes())
    report = dict(schema_version=1, suite=SUITE_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        suite_sha256=digest(dict(sources=SOURCES, tasks=TASKS, history=HISTORY, probes=PROBES)),
        implementation_sha256=source_hash.hexdigest(), data_origin="public_synthetic_only",
        daily_database_used=False, semantic_review="pending", cases=[], pairs=[])
    prompts = {}
    with tempfile.TemporaryDirectory(prefix="knowledgeflow-public-benchmark-") as directory:
        config = replace(settings, data_dir=Path(directory), vector_enabled=False,
            retrieval_strategy="bm25_idf_v2", reranker_enabled=False, ocr_enabled=False,
            llm_max_retries=0, llm_temperature=0, llm_answer_format="json",
            llm_max_output_tokens=max(2048, settings.llm_max_output_tokens),
            llm_enable_thinking=False if settings.llm_model.lower().startswith("qwen") else settings.llm_enable_thinking)
        report["runtime"] = dict(model=config.llm_model, mode=config.llm_mode, temperature=0,
            retrieval_strategy=config.retrieval_strategy, reranker_enabled=False,
            context_window=config.context_window, history_budget=config.history_budget,
            memory_recent_rounds=config.memory_recent_rounds, memory_summary_budget=config.memory_summary_budget,
            llm_max_output_tokens=config.llm_max_output_tokens,
            llm_answer_format=config.llm_answer_format, enable_thinking=config.llm_enable_thinking,
            token_budget_method=ContextBuilder(config).method,
            usage_method="provider_returned_usage_not_budget_estimates")
        store = Store(config.data_dir)
        flow = KnowledgeFlow(store, config)
        recorder = RecordingClient(LLMClient(config))
        flow.llm.client = recorder
        parent = ConversationalFlow(flow)
        flow._conversation_flow = parent
        # 短任务不需要后台压缩；固定历史实验显式压缩一次，确保所有开销可归因。
        parent.memory.schedule = lambda session: None
        try:
            documents = seed_sources(store)
            chunks = {row["id"]: dict(row) for row in store.chunks()}
            report["corpus"] = dict(document_count=len(documents), chunk_count=len(chunks))
            report["documents"] = {key: dict(document_id=documents[key], filename=SOURCES[key][0]) for key in documents}
            for definition in TASKS:
                session = flow.conversations.create(answer_policy="document_enhanced")["id"]
                case = dict(id=definition["id"], category=definition["category"],
                    negative_control=definition["steps"][0]["refusal"], steps=[], review=dict(verdict="pending"))
                report["cases"].append(case)
                for spec in definition["steps"]:
                    start = len(recorder.records)
                    started = time.perf_counter()
                    entry = dict(spec=spec)
                    try:
                        response = await flow.chat(session, spec["question"],
                            [documents[k] for k in spec["scope"]], answer_policy="document_enhanced")
                        entry.update(response=response, checks=check_response(response, spec, documents, chunks))
                        entry["run"] = store.get_run(response["run_id"])
                        generations = [r for r in recorder.records[start:] if
                            json.loads(r["messages"][-1]["content"]).get("task") == "enhanced_answer"]
                        if generations and definition["id"] in PROBES:
                            prompts[definition["id"]] = deepcopy(generations[-1]["messages"])
                    except Exception as exc:
                        entry.update(error_type=type(exc).__name__, checks=dict(passed=False, reasons=["pipeline_error"]))
                        entry["error_trace"] = deepcopy(getattr(exc, "trace", {}))
                        if getattr(exc, "run_id", None):
                            entry["run"] = store.get_run(exc.run_id)
                    entry.update(elapsed_seconds=round(time.perf_counter() - started, 3),
                                 usage=usage_total(recorder.records[start:]),
                                 calls=deepcopy(recorder.records[start:]))
                    case["steps"].append(entry)
                case["automatic_passed"] = all(s["checks"]["passed"] for s in case["steps"])
                print(json.dumps(dict(task=case["id"], passed=case["automatic_passed"],
                    reasons=[s["checks"]["reasons"] for s in case["steps"]]), ensure_ascii=True), flush=True)
                write_report(output, report)

            session = flow.conversations.create(answer_policy="document_enhanced")["id"]
            seed_history(flow.conversations, session)
            start = len(recorder.records)
            outcome = await parent.memory.compact(session)
            compact_calls = deepcopy(recorder.records[start:])
            report["compaction"] = dict(outcome=outcome, calls=compact_calls)
            if outcome.get("status") != "updated":
                raise RuntimeError("固定历史没有成功压缩，不能计算节省率")
            memory = parent.memory.snapshot(session)
            all_rounds = [[dict(role="user", content=q), dict(role="assistant", content=a)] for q, a in HISTORY]
            report["history"] = dict(round_count=len(HISTORY), turns=HISTORY,
                managed_recent_rounds=len(memory["rounds"]), summary=memory["state"]["summary"],
                compaction_scope="one summary reused by all 8 frozen-history answer probes")
            for index, identifier in enumerate(PROBES):
                if identifier not in prompts:
                    prompts[identifier] = probe_prompt(next(c for c in report["cases"] if c["id"] == identifier))
                original = prompts[identifier]
                system, payload = original[0]["content"], json.loads(original[-1]["content"])
                # 两组仅改变历史表达；当前问题、证据、解析结果和最终生成规则完全相同。
                payload.update(final_attempt=True, validation_feedback=None)
                evidence = [EvidenceContext(**item) for item in payload["evidence"]]
                definition = next(t for t in TASKS if t["id"] == identifier)
                spec = definition["steps"][-1]
                resolution = dict(requested_aspects=payload["requested_aspects"], comparison_task=payload.get("comparison_task"))
                pair = dict(id=identifier, same_question_evidence=True, fixed_payload_sha256=digest(
                    {k: v for k, v in payload.items() if k != "conversation_summary"}))
                report["pairs"].append(pair)
                for arm in (("baseline", "managed") if index % 2 == 0 else ("managed", "baseline")):
                    local = deepcopy(payload)
                    local["conversation_summary"] = "" if arm == "baseline" else memory["state"]["summary"]
                    build_config = replace(config, history_budget=config.context_window) if arm == "baseline" else config
                    rounds = all_rounds if arm == "baseline" else memory["rounds"]
                    built = ContextBuilder(build_config).build(system, local, rounds)
                    value = dict(context_budget=built.trace, all_history_retained=(built.trace["kept_rounds"] == len(rounds)),
                                 checks=dict(passed=False, reasons=["generation_not_completed"]))
                    pair[arm] = value
                    start = len(recorder.records)
                    try:
                        generation, _ = await recorder.generate(built.messages)
                        _, result = validate_decision(generation.content, resolution, evidence, final=True)
                        response = canonical_response(result, payload["evidence"], chunks)
                        value.update(response=response, checks=check_response(response, spec, documents, chunks))
                    except Exception as exc:
                        value["error_type"] = type(exc).__name__
                    value["trace"] = deepcopy(recorder.records[start:][-1]["trace"])
                    value["messages"] = built.messages
                print(json.dumps(dict(pair=identifier, baseline=pair["baseline"]["checks"]["passed"],
                    managed=pair["managed"]["checks"]["passed"])), flush=True)
                write_report(output, report)
            report["token_metrics"] = token_metrics(report["pairs"], compact_calls)
        finally:
            await parent.memory.close()
            store.conn.close()
            report["completion_metrics"] = completion_metrics(report["cases"])
            write_report(output, report)
    print(json.dumps(dict(completion=report["completion_metrics"], tokens=report.get("token_metrics")), ensure_ascii=True))
    return report


def write_report(output, report):
    """持续落盘保留失败现场；临时文件替换避免中断留下半个JSON。"""
    target = output / "report.json"
    pending = output / "report.pending.json"
    pending.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/task-memory-v1"))
    args = parser.parse_args()
    asyncio.run(run(args.output))


if __name__ == "__main__":
    main()

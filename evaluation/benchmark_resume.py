"""恢复未完成的固定Token探针；复用原历史、摘要与已完成调用，不重跑挑选答案。"""

import argparse
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

from evaluation.task_memory_benchmark import (
    RecordingClient, canonical_response, check_response, digest, probe_prompt, token_metrics,
)
from evaluation.task_memory_suite import PROBES, TASKS


def source_records(report):
    """仅恢复本报告的公开合成引用，完全不查询日常资料库。"""
    documents = {key: row["document_id"] for key, row in report["documents"].items()}
    chunks = {}
    for case in report["cases"]:
        for step in case["steps"]:
            for citation in step.get("response", {}).get("citations", []):
                chunks[citation["chunk_id"]] = dict(id=citation["chunk_id"],
                    document_id=citation["document_id"], text=citation["snippet"])
    return documents, chunks


def rescore(report):
    """修正评测器误判后重算同一批答案；保存旧规则、旧结果，不生成新答案。"""
    result = deepcopy(report)
    documents, chunks = source_records(result)
    definitions = {t["id"]: t for t in TASKS}
    if set(definitions) != {c["id"] for c in result["cases"]}:
        raise ValueError("原报告必须保留全部20个预定任务")
    result["scoring_revision"] = dict(version="1.1", original_report_sha256=digest(report),
        reason="PDF与中文相邻时不用英文词边界；逐方面核验允许同文档多个依据段落",
        questions_and_generated_answers_unchanged=True)
    for case in result["cases"]:
        definition = definitions[case["id"]]
        if len(case["steps"]) != len(definition["steps"]):
            raise ValueError("原任务步骤不完整")
        case["original_automatic_passed"] = case["automatic_passed"]
        for step, spec in zip(case["steps"], definition["steps"]):
            if step["spec"]["question"] != spec["question"] or step["spec"]["scope"] != spec["scope"]:
                raise ValueError("不允许重计分时修改问题或范围")
            step["original_checks"] = step["checks"]
            step["spec"] = spec
            step["checks"] = check_response(step.get("response"), spec, documents, chunks)
        case["automatic_passed"] = all(s["checks"]["passed"] for s in case["steps"])
    for pair in result["pairs"]:
        spec = definitions[pair["id"]]["steps"][-1]
        for arm in ("baseline", "managed"):
            pair[arm]["original_checks"] = pair[arm]["checks"]
            pair[arm]["checks"] = check_response(pair[arm].get("response"), spec, documents, chunks)
    return result


async def resume(path, output):
    """只执行缺失探针；两组使用原运行参数、原16轮历史和同一摘要。"""
    from app.config import settings
    from app.rag.answer_policy import validate_decision
    from app.rag.context_builder import ContextBuilder
    from app.rag.contracts import EvidenceContext
    from app.rag.llm_client import LLMClient

    if output.exists():
        raise FileExistsError("结果已存在，不覆盖原运行")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("data_origin") != "public_synthetic_only" or report.get("daily_database_used"):
        raise ValueError("恢复脚本仅接受本评测的公开合成报告")
    runtime = report["runtime"]
    if runtime["model"] != settings.llm_model or runtime["mode"] != settings.llm_mode:
        raise ValueError("模型与原对照运行不一致")
    config = replace(settings, llm_temperature=runtime["temperature"], llm_max_retries=0,
        llm_answer_format=runtime["llm_answer_format"], llm_enable_thinking=runtime["enable_thinking"],
        llm_max_output_tokens=runtime["llm_max_output_tokens"], context_window=runtime["context_window"],
        history_budget=runtime["history_budget"])
    recorder = RecordingClient(LLMClient(config))
    documents, chunks = source_records(report)
    existing = {p["id"] for p in report["pairs"]}
    if not existing <= set(PROBES) or len(existing) != len(report["pairs"]):
        raise ValueError("原探针重复或不属于冻结任务")
    history = report["history"]
    rounds = [[dict(role="user", content=q), dict(role="assistant", content=a)] for q, a in history["turns"]]
    if len(rounds) != history["round_count"]:
        raise ValueError("历史轮次数量不一致")
    report["resume_provenance"] = dict(original_report_sha256=digest(report),
        existing_pairs=sorted(existing), summary_regenerated=False,
        reason="对概述路由使用其原始chunk引用补齐普通回答阶段探针")
    for index, identifier in enumerate(PROBES):
        if identifier in existing:
            continue
        case = next(c for c in report["cases"] if c["id"] == identifier)
        messages = probe_prompt(case)
        system, payload = messages[0]["content"], json.loads(messages[-1]["content"])
        evidence = [EvidenceContext(**e) for e in payload["evidence"]]
        resolution = dict(requested_aspects=payload["requested_aspects"], comparison_task=None)
        pair = dict(id=identifier, same_question_evidence=False,
            fixed_payload_sha256=digest({k: v for k, v in payload.items() if k != "conversation_summary"}),
            prompt_origin="overview_original_chunk_references")
        for arm in (("baseline", "managed") if index % 2 == 0 else ("managed", "baseline")):
            local = deepcopy(payload)
            local["conversation_summary"] = "" if arm == "baseline" else history["summary"]
            selected = rounds if arm == "baseline" else rounds[-history["managed_recent_rounds"]:]
            build_config = replace(config, history_budget=config.context_window) if arm == "baseline" else config
            built = ContextBuilder(build_config).build(system, local, selected)
            value = dict(context_budget=built.trace, all_history_retained=built.trace["kept_rounds"] == len(selected),
                fixed_payload_sha256=digest({k: v for k, v in local.items() if k != "conversation_summary"}),
                messages=built.messages, checks=dict(passed=False, reasons=["generation_not_completed"]))
            start = len(recorder.records)
            try:
                generation, _ = await recorder.generate(built.messages)
                _, decision = validate_decision(generation.content, resolution, evidence, True)
                value["response"] = canonical_response(decision, payload["evidence"], chunks)
                value["checks"] = check_response(value["response"], case["steps"][-1]["spec"], documents, chunks)
            except Exception as exc:
                value["error_type"] = type(exc).__name__
            value["trace"] = deepcopy(recorder.records[start:][-1]["trace"])
            pair[arm] = value
        pair["same_question_evidence"] = pair["baseline"]["fixed_payload_sha256"] == pair["managed"]["fixed_payload_sha256"] == pair["fixed_payload_sha256"]
        report["pairs"].append(pair)
        print(json.dumps(dict(pair=identifier, checks={arm: pair[arm]["checks"] for arm in ("baseline", "managed")})), flush=True)
    if {p["id"] for p in report["pairs"]} != set(PROBES):
        raise ValueError("探针未完成，不公布总降幅")
    report = rescore(report)
    # 对既有六组也从实际输入重新核对相同字段，而非仅信任预设True。
    for pair in report["pairs"]:
        values = []
        for arm in ("baseline", "managed"):
            payload = json.loads(pair[arm]["messages"][-1]["content"])
            values.append(digest({k: v for k, v in payload.items() if k != "conversation_summary"}))
        pair["same_question_evidence"] = values[0] == values[1] == pair["fixed_payload_sha256"]
    report["token_metrics"] = token_metrics(report["pairs"], report["compaction"]["calls"])
    from evaluation.task_memory_benchmark import completion_metrics
    report["completion_metrics"] = completion_metrics(report["cases"])
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["token_metrics"], ensure_ascii=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    asyncio.run(resume(args.report, args.output))


if __name__ == "__main__":
    main()

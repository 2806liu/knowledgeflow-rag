"""将逐题原文审阅绑定到冻结报告；审阅者身份与规则结果分别保留。"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from evaluation.task_memory_benchmark import completion_metrics, digest


def review_basis(report):
    """只哈希不可变的输入、答案与规则结果，允许更新汇总但不允许偷换答案。"""
    return digest(dict(suite_sha256=report["suite_sha256"], cases=[
        dict(id=c["id"], steps=[dict(spec=s["spec"], response=s.get("response"), checks=s["checks"])
                               for s in c["steps"]]) for c in report["cases"]],
        pairs=[dict(id=p["id"], **{arm: dict(response=p[arm].get("response"), checks=p[arm]["checks"])
                                  for arm in ("baseline", "managed")}) for p in report["pairs"]]))


def template(report):
    """模板的pending不是通过，必须由审阅者逐项填写。"""
    pending = dict(verdict="pending", notes="")
    return dict(basis_sha256=review_basis(report), reviewer_type="human", reviewer="",
        criteria="正确、完整、引用支持、无额外人工纠正；比较差异同时有两篇依据",
        tasks={c["id"]: deepcopy(pending) for c in report["cases"]},
        pairs={p["id"]: {arm: deepcopy(pending) for arm in ("baseline", "managed")} for p in report["pairs"]})


def apply_review(report, review):
    """只有全部预定题目审阅完且原文未变，才允许生成语义核验完成率。"""
    if review.get("basis_sha256") != review_basis(report):
        raise ValueError("审阅与当前答案不匹配")
    if review.get("reviewer_type") not in {"human", "assistant"} or not review.get("reviewer", "").strip():
        raise ValueError("必须明确审阅者身份，助手审阅不得标为独立人工")
    if set(review["tasks"]) != {c["id"] for c in report["cases"]} or set(review["pairs"]) != {p["id"] for p in report["pairs"]}:
        raise ValueError("审阅必须覆盖全部预定任务和配对题目")
    verdicts = list(review["tasks"].values())
    for arms in review["pairs"].values():
        if set(arms) != {"baseline", "managed"}:
            raise ValueError("每对回答的两组都需核验")
        verdicts.extend(arms.values())
    if any(v.get("verdict") not in {"pass", "fail"} or not v.get("notes", "").strip() for v in verdicts):
        raise ValueError("审阅缺少逐题结论或理由")
    result = deepcopy(report)
    for case in result["cases"]:
        case["review"] = deepcopy(review["tasks"][case["id"]])
    for pair in result["pairs"]:
        for arm in ("baseline", "managed"):
            pair[arm]["review"] = deepcopy(review["pairs"][pair["id"]][arm])
    result["semantic_review"] = dict(status="completed", reviewer_type=review["reviewer_type"],
        reviewer=review["reviewer"], basis_sha256=review["basis_sha256"],
        independent_human_review=review["reviewer_type"] == "human")
    result["completion_metrics"] = completion_metrics(result["cases"])
    if "token_metrics" in result:
        for arm in ("baseline", "managed"):
            result["token_metrics"][arm + "_reviewed_passed"] = sum(
                p[arm]["checks"]["passed"] and p[arm]["review"]["verdict"] == "pass" for p in result["pairs"])
        result["token_metrics"]["quality_gate_passed"] = all(
            p[arm]["checks"]["passed"] and p[arm]["review"]["verdict"] == "pass"
            for p in result["pairs"] for arm in ("baseline", "managed"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    value = apply_review(report, json.loads(args.review.read_text(encoding="utf-8"))) if args.review else template(report)
    if args.output.exists():
        raise FileExistsError("保留原报告，请使用新输出名称")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value.get("completion_metrics", dict(status="review_template_created"))))


if __name__ == "__main__":
    main()

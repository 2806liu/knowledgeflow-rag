"""Compare two live answer reports only when labels and index are identical."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    for key in ("suite", "source_sha256", "index_sha256"):
        if before[key] != after[key]:
            raise ValueError(f"cannot compare reports with different {key}")
    earlier = {case["id"]: case for case in before["cases"]}
    later = {case["id"]: case for case in after["cases"]}
    if earlier.keys() != later.keys():
        raise ValueError("cannot compare reports with different case IDs")
    changed: list[dict[str, Any]] = []
    for case_id in earlier:
        old, new = earlier[case_id], later[case_id]
        if old["expected"] != new["expected"] or old["query"] != new["query"]:
            raise ValueError(f"case label or query changed: {case_id}")
        if "error" in old or "error" in new:
            raise ValueError(f"case {case_id} has an HTTP or model error")
        if old.get("first_relevant_rank") != new.get("first_relevant_rank") or old.get("failure") != new.get("failure"):
            changed.append({
                "id": case_id,
                "old_rank": old.get("first_relevant_rank"),
                "new_rank": new.get("first_relevant_rank"),
                "old_failure": old.get("failure"),
                "new_failure": new.get("failure"),
                "old_answer_status": old.get("answer_status"),
                "new_answer_status": new.get("answer_status"),
            })
    metrics = ("Recall@1", "Recall@2", "Recall@4")
    return {
        "suite": before["suite"],
        "source_sha256": before["source_sha256"],
        "index_sha256": before["index_sha256"],
        "case_count": len(earlier),
        "baseline": before["aggregate"],
        "candidate": after["aggregate"],
        "retrieval_delta": {
            metric: round(after["aggregate"]["retrieval"][metric] - before["aggregate"]["retrieval"][metric], 4)
            for metric in metrics
        },
        "mrr_at_4_delta": round(after["aggregate"]["MRR@4"] - before["aggregate"]["MRR@4"], 4),
        "fixed_bad_cases": sorted(set(before["bad_cases"]) - set(after["bad_cases"])),
        "new_bad_cases": sorted(set(after["bad_cases"]) - set(before["bad_cases"])),
        "changed_cases": changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare answer reports on the same frozen suite and index")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/answer-validation-comparison.json"))
    args = parser.parse_args()
    before = json.loads(args.baseline.read_text(encoding="utf-8"))
    after = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = compare(before, after)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("case_count", "retrieval_delta", "mrr_at_4_delta", "fixed_bad_cases", "new_bad_cases")}, ensure_ascii=False, indent=2))
    print(f"Comparison report: {args.output}")


if __name__ == "__main__":
    main()

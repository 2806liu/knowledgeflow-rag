"""Audit independent-source labels and summarize frozen first-run evidence reviews."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.config import settings
from app.rag.chunking import create_chunker
from app.rag.quality import low_quality_reason
from evaluation.answer_eval import load_suite
from evaluation.corpus import comparable_text, load_documents, load_manifest
from evaluation.evidence_review import check_review
from evaluation.stored_index import load_active_chunks


def audit(suite_path: Path, development_path: Path, db_path: Path) -> dict:
    cases, hashes = load_suite(suite_path, db_path)
    development, _ = load_suite(development_path, db_path)
    raw = json.loads(suite_path.read_text(encoding="utf-8"))
    if not raw.get("freeze"):
        raise ValueError("independent validation requires a freeze record")
    freeze = json.loads((suite_path.parent / raw["freeze"]).read_text(encoding="utf-8"))
    if hashlib.sha256(development_path.read_bytes()).hexdigest() != freeze["development_suite_sha256"]:
        raise ValueError("development suite differs from the one excluded at freeze time")
    if {c.case_id for c in cases} & {c.case_id for c in development}:
        raise ValueError("independent and development IDs overlap")
    if {comparable_text(c.query) for c in cases} & {comparable_text(c.query) for c in development}:
        raise ValueError("independent and development question texts overlap")
    new_ids = {d for c in cases for d in c.scope_document_ids}
    dev_ids = {d for c in development for d in c.scope_document_ids}
    if new_ids & dev_ids:
        raise ValueError("this source-holdout set contains a development source")
    # Different IDs can still represent the identical underlying PDF.
    # Avoid opening a Store (it initializes/migrates); source hashes are read-only.
    import sqlite3
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = dict(conn.execute("SELECT id,content_hash FROM documents"))
    finally:
        conn.close()
    if {rows[d] for d in new_ids} & {rows[d] for d in dev_ids}:
        raise ValueError("independent and development PDF hashes overlap")
    if freeze["positive_count"] != sum(bool(c.evidence) for c in cases) or freeze["negative_count"] != sum(not c.evidence for c in cases):
        raise ValueError("freeze count differs from current labels")
    return {"suite": raw["name"], "frozen_at": freeze["frozen_at"], "case_count": len(cases),
            "positive_count": freeze["positive_count"], "negative_count": freeze["negative_count"],
            "scoped_document_count": len(new_ids), "development_overlap": False,
            "label_author": freeze["label_author"], "independent_human_verified": freeze["independent_human_verified"],
            "source_sha256": hashes}


def source_diagnosis(suite_path: Path, db_path: Path, answers: dict, retrieval: dict) -> list[dict]:
    """Supplement rank diagnostics with index-time loss and split-anchor evidence.
    Recreated chunks are never installed and labels are never changed.
    """
    cases, _ = load_suite(suite_path, db_path)
    raw = json.loads(suite_path.read_text(encoding="utf-8"))
    documents = {}
    saved_strategies = {}
    for group in raw["groups"]:
        manifest = load_manifest(suite_path.parent / group["manifest"])
        documents.update(load_documents(db_path, manifest))
        saved_strategies.update({d["document_id"]: (d.get("chunk_strategy"), d.get("chunk_version"))
                                 for d in manifest["documents"]})
    rows, _, _ = load_active_chunks(db_path)
    chunker = create_chunker(settings.chunk_strategy, target_size=settings.chunk_target_size,
                             overlap=settings.chunk_overlap, min_size=settings.chunk_min_size, max_size=settings.chunk_max_size)
    chunks = {}
    for id, doc in documents.items():
        # Only diagnose with an identical saved strategy; otherwise do not
        # claim the current code reproduces the historical index-time filter.
        chunks[id] = (chunker.split_document(doc.content, list(doc.pages), document_id=id)
                      if saved_strategies[id] == (chunker.name, chunker.version) else [])
    output = []
    strategy = answers.get("runtime", {}).get("retrieval_strategy", "bm25_idf_v2")
    ranks = {c["id"]: c for c in retrieval["retrievers"][strategy]["cases"]}
    labeled = {c.case_id: c for c in cases}
    for case in answers["cases"]:
        if case.get("failure") != "retrieval_miss":
            continue
        reasons = []
        for e in labeled[case["id"]].evidence:
            raw_matches = [c for c in chunks[e.document_id] if c.metadata.get("page") == e.page
                           and comparable_text(e.anchor) in comparable_text(c.text)]
            filtered = sorted({low_quality_reason(c.text) for c in raw_matches if low_quality_reason(c.text)})
            stored = [row for row in rows if row["document_id"] == e.document_id
                      and json.loads(row["metadata"]).get("page") == e.page]
            if filtered:
                reasons.append({"reason": "index_time_quality_filter", "quality_reasons": filtered,
                                "note": "Reproduced using current matching recursive configuration; inspect the PDF before changing rules."})
            elif not any(comparable_text(e.anchor) in comparable_text(r["text"]) for r in stored):
                # This is a diagnostic hint, not a replacement relevance label.
                spans = [s.strip() for s in e.anchor.splitlines() if s.strip()]
                covered = [s for s in spans if any(comparable_text(s) in comparable_text(r["text"]) for r in stored)]
                reasons.append({"reason": "anchor_spans_multiple_chunks_or_text_loss", "anchor_line_count": len(spans),
                                "lines_present_in_stored_chunks": len(covered),
                                "note": "Full-anchor matching cannot award a hit to partial evidence; retain frozen label and inspect all necessary points."})
            else:
                reasons.append({"reason": "stored_evidence_not_in_final_hits",
                                "rank_diagnosis": ranks[case["id"]].get("diagnosis")})
        output.append({"id": case["id"], "query": case["query"], "diagnosis": reasons,
                       "answer_status": case.get("answer_status")})
    return output


def summarize(args) -> dict:
    checked = audit(args.suite, args.development_suite, args.data_dir / "studyflow.db")
    answers = json.loads(args.answers.read_text(encoding="utf-8"))
    retrieval = json.loads(args.retrieval.read_text(encoding="utf-8"))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    if answers["source_sha256"] != checked["source_sha256"]:
        raise ValueError("answer report does not correspond to this frozen suite")
    cases, _ = load_suite(args.suite, args.data_dir / "studyflow.db")
    if len(answers["cases"]) != len(cases) or {c["id"] for c in answers["cases"]} != {c.case_id for c in cases}:
        raise ValueError("answer report case IDs differ from the frozen suite")
    gold_by_id = {c.case_id: c for c in cases}
    for c in answers["cases"]:
        if c["query"] != gold_by_id[c["id"]].query:
            raise ValueError("answer report question differs from frozen question")
    frozen_at = checked["frozen_at"]
    if answers["generated_at"] < frozen_at or retrieval["generated_at"] < frozen_at:
        raise ValueError("evaluation predates freeze record")
    if answers["index_sha256"] != retrieval["index"]["index_sha256"]:
        raise ValueError("retrieval and answer evaluations used different indexes")
    if load_active_chunks(args.data_dir / "studyflow.db")[2] != answers["index_sha256"]:
        raise ValueError("current index changed; cannot diagnose saved results using a new index")
    for key, name in (("manifest_sha256", "corpus.json"), ("queries_sha256", "queries.jsonl")):
        if retrieval["dataset"][key] != hashlib.sha256((args.suite.parent / name).read_bytes()).hexdigest():
            raise ValueError("retrieval labels differ from frozen labels")
    if retrieval["parameters"]["scope_mode"] != "label":
        raise ValueError("answer and retrieval scope modes differ")
    notes_path = args.suite.parent / "REVIEW_NOTES.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    return {"audit": checked, "answer_metrics": answers["aggregate"],
            "analysis_notes": notes,
            "analysis_notes_sha256": hashlib.sha256(notes.encode("utf-8")).hexdigest(),
            "runtime_expected_from_local_config": answers.get("runtime"),
            "retrieval_comparison": {k: v["aggregate"] for k, v in retrieval["retrievers"].items()},
            "sentence_review": check_review(answers, hashlib.sha256(args.answers.read_bytes()).hexdigest(), review),
            "bad_cases": source_diagnosis(args.suite, args.data_dir / "studyflow.db", answers, retrieval),
            "policy": "This is a first-run validation. Do not tune on it and present a rerun as an independent test. "
                      "Assistant-created labels and semantic judgments remain provisional until separately human-confirmed."}


def markdown(report: dict) -> str:
    a, s, audit_result = report["answer_metrics"], report["sentence_review"], report["audit"]
    lines = ["# 独立验证与逐句证据评审", "", f"题集：{audit_result['suite']}", "",
             f"冻结题数：{audit_result['case_count']}；正例：{audit_result['positive_count']}；负例：{audit_result['negative_count']}；检索范围文档：{audit_result['scoped_document_count']}。",
             "先冻结标注再运行，来源与题目和开发集不重叠；助手标注与初审仍待独立人类确认。", "",
             "## 检索对照", "", "|策略|Recall@1|Recall@2|Recall@4|MRR@4|nDCG@4|", "|---|---:|---:|---:|---:|---:|"]
    for name, m in report["retrieval_comparison"].items():
        lines.append(f"|{name}|{m['Recall@1']:.4f}|{m['Recall@2']:.4f}|{m['Recall@4']:.4f}|{m['MRR@4']:.4f}|{m['nDCG@4']:.4f}|")
    lines += ["", "## 实际回答与逐句评审", "",
              f"完成请求：{a['completed']}/{a['total']}；请求错误：{a['request_errors']}；负例拒答率：{a['negative_refusal_rate']}。",
              f"句子单元：{s['sentence_count']}；已评审事实句：{s['reviewed_factual_sentences']}；逐句支持率：{s['sentence_support_rate_on_reviewed']}；引用链接支持率：{s['citation_support_rate_on_reviewed_links']}。",
              f"评审者身份：{s['reviewer']['kind']}；待评审句子：{s['pending_sentences']}；待评审整题：{s['pending_answers']}。", "",
              "|整题判定|数量|", "|---|---:|"]
    for verdict, count in s['answer_verdicts'].items():
        lines.append(f"|{verdict}|{count}|")
    lines += ["", "## 自动诊断（不能代替语义评审）", ""]
    for case in report['bad_cases']:
        lines.append(f"- {case['id']}：" + "；".join(d['reason'] for d in case['diagnosis']))
    lines += ["", "## 本轮初审解释", "", report.get('analysis_notes', '未附来源专属分析。'), "",
              "## 人工复核操作", "",
              "打开生成的 review-template.html，填写姓名和评审身份；逐题判整题正确性，逐句核对实际引用并粘贴原文。下载JSON后运行 evidence_review summarize。程序校验报告指纹和原文引用，不自动判语义。模板保持未判定，助手初审另存，便于独立复核。", ""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "summarize"))
    parser.add_argument("--suite", type=Path, default=Path("evaluation/independent-v1/suite.json"))
    parser.add_argument("--development-suite", type=Path, default=Path("evaluation/answer-validation-suite.json"))
    parser.add_argument("--data-dir", type=Path, default=Path(".studyflow"))
    parser.add_argument("--answers", type=Path, default=Path("evaluation/results/independent-v1-answers.json"))
    parser.add_argument("--retrieval", type=Path, default=Path("evaluation/results/independent-v1-retrieval.json"))
    parser.add_argument("--review", type=Path, default=Path("evaluation/results/independent-v1-assistant-review.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/independent-v1-summary.json"))
    args = parser.parse_args()
    report = audit(args.suite,args.development_suite,args.data_dir/"studyflow.db") if args.command=="audit" else summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if args.command == "summarize":
        args.output.with_suffix('.md').write_text(markdown(report), encoding="utf-8")
    print(f"{args.command} complete: {args.output}")


if __name__ == "__main__":
    main()

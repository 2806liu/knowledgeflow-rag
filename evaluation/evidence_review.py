"""Human/assistant sentence review bound to immutable, saved /chat outputs.

No network calls. Syntax and quote checks are automatic; semantic verdicts
are explicit reviewer labels, never inferred from keyword or anchor overlap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.corpus import comparable_text

SENTENCE_VERDICTS = {"supported", "unsupported", "contradicted", "uncertain", "non_factual"}
ANSWER_VERDICTS = {"correct", "partially_correct", "incorrect", "correct_refusal", "incorrect_refusal", "uncertain"}
CITATION_VERDICTS = {"supported", "irrelevant", "contradicted", "uncertain"}
EXTRACTOR_VERSION = "sentence-v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def split_sentences(answer: str) -> list[str]:
    """Preserve citation suffixes and decimals. These are review units, not
    guaranteed atomic claims; the reviewer must check EVERY claim in a unit.
    Citations on a separate line are attached to the preceding sentence.
    """
    result: list[str] = []
    # Chinese punctuation and English punctuation before whitespace/end.
    boundary = re.compile(r"[。！？!?]+|(?<!\d)\.(?=\s|$)")
    for line in answer.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r"(?:\[\d+\]\s*)+", line) and result:
            result[-1] += line
            continue
        start = 0
        while start < len(line):
            match = boundary.search(line, start)
            if not match:
                remainder = line[start:].strip()
                if remainder:
                    result.append(remainder)
                break
            end = match.end()
            suffix = re.match(r"(?:\s*\[\d+\])+", line[end:])
            if suffix:
                end += suffix.end()
            result.append(line[start:end].strip())
            start = end
    return result


def worksheet(report: dict, report_hash: str) -> dict:
    if report.get("status") == "partial":
        raise ValueError("cannot review a partial evaluation as a final result")
    answers, units = [], []
    for case in report["cases"]:
        answer = case.get("answer") or ""
        answers.append({"case_id": case["id"], "query": case["query"], "answer": answer,
                        "answer_sha256": digest(answer), "expected": case["expected"],
                        "answer_status": case.get("answer_status"), "request_error": case.get("error"),
                        "reference_answer": case.get("reference_answer", ""),
                        "required_points": case.get("required_points", []),
                        "gold_evidence": case.get("gold_evidence", []),
                        "verdict": None, "rationale": ""})
        available = {c["citation_number"]: c for c in case.get("citations", [])}
        for index, text in enumerate(split_sentences(answer), 1):
            numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
            evidence = [{"citation_number": n, **available[n]} for n in numbers if n in available]
            fixed = {"case_id": case["id"], "sentence_number": index, "text": text,
                     "citation_numbers": numbers, "cited_evidence": evidence,
                     "missing_citation_numbers": [n for n in numbers if n not in available]}
            units.append({"unit_id": digest(fixed), **fixed, "verdict": None,
                          "citation_reviews": [], "rationale": ""})
    return {"schema_version": 1, "extractor_version": EXTRACTOR_VERSION,
            "report_sha256": report_hash,
            "reviewer": {"name": "", "kind": "pending"},
            "instructions": "Check every factual claim in each sentence against ONLY its cited text. "
                            "For supported decisions record exact supporting quotes. Correctness and "
                            "completeness against the source gold answer are separate answer decisions. "
                            "Do not use gold anchors as a substitute for model-cited evidence.",
            "answers": answers, "sentences": units}


def check_review(report: dict, report_hash: str, review: dict) -> dict:
    expected = worksheet(report, report_hash)
    if review.get("report_sha256") != report_hash or review.get("extractor_version") != EXTRACTOR_VERSION:
        raise ValueError("stale review: report hash or extractor version changed")
    reviewer = review.get("reviewer", {})
    if reviewer.get("kind") not in {"pending", "assistant", "human", "llm_judge"}:
        raise ValueError("unknown reviewer kind")
    if reviewer.get("kind") != "pending" and not reviewer.get("name", "").strip():
        raise ValueError("completed review requires reviewer name")
    for section, key, editable in (("answers", "case_id", {"verdict", "rationale"}),
                                  ("sentences", "unit_id", {"verdict", "rationale", "citation_reviews"})):
        original = {item[key]: item for item in expected[section]}
        items = review.get(section, [])
        if len(items) != len(original) or {item[key] for item in items} != set(original):
            raise ValueError(f"missing, duplicate, or extra {section}")
        for item in items:
            # Reject edits to the saved question, answer, or evidence, even if
            # someone manually keeps the original unit_id/report hash.
            if {k: v for k, v in item.items() if k not in editable} != {
                k: v for k, v in original[item[key]].items() if k not in editable
            }:
                raise ValueError(f"immutable {section} evidence changed: {item[key]}")
            verdict = item.get("verdict")
            allowed = ANSWER_VERDICTS if section == "answers" else SENTENCE_VERDICTS
            if verdict is not None and verdict not in allowed:
                raise ValueError(f"invalid {section} verdict: {verdict}")
            if verdict is not None and not item.get("rationale", "").strip():
                raise ValueError("a verdict requires an explanation")
            if section == "answers":
                if item.get("request_error") and verdict is not None:
                    raise ValueError("a failed request has no answer to review")
                if verdict == "correct_refusal" and item["expected"] != "refuse":
                    raise ValueError("positive case cannot be marked correct_refusal")
                if verdict == "incorrect_refusal" and item["expected"] != "answer_with_labeled_evidence":
                    raise ValueError("negative case cannot be marked incorrect_refusal")
                if verdict == "correct" and item["expected"] == "refuse":
                    raise ValueError("negative case must be assessed as refusal or incorrect")
                continue
            evidence = {c["citation_number"]: c for c in item["cited_evidence"]}
            links = item.get("citation_reviews", [])
            numbers = [link["citation_number"] for link in links]
            if len(numbers) != len(set(numbers)):
                raise ValueError("duplicate citation reviews")
            for link in links:
                n = link["citation_number"]
                if n not in evidence or link["verdict"] not in CITATION_VERDICTS:
                    raise ValueError("citation review references missing evidence or invalid verdict")
                quotes = link.get("quotes", [])
                if link["verdict"] in {"supported", "contradicted"} and not quotes:
                    raise ValueError("supported/contradicted citation requires exact source quote")
                for quote in quotes:
                    if not comparable_text(quote) or comparable_text(quote) not in comparable_text(evidence[n]["snippet"]):
                        raise ValueError("support quote is absent from cited snippet")
            if verdict and verdict != "non_factual" and set(numbers) != set(evidence):
                raise ValueError("each available citation needs its own relevance review")
            if verdict == "supported" and (item["missing_citation_numbers"] or
                                           not any(link["verdict"] == "supported" for link in links)):
                raise ValueError("supported sentence needs valid cited supporting text")
    labeled = [item for section in ("answers", "sentences") for item in review[section] if item.get("verdict")]
    if labeled and reviewer.get("kind") == "pending":
        raise ValueError("review decisions must identify their reviewer")
    units = review["sentences"]
    factual = [u for u in units if u.get("verdict") not in {None, "non_factual"}]
    links = [link for u in factual for link in u["citation_reviews"]]
    answer_counts = Counter(a.get("verdict") or "pending" for a in review["answers"])
    count = Counter(u.get("verdict") or "pending" for u in units)
    def rate(num, den):
        return round(num / den, 4) if den else None
    return {"report_sha256": report_hash, "reviewer": reviewer,
            "independent_human_confirmed": reviewer.get("kind") == "human",
            "answer_count": len(review["answers"]), "answer_verdicts": dict(answer_counts),
            "sentence_count": len(units), "sentence_verdicts": dict(count),
            "reviewed_factual_sentences": len(factual),
            "sentence_support_rate_on_reviewed": rate(count["supported"], len(factual)),
            "valid_citation_coverage_on_reviewed": rate(sum(bool(u["cited_evidence"]) and not u["missing_citation_numbers"] for u in factual), len(factual)),
            "citation_support_rate_on_reviewed_links": rate(sum(l["verdict"] == "supported" for l in links), len(links)),
            "reviewed_citation_links": len(links),
            "pending_sentences": count["pending"], "pending_answers": answer_counts["pending"],
            "measurement_limits": ["Exact quote containment validates review provenance, not semantic entailment.",
                                   "Sentences may contain several claims; reviewers must check all claims.",
                                   "Assistant or LLM reviews are provisional; human reviewer status is explicit, not inferred.",
                                   "Support rate measures cited evidence fidelity, not source truth or complete answers."]}


def render_markdown(review: dict) -> str:
    lines = ["# 逐句证据评审工作表", "", "判定来自评审者；原文匹配不会自动证明语义正确。",
             "supported=全部事实有引用支持；unsupported=引用不足；contradicted=引用反驳；uncertain=无法确定；non_factual=标题或纯拒答。", ""]
    for answer in review["answers"]:
        lines.extend([f"## {answer['case_id']}：{answer['query']}", "", f"参考答案：{answer['reference_answer']}", "",
                      f"实际回答：{answer['answer']}", "", f"整题判定：{answer['verdict'] or 'pending'}；{answer['rationale']}", ""])
        for u in (u for u in review["sentences"] if u["case_id"] == answer["case_id"]):
            lines.extend([f"### 句 {u['sentence_number']} ({u['unit_id'][:12]})", "", u["text"], "",
                          f"判定：{u['verdict'] or 'pending'}；{u['rationale']}", ""])
            for c in u["cited_evidence"]:
                lines.extend([f"[{c['citation_number']}] {c['filename']}，PDF实际页 {c['page']}，chunk {c['chunk_id']}", "",
                              "> " + c["snippet"].replace("\n", "\n> "), ""])
    return "\n".join(lines) + "\n"


def render_html(review: dict, download_name: str = "evidence-review-reviewed.json") -> str:
    """Local review form; edited verdicts are downloaded as JSON for validation."""
    payload = json.dumps(review, ensure_ascii=False).replace("<", "\\u003c")
    return r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>KnowledgeFlow 逐句证据评审</title><style>
body{font-family:system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#172c3c;background:#f5f7fa}
header,article{background:white;border:1px solid #dbe2e8;padding:20px;margin:16px 0;border-radius:8px}
textarea{width:98%;min-height:55px;margin:6px 0}select,input,button{padding:8px;margin:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f3f6;padding:14px}
.sentence{border-top:2px solid #dbe2e8;margin-top:22px;padding-top:12px}.note{color:#66531a}button{cursor:pointer;background:#e0edf9;border:1px solid #96b6d5}summary{cursor:pointer}
</style><header><h1>逐句证据评审</h1><p>整题是否答对与每句是否有引用支持，分别判断。每句可能包含多个事实，需全部检查。引用相关性逐条判定，并粘贴支持或反驳的原文。标题和纯拒答标为 non_factual。</p>
<p class="note">此页面只保存人工填写的判定，不会自动证明语义正确。点击下载后运行 summarize 校验；不完整的记录仍显示 pending。</p>
<label>评审者姓名<input id="name"></label><label>身份<select id="kind"><option>pending</option><option>human</option><option>assistant</option><option>llm_judge</option></select></label><button id="save">下载评审 JSON</button><span id="progress"></span></header><main id="main"></main>
<script>
const data=PAYLOAD;
function el(tag,text,parent){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(parent)parent.append(e);return e}
function choice(parent,options,value,change){const s=el('select',undefined,parent);for(const v of ['pending',...options]){const o=el('option',v,s);o.value=v} s.value=value||'pending';s.onchange=()=>change(s.value==='pending'?null:s.value);return s}
function area(parent,value,change,placeholder){const t=el('textarea',undefined,parent);t.value=value||'';t.placeholder=placeholder;t.oninput=()=>change(t.value);return t}
function progress(){document.getElementById('progress').textContent=`整题 ${data.answers.filter(a=>a.verdict).length}/${data.answers.length}；句子 ${data.sentences.filter(s=>s.verdict).length}/${data.sentences.length}`}
for(const a of data.answers){const box=el('article',undefined,document.getElementById('main'));el('h2',a.case_id+'：'+a.query,box);el('p','参考答案：'+a.reference_answer,box);el('pre',a.answer||('请求错误：'+a.request_error),box);
const gold=el('details',undefined,box);el('summary','标注原文（用于整题正确性与完整性）',gold);for(const g of a.gold_evidence)el('pre',`文档 ${g.document_id}；实际页 ${g.page}\n${g.anchor}`,gold);
choice(box,['correct','partially_correct','incorrect','correct_refusal','incorrect_refusal','uncertain'],a.verdict,v=>{a.verdict=v;progress()});area(box,a.rationale,v=>a.rationale=v,'整题判定理由：是否回应问题，是否遗漏必要信息，是否正确拒答');
for(const s of data.sentences.filter(s=>s.case_id===a.case_id)){const block=el('section',undefined,box);block.className='sentence';el('h3',`句 ${s.sentence_number}：${s.text}`,block);
choice(block,['supported','unsupported','contradicted','uncertain','non_factual'],s.verdict,v=>{s.verdict=v;progress()});area(block,s.rationale,v=>s.rationale=v,'本句全部事实是否由下面的实际引用支持？');
if(s.missing_citation_numbers.length)el('p','不存在的引用编号：'+s.missing_citation_numbers.join(','),block);
for(const c of s.cited_evidence){const details=el('details',undefined,block);details.open=true;el('summary',`[${c.citation_number}] ${c.filename}；PDF实际页 ${c.page}；chunk ${c.chunk_id}`,details);el('pre',c.snippet,details);
let link=s.citation_reviews.find(l=>l.citation_number===c.citation_number);function ensure(){if(!link){link={citation_number:c.citation_number,verdict:'uncertain',quotes:[]};s.citation_reviews.push(link)}return link}
choice(details,['supported','irrelevant','contradicted','uncertain'],link&&link.verdict,v=>{if(v)ensure().verdict=v;else{s.citation_reviews=s.citation_reviews.filter(l=>l.citation_number!==c.citation_number);link=null}});
area(details,link&&link.quotes.join('\n---\n'),v=>ensure().quotes=v.split('\n---\n').map(x=>x.trim()).filter(Boolean),'粘贴实际引用中的原文；多段原文之间用单独一行 --- 分隔');}}
}
document.getElementById('name').value=data.reviewer.name;document.getElementById('kind').value=data.reviewer.kind;
document.getElementById('save').onclick=()=>{data.reviewer={name:document.getElementById('name').value,kind:document.getElementById('kind').value};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json;charset=utf-8'}));const a=el('a');a.href=url;a.download=DOWNLOAD_NAME;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};progress();
</script></html>'''.replace('PAYLOAD', payload).replace('DOWNLOAD_NAME', json.dumps(download_name).replace('<', '\\u003c'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "summarize"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    report_hash = hashlib.sha256(args.report.read_bytes()).hexdigest()
    if args.command == "export":
        if args.review.exists():
            raise ValueError("review already exists; export to a new path to preserve decisions")
        review = worksheet(report, report_hash)
        args.review.parent.mkdir(parents=True, exist_ok=True)
        args.review.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        review = json.loads(args.review.read_text(encoding="utf-8"))
        summary = check_review(report, report_hash, review)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    args.review.with_suffix(".md").write_text(render_markdown(review), encoding="utf-8")
    download_name = args.review.name.replace("review-template", "human-review")
    args.review.with_suffix(".html").write_text(render_html(review, download_name), encoding="utf-8")


if __name__ == "__main__":
    main()

"""评测计数模块回归：防止缩小分母、遗漏摘要成本、引用错配及伪造Token节省。"""

import unittest

from evaluation.task_memory_benchmark import (
    check_response, completion_metrics, digest, probe_prompt, reduction, token_metrics, usage_total,
)
from evaluation.task_memory_suite import HISTORY, PROBES, SOURCES, TASKS
from evaluation.benchmark_review import apply_review, template


def call(inputs, outputs=10, **changes):
    """注入供应商usage形状，测试统计规则；不代表真实模型性能。"""
    trace = dict(status="ok", attempts=1, usage=dict(input_tokens=inputs,
        output_tokens=outputs, total_tokens=inputs + outputs))
    trace.update(changes)
    return dict(trace=trace, checks=dict(passed=True), all_history_retained=True)


class TaskMemoryBenchmarkTests(unittest.TestCase):
    def test_frozen_suite_has_unique_cases_and_separate_negatives(self):
        """固定18个正例、2个负例、8个探针，避免运行后筛掉失败题。"""
        self.assertEqual(len(TASKS), 20)
        self.assertEqual(len({t["id"] for t in TASKS}), 20)
        self.assertEqual(sum(not t["steps"][0]["refusal"] for t in TASKS), 18)
        self.assertEqual(len(PROBES), 8)
        self.assertEqual(len(HISTORY), 16)
        self.assertEqual(len(SOURCES), 4)
        self.assertEqual(digest(TASKS), digest(TASKS))

    def test_task_denominator_keeps_failed_and_unreviewed_cases(self):
        """失败任务留在分母；负例拒答不计入自主完成率。"""
        cases = [dict(negative_control=False, automatic_passed=True, review=dict(verdict="pass")),
                 dict(negative_control=False, automatic_passed=False, review=dict(verdict="fail")),
                 dict(negative_control=True, automatic_passed=True)]
        result = completion_metrics(cases)
        self.assertEqual(result["task_count"], 2)
        self.assertEqual(result["reviewed_completion_rate_pct"], 50)
        self.assertEqual(result["correct_refusals"], 1)
        cases[1]["review"] = dict(verdict="pending")
        self.assertIsNone(completion_metrics(cases)["reviewed_completion_rate_pct"])

    def test_pending_semantic_review_cannot_become_solved_rate(self):
        """规则全过仍须原文语义核验；避免将结构通过等同解决。"""
        result = completion_metrics([dict(negative_control=False, automatic_passed=True)])
        self.assertEqual(result["rule_pass_rate_pct"], 100)
        self.assertFalse(result["semantic_review_complete"])
        self.assertIsNone(result["reviewed_solved"])

    def test_token_reduction_uses_sum_and_includes_compaction(self):
        """100与900的不同基数按总量比较，摘要额外输入也计入。"""
        pairs = [dict(same_question_evidence=True, baseline=call(100), managed=call(50)),
                 dict(same_question_evidence=True, baseline=call(900), managed=call(450))]
        result = token_metrics(pairs, [call(100)])
        self.assertEqual(result["answer_input_reduction_pct"], 50)
        self.assertEqual(result["managed_plus_compaction_input_tokens"], 600)
        self.assertEqual(result["amortized_input_reduction_pct"], 40)
        self.assertEqual(result["baseline_total_tokens"], 1020)
        self.assertEqual(result["managed_plus_compaction_total_tokens"], 630)

    def test_missing_usage_and_retries_do_not_look_like_savings(self):
        """任一调用缺usage或重试成本未知，都不公布百分比。"""
        self.assertFalse(usage_total([call(100, usage=None)])["complete"])
        self.assertFalse(usage_total([call(100, attempts=2)])["complete"])
        pair = dict(same_question_evidence=True, baseline=call(100), managed=call(10, usage=None))
        result = token_metrics([pair], [call(10)])
        self.assertIsNone(result["answer_input_reduction_pct"])
        self.assertIsNone(result["amortized_input_reduction_pct"])

    def test_truncated_full_history_is_not_a_valid_baseline(self):
        """完整历史被预算裁剪时，不再冒充完整上下文基线。"""
        base = call(100)
        base["all_history_retained"] = False
        result = token_metrics([dict(same_question_evidence=True, baseline=base, managed=call(20))], [call(10)])
        self.assertIsNone(result["answer_input_reduction_pct"])

    def test_no_evidence_mismatch_can_enter_token_comparison(self):
        """题目或原文改变时，不能把差异算成记忆收益。"""
        result = token_metrics([dict(same_question_evidence=False, baseline=call(100), managed=call(20))], [call(10)])
        self.assertIsNone(result["amortized_input_reduction_pct"])

    def test_negative_savings_and_zero_baseline_are_explicit(self):
        """成本增加保留负值；零基线没有合法下降百分比。"""
        self.assertEqual(reduction(100, 150), -50)
        self.assertIsNone(reduction(0, 10))
        self.assertIsNone(reduction(None, 10))

    def example(self):
        """构造一个带明确原文与方面标注的最小回答。"""
        chunks = {"chunk": dict(document_id="doc", text="初始借期30天。")}
        response = dict(answer_status="answered", unresolved_aspects=[], citations=[
            dict(citation_number=1, chunk_id="chunk", document_id="doc", snippet="初始借期30天。")],
            answer_sections=[dict(source_type="document", content="初始借期30天。[1]", citation_numbers=[1])])
        spec = dict(scope=["library"], facts=[dict(document="library", patterns=[r"30天"])])
        return response, spec, {"library": "doc"}, chunks

    def test_source_and_answer_must_both_support_each_fact(self):
        """答案命中关键词但原文不含该事实，仍应失败。"""
        response, spec, docs, chunks = self.example()
        self.assertTrue(check_response(response, spec, docs, chunks)["passed"])
        chunks["chunk"]["text"] = "本段未规定借期。"
        response["citations"][0]["snippet"] = chunks["chunk"]["text"]
        result = check_response(response, spec, docs, chunks)
        self.assertFalse(result["passed"])
        self.assertIn("fact_1_missing_or_unsupported", result["reasons"])

    def test_citation_document_and_current_chunk_are_checked(self):
        """换成其他文档或旧chunk内容都不能通过来源核验。"""
        response, spec, docs, chunks = self.example()
        response["citations"][0]["document_id"] = "wrong"
        self.assertIn("citation_not_current_source", check_response(response, spec, docs, chunks)["reasons"])

    def test_partial_or_uncited_answer_is_not_completed(self):
        """部分回答、未解决方面和不匹配引用均不能算解决。"""
        response, spec, docs, chunks = self.example()
        response["answer_status"] = "partial_answer"
        response["answer_sections"][0]["content"] = "初始借期30天。[2]"
        result = check_response(response, spec, docs, chunks)
        self.assertIn("task_not_fully_answered", result["reasons"])
        self.assertIn("invalid_citation_markers", result["reasons"])

    def test_refusal_does_not_pass_when_it_invents_a_claim(self):
        """缺证据任务必须说明缺口；模型知识填补也视为失败。"""
        response, spec, docs, chunks = self.example()
        spec["refusal"] = True
        response["answer_status"] = "insufficient_evidence"
        self.assertFalse(check_response(response, spec, docs, chunks)["passed"])
        response.update(citations=[], answer_sections=[dict(source_type="limitation", content="缺少依据。", citation_numbers=[])])
        self.assertTrue(check_response(response, spec, docs, chunks)["passed"])

    def review_report(self):
        """包含完整任务和配对回答的最小审阅对象。"""
        case = dict(id="T", negative_control=False, automatic_passed=True,
                    steps=[dict(spec={}, response={}, checks=dict(passed=True))])
        arm = dict(response={}, checks=dict(passed=True))
        return dict(suite_sha256="frozen", cases=[case], pairs=[dict(id="P", baseline=arm, managed=arm)],
                    token_metrics={})

    def test_review_must_be_complete_and_bind_original_answers(self):
        """未填写、漏题或换答案后的旧审阅都被拒绝。"""
        report = self.review_report()
        review = template(report)
        review.update(reviewer="test reviewer", reviewer_type="assistant")
        with self.assertRaises(ValueError):
            apply_review(report, review)
        review["tasks"]["T"] = dict(verdict="pass", notes="逐句核验。")
        review["pairs"]["P"] = {arm: dict(verdict="pass", notes="原文支持。") for arm in ("baseline", "managed")}
        report["cases"][0]["steps"][0]["response"] = dict(answer="changed")
        with self.assertRaises(ValueError):
            apply_review(report, review)

    def test_assistant_review_does_not_become_independent_human_review(self):
        """审阅身份进入成果报告，助手核验不能伪装人工盲评。"""
        report = self.review_report()
        review = template(report)
        review.update(reviewer="test assistant", reviewer_type="assistant")
        review["tasks"]["T"] = dict(verdict="pass", notes="按原文核验。")
        review["pairs"]["P"] = {arm: dict(verdict="pass", notes="两组答案都核验。") for arm in ("baseline", "managed")}
        result = apply_review(report, review)
        self.assertFalse(result["semantic_review"]["independent_human_review"])
        self.assertEqual(result["completion_metrics"]["reviewed_completion_rate_pct"], 100)
        self.assertTrue(result["token_metrics"]["quality_gate_passed"])

    def test_review_cannot_override_failed_rules_as_success(self):
        """即使审阅写通过，已有机械失败也保留为失败。"""
        report = self.review_report()
        report["cases"][0]["automatic_passed"] = False
        review = template(report)
        review.update(reviewer="test assistant", reviewer_type="assistant")
        review["tasks"]["T"] = dict(verdict="pass", notes="规则仍然失败。")
        review["pairs"]["P"] = {arm: dict(verdict="pass", notes="按原文核验。") for arm in ("baseline", "managed")}
        self.assertEqual(apply_review(report, review)["completion_metrics"]["reviewed_solved"], 0)

    def test_same_document_multiple_sections_can_cover_multiple_aspects(self):
        """条件与期限分段回答时，各段有对应原文即可，不强迫合为一段。"""
        response, spec, docs, chunks = self.example()
        spec["facts"][0]["patterns"].append("未预约")
        chunks["chunk"]["text"] = "初始借期30天；未预约可以续借。"
        response["citations"][0]["snippet"] = chunks["chunk"]["text"]
        response["answer_sections"].append(dict(source_type="document", content="未预约可以续借。[1]", citation_numbers=[1]))
        self.assertTrue(check_response(response, spec, docs, chunks)["passed"])

    def test_pdf_next_to_chinese_is_not_a_word_boundary_failure(self):
        """资料中的PDF格式没有英文空格，仍应匹配同一文件格式。"""
        import re
        spec = next(t for t in TASKS if t["id"] == "T06")["steps"][0]
        self.assertTrue(all(re.search(p, "只接收PDF格式") for p in spec["facts"][0]["patterns"]))

    def test_overview_probe_uses_original_sources_not_overview_text(self):
        """概述路线可建立探针，但旧概述文本不能变成原文证据。"""
        import json
        case = dict(steps=[dict(spec=dict(question="研究重点？"), response=dict(answer="不应当作证据的旧摘要", citations=[
            dict(citation_number=1, filename="public.md", page=1, document_id="doc", snippet="真正的公开原文")]))])
        payload = json.loads(probe_prompt(case)[-1]["content"])
        self.assertEqual(payload["evidence"][0]["text"], "真正的公开原文")
        self.assertNotIn("不应当作证据的旧摘要", json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()

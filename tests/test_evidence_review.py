import copy
import unittest

from evaluation.evidence_review import check_review, render_html, split_sentences, worksheet


class EvidenceReviewTests(unittest.TestCase):
    def setUp(self):
        self.report = {"cases": [{"id": "q1", "query": "需要多久？", "expected": "answer_with_labeled_evidence",
            "answer": "需2.5小时[1]。", "answer_status": "answered",
            "citations": [{"citation_number": 1, "document_id": "d1", "chunk_id": "c1",
                           "filename": "x.pdf", "page": 2, "snippet": "要求完成时间为2.5小时。"}]}]}
        self.review = worksheet(self.report, "hash")
        self.review["reviewer"] = {"name": "fixture", "kind": "assistant"}

    def supported(self):
        unit = self.review["sentences"][0]
        unit.update(verdict="supported", rationale="时长由引用直接支持。",
                    citation_reviews=[{"citation_number": 1, "verdict": "supported", "quotes": ["完成时间为2.5小时"]}])

    def test_segmentation_keeps_decimals_suffixes_and_bullets(self):
        self.assertEqual(split_sentences("需2.5小时。[1][2] 下一步。[3]\n- 完成。\n[4]"),
                         ["需2.5小时。[1][2]", "下一步。[3]", "- 完成。[4]"])

    def test_pending_does_not_become_grounded(self):
        result = check_review(self.report, "hash", self.review)
        self.assertEqual(result["pending_sentences"], 1)
        self.assertIsNone(result["sentence_support_rate_on_reviewed"])

    def test_supported_metrics_and_reviewer_provenance(self):
        self.supported()
        result = check_review(self.report, "hash", self.review)
        self.assertEqual(result["sentence_support_rate_on_reviewed"], 1)
        self.assertFalse(result["independent_human_confirmed"])

    def test_quote_must_be_in_actual_citation(self):
        self.supported()
        self.review["sentences"][0]["citation_reviews"][0]["quotes"] = ["编造证据"]
        with self.assertRaisesRegex(ValueError, "absent"):
            check_review(self.report, "hash", self.review)

    def test_cannot_edit_evidence_and_keep_hash(self):
        self.review["sentences"][0]["cited_evidence"][0]["snippet"] = "伪造"
        with self.assertRaisesRegex(ValueError, "immutable"):
            check_review(self.report, "hash", self.review)

    def test_stale_report_rejected(self):
        with self.assertRaisesRegex(ValueError, "stale"):
            check_review(self.report, "newhash", self.review)

    def test_missing_citation_cannot_be_supported(self):
        report = copy.deepcopy(self.report)
        report["cases"][0]["answer"] = "需要2.5小时[9]。"
        review = worksheet(report, "hash")
        review["reviewer"] = self.review["reviewer"]
        review["sentences"][0].update(verdict="supported", rationale="不能冒用引用")
        with self.assertRaisesRegex(ValueError, "valid cited"):
            check_review(report, "hash", review)

    def test_coverage_is_not_semantic_support(self):
        self.review["sentences"][0].update(verdict="unsupported", rationale="引用不支持实际结论",
            citation_reviews=[{"citation_number": 1, "verdict": "irrelevant", "quotes": []}])
        result = check_review(self.report, "hash", self.review)
        self.assertEqual(result["valid_citation_coverage_on_reviewed"], 1)
        self.assertEqual(result["sentence_support_rate_on_reviewed"], 0)

    def test_html_preserves_javascript_quote_separator(self):
        page = render_html(self.review)
        self.assertIn("quotes.join('\\n---\\n')", page)
        self.assertNotIn("quotes.join('\n", page)

    def test_source_text_cannot_close_embedded_script(self):
        self.review["answers"][0]["answer"] = "</script><script>unsafe</script>"
        page = render_html(self.review)
        self.assertEqual(page.count("</script>"), 1)
        self.assertIn("\\u003c/script>", page)


if __name__ == "__main__":
    unittest.main()

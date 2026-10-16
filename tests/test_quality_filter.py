"""Regressions for evidence lost by overbroad index-time quality filtering."""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from dataclasses import replace
from pathlib import Path

from app.config import settings
from app.rag.contracts import DocumentPage
from app.rag.quality import assess_chunk_quality, low_quality_reason
from app.rag.retrieval import BM25Retriever
from app.services import KnowledgeFlow
from app.store import Store


CHOICE_TEXT = (
    "20. 当工程师发现企业的工程活动危害环境、社会和公众人身安全时，应及时反映或揭发。这属于哪种责任？（B）\n"
    "A.职业伦理责任 B.社会伦理责任 C.环境伦理责任 D.家庭伦理责任\n"
    "21. 与公众安全有关的另一道选择题？（A） A.主动报告 B.保持沉默 C.忽略风险 D.延后处理"
)
CONTENTS_TEXT = "目录 第一章 工程伦理基础...........3 第二章 工程风险与公众安全...........18"


class QualityFilterTests(unittest.TestCase):
    def test_multiple_choice_is_not_a_table_of_contents(self) -> None:
        quality = assess_chunk_quality(CHOICE_TEXT)
        self.assertIsNone(quality.exclude_reason)
        self.assertIn("dot_heavy_unconfirmed", quality.flags)

    def test_choice_without_sentence_punctuation_is_not_a_heading(self) -> None:
        text = "20. 以下何者属于社会伦理责任 A.个人偏好 B.危害公众时及时揭发 C.只考虑企业收益 D.完全回避沟通"
        self.assertIsNone(low_quality_reason(text))

    def test_clear_contents_entries_still_filtered(self) -> None:
        self.assertEqual(low_quality_reason(CONTENTS_TEXT), "table_of_contents")

    def test_prose_and_single_page_leader_are_kept(self) -> None:
        prose = "关于目录第3页的说明，A.与B.两个选项各有不同的适用条件，应结合完整证据判断。"
        self.assertIsNone(low_quality_reason(prose))
        ambiguous = "参考材料...........3 其中的内容详细讨论了企业责任、公众安全和风险报告。"
        self.assertIsNone(low_quality_reason(ambiguous))

    def test_english_paper_sentences_are_not_headings(self) -> None:
        """英文摘要即使只含句号也应保留；真正无句子的长标题仍过滤。"""
        abstract = ("This paper studies task priority kinematic control of redundant manipulators. "
                    "The research objective is to organize multiple motion tasks by priority.")
        heading = "RESEARCH FOCUS AND KINEMATIC CONTROL OF REDUNDANT MANIPULATORS UNDER MULTIPLE TASK PRIORITY CONSTRAINTS"
        self.assertIsNone(low_quality_reason(abstract))
        self.assertEqual(low_quality_reason(heading), "heading_only")

    def test_index_keeps_choice_chunk_and_records_quality_flag(self) -> None:
        from app import main

        test_root = Path.cwd() / "tests" / ".tmp"
        test_root.mkdir(parents=True, exist_ok=True)
        temp_path = test_root / str(uuid.uuid4())
        temp_path.mkdir()
        store = Store(temp_path)
        old_store, old_flow = main.store, main.flow
        main.store = store
        main.flow = KnowledgeFlow(store, replace(settings, vector_enabled=False, llm_mode="mock"))
        try:
            document_id = str(uuid.uuid4())
            store.add_document(
                document_id,
                "伦理复习题.pdf",
                f"{CHOICE_TEXT}\n{CONTENTS_TEXT}",
                pages=(
                    DocumentPage(page_number=3, text=CHOICE_TEXT),
                    DocumentPage(page_number=4, text=CONTENTS_TEXT),
                ),
            )
            self.assertGreater(main._index_document(document_id), 0)
            rows = store.chunks((document_id,))
            matches = [row for row in rows if "社会伦理责任" in row["text"]]
            self.assertTrue(matches)
            self.assertEqual(json.loads(matches[0]["metadata"])["page"], 3)
            self.assertTrue(json.loads(matches[0]["metadata"])["is_retrievable"])
            self.assertIn("dot_heavy_unconfirmed", json.loads(matches[0]["metadata"])["quality_flags"])
            self.assertFalse(any("目录 第一章" in row["text"] for row in rows))
            self.assertTrue(BM25Retriever(store).search("工程师危害公众及时揭发", document_ids=(document_id,)))
        finally:
            main.store, main.flow = old_store, old_flow
            store.conn.close()
            if not temp_path.resolve().is_relative_to(Path.cwd().resolve()):
                raise RuntimeError("refusing to clean a test directory outside the workspace")
            shutil.rmtree(temp_path)


if __name__ == "__main__":
    unittest.main()

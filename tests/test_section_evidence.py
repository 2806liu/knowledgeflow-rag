"""Semantic list boundaries, deterministic sections and bounded source evidence."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

from fastapi.testclient import TestClient
from app.config import settings
from app.rag.chunking import RecursiveChunker
from app.rag.contracts import DocumentPage, RetrievalHit
from app.rag.evidence_expansion import EvidenceExpander, token_counter
from app.services import KnowledgeFlow
from app.store import Store


RESPONSIBILITIES = (
    "9. 工程实践中环境相关伦理责任\n"
    "1. 底线责任（守法合规）严格执行环评、排污许可、生态保护红线等法律法规。\n"
    "2. 预防责任（风险防范）采用最佳可行技术（BAT）减少排放，实施清洁生产，建立环境应急预案。\n"
    "3. 恢复责任（损害修复）对已造成的生态破坏（如水土流失、植被破坏）进行补\n偿性恢复或异地修复。\n"
    "4. 超越责任（可持续发展）主动选择低碳循环工艺，考虑全生命周期环境影响，推动绿色设计。\n"
    "10. 工程师的职业特征\n专业职责需要诚信和独立判断。"
)


class SectionChunkingTests(unittest.TestCase):
    def split(self, text, **kwargs):
        return RecursiveChunker(**kwargs).split_document(text, (DocumentPage(3, text),), document_id="a")

    def test_wrapped_third_item_keeps_four_responsibilities_and_parent_title(self):
        chunks = self.split(RESPONSIBILITIES)
        evidence = next(c for c in chunks if "底线责任" in c.text)
        self.assertTrue(all(name in evidence.text for name in ("底线责任", "预防责任", "恢复责任", "超越责任")))
        self.assertEqual(evidence.metadata["section_title"], "9. 工程实践中环境相关伦理责任")
        self.assertIn("补偿性恢复", evidence.text)
        self.assertNotEqual(evidence.metadata["section_id"], chunks[-1].metadata["section_id"])

    def test_short_consecutive_items_after_list_intro_remain_list(self):
        chunks = self.split("# 环境责任\n包括以下四项：\n1. 底线责任\n2. 预防责任\n3. 恢复责任\n4. 超越责任")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["section_title"], "# 环境责任")

    def test_real_numbered_chapters_remain_separate(self):
        chunks = self.split("1. 项目介绍\n项目承担文档知识库的检索任务。\n2. 技术设计\n采用结构化元数据支持引用回溯。\n3. 测试结果\n完成回归检验。")
        self.assertEqual([c.metadata["section_title"] for c in chunks], ["1. 项目介绍", "2. 技术设计", "3. 测试结果"])
        self.assertEqual(len({c.metadata["section_id"] for c in chunks}), 3)

    def test_complete_list_can_exceed_soft_target_without_exceeding_hard_limit(self):
        text = "# 操作步骤\n1. " + "开始准备文档材料，"*10 + "完成。\n2. " + "执行索引构建操作，"*10 + "完成。"
        chunks = self.split(text, target_size=100, overlap=10, min_size=10, max_size=300)
        self.assertEqual(len(chunks), 1)
        self.assertGreater(len(chunks[0].text), 100)
        self.assertLessEqual(len(chunks[0].text), 300)

    def test_oversized_list_keeps_parent_id_and_hard_limit_including_overlap(self):
        text = "# 完整处理步骤\n1. " + "文档处理说明，"*50 + "完成。\n2. " + "继续执行索引，"*50 + "完成。"
        chunks = self.split(text, target_size=90, overlap=20, min_size=10, max_size=120)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(c.text) <= 120 for c in chunks))
        self.assertEqual(len({c.metadata["section_id"] for c in chunks}), 1)
        self.assertEqual(len({c.metadata["list_group_id"] for c in chunks}), 1)
        self.assertTrue(all(c.metadata["list_part_count"] == len(chunks) for c in chunks))
        self.assertTrue(all(c.metadata["section_title"] == "# 完整处理步骤" for c in chunks))

    def test_section_identity_stable_across_rechunk_and_distinct_for_repeated_titles(self):
        text = "# 相同标题\n" + "正文内容需要保留。"*50 + "\n# 相同标题\n另一段内容。"
        first = self.split(text, target_size=90, overlap=10, max_size=160)
        second = self.split(text, target_size=120, overlap=10, max_size=180)
        self.assertEqual({c.metadata["section_id"] for c in first}, {c.metadata["section_id"] for c in second})
        self.assertEqual(len({c.metadata["section_id"] for c in first}), 2)
        other = RecursiveChunker().split_document(text, (), document_id="b")
        self.assertTrue({c.metadata["section_id"] for c in first}.isdisjoint({c.metadata["section_id"] for c in other}))

    def test_cross_page_list_continuation_keeps_page_and_same_section(self):
        pages = (DocumentPage(4, "# 环境责任\n1. 底线责任需要依法履行。\n2. 预防责任需要提前防范。"),
                 DocumentPage(5, "3. 恢复责任\n4. 超越责任\n# 其他章节\n后续正文。"))
        chunks = RecursiveChunker().split_document("", pages, document_id="a")
        first, second = chunks[:2]
        self.assertEqual((first.metadata["page"], second.metadata["page"]), (4, 5))
        self.assertEqual(first.metadata["section_id"], second.metadata["section_id"])
        self.assertEqual(first.metadata["list_group_id"], second.metadata["list_group_id"])
        self.assertEqual(second.metadata["section_title"], "# 环境责任")
        self.assertNotEqual(second.metadata["section_id"], chunks[-1].metadata["section_id"])

    def test_bullet_list_and_chinese_numbering_stay_under_parent(self):
        for marker1, marker2 in (("• ", "• "), ("一、", "二、"), ("（1）", "（2）")):
            text = "# 操作步骤\n包括以下内容：\n" + marker1 + "开始准备文档材料，"*10 + "完成。\n" + marker2 + "执行索引构建操作，"*10 + "完成。"
            chunks = self.split(text, target_size=100, overlap=10, min_size=10, max_size=300)
            self.assertTrue(all(c.metadata["section_title"] == "# 操作步骤" for c in chunks))
            self.assertTrue(any(marker1 in c.text and marker2 in c.text for c in chunks))


class EvidenceExpansionTests(unittest.TestCase):
    def setUp(self):
        self.path = Path.cwd()/"tests/.tmp"/str(uuid.uuid4())
        self.store = Store(self.path)
        self.config = replace(settings, vector_enabled=False, llm_mode="mock", reranker_enabled=False,
            evidence_expansion_enabled=True, evidence_neighbor_radius=2, evidence_max_extra_chunks=2,
            evidence_max_extra_tokens=1000, evidence_max_context_tokens=5000)
        for doc in ("a", "b"):
            self.store.add_document(doc, doc+".pdf", "document content", pages=(DocumentPage(4, "source"),))
        self.publish("a", [("a0", 0, "section", 4, "底线责任、预防责任。"),
                           ("a1", 1, "section", 5, "恢复责任、超越责任。"),
                           ("a2", 2, "other", 5, "其他章节不能补入。")])
        self.publish("b", [("b0", 0, "section", 4, "底线责任、预防责任。"),
                           ("b1", 1, "section", 6, "恢复责任、超越责任。")])

    def tearDown(self):
        self.store.conn.close()
        if not self.path.resolve().is_relative_to((Path.cwd()/"tests/.tmp").resolve()):
            raise RuntimeError("test cleanup outside workspace")
        shutil.rmtree(self.path)

    def publish(self, doc, entries, extra=None):
        chunks = [(cid, doc, text, pos, {"page": page, "section_id": section,
            "section_title": "同名标题", "section_chunk_count": 2,
            "extraction_method": "mixed", "ocr_mean_confidence": .99, **(extra or {})})
                  for cid, pos, section, page, text in entries]
        self.store.replace_document_chunks(doc, chunks, chunk_strategy="recursive", chunk_version="3.0")

    def hit(self, cid):
        row = self.store.chunks_by_ids((cid,))[0]
        meta = json.loads(row["metadata"])
        return RetrievalHit(cid, row["document_id"], row["text"], 3.0, row["position"], row["filename"], meta["page"], meta)

    def expand(self, seeds, config=None, query="包括哪四个层次？", scope=("a", "b")):
        return EvidenceExpander(self.store, config or self.config, counter=("test_characters", len)).expand(query, seeds, scope)

    def test_only_same_document_section_neighbors_keep_original_independent_citation(self):
        result, trace = self.expand([self.hit("a0")])
        self.assertEqual([h.chunk_id for h in result], ["a0", "a1"])
        citation = result[1].citation()
        self.assertEqual((citation["filename"], citation["page"], citation["snippet"]), ("a.pdf", 5, "恢复责任、超越责任。"))
        self.assertEqual(citation["evidence_role"], "neighbor")
        self.assertEqual(citation["expanded_from_chunk_id"], "a0")
        self.assertIsNone(citation["retrieval_score"])
        self.assertIsNone(citation["rerank_score"])
        self.assertEqual(citation["ocr_confidence"], .99)
        self.assertEqual(trace["added_count"], 1)

    def test_same_text_from_two_documents_keeps_both_neighbor_sources(self):
        result, trace = self.expand([self.hit("a0"), self.hit("b0")])
        self.assertEqual([h.chunk_id for h in result], ["a0", "b0", "a1", "b1"])
        self.assertEqual(trace["added_count"], 2)

    def test_count_and_token_limits_never_truncate_original_passages(self):
        seeds = [self.hit("a0"), self.hit("b0")]
        result, trace = self.expand(seeds, replace(self.config, evidence_max_extra_chunks=1))
        self.assertEqual(len(result), 3)
        result, trace = self.expand(seeds, replace(self.config, evidence_max_extra_tokens=1))
        self.assertEqual(result, seeds)
        self.assertTrue(all(s["reason"] == "token_budget" for s in trace["skipped"]))
        result, trace = self.expand(seeds, replace(self.config, evidence_max_context_tokens=1))
        self.assertEqual(result, seeds)
        self.assertEqual(trace["status"], "skipped_seed_budget_exceeded")

    def test_disabled_empty_and_zero_budget_do_not_expand(self):
        seeds = [self.hit("a0")]
        for config in (replace(self.config, evidence_expansion_enabled=False),
                       replace(self.config, evidence_max_extra_chunks=0),
                       replace(self.config, evidence_max_extra_tokens=0)):
            result, _ = self.expand(seeds, config)
            self.assertEqual(result, seeds)
        self.assertEqual(self.expand([])[0], [])

    def test_legacy_and_stale_seed_metadata_do_not_authorize_neighbors(self):
        self.publish("a", [("a0", 0, None, 4, "legacy"), ("a1", 1, None, 5, "legacy neighbor")])
        result, trace = self.expand([self.hit("a0")])
        self.assertEqual(len(result), 1)
        self.assertEqual(trace["skipped"][0]["reason"], "legacy_missing_section_id")
        result, trace = self.expand([replace(self.hit("a0"), text="stale text")])
        self.assertEqual(len(result), 1)
        self.assertEqual(trace["skipped"][0]["reason"], "stale_or_missing_seed")

    def test_detail_question_does_not_expand_unless_a_list_is_split(self):
        self.assertEqual(len(self.expand([self.hit("a0")], query="什么是底线责任？")[0]), 1)
        self.publish("a", [("a0", 0, "section", 4, "底线责任。"), ("a1", 1, "section", 5, "恢复责任。")],
                     {"list_group_id": "list", "list_part_count": 2})
        self.assertEqual(len(self.expand([self.hit("a0")], query="什么是底线责任？")[0]), 2)

    def test_deleted_docs_and_different_list_groups_are_not_added(self):
        self.store.soft_delete_document("b")
        self.assertEqual(self.store.section_neighbors("b", "section", 0, 2), [])
        self.publish("a", [("a0", 0, "section", 4, "first"), ("a1", 1, "section", 5, "second")],
                     {"list_group_id": "list", "list_part_count": 2})
        meta = json.loads(self.store.chunks_by_ids(("a1",))[0]["metadata"])
        meta["list_group_id"] = "different-list"
        self.store.conn.execute("UPDATE chunks SET metadata=? WHERE id='a1'", (json.dumps(meta),))
        self.store.conn.commit()
        self.assertEqual(len(self.expand([self.hit("a0")])[0]), 1)

    def test_scope_and_radius_are_enforced(self):
        self.assertEqual(len(self.expand([self.hit("a0")], scope=("b",))[0]), 1)
        self.publish("a", [("a0", 0, "section", 4, "first"), ("a3", 3, "section", 5, "outside radius")])
        self.assertEqual(len(self.expand([self.hit("a0")])[0]), 1)

    def test_unknown_model_uses_conservative_budget_without_remote_tokenizer(self):
        name, count = token_counter("unknown-local-model")
        self.assertEqual(name, "utf8_bytes_conservative_budget")
        self.assertEqual(count("中文"), 6)

    def test_api_numbers_seed_and_neighbor_independently_and_persists_trace(self):
        from app import main
        old = main.store, main.flow, main.settings
        try:
            main.store, main.settings = self.store, self.config
            main.flow = KnowledgeFlow(self.store, self.config)
            seed = self.hit("a0")
            class SeedRetriever:
                def search(self, *args):
                    return [seed]
            main.flow.retriever = SeedRetriever()
            with TestClient(main.app) as client:
                response = client.post("/chat", json={"session_id": "expansion", "message": "责任包括哪四个层次？", "document_ids": ["a"]})
                self.assertEqual(response.status_code, 200, response.text)
                citations = response.json()["citations"]
                self.assertEqual([(c["citation_number"], c["chunk_id"], c["page"]) for c in citations], [(1, "a0", 4), (2, "a1", 5)])
                run = client.get("/runs/" + response.json()["run_id"]).json()
                event = next(e for e in run["events"] if e["node"] == "evidence_expansion")
                self.assertEqual(event["status"], "expanded")
                self.assertEqual(event["output"]["added_count"], 1)
        finally:
            main.store, main.flow, main.settings = old


if __name__ == "__main__":
    unittest.main()

"""Ranking correctness, source fidelity, isolation, and request failure policy."""

from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

from fastapi.testclient import TestClient
from app.config import settings
from app.rag.contracts import RetrievalHit
from app.rag.hybrid import HybridRetriever
from app.rag.retrieval import LexicalRetriever, BalancedLexicalRetriever, BM25Retriever
from app.rag.reranker import ONNXCrossEncoder, RerankingRetriever, RerankerUnavailable, RerankerExecutionError, MODEL_ID, MODEL_REVISION, prepare_models
from app.services import KnowledgeFlow
from app.store import Store


def hit(cid, doc, text, score, page=1):
    return RetrievalHit(cid, doc, text, score, 0, doc + ".pdf", page,
                        {"page": page, "extraction_method": "mixed", "ocr_mean_confidence": .99})


class Base:
    max_hits_per_document = 2
    def __init__(self, hits):
        self.hits, self.calls = hits, []
    def search(self, query, top_k, document_ids):
        self.calls.append(("final", query, top_k, document_ids))
        return [h for h in self.hits if document_ids is None or h.document_id in document_ids][:top_k]
    def search_candidates(self, query, candidate_k, document_ids):
        self.calls.append(("candidates", query, candidate_k, document_ids))
        return self.hits[:candidate_k]


class Scores:
    model = "test-cross-encoder"
    def __init__(self, values):
        self.values, self.calls = values, []
    def score(self, query, texts):
        self.calls.append((query, texts))
        return self.values


class RerankerTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(settings, vector_enabled=False, llm_mode="mock", reranker_enabled=True,
                              reranker_candidate_k=8, reranker_failure_policy="error")

    def test_disabled_is_identical_and_does_not_load_models(self):
        hits = [hit("one", "a", "original", 9)]
        base = Base(hits)
        with patch("app.rag.reranker.get_reranker") as factory:
            result, trace = RerankingRetriever(base, replace(self.config, reranker_enabled=False)).search_with_trace("query")
        self.assertEqual(result, hits)
        factory.assert_not_called()
        self.assertEqual(base.calls[0][0], "final")
        self.assertEqual(trace["status"], "disabled")

    def test_reranks_candidates_before_topk_and_keeps_score_types(self):
        hits = [hit("one", "a", "wrong", 100), hit("two", "a", "weak", 50), hit("three", "a", "answer", 1)]
        result, trace = RerankingRetriever(Base(hits), self.config, reranker=Scores([-2, -1, 7])).search_with_trace("query", 1)
        self.assertEqual(result[0].chunk_id, "three")
        self.assertEqual(result[0].text, "answer")
        self.assertEqual(result[0].score, 7)
        self.assertEqual(result[0].citation()["retrieval_score"], 1)
        self.assertEqual(result[0].citation()["rerank_score"], 7)
        self.assertEqual(trace["rankings"][0]["candidate_rank"], 3)
        self.assertEqual(trace["status"], "ok")
        self.assertGreaterEqual(trace["total_ms"], trace["retrieval_ms"])

    def test_equal_text_preserves_independent_sources_and_page_records(self):
        hits = [hit("one", "a", "same", 3, 1), hit("duplicate", "a", "same  ", 2, 1),
                hit("two", "b", "same", 2, 6), hit("three", "a", "same", 1, 2)]
        result, _ = RerankingRetriever(Base(hits), self.config, reranker=Scores([4, 4, 4])).search_with_trace("q")
        self.assertEqual([(h.document_id, h.page) for h in result], [("a", 1), ("b", 6), ("a", 2)])
        self.assertEqual(result[1].filename, "b.pdf")
        self.assertEqual(result[1].metadata["ocr_mean_confidence"], .99)
        self.assertNotIn("rerank_score", hits[0].metadata)

    def test_final_document_cap_applied_after_ranking(self):
        hits = [hit(str(i), "a" if i < 3 else "b", str(i), 10-i) for i in range(4)]
        result, trace = RerankingRetriever(Base(hits), self.config, reranker=Scores([1, 2, 3, 0])).search_with_trace("q", 4)
        self.assertEqual([h.chunk_id for h in result], ["2", "1", "3"])
        self.assertEqual(trace["rankings"][2]["selection"], "document_cap")

    def test_request_local_document_cap_uses_candidates_without_changing_default(self):
        """单文档多方面取证只放宽当前请求；关闭或开启重排都先看未截断候选。"""
        hits = [hit(str(i), "a", "不同证据" + str(i), 4-i) for i in range(4)]
        base = Base(hits)
        disabled = RerankingRetriever(base, replace(self.config, reranker_enabled=False))
        selected, trace = disabled.search_with_trace("方面问题", 4, ("a",), per_document_cap=4)
        self.assertEqual(len(selected), 4)
        self.assertEqual(base.calls[-1][0], "candidates")
        self.assertEqual(trace["candidate_count"], 4)
        enabled = RerankingRetriever(base, self.config, reranker=Scores([4, 3, 2, 1]))
        selected, _ = enabled.search_with_trace("方面问题", 4, ("a",), per_document_cap=4)
        self.assertEqual(len(selected), 4)
        disabled.search("普通问题", 4, ("a",))
        self.assertEqual(base.calls[-1][0], "final")

    def test_scope_defense_and_empty_queries_do_not_infer(self):
        engine = Scores([4])
        hits = [hit("bad", "outside", "outside", 10), hit("good", "inside", "inside", 1)]
        pipeline = RerankingRetriever(Base(hits), self.config, reranker=engine)
        result, _ = pipeline.search_with_trace("q", document_ids=("inside",))
        self.assertEqual([h.chunk_id for h in result], ["good"])
        self.assertEqual(engine.calls[0][1], ["inside"])
        self.assertEqual(pipeline.search("q", document_ids=()), [])
        self.assertEqual(pipeline.search("q", top_k=0), [])
        with patch("app.rag.reranker.get_reranker") as factory:
            result, trace = RerankingRetriever(Base([]), self.config).search_with_trace("q")
        factory.assert_not_called()
        self.assertEqual(trace["status"], "skipped_no_candidates")

    def test_failure_fallback_is_exact_legacy_result_and_is_visible(self):
        base = Base([hit("one", "a", "original", 9)])
        for error in (RerankerUnavailable("missing"), RerankerExecutionError("inference")):
            with patch("app.rag.reranker.get_reranker", side_effect=error):
                result, trace = RerankingRetriever(base, replace(self.config, reranker_failure_policy="fallback")).search_with_trace("q")
            self.assertEqual(result, base.hits)
            self.assertEqual(trace["status"], "fallback")
            self.assertIn("error_type", trace)
            with patch("app.rag.reranker.get_reranker", side_effect=error), self.assertRaises(type(error)):
                RerankingRetriever(base, self.config).search("q")

    def test_malformed_scores_never_silently_truncate_or_accept_nan(self):
        for scores in ([], [math.nan], [math.inf], [1, 2]):
            with self.subTest(scores=scores), self.assertRaises(RerankerExecutionError):
                RerankingRetriever(Base([hit("one", "a", "text", 1)]), self.config, reranker=Scores(scores)).search("q")

    def test_missing_models_does_not_initialize_network_or_runtime(self):
        with self.assertRaises(RerankerUnavailable):
            ONNXCrossEncoder(Path("tests/.tmp/nonexistent-reranker-assets"))

    def test_candidate_providers_do_not_apply_final_document_cap(self):
        class StoreFixture:
            index_generation = 0
            def chunks(self, document_ids=None):
                return [{"id": str(i), "document_id": "a", "filename": "a.pdf", "text": "report name " + str(i),
                         "position": i, "metadata": {"page": i + 1}} for i in range(4)]
        for cls in (LexicalRetriever, BalancedLexicalRetriever, BM25Retriever):
            base = cls(StoreFixture(), max_hits_per_document=1)
            self.assertEqual(len(base.search("report", 4)), 1)
            self.assertEqual(len(base.search_candidates("report", 4)), 4)
            self.assertEqual(base.max_hits_per_document, 1)
        class Raw(Base):
            pass
        values = [hit(str(i), "a", "report " + str(i), 10-i) for i in range(4)]
        hybrid = HybridRetriever(Raw(values), Raw(values), max_hits_per_document=1)
        self.assertEqual(len(hybrid.search("q", 4)), 1)
        self.assertEqual(len(hybrid.search_candidates("q", 4)), 4)
        self.assertEqual(hybrid.max_hits_per_document, 1)

    def test_chat_citation_numbers_follow_reranked_order_and_trace_is_persisted(self):
        from app import main
        path = Path.cwd()/"tests/.tmp"/str(uuid.uuid4())
        store = Store(path)
        old = main.store, main.flow, main.settings
        try:
            main.store, main.settings = store, self.config
            main.flow = KnowledgeFlow(store, self.config)
            main.flow.retriever = Base([hit("a1", "a", "first original", 9), hit("b1", "b", "second reranked", 1)])
            with TestClient(main.app) as client, patch("app.rag.reranker.get_reranker", return_value=Scores([1, 9])):
                response = client.post("/chat", json={"session_id": "rerank", "message": "question"})
                self.assertEqual(response.status_code, 200, response.text)
                citations = response.json()["citations"]
                self.assertEqual([(c["citation_number"], c["chunk_id"]) for c in citations], [(1, "b1"), (2, "a1")])
                run = store.get_run(response.json()["run_id"])
                event = next(e for e in run["events"] if e["node"] == "reranker")
                self.assertEqual(event["output"]["rankings"][0]["chunk_id"], "b1")
            for error, code in ((RerankerUnavailable("missing"), 503), (RerankerExecutionError("failed"), 502)):
                with TestClient(main.app) as client, patch("app.rag.reranker.get_reranker", side_effect=error):
                    self.assertEqual(client.post("/chat", json={"session_id":"failure","message":"q"}).status_code, code)
            self.assertEqual(store.conn.execute("SELECT COUNT(*) FROM runs WHERE status='failed'").fetchone()[0], 2)
        finally:
            main.store, main.flow, main.settings = old
            store.conn.close()
            if path.resolve().is_relative_to((Path.cwd()/"tests/.tmp").resolve()):
                shutil.rmtree(path)

    def test_request_local_diagnostics_do_not_overwrite_each_other(self):
        pipeline = RerankingRetriever(Base([hit("a", "a", "same", 1)]), self.config, reranker=Scores([2]))
        first, second = pipeline.search("first"), pipeline.search("second")
        first.trace["status"] = "changed"
        self.assertEqual(second.trace["status"], "ok")

    def test_corrupted_model_fingerprint_cannot_be_blessed_by_prepare(self):
        path = Path.cwd()/"tests/.tmp"/str(uuid.uuid4())
        path.mkdir(parents=True)
        manifest = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                    "sha256": {"model.onnx": "0"*64, "tokenizer.json": "0"*64}}
        original = json.dumps(manifest)
        try:
            (path/"models.json").write_text(original, encoding="utf-8")
            (path/"model.onnx").write_bytes(b"corrupted")
            (path/"tokenizer.json").write_bytes(b"corrupted")
            with patch("httpx.Client") as network:
                with self.assertRaises(RerankerUnavailable):
                    prepare_models(path)
                network.assert_not_called()
            self.assertEqual((path/"models.json").read_text(encoding="utf-8"), original)
        finally:
            if path.resolve().is_relative_to((Path.cwd()/"tests/.tmp").resolve()):
                shutil.rmtree(path)


if __name__ == "__main__":
    unittest.main()

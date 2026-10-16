"""Focused regression tests for scope, BM25, and the optional vector index."""

from __future__ import annotations

import shutil
import json
import httpx
import unittest
import uuid
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.rag.retrieval import BM25RareTermRetriever, BM25Retriever, BalancedLexicalRetriever, LexicalRetriever
from app.rag.contracts import RetrievalHit
from app.rag.hybrid import HybridRetriever
from app.rag.embedding import OpenAICompatibleEmbeddingProvider
from app.rag.vector_index import QdrantVectorIndex
from app.services import KnowledgeFlow
from app.store import Store


def add_indexed_document(store: Store, document_id: str, filename: str, text: str) -> str:
    store.add_document(document_id, filename, text)
    chunk_id = str(uuid.uuid4())
    store.replace_document_chunks(
        document_id,
        [(chunk_id, document_id, text, 0, {"page": 1})],
        chunk_strategy="recursive",
        chunk_version="2.1",
    )
    return chunk_id


class FakeEmbedding:
    model = "fixture-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [float("推免" in text), float("考勤" in text), float("奖学金" in text)]
            for text in texts
        ]


class FakeCandidates:
    def __init__(self, hits: list[RetrievalHit]):
        self.hits = hits

    def search_candidates(self, query: str, candidate_k: int, document_ids=None) -> list[RetrievalHit]:
        return [hit for hit in self.hits if document_ids is None or hit.document_id in document_ids][:candidate_k]


class RetrievalPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        test_root = Path.cwd() / "tests" / ".tmp"
        test_root.mkdir(parents=True, exist_ok=True)
        self.temp_path = test_root / str(uuid.uuid4())
        self.temp_path.mkdir()
        self.store = Store(self.temp_path)

    def tearDown(self) -> None:
        self.store.conn.close()
        if not self.temp_path.resolve().is_relative_to(Path.cwd().resolve()):
            raise RuntimeError("refusing to clean a test directory outside the workspace")
        shutil.rmtree(self.temp_path)

    def test_bm25_scope_and_index_invalidation(self) -> None:
        first = str(uuid.uuid4())
        second = str(uuid.uuid4())
        first_chunk = add_indexed_document(self.store, first, "推免.pdf", "推免申请表两份")
        add_indexed_document(self.store, second, "其他.pdf", "推免申请表三份")
        retriever = BM25Retriever(self.store)
        self.assertEqual(retriever.search("推免申请表", document_ids=(first,))[0].chunk_id, first_chunk)
        self.assertTrue(retriever.search("推免申请表", document_ids=(second,)))
        self.store.soft_delete_document(first)
        self.assertEqual(retriever.search("推免申请表", document_ids=(first,)), [])
        self.assertTrue(all(hit.document_id != first for hit in retriever.search("推免申请表")))

    def test_equal_text_keeps_distinct_document_and_page_sources(self) -> None:
        same_text = "电子版报告和程序的命名格式为学号-姓名-报告/程序。"
        first, second = str(uuid.uuid4()), str(uuid.uuid4())
        first_chunk, second_chunk, same_page_copy = [str(uuid.uuid4()) for _ in range(3)]
        self.store.add_document(first, "实验三.pdf", same_text)
        self.store.add_document(second, "实验四.pdf", same_text)
        self.store.replace_document_chunks(first, [(first_chunk, first, same_text, 0, {"page": 6})])
        self.store.replace_document_chunks(second, [
            (second_chunk, second, same_text, 0, {"page": 6}),
            (same_page_copy, second, f"电子版报告和程序的命名格式为学号-姓名-报告/程序。  ", 1, {"page": 6}),
        ])
        for retriever_type in (LexicalRetriever, BalancedLexicalRetriever, BM25Retriever, BM25RareTermRetriever):
            with self.subTest(retriever=retriever_type.__name__):
                retriever = retriever_type(self.store)
                hits = retriever.search("报告程序命名格式", top_k=4)
                self.assertEqual({hit.document_id for hit in hits}, {first, second})
                self.assertEqual(len(hits), 2)
                citations = {hit.document_id: hit.citation() for hit in hits}
                self.assertEqual(citations[first]["chunk_id"], first_chunk)
                self.assertIn(citations[second]["chunk_id"], {second_chunk, same_page_copy})
                self.assertEqual(citations[second]["filename"], "实验四.pdf")
                self.assertEqual(citations[second]["page"], 6)
                self.assertEqual(citations[second]["snippet"].strip(), same_text)
                self.assertEqual([hit.document_id for hit in retriever.search("报告程序命名格式", document_ids=(second,))], [second])
                self.store.soft_delete_document(first)
                self.assertEqual([hit.document_id for hit in retriever.search("报告程序命名格式")], [second])
                # Restore first for the next retriever to test the same corpus.
                self.store.add_document(first, "实验三.pdf", same_text)
                self.store.replace_document_chunks(first, [(first_chunk, first, same_text, 0, {"page": 6})])

    def test_dense_and_hybrid_keep_equal_text_sources(self) -> None:
        from qdrant_client import QdrantClient

        same_text = "推免材料电子版按学号姓名报告格式命名。"
        first, second = str(uuid.uuid4()), str(uuid.uuid4())
        first_chunk = add_indexed_document(self.store, first, "实验三.pdf", same_text)
        second_chunk = add_indexed_document(self.store, second, "实验四.pdf", same_text)
        index = QdrantVectorIndex(self.store, FakeEmbedding(), "source_chunks", client=QdrantClient(":memory:"))
        index.replace_document(first)
        index.replace_document(second)
        dense_hits = index.search("推免", top_k=4)
        self.assertEqual({hit.document_id for hit in dense_hits}, {first, second})
        self.assertEqual({hit.chunk_id for hit in dense_hits}, {first_chunk, second_chunk})

        hybrid = HybridRetriever(BM25Retriever(self.store), index, candidate_k=8)
        hybrid_hits, trace = hybrid.search_with_trace("推免", top_k=4)
        self.assertEqual({hit.document_id for hit in hybrid_hits}, {first, second})
        self.assertEqual({hit.chunk_id for hit in hybrid_hits}, {first_chunk, second_chunk})
        self.assertTrue(all(entry["selection"] == "selected" for entry in trace["fused"]))
        self.assertEqual([hit.document_id for hit in index.search("推免", document_ids=(second,))], [second])
        self.assertEqual([hit.document_id for hit in hybrid.search("推免", document_ids=(second,))], [second])
        index.delete_document(first)
        self.store.soft_delete_document(first)
        self.assertEqual([hit.document_id for hit in index.search("推免")], [second])
        self.assertEqual([hit.document_id for hit in hybrid.search("推免")], [second])

    def test_mock_chat_citations_preserve_both_document_sources(self) -> None:
        from app import main

        text = "电子版报告和程序按照学号姓名报告格式命名。"
        first, second = str(uuid.uuid4()), str(uuid.uuid4())
        first_chunk = add_indexed_document(self.store, first, "实验三.pdf", text)
        second_chunk = add_indexed_document(self.store, second, "实验四.pdf", text)
        old_store, old_flow = main.store, main.flow
        main.store = self.store
        main.flow = KnowledgeFlow(
            self.store,
            replace(settings, retrieval_strategy="bm25_v1", vector_enabled=False, llm_mode="mock"),
        )
        try:
            response = TestClient(main.app).post(
                "/chat", json={"session_id": "source-test", "message": "电子版报告程序命名格式是什么"}
            )
            self.assertEqual(response.status_code, 200, response.text)
            citations = {item["document_id"]: item for item in response.json()["citations"]}
            self.assertEqual(set(citations), {first, second})
            self.assertEqual(citations[first]["chunk_id"], first_chunk)
            self.assertEqual(citations[second]["chunk_id"], second_chunk)
            self.assertEqual(citations[second]["filename"], "实验四.pdf")
            self.assertEqual(citations[second]["page"], 1)
            self.assertEqual(citations[second]["snippet"], text)
        finally:
            main.store, main.flow = old_store, old_flow

    def test_hybrid_fuses_before_document_cap_and_reports_candidate_ranks(self) -> None:
        a1, a2, a3, b1 = [
            RetrievalHit(chunk_id=name, document_id=doc, text=name, score=1.0, page=1)
            for name, doc in (("a1", "a"), ("a2", "a"), ("a3", "a"), ("b1", "b"))
        ]
        hybrid = HybridRetriever(
            FakeCandidates([a1, a2, a3, b1]),
            FakeCandidates([a3, b1, a2]),
            candidate_k=4,
            max_hits_per_document=1,
        )
        hits, trace = hybrid.search_with_trace("question", top_k=2)
        self.assertEqual([hit.chunk_id for hit in hits], ["a3", "b1"])
        self.assertEqual(trace["bm25"], ["a1", "a2", "a3", "b1"])
        self.assertEqual(trace["dense"], ["a3", "b1", "a2"])
        by_id = {entry["chunk_id"]: entry for entry in trace["fused"]}
        self.assertEqual(by_id["a2"]["selection"], "document_cap")
        self.assertEqual(by_id["a3"]["bm25_rank"], 3)
        self.assertEqual(by_id["a3"]["dense_rank"], 1)

    def test_hybrid_dedupes_same_page_but_keeps_other_sources(self) -> None:
        hits = [
            RetrievalHit(chunk_id="a1", document_id="a", text="相同的报告要求", score=1.0, page=1),
            RetrievalHit(chunk_id="a1_copy", document_id="a", text="相同的报告要求  ", score=1.0, page=1),
            RetrievalHit(chunk_id="a2", document_id="a", text="相同的报告要求", score=1.0, page=2),
            RetrievalHit(chunk_id="b1", document_id="b", text="相同的报告要求", score=1.0, page=1),
        ]
        hybrid = HybridRetriever(FakeCandidates(hits), FakeCandidates(hits), candidate_k=4)
        selected, trace = hybrid.search_with_trace("报告要求", top_k=3)
        self.assertEqual({hit.chunk_id for hit in selected}, {"a1", "a2", "b1"})
        by_id = {entry["chunk_id"]: entry for entry in trace["fused"]}
        self.assertEqual(by_id["a1_copy"]["selection"], "duplicate_source_text")
        self.assertEqual(by_id["b1"]["selection"], "selected")

    def test_bm25_candidate_pool_is_not_capped_by_document(self) -> None:
        document_id = str(uuid.uuid4())
        self.store.add_document(document_id, "rules.txt", "打卡规则")
        chunks = [
            (str(uuid.uuid4()), document_id, text, index, {"page": 1})
            for index, text in enumerate(("周六打卡两次", "周日打卡一次", "工作日打卡三次"))
        ]
        self.store.replace_document_chunks(document_id, chunks)
        retriever = BM25Retriever(self.store, max_hits_per_document=1)
        self.assertEqual(len(retriever.search_candidates("打卡", candidate_k=3)), 3)
        self.assertEqual(len(retriever.search("打卡", top_k=3)), 1)

    def test_chat_scope_is_visible_and_invalid_ids_fail(self) -> None:
        from app import main

        first = str(uuid.uuid4())
        second = str(uuid.uuid4())
        add_indexed_document(self.store, first, "推免.pdf", "推免申请表两份")
        add_indexed_document(self.store, second, "其他.pdf", "推免申请表三份")
        old_store, old_flow = main.store, main.flow
        main.store = self.store
        main.flow = KnowledgeFlow(
            self.store,
            replace(settings, retrieval_strategy="bm25_v1", vector_enabled=False, llm_mode="mock"),
        )
        try:
            client = TestClient(main.app)
            response = client.post(
                "/chat",
                json={"session_id": "scope-test", "message": "推免申请表几份", "document_ids": [first]},
            )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertTrue(body["citations"])
            self.assertTrue(all(citation["document_id"] == first for citation in body["citations"]))
            self.assertEqual(body["tool_calls"][1]["input"]["document_ids"], [first])
            self.assertEqual(client.post("/chat", json={"session_id": "x", "message": "a", "document_ids": ["missing"]}).status_code, 404)
            self.assertEqual(client.post("/chat", json={"session_id": "x", "message": "a", "document_ids": []}).status_code, 422)
        finally:
            main.store, main.flow = old_store, old_flow

    def test_qdrant_local_upsert_scope_reindex_and_delete(self) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            self.skipTest("install qdrant-client to run the local vector index test")
        first = str(uuid.uuid4())
        second = str(uuid.uuid4())
        first_chunk = add_indexed_document(self.store, first, "推免.pdf", "推免申请表两份")
        add_indexed_document(self.store, second, "考勤.pdf", "考勤每周记录")
        index = QdrantVectorIndex(
            self.store, FakeEmbedding(), "test_chunks", client=QdrantClient(":memory:")
        )
        self.assertEqual(index.replace_document(first), 1)
        self.assertEqual(index.replace_document(second), 1)
        self.assertEqual(index.search("推免", document_ids=(first,))[0].chunk_id, first_chunk)
        self.assertTrue(all(hit.document_id == second for hit in index.search("考勤", document_ids=(second,))))

        new_chunk = str(uuid.uuid4())
        self.store.replace_document_chunks(
            first,
            [(new_chunk, first, "推免申请表三份", 0, {"page": 2})],
        )
        self.assertEqual(index.replace_document(first), 1)
        self.assertEqual(index.search("推免", document_ids=(first,))[0].chunk_id, new_chunk)
        index.delete_document(first)
        self.store.soft_delete_document(first)
        self.assertEqual(index.search("推免", document_ids=(first,)), [])

    def test_embedding_provider_restores_response_order(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.host, "api.openai-proxy.org")
            self.assertEqual(request.url.path, "/v1/embeddings")
            self.assertEqual(request.headers["Authorization"], "Bearer fixture-key")
            self.assertEqual(
                json.loads(request.content),
                {"model": "fixture-v1", "input": ["first", "second"], "encoding_format": "float"},
            )
            return httpx.Response(
                200,
                json={"data": [
                    {"index": 1, "embedding": [0, 1]},
                    {"index": 0, "embedding": [1, 0]},
                ]},
            )

        provider = OpenAICompatibleEmbeddingProvider(
            "https://api.openai-proxy.org/v1", "fixture-v1", "fixture-key",
            transport=httpx.MockTransport(respond),
        )
        self.assertEqual(provider.embed(["first", "second"]), [[1.0, 0.0], [0.0, 1.0]])

    def test_openai_embedding_requires_key_and_nonempty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
            OpenAICompatibleEmbeddingProvider("https://api.openai.com/v1", "text-embedding-3-small")
        provider = OpenAICompatibleEmbeddingProvider(
            "https://api.openai.com/v1", "text-embedding-3-small", "fixture-key",
            transport=httpx.MockTransport(lambda request: httpx.Response(500)),
        )
        with self.assertRaisesRegex(ValueError, "empty or whitespace"):
            provider.embed([" "])

    def test_upload_and_delete_synchronize_qdrant(self) -> None:
        from qdrant_client import QdrantClient
        from app import main

        old_store, old_flow = main.store, main.flow
        main.store = self.store
        main.flow = KnowledgeFlow(
            self.store,
            replace(settings, retrieval_strategy="bm25_v1", vector_enabled=False, llm_mode="mock"),
        )
        main.flow.vector_index = QdrantVectorIndex(
            self.store, FakeEmbedding(), "endpoint_chunks", client=QdrantClient(":memory:")
        )
        try:
            client = TestClient(main.app)
            response = client.post(
                "/documents/upload",
                files={"file": ("rules.txt", ("推免申请表需要提交两份。请按制度要求提交。" * 4).encode("utf-8"), "text/plain")},
            )
            self.assertEqual(response.status_code, 200, response.text)
            document_id = response.json()["document_id"]
            self.assertEqual(client.get(f"/documents/{document_id}").json()["vector_status"], "indexed")
            self.assertTrue(main.flow.vector_index.search("推免", document_ids=(document_id,)))
            main.flow.retriever = main.flow.vector_index
            chat = client.post(
                "/chat",
                json={"session_id": "dense-test", "message": "推免", "document_ids": [document_id]},
            )
            self.assertEqual(chat.status_code, 200, chat.text)
            self.assertEqual(chat.json()["citations"][0]["document_id"], document_id)
            self.assertEqual(client.post(f"/documents/{document_id}/reindex").status_code, 200)
            self.assertTrue(main.flow.vector_index.search("推免", document_ids=(document_id,)))
            self.assertEqual(client.delete(f"/documents/{document_id}").status_code, 200)
            self.assertEqual(main.flow.vector_index.search("推免", document_ids=(document_id,)), [])
        finally:
            main.store, main.flow = old_store, old_flow


if __name__ == "__main__":
    unittest.main()

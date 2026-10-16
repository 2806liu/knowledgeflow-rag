import time
import uuid
from typing import Any
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.rag.contracts import EvidenceContext
from app.rag.embedding import OpenAICompatibleEmbeddingProvider
from app.rag.hybrid import HybridRetriever
from app.rag.quality import normalize_text
from app.rag.retrieval import BM25RareTermRetriever, BM25Retriever, BalancedLexicalRetriever, LexicalRetriever
from app.rag.solver import EvidenceSolver
from app.rag.llm_client import LLMCallError
from app.rag.vector_index import QdrantVectorIndex
from app.rag.reranker import RerankingRetriever
from app.rag.evidence_expansion import EvidenceExpander
from app.store import Store
from app.tools import RetrieveKnowledgeInput, SearchHistoryInput, Tool, ToolRegistry


class KnowledgeFlow:
    """编排规划、检索、回答、评审和运行记录保存的完整流程。"""

    def __init__(self, store: Store, settings: Settings):
        # 新会话状态与旧客户端兼容路径共存，业务模块独立实例化。
        from app.conversation import ConversationService
        self.conversations = ConversationService(store)
        self._conversation_flow = None
        self.store = store
        self.settings = settings
        self.vector_index: QdrantVectorIndex | None = None
        if settings.vector_enabled or settings.retrieval_strategy in {"dense_v1", "hybrid_rrf_v1"}:
            embedding = OpenAICompatibleEmbeddingProvider(
                settings.embedding_base_url,
                settings.embedding_model,
                settings.embedding_api_key,
            )
            self.vector_index = QdrantVectorIndex(
                store, embedding, settings.qdrant_collection, url=settings.qdrant_url
            )
        if settings.retrieval_strategy == "lexical_v1":
            self.retriever = LexicalRetriever(
                store,
                min_query_coverage=settings.retrieval_min_query_coverage,
                max_hits_per_document=settings.retrieval_max_hits_per_document,
            )
        elif settings.retrieval_strategy == "lexical_v2":
            self.retriever = BalancedLexicalRetriever(
                store,
                min_query_coverage=settings.retrieval_min_query_coverage,
                max_hits_per_document=settings.retrieval_v2_max_hits_per_document,
            )
        elif settings.retrieval_strategy == "bm25_v1":
            self.retriever = BM25Retriever(
                store,
                max_hits_per_document=settings.retrieval_bm25_max_hits_per_document,
            )
        elif settings.retrieval_strategy == "bm25_idf_v2":
            self.retriever = BM25RareTermRetriever(
                store,
                max_hits_per_document=settings.retrieval_bm25_max_hits_per_document,
            )
        elif settings.retrieval_strategy == "dense_v1":
            assert self.vector_index is not None
            self.retriever = self.vector_index
        elif settings.retrieval_strategy == "hybrid_rrf_v1":
            assert self.vector_index is not None
            self.retriever = HybridRetriever(
                BM25Retriever(store, max_hits_per_document=settings.retrieval_bm25_max_hits_per_document),
                self.vector_index,
                candidate_k=settings.retrieval_hybrid_candidate_k,
                rrf_k=settings.retrieval_hybrid_rrf_k,
                max_hits_per_document=settings.retrieval_hybrid_max_hits_per_document,
            )
        else:
            raise ValueError(f"unsupported retrieval strategy: {settings.retrieval_strategy}")
        self.llm = EvidenceSolver(settings)
        self.tools = ToolRegistry()
        self.tools.register(
            Tool(
                name="retrieve_knowledge",
                description="Search indexed document chunks with the configured retrieval strategy.",
                input_model=RetrieveKnowledgeInput,
                handler=self._retrieve_knowledge,
            )
        )
        self.tools.register(
            Tool(
                name="search_history",
                description="Load recent messages for a conversation session.",
                input_model=SearchHistoryInput,
                handler=lambda args: self.store.history(args.session_id, args.limit),
            )
        )

    async def _retrieve_knowledge(self, args):
        # Construct around the current retriever: test/runtime swaps still work.
        pipeline = RerankingRetriever(self.retriever, self.settings)
        scope = tuple(dict.fromkeys(args.document_ids)) if args.document_ids is not None else None
        def retrieve():
            # 按方面检索可局部提高单文档证据上限；普通问答仍使用原有配置。
            batch = pipeline.search(args.query, args.top_k, scope,
                                    per_document_cap=getattr(args, "per_document_cap", None))
            expanded, trace = EvidenceExpander(self.store, self.settings).expand(args.query, batch, scope)
            batch[:] = expanded
            batch.expansion_trace = trace
            return batch
        return await run_in_threadpool(retrieve)

    async def _legacy_chat(
        self, session_id: str, question: str, document_ids: tuple[str, ...] | None = None
    ) -> dict[str, Any]:
        """执行一次问答工作流，并返回答案、引用和工具调用摘要。"""

        run_id = str(uuid.uuid4())
        events: list[dict[str, Any]] = []
        started = time.perf_counter()
        try:
            # 先保存用户消息，随后检索知识库并记录每个流程节点的事件。
            self.store.add_message(session_id, "user", question)
            plan = {"task_type": "knowledge_qa", "steps": ["retrieve", "solve", "critic"]}
            events.append({"node": "planner", "status": "ok", "output": plan})
            history = await self.tools.execute(
                "search_history", {"session_id": session_id, "limit": 6}
            )
            events.append(
                {
                    "node": "tool",
                    "tool": "search_history",
                    "status": "ok",
                    "result_count": len(history),
                }
            )
            hits = await self.tools.execute(
                "retrieve_knowledge",
                {"query": question, "top_k": 4, "document_ids": list(document_ids) if document_ids is not None else None},
            )
            events.append({"node": "reranker", "status": hits.trace["status"], "output": hits.trace})
            events.append({"node": "evidence_expansion", "status": hits.expansion_trace["status"],
                           "output": hits.expansion_trace})
            # 显式构造 Solver 的证据契约，Mock 与真实 LLM 共用同一结构。
            evidence = [
                EvidenceContext(
                    citation_number=index,
                    filename=hit.filename,
                    page=hit.page,
                    text=hit.text,
                )
                for index, hit in enumerate(hits, start=1)
            ]
            events.append(
                {
                    "node": "tool",
                    "tool": "retrieve_knowledge",
                    "status": "ok",
                    "result_count": len(hits),
                    "citations": [h.citation() for h in hits],
                }
            )
            # The current user message was just persisted; the structured user
            # message below already contains it, so do not send it twice.
            prior_history = history[:-1] if history and history[-1] == {"role": "user", "content": question} else history
            result = await self.llm.answer(question, evidence, prior_history)
            # 单独保存脱敏调用诊断，失败调用在异常分支同样记录。
            events.append({"node": "llm", "status": result.trace.get("status", "unknown"), "output": result.trace})
            answer = result.answer
            citation_records = [
                {"citation_number": index, **hit.citation()}
                for index, hit in enumerate(hits, 1)
                if index in result.cited_numbers
            ]
            events.append({
                "node": "solver",
                "status": "ok",
                "mode": self.settings.llm_mode,
                "answer_status": result.status,
                "cited_numbers": list(result.cited_numbers),
            })
            # 评审节点检查证据覆盖、引用标记和重复率，而不是只看 hits 是否非空。
            normalized_hits = [normalize_text(hit.text) for hit in hits]
            unique_count = len(set(normalized_hits))
            lexical_match_count = sum(
                1
                for hit in hits
                if hit.metadata.get("query_coverage", 0.0) >= self.settings.retrieval_min_query_coverage
            )
            citation_markers = len(result.cited_numbers)
            critic = {
                "grounded": None,
                "grounding_status": "citation_numbers_validated_semantics_not_evaluated",
                "answer_status": result.status,
                "retrieved_count": len(hits),
                "citation_count": len(citation_records),
                "citation_coverage": round(citation_markers / max(len(hits), 1), 4),
                "lexical_match_count": lexical_match_count,
                "lexical_weak_match_count": len(hits) - lexical_match_count,
                "duplicate_ratio": round(1 - unique_count / max(len(hits), 1), 4),
                "max_score": max((hit.score for hit in hits), default=0.0),
                "needs_followup": result.status not in {"answered", "mock"},
            }
            events.append({"node": "critic", "status": "ok", "output": critic})
            events.append({"node": "workflow", "status": "ok", "latency_ms": round((time.perf_counter() - started) * 1000, 2)})
            self.store.add_message(session_id, "assistant", answer)
            self.store.save_run(run_id, session_id, "completed", events)
            return {
                "run_id": run_id,
                "answer": answer,
                "answer_status": result.status,
                "citations": citation_records,
                "tool_calls": [
                    {
                        "name": "search_history",
                        "input": {"session_id": session_id, "limit": 6},
                        "result_count": len(history),
                    },
                    {
                        "name": "retrieve_knowledge",
                        "input": {
                            "query": question,
                            "top_k": 4,
                            "document_ids": list(document_ids) if document_ids is not None else None,
                        },
                        "result_count": len(hits),
                    },
                ],
            }
        except Exception as exc:
            # 无论失败原因是什么，都保存失败事件，便于通过运行详情接口排查。
            if isinstance(exc, LLMCallError):
                # 错误响应携带关联 ID，调用者可查询本次失败的脱敏诊断。
                exc.run_id = run_id
                events.append({"node": "llm", "status": "error", "output": exc.trace})
            elif getattr(exc, "trace", None) is not None:
                events.append({"node": "reranker", "status": "error", "output": exc.trace})
            events.append({"node": "workflow", "status": "error", "latency_ms": round((time.perf_counter() - started) * 1000, 2)})
            self.store.save_run(run_id, session_id, "failed", events, str(exc))
            raise

    async def chat(self, session_id, question, document_ids=None, request_id=None, scope_revision=None, answer_policy=None):
        """已创建会话使用新编排；既有评测客户端保持原检索契约。"""
        conversation=self.conversations.get(session_id)
        # 旧API未指定模式仍走严格行为；显式增强会话从空范围开始，不能回落全库。
        if conversation is None and answer_policy=="document_enhanced":
            conversation=self.conversations.create(session_id,answer_policy=answer_policy)
        policy=answer_policy or (conversation or {}).get("answer_policy","document_strict")
        if conversation is not None:
            if self._conversation_flow is None:
                from app.conversational_flow import ConversationalFlow
                self._conversation_flow = ConversationalFlow(self)
            return await self._conversation_flow.chat(session_id,question,document_ids,request_id,scope_revision,policy)
        return await self._legacy_chat(session_id,question,document_ids)

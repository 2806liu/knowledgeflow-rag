"""Qdrant dense index with SQLite as the canonical source for citations."""

from __future__ import annotations

import json
from typing import Any

from app.rag.contracts import RetrievalHit, source_dedupe_key
from app.rag.embedding import EmbeddingProvider
from app.store import Store


class QdrantVectorIndex:
    min_query_coverage = 0.0

    def __init__(
        self,
        store: Store,
        embedding: EmbeddingProvider,
        collection: str,
        *,
        url: str = "http://127.0.0.1:6333",
        client: Any | None = None,
    ):
        if not collection:
            raise ValueError("Qdrant collection is required")
        if client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise RuntimeError("install the vector extra: pip install -e '.[vector]'") from exc
            client = QdrantClient(url=url)
        self.store = store
        self.embedding = embedding
        self.collection = collection
        self.client = client

    @staticmethod
    def _models():
        try:
            from qdrant_client import models
        except ImportError as exc:
            raise RuntimeError("install the vector extra: pip install -e '.[vector]'") from exc
        return models

    def _ensure_collection(self, dimension: int) -> None:
        models = self._models()
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )
            return
        current = self.client.get_collection(self.collection).config.params.vectors
        if getattr(current, "size", None) != dimension:
            raise ValueError(
                f"Qdrant collection {self.collection!r} has a different vector dimension; "
                "use a collection for the configured embedding model"
            )

    def _document_filter(self, document_id: str):
        models = self._models()
        return models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
        )

    def delete_document(self, document_id: str) -> None:
        if not self.client.collection_exists(self.collection):
            return
        models = self._models()
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=self._document_filter(document_id)),
            wait=True,
        )

    def replace_document(self, document_id: str) -> int:
        """Embed current SQLite chunks and replace only this document's vectors."""

        rows = self.store.chunks((document_id,))
        vectors = self.embedding.embed([str(row["text"]) for row in rows])
        if len(vectors) != len(rows):
            raise ValueError("embedding provider returned the wrong vector count")
        if not rows:
            self.delete_document(document_id)
            self.store.set_vector_status(document_id, "not_indexed")
            return 0
        self._ensure_collection(len(vectors[0]))
        models = self._models()
        points = [
            models.PointStruct(
                id=str(row["id"]),
                vector=vector,
                payload={
                    "chunk_id": str(row["id"]),
                    "document_id": document_id,
                    "embedding_model": self.embedding.model,
                },
            )
            for row, vector in zip(rows, vectors)
        ]
        self.delete_document(document_id)
        for start in range(0, len(points), 128):
            self.client.upsert(collection_name=self.collection, points=points[start : start + 128], wait=True)
        self.store.set_vector_status(document_id, "indexed", self.embedding.model)
        return len(points)

    def stage_chunks(self, document_id, chunks):
        """为尚未发布的候选生成向量；无SQLite规范行的点不会进入引用结果。"""
        vectors = self.embedding.embed([row[2] for row in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("embedding provider returned wrong count")
        if not chunks:
            return
        self._ensure_collection(len(vectors[0]))
        models = self._models()
        points = [models.PointStruct(id=row[0], vector=vector,
                  payload={"chunk_id":row[0], "document_id":document_id,"embedding_model":self.embedding.model})
                  for row,vector in zip(chunks,vectors)]
        for start in range(0,len(points),128):
            self.client.upsert(collection_name=self.collection,points=points[start:start+128],wait=True)

    def search_candidates(
        self, query: str, candidate_k: int = 32, document_ids: tuple[str, ...] | None = None
    ) -> list[RetrievalHit]:
        """Return active Qdrant matches before text dedupe or diversity limits."""

        if candidate_k <= 0 or document_ids == () or not self.client.collection_exists(self.collection):
            return []
        vectors = self.embedding.embed([query])
        models = self._models()
        conditions = [
            models.FieldCondition(
                key="embedding_model", match=models.MatchValue(value=self.embedding.model)
            )
        ]
        if document_ids is not None:
            conditions.append(
                models.FieldCondition(
                    key="document_id", match=models.MatchAny(any=list(document_ids))
                )
            )
        scope_filter = models.Filter(must=conditions)
        points = self.client.query_points(
            collection_name=self.collection,
            query=vectors[0],
            query_filter=scope_filter,
            limit=max(candidate_k * 5, 20),
            with_payload=True,
        ).points
        rows = self.store.chunks_by_ids(tuple(str(point.id) for point in points))
        active = {str(row["id"]): row for row in rows}
        hits: list[RetrievalHit] = []
        for point in points:
            row = active.get(str(point.id))
            if row is None or row["vector_status"] != "indexed" or (
                document_ids is not None and row["document_id"] not in document_ids
            ):
                continue
            payload = point.payload or {}
            if payload.get("embedding_model") != self.embedding.model:
                continue
            body = str(row["text"])
            metadata = json.loads(row["metadata"] or "{}")
            metadata.update({"retriever_version": "dense_v1", "query_coverage": 0.0})
            hits.append(
                RetrievalHit(
                    chunk_id=str(row["id"]),
                    document_id=str(row["document_id"]),
                    text=body,
                    score=round(float(point.score), 4),
                    position=int(row["position"]),
                    filename=str(row["filename"]),
                    page=int(metadata["page"]) if metadata.get("page") is not None else None,
                    metadata=metadata,
                )
            )
            if len(hits) >= candidate_k:
                break
        return hits

    def search(
        self, query: str, top_k: int = 4, document_ids: tuple[str, ...] | None = None
    ) -> list[RetrievalHit]:
        if top_k <= 0:
            return []
        results: list[RetrievalHit] = []
        seen_sources: set[tuple[str, int | None, str]] = set()
        for hit in self.search_candidates(query, candidate_k=max(top_k * 5, 20), document_ids=document_ids):
            source_key = source_dedupe_key(hit)
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            results.append(hit)
            if len(results) >= top_k:
                break
        return results

"""Embedding provider contract and an OpenAI-compatible HTTP implementation."""

from __future__ import annotations

import math
from typing import Protocol
from urllib.parse import urlparse

import httpx


class EmbeddingProvider(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        batch_size: int = 32,
        transport: httpx.BaseTransport | None = None,
    ):
        if not base_url or not model or batch_size <= 0:
            raise ValueError("embedding base URL, model, and positive batch size are required")
        if urlparse(base_url).hostname == "api.openai.com" and not api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI Embeddings API")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size
        self.transport = transport

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("embedding input cannot be empty or whitespace")
        vectors: list[list[float]] = []
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        with httpx.Client(timeout=90, transport=self.transport) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                response = client.post(
                    f"{self.base_url}/embeddings",
                    json={"model": self.model, "input": batch, "encoding_format": "float"},
                    headers=headers,
                )
                response.raise_for_status()
                payload = response.json()
                items = payload.get("data")
                if not isinstance(items, list) or len(items) != len(batch):
                    raise ValueError("embedding response count does not match request")
                by_index = sorted(items, key=lambda item: item["index"])
                if [item["index"] for item in by_index] != list(range(len(batch))):
                    raise ValueError("embedding response indexes are invalid")
                for item in by_index:
                    vector = [float(value) for value in item["embedding"]]
                    if not vector or not all(math.isfinite(value) for value in vector):
                        raise ValueError("embedding response contains an invalid vector")
                    vectors.append(vector)
        if len({len(vector) for vector in vectors}) != 1:
            raise ValueError("embedding response dimensions are inconsistent")
        return vectors

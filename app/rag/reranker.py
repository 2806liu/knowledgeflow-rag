"""Optional local ONNX cross-encoder and a source-preserving ranking stage."""

from __future__ import annotations

import argparse
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import logging
import math
from pathlib import Path
import threading
import time
from typing import Protocol

from app.rag.contracts import RetrievalHit, source_dedupe_key

MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MODEL_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
MODEL_FILE = "onnx/model_quint8_avx2.onnx"
logger = logging.getLogger(__name__)


class RerankerUnavailable(RuntimeError):
    pass


class RerankerExecutionError(RuntimeError):
    pass


class Reranker(Protocol):
    model: str
    def score(self, query: str, texts: list[str]) -> list[float]: ...


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ONNXCrossEncoder:
    """Loads only pinned, prepared local assets; no Hugging Face remote code."""

    def __init__(self, model_dir: Path, batch_size: int = 8, max_length: int = 512):
        if batch_size <= 0 or not 8 <= max_length <= 512:
            raise ValueError("batch size must be positive and max length must be 8..512")
        manifest_path = model_dir / "models.json"
        if not manifest_path.is_file():
            raise RerankerUnavailable("reranker models not prepared; run python -m app.rag.reranker --prepare")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["model_id"] != MODEL_ID or manifest["revision"] != MODEL_REVISION:
                raise ValueError("unexpected model identity or revision")
            paths = {}
            for name in ("model.onnx", "tokenizer.json"):
                path = (model_dir / name).resolve()
                if not path.is_relative_to(model_dir.resolve()) or not path.is_file() or file_hash(path) != manifest["sha256"][name]:
                    raise ValueError("missing model or fingerprint mismatch")
                paths[name] = path
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
            self.np = np
            self.tokenizer = Tokenizer.from_file(str(paths["tokenizer.json"]))
            self.tokenizer.enable_truncation(max_length=max_length, strategy="longest_first")
            self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(paths["model.onnx"]), options, providers=["CPUExecutionProvider"])
            self.input_names = {value.name for value in self.session.get_inputs()}
            if not {"input_ids", "attention_mask"} <= self.input_names or self.input_names - {"input_ids", "attention_mask", "token_type_ids"}:
                raise ValueError("unsupported ONNX input contract")
            self.model = MODEL_ID
            self.manifest = manifest
            self.batch_size = batch_size
            self.max_length = max_length
            self._lock = threading.Lock()
        except Exception as exc:
            raise RerankerUnavailable(f"reranker initialization failed: {type(exc).__name__}: {exc}") from exc

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        # Identical text shares inference, but every document's hit is retained.
        unique = list(dict.fromkeys(texts))
        values = []
        try:
            with self._lock:
                for start in range(0, len(unique), self.batch_size):
                    encoded = self.tokenizer.encode_batch([(query, text) for text in unique[start:start + self.batch_size]])
                    fields = {"input_ids": "ids", "attention_mask": "attention_mask", "token_type_ids": "type_ids"}
                    feed = {name: self.np.array([getattr(e, fields[name]) for e in encoded], dtype=self.np.int64)
                            for name in self.input_names}
                    output = self.session.run(None, feed)[0]
                    if output.shape != (len(encoded), 1):
                        raise ValueError("reranker returned unexpected logits shape")
                    values.extend(float(x) for x in output[:, 0])
            if len(values) != len(unique) or not all(math.isfinite(x) for x in values):
                raise ValueError("reranker returned invalid scores")
            by_text = dict(zip(unique, values))
            return [by_text[text] for text in texts]
        except Exception as exc:
            raise RerankerExecutionError(f"reranker inference failed: {type(exc).__name__}: {exc}") from exc


@lru_cache(maxsize=4)
def get_reranker(model_dir: str, batch_size: int = 8, max_length: int = 512):
    return ONNXCrossEncoder(Path(model_dir), batch_size, max_length)


class RetrievalBatch(list):
    """A normal hit list with request-local diagnostics (no shared last_trace)."""
    def __init__(self, hits, trace):
        super().__init__(hits)
        self.trace = trace


class RerankingRetriever:
    def __init__(self, base, config, *, reranker: Reranker | None = None):
        if not 1 <= config.reranker_candidate_k <= 256 or config.reranker_batch_size <= 0 or not 8 <= config.reranker_max_length <= 512:
            raise ValueError("invalid reranker candidate/batch/token budget")
        if config.reranker_failure_policy not in {"fallback", "error"}:
            raise ValueError("reranker failure policy must be fallback or error")
        self.base, self.config, self.reranker = base, config, reranker

    def search(self, query, top_k=4, document_ids=None, per_document_cap=None):
        hits, trace = self.search_with_trace(query, top_k, document_ids, per_document_cap)
        return RetrievalBatch(hits, trace)

    def search_with_trace(self, query, top_k=4, document_ids=None, per_document_cap=None):
        """普通请求沿用原上限；按方面取证时可对本次请求单独放宽单文档上限。"""
        started = time.perf_counter()
        trace = {"enabled": self.config.reranker_enabled, "status": "disabled", "candidate_count": 0,
                 "retrieval_ms": 0.0, "rerank_ms": 0.0, "total_ms": 0.0, "rankings": []}
        if top_k <= 0 or document_ids == ():
            trace["status"] = "skipped_empty_scope"
            return [], trace
        def candidate_fallback():
            # 单文档多方面问答须先看候选池，再做去重与本轮上限；不修改全局检索器。
            raw = self.base.search_candidates(query, max(top_k, self.config.reranker_candidate_k), document_ids)
            selected, seen, counts = [], set(), {}
            for hit in raw:
                if document_ids is not None and hit.document_id not in document_ids:
                    continue
                key = source_dedupe_key(hit)
                if key in seen or counts.get(hit.document_id, 0) >= per_document_cap:
                    continue
                selected.append(hit)
                seen.add(key)
                counts[hit.document_id] = counts.get(hit.document_id, 0) + 1
                if len(selected) >= top_k:
                    break
            trace["candidate_count"] = len(raw)
            return selected

        if not self.config.reranker_enabled:
            hits = self.base.search(query, top_k, document_ids) if per_document_cap is None else candidate_fallback()
            trace["retrieval_ms"] = trace["total_ms"] = round((time.perf_counter() - started) * 1000, 3)
            return hits, trace
        budget = max(top_k, self.config.reranker_candidate_k)
        raw = self.base.search_candidates(query, budget, document_ids)
        seen = set()
        candidates = []
        for hit in raw:
            if document_ids is not None and hit.document_id not in document_ids:
                continue
            key = source_dedupe_key(hit)
            if key not in seen:
                seen.add(key)
                candidates.append(hit)
        trace["candidate_count"] = len(candidates)
        trace["retrieval_ms"] = round((time.perf_counter() - started) * 1000, 3)
        if not candidates:
            trace.update(status="skipped_no_candidates", total_ms=trace["retrieval_ms"])
            return [], trace
        rerank_started = time.perf_counter()
        try:
            engine = self.reranker or get_reranker(str(self.config.reranker_model_dir.resolve()),
                                                  self.config.reranker_batch_size, self.config.reranker_max_length)
            scores = engine.score(query, [h.text for h in candidates])
            if len(scores) != len(candidates) or not all(math.isfinite(s) for s in scores):
                raise RerankerExecutionError("reranker returned invalid score count or values")
            ranked = sorted(enumerate(zip(candidates, scores)), key=lambda item: (-item[1][1], item[0]))
            results, counts = [], {}
            cap = per_document_cap if per_document_cap is not None else getattr(self.base, "max_hits_per_document", None)
            for rank, (before_rank, (hit, score)) in enumerate(ranked, 1):
                reason = "selected"
                if cap is not None and counts.get(hit.document_id, 0) >= cap:
                    reason = "document_cap"
                elif len(results) >= top_k:
                    reason = "top_k"
                if reason == "selected":
                    counts[hit.document_id] = counts.get(hit.document_id, 0) + 1
                    results.append(replace(hit, score=float(score), metadata={**hit.metadata,
                        "retrieval_score": hit.score, "rerank_score": float(score), "reranker_model": engine.model,
                        "candidate_rank": before_rank + 1, "rerank_rank": rank}))
                trace["rankings"].append({"chunk_id": hit.chunk_id, "document_id": hit.document_id, "page": hit.page,
                    "candidate_rank": before_rank + 1, "rerank_rank": rank, "retrieval_score": hit.score,
                    "rerank_score": float(score), "selection": reason})
            trace.update(status="ok", model=engine.model)
        except (RerankerUnavailable, RerankerExecutionError) as exc:
            trace.update(status="fallback", error_type=type(exc).__name__, error=str(exc))
            if self.config.reranker_failure_policy == "error":
                trace.update(status="error", rerank_ms=round((time.perf_counter() - rerank_started) * 1000, 3),
                             total_ms=round((time.perf_counter() - started) * 1000, 3))
                exc.trace = trace
                logger.info(json.dumps({"event": "retrieval_rerank", **trace}, ensure_ascii=False))
                raise
            # Return precisely the legacy result, never an unscored large pool.
            fallback_started = time.perf_counter()
            results = self.base.search(query, top_k, document_ids) if per_document_cap is None else candidate_fallback()
            trace["fallback_retrieval_ms"] = round((time.perf_counter() - fallback_started) * 1000, 3)
        trace["rerank_ms"] = round((time.perf_counter() - rerank_started) * 1000, 3)
        if "fallback_retrieval_ms" in trace:
            trace["rerank_ms"] = round(max(0.0, trace["rerank_ms"] - trace["fallback_retrieval_ms"]), 3)
            trace["retrieval_ms"] = round(trace["retrieval_ms"] + trace["fallback_retrieval_ms"], 3)
        trace["total_ms"] = round((time.perf_counter() - started) * 1000, 3)
        logger.info(json.dumps({"event": "retrieval_rerank", **trace}, ensure_ascii=False))
        return results, trace


def prepare_models(model_dir: Path):
    """Explicit provisioning only; stage assets before publishing the manifest."""
    # Never replace a trusted fingerprint with a hash of an altered local file.
    if (model_dir / "models.json").is_file():
        engine = ONNXCrossEncoder(model_dir)
        print(json.dumps({"prepared": True, "reused": True, "model": engine.model,
                          "revision": MODEL_REVISION}))
        return
    import httpx
    model_dir.mkdir(parents=True, exist_ok=True)
    files = {"model.onnx": MODEL_FILE, "tokenizer.json": "tokenizer.json"}
    hashes = {}
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        for target_name, remote_name in files.items():
            target = model_dir / target_name
            # Without a published manifest, local assets are untrusted remnants.
            staging = model_dir / (target_name + ".partial")
            try:
                with client.stream("GET", f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{remote_name}") as response:
                    response.raise_for_status()
                    with staging.open("wb") as stream:
                        for block in response.iter_bytes():
                            stream.write(block)
                staging.replace(target)
            finally:
                staging.unlink(missing_ok=True)
            hashes[target_name] = file_hash(target)
    manifest = {"model_id": MODEL_ID, "revision": MODEL_REVISION, "remote_model_file": MODEL_FILE,
                "sha256": hashes, "score_type": "raw_cross_encoder_logit", "license": "Apache-2.0"}
    staging = model_dir / "models.json.partial"
    staging.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    staging.replace(model_dir / "models.json")
    engine = ONNXCrossEncoder(model_dir)
    scores = engine.score("可持续发展原则", ["可持续发展的原则包括公平性、持续性和共同性。"])
    print(json.dumps({"prepared": True, "model": engine.model, "revision": MODEL_REVISION, "smoke_score": scores[0]}))


if __name__ == "__main__":
    from app.config import settings
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", required=True)
    parser.add_argument("--model-dir", type=Path, default=settings.reranker_model_dir)
    args = parser.parse_args()
    prepare_models(args.model_dir)

from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os
import re
import sys
import time
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    _model_cache: ClassVar[dict[str, object]] = {}

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            if self.model_name in self._model_cache:
                self._model = self._model_cache[self.model_name]
                return self._model
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except Exception as exc:  # noqa: BLE001 - optional model boundary
                print(f"  ⚠️  Cross-encoder unavailable; using lexical reranker: {exc}")
                self._model = _LexicalCrossEncoder()
            self._model_cache[self.model_name] = self._model
        return self._model

    def rerank(
        self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K
    ) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents or top_k <= 0:
            return []
        pairs = [(query, document["text"]) for document in documents]
        scores = self._load_model().predict(pairs)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if len(scores) != len(documents):
            raise ValueError(
                "Reranker returned a different number of scores than documents"
            )
        scored = sorted(
            zip(scores, documents), key=lambda item: float(item[0]), reverse=True
        )
        return [
            RerankResult(
                text=document["text"],
                original_score=float(document.get("score", 0.0)),
                rerank_score=float(score),
                metadata=dict(document.get("metadata", {})),
                rank=rank,
            )
            for rank, (score, document) in enumerate(scored[:top_k], start=1)
        ]


class _LexicalCrossEncoder:
    """Transparent offline fallback based on token and numeric overlap."""

    @staticmethod
    def predict(pairs):
        scores = []
        for query, document in pairs:
            query_tokens = set(re.findall(r"\w+", query.casefold()))
            document_tokens = set(re.findall(r"\w+", document.casefold()))
            overlap = len(query_tokens & document_tokens) / max(len(query_tokens), 1)
            query_numbers = set(re.findall(r"\d+(?:[.,]\d+)*", query))
            number_bonus = 0.1 * len(
                query_numbers & set(re.findall(r"\d+(?:[.,]\d+)*", document))
            )
            scores.append(overlap + number_bonus)
        return np.asarray(scores, dtype=float)


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""

    def __init__(self):
        self._model = None

    def rerank(
        self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K
    ) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []
        try:
            from flashrank import Ranker, RerankRequest

            if self._model is None:
                self._model = Ranker()
            passages = [
                {
                    "id": index,
                    "text": document["text"],
                    "meta": document.get("metadata", {}),
                }
                for index, document in enumerate(documents)
            ]
            results = self._model.rerank(RerankRequest(query=query, passages=passages))
            return [
                RerankResult(
                    text=result["text"],
                    original_score=float(
                        documents[int(result["id"])].get("score", 0.0)
                    ),
                    rerank_score=float(result["score"]),
                    metadata=dict(documents[int(result["id"])].get("metadata", {})),
                    rank=rank,
                )
                for rank, result in enumerate(results[:top_k], start=1)
            ]
        except (ImportError, OSError):
            fallback = _LexicalCrossEncoder().predict(
                [(query, d["text"]) for d in documents]
            )
            scored = sorted(
                zip(fallback, documents), key=lambda item: item[0], reverse=True
            )
            return [
                RerankResult(
                    d["text"],
                    float(d.get("score", 0.0)),
                    float(score),
                    dict(d.get("metadata", {})),
                    rank,
                )
                for rank, (score, d) in enumerate(scored[:top_k], start=1)
            ]


def benchmark_reranker(
    reranker, query: str, documents: list[dict], n_runs: int = 5
) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")

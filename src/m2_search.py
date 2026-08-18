from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os
import re
import sys
import unicodedata
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    try:
        from underthesea import word_tokenize

        normalized = word_tokenize(normalized, format="text")
    except (ImportError, OSError):
        # Unicode-aware tokenization is a predictable no-dependency fallback.
        normalized = " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))
    return " ".join(normalized.replace("_", " ").split())


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = list(chunks)
        self.corpus_tokens = [segment_vietnamese(c["text"]).split() for c in chunks]
        if not self.corpus_tokens:
            self.bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.corpus_tokens)
        except ImportError:
            self.bm25 = _SimpleBM25(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0:
            return []
        scores = self.bm25.get_scores(segment_vietnamese(query).split())
        top_indices = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[
            :top_k
        ]
        return [
            SearchResult(
                text=self.documents[index]["text"],
                score=float(scores[index]),
                metadata=dict(self.documents[index].get("metadata", {})),
                method="bm25",
            )
            for index in top_indices
            if float(scores[index]) > 0
        ]


class _SimpleBM25:
    """Small BM25 implementation used only when rank-bm25 is unavailable."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        from collections import Counter
        from math import log

        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.lengths) / max(len(self.lengths), 1)
        self.frequencies = [Counter(doc) for doc in corpus]
        document_frequency = Counter(token for doc in corpus for token in set(doc))
        n_docs = len(corpus)
        self.idf = {
            token: log(1 + (n_docs - count + 0.5) / (count + 0.5))
            for token, count in document_frequency.items()
        }

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.corpus), dtype=float)
        for index, frequencies in enumerate(self.frequencies):
            normalization = self.k1 * (
                1 - self.b + self.b * self.lengths[index] / max(self.avgdl, 1.0)
            )
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if frequency:
                    scores[index] += self.idf.get(token, 0.0) * (
                        frequency * (self.k1 + 1) / (frequency + normalization)
                    )
        return scores


class DenseSearch:
    def __init__(self):
        try:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
        except ImportError:
            self.client = None
        self._encoder = None
        self._local_collections: dict[str, tuple[np.ndarray, list[dict]]] = {}

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(EMBEDDING_MODEL)
            except Exception as exc:  # noqa: BLE001 - optional model boundary
                print(f"  ⚠️  Dense model unavailable; using hashing embeddings: {exc}")
                self._encoder = _HashingEncoder(EMBEDDING_DIM)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        documents = list(chunks)
        texts = [chunk["text"] for chunk in documents]
        if not texts:
            self._local_collections[collection] = (np.empty((0, EMBEDDING_DIM)), [])
            return
        vectors = np.asarray(
            self._get_encoder().encode(
                texts, show_progress_bar=False, normalize_embeddings=True
            ),
            dtype=np.float32,
        )
        payloads = [
            {**chunk.get("metadata", {}), "text": chunk["text"]} for chunk in documents
        ]

        if self.client is not None:
            try:
                from qdrant_client.models import Distance, PointStruct, VectorParams

                if hasattr(
                    self.client, "collection_exists"
                ) and self.client.collection_exists(collection):
                    self.client.delete_collection(collection)
                self.client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(
                        size=int(vectors.shape[1]), distance=Distance.COSINE
                    ),
                )
                points = [
                    PointStruct(
                        id=index, vector=vector.tolist(), payload=payloads[index]
                    )
                    for index, vector in enumerate(vectors)
                ]
                # Batch writes avoid request-size limits for larger corpora.
                for start in range(0, len(points), 128):
                    self.client.upsert(
                        collection_name=collection, points=points[start : start + 128]
                    )
                self._local_collections.pop(collection, None)
                return
            except Exception as exc:  # noqa: BLE001 - remote client boundary
                print(f"  ⚠️  Qdrant unavailable; using in-memory dense index: {exc}")
        self._local_collections[collection] = (vectors, payloads)

    def search(
        self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME
    ) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0:
            return []
        query_vector = np.asarray(
            self._get_encoder().encode(query, normalize_embeddings=True),
            dtype=np.float32,
        )
        if collection in self._local_collections:
            vectors, payloads = self._local_collections[collection]
            if not payloads:
                return []
            scores = vectors @ query_vector
            indices = np.argsort(-scores, kind="stable")[:top_k]
            return [
                SearchResult(
                    text=payloads[index]["text"],
                    score=float(scores[index]),
                    metadata=dict(payloads[index]),
                    method="dense",
                )
                for index in indices
            ]
        if self.client is None:
            return []
        try:
            response = self.client.query_points(
                collection_name=collection, query=query_vector.tolist(), limit=top_k
            )
            return [
                SearchResult(
                    text=point.payload.get("text", ""),
                    score=float(point.score),
                    metadata=dict(point.payload or {}),
                    method="dense",
                )
                for point in response.points
            ]
        except Exception as exc:  # noqa: BLE001 - remote client boundary
            print(f"  ⚠️  Dense search failed: {exc}")
            return []


class _HashingEncoder:
    """Deterministic, normalized bag-of-token embeddings for offline execution."""

    def __init__(self, dimension: int):
        self.dimension = dimension

    def encode(self, texts, **_kwargs):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vectors = np.zeros((len(items), self.dimension), dtype=np.float32)
        for row, text in enumerate(items):
            for token in segment_vietnamese(text).split():
                # Python's hash is randomized, so use a stable byte hash.
                import hashlib

                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest, "little") % self.dimension
                vectors[row, index] += 1.0
            norm = np.linalg.norm(vectors[row])
            if norm:
                vectors[row] /= norm
        return vectors[0] if single else vectors


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]], k: int = 60, top_k: int = HYBRID_TOP_K
) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if top_k <= 0:
        return []
    fused: dict[str, dict] = {}
    for result_list in results_list:
        seen: set[str] = set()
        for rank, result in enumerate(result_list):
            # A document contributes at most once per retrieval method.
            if result.text in seen:
                continue
            seen.add(result.text)
            item = fused.setdefault(result.text, {"score": 0.0, "result": result})
            item["score"] += 1.0 / (k + rank + 1)
    ranked = sorted(fused.values(), key=lambda item: -item["score"])[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=dict(item["result"].metadata),
            method="hybrid",
        )
        for item in ranked
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""

    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")

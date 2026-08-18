from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import glob
import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
    SEMANTIC_THRESHOLD,
)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append(
                {"text": f.read(), "metadata": {"source": os.path.basename(fp)}}
            )

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(
                f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR)."
            )

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(
    text: str, chunk_size: int = 500, metadata: dict | None = None
) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(
                Chunk(
                    text=current.strip(),
                    metadata={**metadata, "chunk_index": len(chunks)},
                )
            )
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(
            Chunk(
                text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}
            )
        )
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(
    text: str, threshold: float = SEMANTIC_THRESHOLD, metadata: dict | None = None
) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = dict(metadata or {})
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n\s*\n+", text.strip())
        if sentence.strip()
    ]
    if not sentences:
        return []

    try:
        model = _semantic_model()
        if model is None:
            raise RuntimeError("all-MiniLM-L6-v2 could not be loaded")
        embeddings = model.encode(
            sentences, convert_to_numpy=True, normalize_embeddings=True
        )
    except Exception as exc:  # noqa: BLE001 - optional model boundary
        # A lexical vectorizer keeps the module usable in offline deployments.
        print(f"  ⚠️  Semantic model unavailable; using lexical fallback: {exc}")
        embeddings = _lexical_embeddings(sentences)

    groups: list[list[str]] = [[sentences[0]]]
    for index in range(1, len(sentences)):
        similarity = float(np.dot(embeddings[index - 1], embeddings[index]))
        if similarity < threshold:
            groups.append([sentences[index]])
        else:
            groups[-1].append(sentences[index])

    return [
        Chunk(
            text=" ".join(group),
            metadata={**metadata, "strategy": "semantic", "chunk_index": index},
        )
        for index, group in enumerate(groups)
    ]


@lru_cache(maxsize=1)
def _semantic_model():
    """Load the embedding model once instead of once per document."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:  # noqa: BLE001 - model loaders expose heterogeneous errors
        return None


def _lexical_embeddings(sentences: list[str]) -> np.ndarray:
    tokens = [re.findall(r"\w+", sentence.casefold()) for sentence in sentences]
    vocabulary = {token for sentence in tokens for token in sentence}
    if not vocabulary:
        return np.zeros((len(sentences), 1), dtype=float)
    token_index = {token: index for index, token in enumerate(sorted(vocabulary))}
    vectors = np.zeros((len(sentences), len(token_index)), dtype=float)
    for row, sentence in enumerate(tokens):
        for token in sentence:
            vectors[row, token_index[token]] += 1.0
        norm = np.linalg.norm(vectors[row])
        if norm:
            vectors[row] /= norm
    return vectors


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")
    metadata = dict(metadata or {})
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]

    parent_texts = _pack_units(paragraphs, parent_size)
    parents: list[Chunk] = []
    children: list[Chunk] = []
    source = str(metadata.get("source", "document"))
    safe_source = re.sub(r"[^\w.-]+", "_", source)

    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"{safe_source}:parent_{parent_index}"
        parent_metadata = {
            **metadata,
            "strategy": "hierarchical",
            "chunk_type": "parent",
            "chunk_index": parent_index,
            "parent_id": parent_id,
        }
        parents.append(Chunk(text=parent_text, metadata=parent_metadata))

        child_units = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+|\n+", parent_text)
            if part.strip()
        ]
        for child_text in _pack_units(child_units, child_size):
            children.append(
                Chunk(
                    text=child_text,
                    metadata={
                        **metadata,
                        "strategy": "hierarchical",
                        "chunk_type": "child",
                        "chunk_index": len(children),
                    },
                    parent_id=parent_id,
                )
            )

    return parents, children


def _pack_units(units: list[str], max_size: int) -> list[str]:
    """Pack logical units without exceeding max_size when possible."""
    expanded: list[str] = []
    for unit in units:
        if len(unit) <= max_size:
            expanded.append(unit)
            continue
        words = unit.split()
        current = ""
        for word in words:
            # Hard-split a pathological token while retaining all content.
            pieces = [word[i : i + max_size] for i in range(0, len(word), max_size)]
            for piece in pieces:
                candidate = f"{current} {piece}".strip()
                if current and len(candidate) > max_size:
                    expanded.append(current)
                    current = piece
                else:
                    current = candidate
        if current:
            expanded.append(current)

    packed: list[str] = []
    current = ""
    for unit in expanded:
        candidate = f"{current}\n\n{unit}".strip()
        if current and len(candidate) > max_size:
            packed.append(current)
            current = unit
        else:
            current = candidate
    if current:
        packed.append(current)
    return packed


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = dict(metadata or {})
    header_pattern = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)
    matches = list(header_pattern.finditer(text))
    chunks: list[Chunk] = []

    def add_chunk(chunk_text: str, section: str) -> None:
        chunk_text = chunk_text.strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        **metadata,
                        "strategy": "structure",
                        "section": section,
                        "chunk_index": len(chunks),
                    },
                )
            )

    if not matches:
        add_chunk(text, "Preamble")
        return chunks

    add_chunk(text[: matches[0].start()], "Preamble")
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        header = match.group(0).strip()
        add_chunk(text[match.start() : end], re.sub(r"^#{1,6}\s+", "", header))
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """

    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(
            f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}"
        )

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")

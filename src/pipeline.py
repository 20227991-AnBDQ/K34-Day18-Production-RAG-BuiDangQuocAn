from __future__ import annotations

"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RERANK_TOP_K
from src.m1_chunking import chunk_hierarchical, load_documents
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import evaluate_ragas, failure_analysis, load_test_set, save_report
from src.m5_enrichment import enrich_chunks


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    timings = {}
    t0 = time.perf_counter()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        parent_by_id = {parent.metadata["parent_id"]: parent.text for parent in parents}
        for child in children:
            all_chunks.append(
                {
                    "text": child.text,
                    "metadata": {
                        **child.metadata,
                        "parent_id": child.parent_id,
                        "parent_text": parent_by_id[child.parent_id],
                    },
                }
            )
    timings["chunking"] = (time.perf_counter() - t0) * 1000
    print(
        f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({timings['chunking'] / 1000:.1f}s)",
        flush=True,
    )

    # Step 2: Enrichment (M5)
    t0 = time.perf_counter()
    print(
        f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...",
        flush=True,
    )
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [
            {"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched
        ]
        timings["enrichment"] = (time.perf_counter() - t0) * 1000
        print(
            f"  ✓ Enriched {len(enriched)} chunks ({timings['enrichment'] / 1000:.1f}s)",
            flush=True,
        )
    else:
        timings["enrichment"] = (time.perf_counter() - t0) * 1000
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.perf_counter()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    timings["indexing"] = (time.perf_counter() - t0) * 1000
    print(f"  ✓ Indexed ({timings['indexing'] / 1000:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.perf_counter()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    # Model loading remains lazy so startup does not pay for unused queries.
    timings["reranker_initialization"] = (time.perf_counter() - t0) * 1000
    print(
        f"  ✓ Reranker ready ({timings['reranker_initialization'] / 1000:.1f}s)",
        flush=True,
    )

    search.latency_breakdown_ms = timings

    return search, reranker


def run_query(
    query: str, search: HybridSearch, reranker: CrossEncoderReranker
) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    selected = reranked if reranked else results[:3]
    contexts = []
    seen_contexts = set()
    for result in selected:
        metadata = result.metadata
        # Hierarchical retrieval uses a precise child for ranking but supplies
        # its full parent to answer generation.
        context = metadata.get("parent_text") or result.text
        if context not in seen_contexts:
            contexts.append(context)
            seen_contexts.add(context)

    from config import OPENAI_API_KEY, OPENAI_MODEL

    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI

            client = OpenAI()
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'",
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}",
                    },
                ],
            )
            answer = resp.choices[0].message.content
        except Exception as e:  # noqa: BLE001 - OpenAI boundary
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = _extractive_answer(query, contexts)
    return answer, contexts


def _extractive_answer(query: str, contexts: list[str]) -> str:
    """Select grounded sentences when an LLM key is not configured."""
    if not contexts:
        return "Không tìm thấy thông tin."
    query_tokens = set(re.findall(r"\w+", query.casefold()))
    candidates = []
    for context_index, context in enumerate(contexts):
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", context)
            if sentence.strip() and not sentence.lstrip().startswith("#")
        ]
        for sentence_index, sentence in enumerate(sentences):
            sentence_tokens = set(re.findall(r"\w+", sentence.casefold()))
            overlap = len(query_tokens & sentence_tokens) / max(len(query_tokens), 1)
            version_bonus = (
                0.05
                if any(
                    term in sentence.casefold()
                    for term in ("hiện hành", "v2024", "v2.0")
                )
                else 0
            )
            candidates.append(
                (overlap + version_bonus, -context_index, -sentence_index, sentence)
            )
    if not candidates:
        return contexts[0]
    ranked = sorted(candidates, reverse=True)
    chosen = []
    for score, _, _, sentence in ranked:
        if score <= 0 and chosen:
            break
        if sentence not in chosen:
            chosen.append(sentence)
        if len(chosen) == 3:
            break
    return " ".join(chosen) if chosen else contexts[0]


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    query_times = []
    for i, item in enumerate(test_set):
        query_start = time.perf_counter()
        answer, contexts = run_query(item["question"], search, reranker)
        query_times.append((time.perf_counter() - query_start) * 1000)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i + 1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.perf_counter()
    print(
        f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True
    )
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    evaluation_ms = (time.perf_counter() - t0) * 1000
    print(f"  ✓ RAGAS done ({evaluation_ms / 1000:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    os.makedirs("reports", exist_ok=True)
    save_report(results, failures, path="reports/ragas_report.json")
    latency = {
        **getattr(search, "latency_breakdown_ms", {}),
        "query_average": sum(query_times) / max(len(query_times), 1),
        "query_p95": sorted(query_times)[max(0, int(len(query_times) * 0.95) - 1)]
        if query_times
        else 0.0,
        "evaluation": evaluation_ms,
    }
    with open("reports/latency_report.json", "w", encoding="utf-8") as report_file:
        json.dump(
            {"unit": "milliseconds", "steps": latency},
            report_file,
            ensure_ascii=False,
            indent=2,
        )
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")

from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, OPENAI_MODEL


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""

    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if not text.strip():
        return ""
    if OPENAI_API_KEY:
        try:
            return _chat_text(
                "Tóm tắt đoạn văn sau trong 2 câu ngắn gọn bằng tiếng Việt. "
                "Giữ nguyên con số và điều kiện quan trọng.",
                text,
                max_tokens=150,
            )
        except Exception as exc:  # noqa: BLE001 - OpenAI boundary
            print(f"  ⚠️  OpenAI summarize failed: {exc}")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    return " ".join(sentences[:2]) if sentences else text.strip()


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if n_questions <= 0 or not text.strip():
        return []
    if OPENAI_API_KEY:
        try:
            response = _chat_text(
                f"Dựa trên đoạn văn, tạo đúng {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
                "Mỗi dòng chỉ chứa một câu hỏi tiếng Việt.",
                text,
                max_tokens=200,
            )
            questions = [_clean_list_item(line) for line in response.splitlines()]
            return [question for question in questions if question][:n_questions]
        except Exception as exc:  # noqa: BLE001 - OpenAI boundary
            print(f"  ⚠️  OpenAI HyQA failed: {exc}")
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", text) if len(s.strip()) > 10]
    questions = []
    for sentence in sentences[:n_questions]:
        topic = " ".join(sentence.split()[:12]).rstrip(".,;:")
        questions.append(f"Thông tin nào được quy định về {topic}?")
    return questions


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if not text:
        return ""
    if OPENAI_API_KEY:
        try:
            context = _chat_text(
                "Viết một câu ngắn mô tả vị trí và chủ đề của đoạn trích. "
                "Không thêm dữ kiện không có trong văn bản.",
                f"Tài liệu: {document_title or 'không rõ'}\n\nĐoạn văn:\n{text}",
                max_tokens=80,
            )
            return f"{context}\n\n{text}" if context else text
        except Exception as exc:  # noqa: BLE001 - OpenAI boundary
            print(f"  ⚠️  OpenAI contextual failed: {exc}")
    prefix = (
        f"Trích từ tài liệu {document_title}."
        if document_title
        else "Trích từ tài liệu nội bộ."
    )
    return f"{prefix}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if OPENAI_API_KEY:
        try:
            result = _chat_json(
                "Trích xuất metadata và chỉ trả JSON hợp lệ: "
                '{"topic":"...","entities":["..."],'
                '"category":"policy|hr|it|finance","language":"vi|en"}.',
                text,
                max_tokens=150,
            )
            return _normalize_metadata(result, text)
        except Exception as exc:  # noqa: BLE001 - OpenAI boundary
            print(f"  ⚠️  OpenAI metadata failed: {exc}")
    return _fallback_metadata(text)


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    if OPENAI_API_KEY:
        try:
            result = _chat_json(
                """Phân tích đoạn văn và chỉ trả về JSON hợp lệ:
{"summary":"tóm tắt 2 câu","questions":["câu hỏi 1","câu hỏi 2","câu hỏi 3"],
"context":"một câu mô tả vị trí và chủ đề đoạn văn",
"metadata":{"topic":"...","entities":["..."],"category":"policy|hr|it|finance","language":"vi|en"}}""",
                f"Tài liệu: {source or 'không rõ'}\n\nĐoạn văn:\n{text}",
                max_tokens=400,
            )
            return {
                "summary": str(result.get("summary", "")).strip(),
                "questions": [
                    str(q).strip()
                    for q in result.get("questions", [])
                    if str(q).strip()
                ][:3],
                "context": str(result.get("context", "")).strip(),
                "metadata": _normalize_metadata(result.get("metadata", {}), text),
            }
        except Exception as exc:  # noqa: BLE001 - OpenAI boundary
            print(f"  ⚠️  Enrichment API failed: {exc}")

    # Full deterministic fallback preserves the same schema and makes the
    # enriched representation useful to both sparse and dense retrieval.
    summary = summarize_chunk(text)
    questions = generate_hypothesis_questions(text, n_questions=3)
    context = (
        f"Đoạn trích thuộc tài liệu {source}."
        if source
        else "Đoạn trích thuộc tài liệu nội bộ."
    )
    return {
        "summary": summary,
        "questions": questions,
        "context": context,
        "metadata": _fallback_metadata(text),
    }


def _chat_text(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    from openai import OpenAI

    response = OpenAI().chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def _chat_json(system_prompt: str, user_prompt: str, max_tokens: int) -> dict:
    from openai import OpenAI

    response = OpenAI().chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = (response.choices[0].message.content or "{}").strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
    result = json.loads(content)
    if not isinstance(result, dict):
        raise TypeError("Enrichment response must be a JSON object")
    return result


def _clean_list_item(line: str) -> str:
    return re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()


def _fallback_metadata(text: str) -> dict:
    folded = unicodedata.normalize("NFC", text).casefold()
    keyword_groups = {
        "it": ("mật khẩu", "vpn", "malware", "dữ liệu", "bảo mật", "mfa"),
        "finance": ("lương", "chi phí", "vnđ", "thanh toán", "tạm ứng", "mua sắm"),
        "hr": ("nhân viên", "nghỉ", "thử việc", "đào tạo", "mentor", "bảo hiểm"),
    }
    category = next(
        (
            name
            for name, keywords in keyword_groups.items()
            if any(word in folded for word in keywords)
        ),
        "policy",
    )
    heading = next(
        (
            line.lstrip("# ").strip()
            for line in text.splitlines()
            if line.startswith("#")
        ),
        "",
    )
    topic = heading or " ".join(text.strip().split()[:8]).rstrip(".,;:") or "general"
    entities = list(dict.fromkeys(re.findall(r"\b[A-ZÀ-Ỹ][A-ZÀ-Ỹ0-9-]{1,}\b", text)))[
        :10
    ]
    language = "vi" if re.search(r"[ăâđêôơưĂÂĐÊÔƠƯ]", text) else "en"
    return {
        "topic": topic,
        "entities": entities,
        "category": category,
        "language": language,
    }


def _normalize_metadata(value, text: str) -> dict:
    fallback = _fallback_metadata(text)
    if not isinstance(value, dict):
        return fallback
    category = value.get("category")
    if category not in {"policy", "hr", "it", "finance"}:
        category = fallback["category"]
    language = value.get("language")
    if language not in {"vi", "en"}:
        language = fallback["language"]
    entities = value.get("entities", [])
    if not isinstance(entities, list):
        entities = []
    return {
        "topic": str(value.get("topic") or fallback["topic"]),
        "entities": [str(entity) for entity in entities],
        "category": category,
        "language": language,
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enrichment_parts = []
            if context_line:
                enrichment_parts.append(context_line)
            if summary:
                enrichment_parts.append(f"Tóm tắt: {summary}")
            if questions:
                enrichment_parts.append("Câu hỏi liên quan: " + " ".join(questions))
            enrichment_parts.append(text)
            enriched_text = "\n\n".join(enrichment_parts)
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = (
                contextual_prepend(text, source) if "contextual" in methods else text
            )
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata={**chunk.get("metadata", {}), **auto_meta},
                method="+".join(methods),
            )
        )

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")

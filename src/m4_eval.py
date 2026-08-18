from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import math
import os
import re
import sys
import unicodedata
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Run RAGAS evaluation."""
    lengths = {len(questions), len(answers), len(contexts), len(ground_truths)}
    if len(lengths) != 1:
        raise ValueError(
            "questions, answers, contexts and ground_truths must have equal lengths"
        )
    if not questions:
        return _aggregate([], backend="empty")

    try:
        # RAGAS' default metrics use an OpenAI judge, so do not trigger hidden
        # network calls when credentials are intentionally absent.
        from config import OPENAI_API_KEY

        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            raise_exceptions=False,
        )
        frame = result.to_pandas()
        per_question = [
            EvalResult(
                question=str(row["question"]),
                answer=str(row["answer"]),
                contexts=list(row["contexts"]),
                ground_truth=str(row["ground_truth"]),
                faithfulness=_safe_score(row.get("faithfulness")),
                answer_relevancy=_safe_score(row.get("answer_relevancy")),
                context_precision=_safe_score(row.get("context_precision")),
                context_recall=_safe_score(row.get("context_recall")),
            )
            for _, row in frame.iterrows()
        ]
        return _aggregate(per_question, backend="ragas")
    except Exception as exc:  # noqa: BLE001 - evaluator/network boundary
        print(f"  ⚠️  RAGAS evaluation unavailable: {exc}")
        print("  ℹ️  Using deterministic lexical proxy metrics for this offline run.")
        proxy_results = [
            _proxy_evaluate(question, answer, context, ground_truth)
            for question, answer, context, ground_truth in zip(
                questions, answers, contexts, ground_truths
            )
        ]
        return _aggregate(proxy_results, backend="lexical_proxy")


METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def _safe_score(value) -> float:
    try:
        score = float(value)
        return score if math.isfinite(score) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _aggregate(per_question: list[EvalResult], backend: str) -> dict:
    return {
        **{
            metric: (
                sum(getattr(item, metric) for item in per_question) / len(per_question)
                if per_question
                else 0.0
            )
            for metric in METRIC_NAMES
        },
        "per_question": per_question,
        "evaluation_backend": backend,
    }


_STOPWORDS = {
    "a",
    "an",
    "the",
    "và",
    "là",
    "có",
    "được",
    "cho",
    "của",
    "khi",
    "bao",
    "nhiêu",
    "một",
    "nhân",
    "viên",
    "theo",
    "trong",
    "với",
    "thì",
    "phải",
}


def _tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFC", text).casefold()
    return {
        token for token in re.findall(r"\w+", normalized) if token not in _STOPWORDS
    }


def _coverage(expected: set[str], observed: set[str]) -> float:
    return min(1.0, len(expected & observed) / max(len(expected), 1))


def _proxy_evaluate(
    question: str, answer: str, contexts: list[str], ground_truth: str
) -> EvalResult:
    answer_tokens = _tokens(answer)
    question_tokens = _tokens(question)
    truth_tokens = _tokens(ground_truth)
    context_token_sets = [_tokens(context) for context in contexts]
    all_context_tokens = (
        set().union(*context_token_sets) if context_token_sets else set()
    )
    relevant_basis = truth_tokens | question_tokens

    faithfulness_score = _coverage(answer_tokens, all_context_tokens)
    answer_relevancy_score = max(
        _coverage(question_tokens, answer_tokens),
        _coverage(truth_tokens, answer_tokens),
    )
    relevant_contexts = [
        tokens for tokens in context_token_sets if tokens & relevant_basis
    ]
    context_precision_score = len(relevant_contexts) / max(len(context_token_sets), 1)
    context_recall_score = _coverage(truth_tokens, all_context_tokens)
    return EvalResult(
        question,
        answer,
        list(contexts),
        ground_truth,
        faithfulness_score,
        answer_relevancy_score,
        context_precision_score,
        context_recall_score,
    )


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0:
        return []
    diagnostic_tree = {
        "faithfulness": (
            "The answer contains claims unsupported by the retrieved context.",
            "Tighten the grounded-answer prompt, cite evidence, and lower generation temperature.",
        ),
        "context_recall": (
            "The retriever missed information needed for the reference answer.",
            "Improve chunk boundaries, retrieve more candidates, or add query expansion/BM25.",
        ),
        "context_precision": (
            "The retrieved set contains too many irrelevant chunks.",
            "Strengthen cross-encoder reranking and add version or metadata filters.",
        ),
        "answer_relevancy": (
            "The answer does not directly address the question.",
            "Use a question-focused answer template and remove unrelated context from the response.",
        ),
    }
    analyzed = []
    for result in eval_results:
        scores = {
            metric: _safe_score(getattr(result, metric)) for metric in METRIC_NAMES
        }
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append(
            {
                "question": result.question,
                "expected": result.ground_truth,
                "answer": result.answer,
                "average_score": sum(scores.values()) / len(scores),
                "worst_metric": worst_metric,
                "score": scores[worst_metric],
                "diagnosis": diagnosis,
                "suggested_fix": suggested_fix,
            }
        )
    analyzed.sort(
        key=lambda item: (item["average_score"], item["score"], item["question"])
    )
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {
            metric: _safe_score(results.get(metric)) for metric in METRIC_NAMES
        },
        "evaluation_backend": results.get("evaluation_backend", "unknown"),
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")

from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
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


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    # RAGAS cần OPENAI_API_KEY và Python 3.11+ → wrap try/except để pipeline
    # vẫn chạy được (trả zeros) khi thiếu key hoặc lỗi version.
    try:
        from ragas import evaluate
        from ragas.metrics import (faithfulness, answer_relevancy,
                                   context_precision, context_recall)
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall])
        df = result.to_pandas()

        def _num(row, key):
            """NaN xuất hiện khi RAGAS không chấm được 1 câu → coi như 0.0."""
            value = row.get(key, 0.0)
            try:
                value = float(value)
            except (TypeError, ValueError):
                return 0.0
            return 0.0 if value != value else value  # NaN != NaN

        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=list(row["contexts"]),
                ground_truth=row["ground_truth"],
                faithfulness=_num(row, "faithfulness"),
                answer_relevancy=_num(row, "answer_relevancy"),
                context_precision=_num(row, "context_precision"),
                context_recall=_num(row, "context_recall"),
            )
            for _, row in df.iterrows()
        ]

        def _mean(attr: str) -> float:
            values = [getattr(r, attr) for r in per_question]
            return sum(values) / len(values) if values else 0.0

        return {
            "faithfulness": _mean("faithfulness"),
            "answer_relevancy": _mean("answer_relevancy"),
            "context_precision": _mean("context_precision"),
            "context_recall": _mean("context_recall"),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    # Diagnostic Tree: metric thấp nhất → nguyên nhân gốc → fix tương ứng.
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating — câu trả lời không bám context",
                         "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks — retrieval bỏ sót thông tin cần thiết",
                           "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks — context bị nhiễu",
                              "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question — trả lời lạc đề",
                             "Improve prompt template"),
    }

    scored = []
    for r in eval_results:
        metrics = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        avg = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        scored.append((avg, r, metrics, worst_metric))

    scored.sort(key=lambda x: x[0])  # tệ nhất lên đầu

    failures = []
    for avg, r, metrics, worst_metric in scored[:bottom_n]:
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        failures.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "worst_metric": worst_metric,
            "score": round(metrics[worst_metric], 4),
            "avg_score": round(avg, 4),
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")

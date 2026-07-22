"""Tests for frozen composition-QA metrics and paired statistics."""

from src.observability.evaluation.agentic_bridge_evaluator import (
    evaluate_case,
    paired_comparison,
    summarize,
)


def _case() -> dict:
    return {
        "required_chunk_ids": ["c1", "c2"],
        "answer_facts": [
            {"fact": "alpha uses retrieval", "aliases": ["retrieval alpha"]},
            {"fact": "beta uses reflection", "aliases": ["reflection beta"]},
        ],
    }


def test_evaluate_case_scores_complete_evidence_and_citations():
    metrics = evaluate_case(
        _case(),
        {
            "answer": "Alpha uses retrieval, while beta uses reflection.",
            "retrieved_chunk_ids": ["c1", "c2", "noise"],
            "cited_chunk_ids": ["c1", "c2"],
        },
    )

    assert metrics["evidence_chain_complete"] == 1.0
    assert metrics["evidence_recall"] == 1.0
    assert metrics["answer_fact_f1"] > 0.5
    assert metrics["citation_precision"] == 1.0
    assert metrics["citation_recall"] == 1.0
    assert metrics["invalid_citation_rate"] == 0.0


def test_failure_is_counted_as_zero():
    metrics = evaluate_case(_case(), {"error": "timeout"})
    assert set(metrics.values()) == {0.0}


def test_summarize_keeps_failures_in_denominator():
    rows = [
        {
            "metrics": evaluate_case(
                _case(),
                {
                    "answer": "alpha uses retrieval and beta uses reflection",
                    "retrieved_chunk_ids": ["c1", "c2"],
                    "cited_chunk_ids": ["c1", "c2"],
                },
            ),
            "latency_ms": 10,
            "retrieval_calls": 1,
            "llm_calls": 1,
            "error": "",
            "fallback": False,
        },
        {
            "metrics": evaluate_case(_case(), {"error": "timeout"}),
            "latency_ms": 30,
            "retrieval_calls": 0,
            "llm_calls": 0,
            "error": "timeout",
            "fallback": True,
        },
    ]

    summary = summarize(rows)

    assert summary["evidence_chain_complete"] == 0.5
    assert summary["failure_rate"] == 0.5
    assert summary["fallback_rate"] == 0.5
    assert summary["avg_latency_ms"] == 20


def test_paired_comparison_reports_delta_ci_and_signs():
    baseline = [
        {"case_id": "a", "metrics": {"evidence_chain_complete": 0.0, "answer_fact_f1": 0.2}},
        {"case_id": "b", "metrics": {"evidence_chain_complete": 1.0, "answer_fact_f1": 0.5}},
    ]
    treatment = [
        {"case_id": "a", "metrics": {"evidence_chain_complete": 1.0, "answer_fact_f1": 0.4}},
        {"case_id": "b", "metrics": {"evidence_chain_complete": 1.0, "answer_fact_f1": 0.4}},
    ]

    comparison = paired_comparison(baseline, treatment, samples=100, seed=7)

    chain = comparison["metrics"]["evidence_chain_complete"]
    facts = comparison["metrics"]["answer_fact_f1"]
    assert chain["delta"] == 0.5
    assert chain["win_tie_loss"] == {"win": 1, "tie": 1, "loss": 0}
    assert round(facts["delta"], 6) == 0.05
    assert facts["win_tie_loss"] == {"win": 1, "tie": 0, "loss": 1}

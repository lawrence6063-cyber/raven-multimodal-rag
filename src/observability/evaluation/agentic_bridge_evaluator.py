"""Deterministic metrics for frozen multi-evidence Agentic RAG experiments."""

from __future__ import annotations

import random
import re
from collections import Counter
from typing import Any

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.IGNORECASE)


def evaluate_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, float]:
    """Score one arm result against a frozen evidence-composition case."""
    if result.get("error"):
        return _zero_metrics()

    answer = str(result.get("answer", ""))
    retrieved = set(result.get("retrieved_chunk_ids") or [])
    cited = set(result.get("cited_chunk_ids") or [])
    required = set(case.get("required_chunk_ids") or [])
    evidence_hits = len(retrieved & required)
    evidence_recall = evidence_hits / len(required) if required else 0.0
    chain_complete = float(bool(required) and required.issubset(retrieved))

    fact_scores = [_fact_score(answer, fact) for fact in case.get("answer_facts") or []]
    answer_fact_f1 = sum(fact_scores) / len(fact_scores) if fact_scores else 0.0
    citation_precision = len(cited & required) / len(cited) if cited else 0.0
    citation_recall = len(cited & required) / len(required) if required else 0.0
    invalid_citation_rate = len(cited - set(result.get("retrieved_chunk_ids") or [])) / len(cited) if cited else 0.0
    return {
        "evidence_chain_complete": chain_complete,
        "evidence_recall": evidence_recall,
        "answer_fact_f1": answer_fact_f1,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "invalid_citation_rate": invalid_citation_rate,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Average metrics and operational measurements over all frozen cases."""
    if not rows:
        return {}
    metric_names = list(_zero_metrics())
    summary = {
        name: sum(float(row["metrics"].get(name, 0.0)) for row in rows) / len(rows)
        for name in metric_names
    }
    latencies = sorted(float(row.get("latency_ms", 0.0)) for row in rows)
    summary.update(
        {
            "n_cases": len(rows),
            "failure_rate": sum(bool(row.get("error")) for row in rows) / len(rows),
            "fallback_rate": sum(bool(row.get("fallback")) for row in rows) / len(rows),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p50_latency_ms": _percentile(latencies, 0.50),
            "p95_latency_ms": _percentile(latencies, 0.95),
            "avg_retrieval_calls": sum(float(row.get("retrieval_calls", 0)) for row in rows)
            / len(rows),
            "avg_llm_calls": sum(float(row.get("llm_calls", 0)) for row in rows) / len(rows),
        }
    )
    return summary


def paired_comparison(
    baseline_rows: list[dict[str, Any]],
    treatment_rows: list[dict[str, Any]],
    metrics: tuple[str, ...] = ("evidence_chain_complete", "answer_fact_f1"),
    samples: int = 10000,
    seed: int = 20260721,
) -> dict[str, Any]:
    """Compute paired deltas, percentile bootstrap CIs, and win/tie/loss."""
    baseline = {row["case_id"]: row for row in baseline_rows}
    treatment = {row["case_id"]: row for row in treatment_rows}
    case_ids = sorted(set(baseline) & set(treatment))
    output: dict[str, Any] = {"n_pairs": len(case_ids), "metrics": {}}
    for metric_index, metric in enumerate(metrics):
        diffs = [
            float(treatment[case_id]["metrics"].get(metric, 0.0))
            - float(baseline[case_id]["metrics"].get(metric, 0.0))
            for case_id in case_ids
        ]
        output["metrics"][metric] = {
            "delta": sum(diffs) / len(diffs) if diffs else 0.0,
            "bootstrap_ci95": _bootstrap_ci(diffs, samples, seed + metric_index),
            "win_tie_loss": {
                "win": sum(diff > 1e-12 for diff in diffs),
                "tie": sum(abs(diff) <= 1e-12 for diff in diffs),
                "loss": sum(diff < -1e-12 for diff in diffs),
            },
        }
    return output


def _fact_score(answer: str, fact: dict[str, Any]) -> float:
    candidates = [str(fact.get("fact", ""))]
    candidates.extend(str(alias) for alias in fact.get("aliases") or [])
    return max((_token_f1(answer, candidate) for candidate in candidates), default=0.0)


def _token_f1(left: str, right: str) -> float:
    left_tokens = Counter(_TOKEN_PATTERN.findall(left.lower()))
    right_tokens = Counter(_TOKEN_PATTERN.findall(right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = sum((left_tokens & right_tokens).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(left_tokens.values())
    recall = overlap / sum(right_tokens.values())
    return 2 * precision * recall / (precision + recall)


def _zero_metrics() -> dict[str, float]:
    return {
        "evidence_chain_complete": 0.0,
        "evidence_recall": 0.0,
        "answer_fact_f1": 0.0,
        "citation_precision": 0.0,
        "citation_recall": 0.0,
        "invalid_citation_rate": 0.0,
    }


def _bootstrap_ci(diffs: list[float], samples: int, seed: int) -> list[float]:
    if not diffs or samples <= 0:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = sorted(
        sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs)
        for _ in range(samples)
    )
    return [means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]]


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round((len(values) - 1) * quantile)))
    return values[index]

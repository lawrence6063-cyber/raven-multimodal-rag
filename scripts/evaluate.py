#!/usr/bin/env python3
"""Evaluate script — runs RAG retrieval quality evaluation over a golden set.

Usage:
    # Full pipeline (BM25 + Dense + RRF + Rerank)
    python scripts/evaluate.py [--test-set data/golden_papers.json] [--json]

    # Ablation: disable specific components via env vars
    COGENT_EVAL_NO_RERANK=1 python scripts/evaluate.py --json
    COGENT_EVAL_NO_DENSE=1 python scripts/evaluate.py --json
    COGENT_EVAL_NO_FUSION=1 python scripts/evaluate.py --json
    COGENT_EVAL_NO_SPARSE=1 python scripts/evaluate.py --json

    # Agentic RAG comparison
    COGENT_EVAL_AGENTIC=1 python scripts/evaluate.py --json
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.settings import SettingsError, load_settings
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.observability.evaluation.eval_runner import EvalRunner
from src.observability.logger import get_logger

logger = get_logger("evaluate")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument(
        "--test-set",
        default=None,
        help="Path to golden test set JSON (default: settings.evaluation.golden_test_set)",
    )
    parser.add_argument(
        "--backends",
        default=None,
        help="Comma-separated evaluator backends (default: settings.evaluation.backends)",
    )
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON")
    args = parser.parse_args()

    try:
        settings = load_settings()
    except SettingsError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Ablation flags via env vars
    no_rerank = os.getenv("COGENT_EVAL_NO_RERANK", "").strip() in ("1", "true", "yes")
    no_dense = os.getenv("COGENT_EVAL_NO_DENSE", "").strip() in ("1", "true", "yes")
    no_sparse = os.getenv("COGENT_EVAL_NO_SPARSE", "").strip() in ("1", "true", "yes")
    no_fusion = os.getenv("COGENT_EVAL_NO_FUSION", "").strip() in ("1", "true", "yes")
    use_agentic = os.getenv("COGENT_EVAL_AGENTIC", "").strip() in ("1", "true", "yes")

    # Apply ablation overrides to settings
    if no_rerank:
        settings.rerank.enabled = False
    if no_dense:
        settings.retrieval.dense_weight = 0.0
        settings.retrieval.sparse_weight = 1.0
    if no_sparse:
        settings.retrieval.dense_weight = 1.0
        settings.retrieval.sparse_weight = 0.0
    if no_fusion:
        # Disable fusion by setting dense_weight to 1.0 (fusion degrades to dense_only)
        settings.retrieval.dense_weight = 1.0
        settings.retrieval.sparse_weight = 0.0

    # Determine pipeline label for logging
    label_parts = []
    if not no_dense:
        label_parts.append("Dense")
    if not no_sparse:
        label_parts.append("Sparse")
    if not no_fusion and not no_dense and not no_sparse:
        label_parts.append("RRF")
    if not no_rerank:
        label_parts.append("Rerank")
    pipeline_label = "+".join(label_parts) if label_parts else "empty"
    if use_agentic:
        pipeline_label = "Agentic"

    backends = (
        [b.strip() for b in args.backends.split(",") if b.strip()]
        if args.backends
        else settings.evaluation.backends
    )

    try:
        evaluator = EvaluatorFactory.create_composite(backends)

        if use_agentic:
            report = _run_agentic_eval(settings, evaluator, args.test_set)
        else:
            report = _run_hybrid_eval(settings, evaluator, args.test_set, no_rerank)

    except FileNotFoundError as e:
        logger.error(str(e))
        print(f"\n❌ {e}")
        print("💡 Hint: create the golden test set or pass --test-set <path>.")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001 - surface a friendly message
        logger.error(f"Evaluation failed: {e}")
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(1)

    if args.json:
        report_dict = report.to_dict()
        report_dict["pipeline"] = pipeline_label
        print(json.dumps(report_dict, ensure_ascii=False, indent=2))
        return

    print(f"\n📏 Evaluation Report [{pipeline_label}] — {report.test_set_path}")
    print(f"   Backends: {', '.join(report.backends)}")
    print(f"   Queries:  {report.total_queries}")
    print("=" * 60)
    print("\nMetrics:")
    for name, value in report.metrics.items():
        print(f"  {name:<32} {value:.4f}")

    # Per-category breakdown
    categories = {}
    for item in report.per_query:
        cat = item.get("category", "default")
        if cat not in categories:
            categories[cat] = {"total": 0, "hits": 0}
        categories[cat]["total"] += 1
        if item["hit"]:
            categories[cat]["hits"] += 1

    if len(categories) > 1:
        print("\nPer-category:")
        for cat, stats in sorted(categories.items()):
            rate = stats["hits"] / stats["total"] if stats["total"] else 0
            print(f"  {cat:<20} {stats['hits']}/{stats['total']}  ({rate:.1%})")

    print("\nPer-query:")
    for item in report.per_query:
        flag = "✅" if item["hit"] else "❌"
        cat = item.get("category", "")
        lat = item.get("latency_ms", 0)
        print(f"  {flag} [{cat}] {item['query'][:60]}  ({item['num_retrieved']} results, {lat:.0f}ms)")
    print("=" * 60)


def _run_hybrid_eval(settings, evaluator, test_set, no_rerank):
    """Run standard hybrid search evaluation with optional rerank."""
    from src.core.query_engine.hybrid_search import HybridSearch
    from src.core.query_engine.reranker import QueryReranker

    hybrid = HybridSearch(settings)
    reranker = None
    if settings.rerank.enabled and not no_rerank:
        try:
            reranker = QueryReranker(settings)
        except Exception as e:
            logger.warning(f"Reranker init failed, skipping: {e}")

    runner = EvalRunner(settings, hybrid, evaluator, reranker=reranker)
    return runner.run(test_set)


def _run_agentic_eval(settings, evaluator, test_set):
    """Run Agentic RAG evaluation over the golden set."""
    import time

    from src.core.agent.agentic_rag import AgenticRAG
    from src.core.query_engine.hybrid_search import HybridSearch
    from src.core.query_engine.reranker import QueryReranker
    from src.libs.evaluator.base_evaluator import EvalInput
    from src.observability.evaluation.eval_runner import EvalReport

    hybrid = HybridSearch(settings)
    reranker = None
    if settings.rerank.enabled:
        try:
            reranker = QueryReranker(settings)
        except Exception:
            pass

    agent = AgenticRAG(settings, hybrid_search=hybrid, reranker=reranker)

    path = test_set or settings.evaluation.golden_test_set
    from src.observability.evaluation.eval_runner import EvalRunner

    test_cases = EvalRunner._load_test_set(path)

    eval_inputs = []
    per_query = []
    latencies = []

    for case in test_cases:
        query = case.get("query", "")
        expected_sources = case.get("expected_sources") or []
        category = case.get("category", "default")

        start = time.perf_counter()
        try:
            result = agent.run(query)
            results = result.results
        except Exception as exc:
            logger.warning(f"Agentic query failed '{query}': {exc}")
            results = []
        total_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(total_ms)

        retrieved_sources = [EvalRunner._resolve_source(r) for r in results]
        retrieved_texts = [r.text for r in results]

        golden = expected_sources
        ids = retrieved_sources

        eval_inputs.append(
            EvalInput(
                query=query,
                retrieved_ids=ids,
                golden_ids=golden,
                retrieved_texts=retrieved_texts,
                contexts=retrieved_texts,
            )
        )

        hit = bool(set(ids) & set(golden))
        per_query.append(
            {
                "query": query,
                "category": category,
                "retrieved_sources": retrieved_sources,
                "expected_sources": expected_sources,
                "num_retrieved": len(results),
                "num_expected": len(golden),
                "hit": hit,
                "latency_ms": round(total_ms, 2),
                "fallback": getattr(result, "fallback", False),
                "n_steps": len(getattr(result, "steps", [])),
            }
        )

    eval_result = evaluator.evaluate(eval_inputs)
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    metrics = dict(eval_result.metrics)
    metrics["avg_latency_ms"] = round(avg_lat, 2)

    return EvalReport(
        metrics=metrics,
        per_query=per_query,
        backends=[evaluator.provider_name],
        test_set_path=str(path),
        total_queries=len(test_cases),
    )


if __name__ == "__main__":
    main()

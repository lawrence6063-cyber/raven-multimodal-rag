#!/usr/bin/env python3
"""Evaluate script — runs RAG retrieval quality evaluation over a golden set.

Usage:
    python scripts/evaluate.py [--test-set tests/fixtures/golden_test_set.json] \
        [--backends custom,ragas] [--json]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.query_engine.hybrid_search import HybridSearch
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

    backends = (
        [b.strip() for b in args.backends.split(",") if b.strip()]
        if args.backends
        else settings.evaluation.backends
    )

    try:
        evaluator = EvaluatorFactory.create_composite(backends)
        hybrid = HybridSearch(settings)
        runner = EvalRunner(settings, hybrid, evaluator)
        report = runner.run(args.test_set)
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
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return

    print(f"\n📏 Evaluation Report — {report.test_set_path}")
    print(f"   Backends: {', '.join(report.backends)}")
    print(f"   Queries:  {report.total_queries}")
    print("=" * 60)
    print("\nMetrics:")
    for name, value in report.metrics.items():
        print(f"  {name:<32} {value:.4f}")

    print("\nPer-query:")
    for item in report.per_query:
        flag = "✅" if item["hit"] else "❌"
        print(f"  {flag} {item['query']}  (retrieved {item['num_retrieved']})")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""只跑 multihop 类 query 的 Agentic RAG 评估"""

import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.settings import load_settings
from src.core.agent.agentic_rag import AgenticRAG
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.reranker import QueryReranker
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.libs.evaluator.base_evaluator import EvalInput
from src.observability.evaluation.eval_runner import EvalRunner, EvalReport

def main():
    settings = load_settings()

    # Load only multihop queries
    data = json.load(open('data/golden_papers.json', encoding='utf-8'))
    multihop_cases = [c for c in data['test_cases'] if c.get('category') == 'multihop']

    print(f"Running Agentic RAG on {len(multihop_cases)} multihop queries...\n", flush=True)

    # Setup
    hybrid = HybridSearch(settings)
    reranker = None
    if settings.rerank.enabled:
        try:
            reranker = QueryReranker(settings)
        except Exception:
            pass

    agent = AgenticRAG(settings, hybrid_search=hybrid, reranker=reranker)
    evaluator = EvaluatorFactory.create_composite(["custom"])

    eval_inputs = []
    per_query = []
    latencies = []

    for idx, case in enumerate(multihop_cases, 1):
        query = case.get("query", "")
        expected_sources = case.get("expected_sources") or []
        category = case.get("category", "multihop")

        print(f"[{idx}/{len(multihop_cases)}] {query[:70]}...", flush=True)

        start = time.perf_counter()
        try:
            result = agent.run(query)
            results = result.results
            fallback = getattr(result, "fallback", False)
            n_steps = len(getattr(result, "steps", []))
        except Exception as exc:
            print(f"  ❌ Failed: {exc}", flush=True)
            results = []
            fallback = True
            n_steps = 0

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

        # Compute per-query metrics
        hit = bool(set(ids) & set(golden))
        golden_set = set(golden)
        retrieved_set = set(ids)
        if len(golden_set) > 0:
            query_recall = len(retrieved_set & golden_set) / len(golden_set)
        else:
            query_recall = 0.0

        per_query.append({
            "query": query,
            "category": category,
            "retrieved_sources": retrieved_sources,
            "expected_sources": expected_sources,
            "num_retrieved": len(results),
            "num_expected": len(golden),
            "hit": hit,
            "recall_completeness": round(query_recall, 4),
            "latency_ms": round(total_ms, 2),
            "fallback": fallback,
            "n_steps": n_steps,
        })

        flag = "✅" if hit else "❌"
        print(f"  {flag} {len(results)} results, RC={query_recall:.3f}, {total_ms/1000:.1f}s, steps={n_steps}, fallback={fallback}", flush=True)

    # Overall metrics
    eval_result = evaluator.evaluate(eval_inputs)
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    metrics = dict(eval_result.metrics)
    metrics["avg_latency_ms"] = round(avg_lat, 2)

    report = EvalReport(
        metrics=metrics,
        per_query=per_query,
        backends=[evaluator.provider_name],
        test_set_path="data/golden_papers.json (multihop only)",
        total_queries=len(multihop_cases),
    )

    # Save JSON
    output = report.to_dict()
    output["pipeline"] = "agentic_multihop"
    with open("eval_agentic_multihop.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("Agentic RAG Summary (multihop queries only)")
    print("="*60)
    print(f"Queries: {len(multihop_cases)}")
    print(f"Hit Rate: {metrics['hit_rate']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Recall Completeness: {metrics['recall_completeness']:.4f}")
    print(f"Avg Latency: {avg_lat:.0f}ms ({avg_lat/1000:.1f}s)")
    print(f"\nPer-query:")
    for item in per_query:
        flag = "✅" if item["hit"] else "❌"
        print(f"  {flag} RC={item['recall_completeness']:.3f}  {item['latency_ms']/1000:.1f}s  {item['query'][:60]}...")
    print("="*60)
    print(f"\n✅ Results saved to eval_agentic_multihop.json")

if __name__ == "__main__":
    main()

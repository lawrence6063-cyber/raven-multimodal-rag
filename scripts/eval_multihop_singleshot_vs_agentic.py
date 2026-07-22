#!/usr/bin/env python3
"""受控对比实验：单次混合检索 vs Agentic RAG（多跳 query 上的 Recall Completeness）。

实验目的
    验证 Agentic RAG 编排（route → 查询改写/子问题分解 → 多跳检索 → 反思回灌）
    是否能在**跨域多跳 query** 上提升 Recall Completeness（对多篇预期论文的覆盖度）。

实验设计（配对、单变量）
    - 数据：data/golden_papers.json 中 category == "multihop" 的全部 query。
    - 受控变量：两个 arm 共用同一个 HybridSearch、同一个 reranker，并将最终上下文
      预算对齐为 settings.agent.max_context_chunks（single-shot 的 top_k 与 agentic
      的上下文上限相同）——保证两 arm 输出的 chunk 数量一致，唯一差异是 agentic 编排。
    - 对照组 (single-shot)：一次 hybrid.search + rerank（等价于 agentic 的 fallback 路径）。
    - 实验组 (agentic)：完整 AgenticRAG.run()。
    - 指标：
        * Recall Completeness = |去重后检索到的 source ∩ expected_sources| / |expected_sources|
          （多跳场景下衡量"覆盖了多少篇预期论文"）。
        * Hit Rate = 至少命中一篇预期 source 的 query 占比。
    - 统计（n 较小，配对）：逐 query 差值 + 配对 bootstrap 95% CI + 符号检验（win/tie/loss）。
      n=8 时结论按方向性证据解读（与 docs/EVAL_OPTIMIZATION_SPEC 的谨慎口径一致）。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.agent.agentic_rag import AgenticRAG
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.reranker import QueryReranker
from src.core.settings import load_settings
from src.observability.evaluation.eval_runner import EvalRunner


def _unique_sources(results: list) -> list[str]:
    """去重后按首次出现顺序返回检索结果的 source。"""
    seen: list[str] = []
    for r in results:
        src = EvalRunner._resolve_source(r)
        if src and src not in seen:
            seen.append(src)
    return seen


def _recall_completeness(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """source 级召回完整度：预期论文被覆盖的比例。"""
    golden = set(expected_sources)
    if not golden:
        return 0.0
    return len(golden & set(retrieved_sources)) / len(golden)


def _single_shot(hybrid: HybridSearch, reranker, query: str, top_k: int) -> list:
    """对照组：一次混合检索 + 可选精排。"""
    results = hybrid.search(query=query, top_k=top_k)
    if reranker is not None and results:
        results = reranker.rerank(query, results)
    return results


def _paired_bootstrap_ci(diffs: list[float], n_boot: int = 10000, seed: int = 42) -> tuple[float, float]:
    """对逐 query 差值做配对 bootstrap，返回均值差的 95% CI。"""
    if not diffs:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(diffs)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return round(lo, 4), round(hi, 4)


def _sign_test(diffs: list[float], eps: float = 1e-9) -> dict:
    """符号检验：统计 agentic 相对 single-shot 的 win/tie/loss。"""
    win = sum(1 for d in diffs if d > eps)
    loss = sum(1 for d in diffs if d < -eps)
    tie = len(diffs) - win - loss
    return {"win": win, "tie": tie, "loss": loss}


def _run_arm(name: str, run_fn, cases: list[dict]) -> dict:
    """在全部 case 上跑一个 arm，返回逐 query 明细 + 汇总指标。"""
    per_query = []
    for idx, case in enumerate(cases, 1):
        query = case.get("query", "")
        expected = case.get("expected_sources") or []

        start = time.perf_counter()
        results, extra = run_fn(query)
        latency_ms = (time.perf_counter() - start) * 1000.0

        sources = _unique_sources(results)
        rc = _recall_completeness(sources, expected)
        hit = bool(set(sources) & set(expected))

        row = {
            "query": query,
            "num_expected": len(expected),
            "num_retrieved_chunks": len(results),
            "num_unique_sources": len(sources),
            "covered_sources": sorted(set(sources) & set(expected)),
            "recall_completeness": round(rc, 4),
            "hit": hit,
            "latency_ms": round(latency_ms, 2),
        }
        row.update(extra)
        per_query.append(row)
        flag = "✅" if hit else "❌"
        print(
            f"  [{name}] [{idx}/{len(cases)}] {flag} RC={rc:.3f} "
            f"({len(set(sources) & set(expected))}/{len(expected)} papers) "
            f"{latency_ms / 1000:.1f}s | {query[:48]}...",
            flush=True,
        )

    n = len(per_query)
    mean_rc = sum(r["recall_completeness"] for r in per_query) / n if n else 0.0
    hit_rate = sum(1 for r in per_query if r["hit"]) / n if n else 0.0
    avg_lat = sum(r["latency_ms"] for r in per_query) / n if n else 0.0
    return {
        "arm": name,
        "recall_completeness": round(mean_rc, 4),
        "hit_rate": round(hit_rate, 4),
        "avg_latency_ms": round(avg_lat, 2),
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-set", default="data/golden_papers.json", help="golden 测试集路径"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="最终上下文预算（两 arm 对齐）；默认取 settings.agent.max_context_chunks",
    )
    parser.add_argument(
        "--out",
        default="experiments/results/multihop_singleshot_vs_agentic.json",
        help="结果 JSON 输出路径",
    )
    args = parser.parse_args()

    settings = load_settings()
    top_k = args.top_k or settings.agent.max_context_chunks

    data = json.load(open(args.test_set, encoding="utf-8"))
    cases = [c for c in data["test_cases"] if c.get("category") == "multihop"]
    if not cases:
        print("未找到 multihop query，退出", flush=True)
        sys.exit(2)

    print(
        f"多跳 query 数: {len(cases)} | 对齐上下文预算 top_k={top_k} "
        f"| rerank={settings.rerank.enabled}({settings.rerank.provider})",
        flush=True,
    )
    print(
        f"agentic 配置: route={settings.agent.route_enabled} "
        f"rewrite={settings.agent.rewrite_enabled} multihop={settings.agent.multihop_enabled} "
        f"reflect={settings.agent.reflect_enabled} max_hops={settings.agent.max_hops}\n",
        flush=True,
    )

    hybrid = HybridSearch(settings)
    reranker = None
    if settings.rerank.enabled:
        try:
            reranker = QueryReranker(settings)
        except Exception as exc:  # noqa: BLE001 - 精排不可用时降级为纯检索
            print(f"reranker 不可用，降级为纯检索: {exc}", flush=True)

    agent = AgenticRAG(settings, hybrid_search=hybrid, reranker=reranker)

    def run_single(query: str):
        try:
            results = _single_shot(hybrid, reranker, query, top_k)
            return results, {}
        except Exception as exc:  # noqa: BLE001
            print(f"    single-shot 失败: {exc}", flush=True)
            return [], {"error": str(exc)[:200]}

    def run_agentic(query: str):
        try:
            res = agent.run(query, top_k=top_k)
            return res.results, {
                "n_steps": len(getattr(res, "steps", [])),
                "fallback": bool(getattr(res, "fallback", False)),
            }
        except Exception as exc:  # noqa: BLE001
            print(f"    agentic 失败: {exc}", flush=True)
            return [], {"error": str(exc)[:200], "fallback": True}

    print("=== 对照组：单次混合检索 (single-shot) ===", flush=True)
    single = _run_arm("single", run_single, cases)
    print("\n=== 实验组：Agentic RAG ===", flush=True)
    agentic = _run_arm("agentic", run_agentic, cases)

    # 配对统计（逐 query 差值：agentic - single）
    diffs = [
        a["recall_completeness"] - s["recall_completeness"]
        for a, s in zip(agentic["per_query"], single["per_query"])
    ]
    ci_lo, ci_hi = _paired_bootstrap_ci(diffs)
    signs = _sign_test(diffs)

    comparison = {
        "metric": "recall_completeness (source-level coverage of expected papers)",
        "n_queries": len(cases),
        "top_k": top_k,
        "single_shot": {
            "recall_completeness": single["recall_completeness"],
            "hit_rate": single["hit_rate"],
            "avg_latency_ms": single["avg_latency_ms"],
        },
        "agentic": {
            "recall_completeness": agentic["recall_completeness"],
            "hit_rate": agentic["hit_rate"],
            "avg_latency_ms": agentic["avg_latency_ms"],
        },
        "delta_recall_completeness": round(
            agentic["recall_completeness"] - single["recall_completeness"], 4
        ),
        "delta_hit_rate": round(agentic["hit_rate"] - single["hit_rate"], 4),
        "paired_diff_bootstrap_ci95": [ci_lo, ci_hi],
        "sign_test": signs,
    }

    output = {
        "experiment": "multihop_singleshot_vs_agentic",
        "test_set_path": args.test_set,
        "comparison": comparison,
        "single_shot_detail": single,
        "agentic_detail": agentic,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 汇总打印
    print("\n" + "=" * 66)
    print("Recall Completeness 对比（单次混合检索 vs Agentic RAG）")
    print("=" * 66)
    print(f"多跳 query 数: {len(cases)}  |  对齐 top_k={top_k}")
    print(f"{'':<14}{'RecallCompl.':>14}{'HitRate':>10}{'AvgLatency':>14}")
    print(
        f"{'single-shot':<14}{single['recall_completeness']:>14.4f}"
        f"{single['hit_rate']:>10.2%}{single['avg_latency_ms'] / 1000:>12.1f}s"
    )
    print(
        f"{'agentic':<14}{agentic['recall_completeness']:>14.4f}"
        f"{agentic['hit_rate']:>10.2%}{agentic['avg_latency_ms'] / 1000:>12.1f}s"
    )
    print("-" * 66)
    print(
        f"ΔRecallCompleteness = {comparison['delta_recall_completeness']:+.4f}  "
        f"(配对 bootstrap 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}])"
    )
    print(f"ΔHitRate            = {comparison['delta_hit_rate']:+.4f}")
    print(
        f"符号检验            = win {signs['win']} / tie {signs['tie']} / loss {signs['loss']}"
    )
    print("=" * 66)
    print(f"\n✅ 结果已写入 {out_path}")


if __name__ == "__main__":
    main()

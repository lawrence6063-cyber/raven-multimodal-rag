#!/usr/bin/env python3
"""Run frozen composition-QA arms and merge their paired experiment report."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.agent.agentic_rag import AgenticRAG
from src.core.agent.answer_synthesizer import AnswerSynthesizer
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.reranker import QueryReranker
from src.core.settings import load_settings
from src.observability.evaluation.agentic_bridge_evaluator import (
    evaluate_case,
    paired_comparison,
    summarize,
)

_DEFAULT_DATASET = "data/agentic_bridge_qa.json"
_DEFAULT_RESULT = "experiments/results/agentic_bridge_qa.json"
_ARMS = ("single", "decompose", "agentic")


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _citation_chunks(context: list, citation_ids: list[int]) -> list[str]:
    return [
        context[index - 1].chunk_id
        for index in citation_ids
        if 1 <= index <= len(context)
    ]


def _single_shot(hybrid, reranker, synthesizer, query: str, retrieval_top_k: int, budget: int):
    results = hybrid.search(query=query, top_k=retrieval_top_k)
    if reranker is not None and results:
        results = reranker.rerank(query, results)
    context = results[:budget]
    synth = synthesizer.answer(query, context)
    return {
        "answer": synth.answer,
        "results": context,
        "used_citation_ids": synth.used_citation_ids,
        "cited_chunk_ids": _citation_chunks(context, synth.used_citation_ids),
        "steps": [],
        "fallback": False,
        "retrieval_calls": 1,
        "llm_calls": 1,
        "audit": {},
    }


def _agent_run(agent: AgenticRAG, query: str, retrieval_top_k: int) -> dict[str, Any]:
    result = agent.run(query, top_k=retrieval_top_k)
    retrieval_calls = sum(
        len(step.get("subqueries", []))
        for step in result.steps
        if str(step.get("stage", "")).startswith("hop_")
    )
    llm_calls = sum(
        1
        for step in result.steps
        if step.get("stage") in {"route", "rewrite", "synthesize"}
        or str(step.get("stage", "")).startswith("reflect_")
    )
    return {
        "answer": result.answer,
        "results": result.results,
        "used_citation_ids": result.used_citation_ids,
        "cited_chunk_ids": result.cited_chunk_ids,
        "steps": result.steps,
        "fallback": result.fallback,
        "retrieval_calls": retrieval_calls,
        "llm_calls": llm_calls,
        "audit": result.audit,
    }


def _build_agent(settings, hybrid, reranker, arm: str) -> AgenticRAG:
    arm_settings = copy.deepcopy(settings)
    arm_settings.agent.synthesize_answer = True
    if arm == "decompose":
        arm_settings.agent.route_enabled = False
        arm_settings.agent.rewrite_enabled = True
        arm_settings.agent.multihop_enabled = False
        arm_settings.agent.reflect_enabled = False
    return AgenticRAG(arm_settings, hybrid_search=hybrid, reranker=reranker)


def _serialize_row(case: dict[str, Any], raw: dict[str, Any], latency_ms: float) -> dict[str, Any]:
    results = raw.get("results") or []
    row = {
        "case_id": case["id"],
        "query": case["query"],
        "answer": raw.get("answer", ""),
        "retrieved_chunk_ids": [result.chunk_id for result in results],
        "retrieved_sources": [result.metadata.get("source_path", "") for result in results],
        "used_citation_ids": raw.get("used_citation_ids") or [],
        "cited_chunk_ids": raw.get("cited_chunk_ids") or [],
        "latency_ms": round(latency_ms, 2),
        "retrieval_calls": raw.get("retrieval_calls", 0),
        "llm_calls": raw.get("llm_calls", 0),
        "fallback": bool(raw.get("fallback", False)),
        "steps": raw.get("steps") or [],
        "audit": raw.get("audit") or {},
        "error": raw.get("error", ""),
    }
    row["metrics"] = evaluate_case(case, row)
    return row


def run_arm(args: argparse.Namespace) -> None:
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    cases = [case for case in dataset["cases"] if case.get("split") == args.split]
    settings = load_settings()
    settings.agent.retrieval_top_k = args.retrieval_top_k
    settings.agent.max_context_chunks = args.context_budget
    hybrid = HybridSearch(settings)
    try:
        reranker = QueryReranker(settings) if settings.rerank.enabled else None
    except Exception as exc:  # noqa: BLE001
        print(f"reranker unavailable, using retrieval order: {exc}", flush=True)
        reranker = None
    synthesizer = AnswerSynthesizer(settings)
    agent = _build_agent(settings, hybrid, reranker, args.arm) if args.arm != "single" else None

    output_path = Path(args.out)
    rows: list[dict[str, Any]] = []
    if output_path.exists() and not args.restart:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("dataset_sha256") != dataset.get("dataset_sha256"):
            raise RuntimeError("checkpoint dataset hash does not match frozen dataset")
        rows = existing.get("rows") or []
    completed = {row["case_id"] for row in rows}

    config = {
        "arm": args.arm,
        "split": args.split,
        "retrieval_top_k": args.retrieval_top_k,
        "context_budget": args.context_budget,
        "llm_provider": settings.llm.provider,
        "llm_model": settings.llm.model,
        "embedding_provider": settings.embedding.provider,
        "embedding_model": settings.embedding.model,
        "rerank_provider": settings.rerank.provider if reranker is not None else "disabled",
        "max_hops": settings.agent.max_hops,
        "max_subqueries": settings.agent.max_subqueries,
        "max_reflect_rounds": settings.agent.max_reflect_rounds,
    }
    payload = {
        "experiment": "agentic_composition_qa_arm",
        "dataset": args.dataset,
        "dataset_sha256": dataset.get("dataset_sha256"),
        "config": config,
        "rows": rows,
        "summary": summarize(rows),
    }
    for index, case in enumerate(cases, start=1):
        if case["id"] in completed:
            continue
        start = time.perf_counter()
        try:
            if args.arm == "single":
                raw = _single_shot(
                    hybrid,
                    reranker,
                    synthesizer,
                    case["query"],
                    args.retrieval_top_k,
                    args.context_budget,
                )
            else:
                raw = _agent_run(agent, case["query"], args.retrieval_top_k)
        except Exception as exc:  # noqa: BLE001
            raw = {"error": f"{type(exc).__name__}: {str(exc)[:300]}", "results": []}
        latency_ms = (time.perf_counter() - start) * 1000.0
        row = _serialize_row(case, raw, latency_ms)
        rows.append(row)
        payload["rows"] = rows
        payload["summary"] = summarize(rows)
        _atomic_write(output_path, payload)
        print(
            f"[{args.arm}] [{index}/{len(cases)}] {case['id']} "
            f"chain={row['metrics']['evidence_chain_complete']:.0f} "
            f"fact_f1={row['metrics']['answer_fact_f1']:.3f} "
            f"latency={latency_ms / 1000:.1f}s",
            flush=True,
        )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


def merge_arms(args: argparse.Namespace) -> None:
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.merge]
    hashes = {report.get("dataset_sha256") for report in reports}
    if len(hashes) != 1:
        raise RuntimeError("arm reports use different frozen datasets")
    by_arm = {report["config"]["arm"]: report for report in reports}
    missing = set(_ARMS) - set(by_arm)
    if missing:
        raise RuntimeError(f"missing arms: {sorted(missing)}")
    output = {
        "experiment": "agentic_composition_qa",
        "dataset": by_arm["single"]["dataset"],
        "dataset_sha256": hashes.pop(),
        "preregistered_primary_metrics": ["evidence_chain_complete", "answer_fact_f1"],
        "arms": {
            arm: {
                "config": by_arm[arm]["config"],
                "summary": by_arm[arm]["summary"],
                "rows": by_arm[arm]["rows"],
            }
            for arm in _ARMS
        },
        "comparisons": {
            "agentic_vs_single": paired_comparison(
                by_arm["single"]["rows"], by_arm["agentic"]["rows"]
            ),
            "decompose_vs_single": paired_comparison(
                by_arm["single"]["rows"], by_arm["decompose"]["rows"]
            ),
        },
    }
    _atomic_write(Path(args.out), output)
    print(json.dumps(output["comparisons"], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=_DEFAULT_DATASET)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--arm", choices=_ARMS)
    parser.add_argument("--merge", nargs=3, metavar=("SINGLE", "DECOMPOSE", "AGENTIC"))
    parser.add_argument("--out", default=_DEFAULT_RESULT)
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--context-budget", type=int, default=20)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    if bool(args.arm) == bool(args.merge):
        parser.error("provide exactly one of --arm or --merge")
    if args.merge:
        merge_arms(args)
    else:
        run_arm(args)


if __name__ == "__main__":
    main()

"""AgenticRAG — the Agentic RAG orchestrator (OPTIMIZATION_SPEC §3).

Coordinates the LLM-driven retrieval pipeline:

    route → (transform → multi-hop retrieve → reflect)* → synthesize

This module lands incrementally per ``docs/P1_AGENTIC_RAG_SPEC.md`` milestones.
M-C1 wires the minimal closed loop: route → single retrieval → synthesize, with
a global try/except that degrades to a single hybrid search (+ best-effort
synthesis) on any failure — the user never sees an agent error.

All collaborators are injectable for offline testing; when omitted they are
lazily constructed from ``Settings``. The retrieval/rerank/LLM internals are
reused unchanged (HybridSearch / QueryReranker / LLMFactory).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.agent.agent_types import AgentResult, RouteDecision, SynthResult
from src.core.agent.context_selector import ContextSelection, RetrievalBatch, select_context
from src.observability.logger import get_logger

if TYPE_CHECKING:
    from src.core.agent.agent_types import SubQuery
    from src.core.agent.answer_synthesizer import AnswerSynthesizer
    from src.core.agent.collection_registry import CollectionRegistry
    from src.core.agent.query_transformer import QueryTransformer
    from src.core.agent.reflector import Reflector
    from src.core.agent.router import QueryRouter
    from src.core.query_engine.hybrid_search import HybridSearch
    from src.core.query_engine.reranker import QueryReranker
    from src.core.settings import Settings
    from src.core.trace.trace_context import TraceContext
    from src.core.types import RetrievalResult

logger = get_logger("core.agent.agentic_rag")


class AgenticRAG:
    """LLM-driven retrieval orchestrator with global degradation."""

    def __init__(
        self,
        settings: "Settings",
        hybrid_search: "HybridSearch | None" = None,
        reranker: "QueryReranker | None" = None,
        router: "QueryRouter | None" = None,
        transformer: "QueryTransformer | None" = None,
        reflector: "Reflector | None" = None,
        synthesizer: "AnswerSynthesizer | None" = None,
        registry: "CollectionRegistry | None" = None,
    ):
        self._settings = settings
        self._hybrid = hybrid_search
        self._reranker = reranker
        self._router = router
        self._transformer = transformer
        self._reflector = reflector
        self._synthesizer = synthesizer
        self._registry = registry

    # ------------------------------------------------------------------ run
    def run(
        self,
        query: str,
        collection: str | None = None,
        image: str | bytes | None = None,
        top_k: int | None = None,
        trace: "TraceContext | None" = None,
    ) -> AgentResult:
        """Execute the agentic pipeline for ``query``.

        Never raises: any failure degrades to a single hybrid retrieval with a
        best-effort synthesized answer (``AgentResult.fallback=True``).

        Args:
            query: The user question.
            collection: Optional explicit collection filter (overrides routing).
            image: Optional query image for cross-modal retrieval.
            top_k: Optional per-query result count override.
            trace: Optional TraceContext for stage instrumentation.

        Returns:
            AgentResult with the synthesized answer, the results used for
            citation, and per-step decision metadata.
        """
        steps: list[dict[str, Any]] = []
        try:
            return self._run_agentic(query, collection, image, top_k, trace, steps)
        except Exception as e:  # global degradation — never surface agent errors
            logger.warning(f"Agentic pipeline failed, degrading to single search: {e}")
            return self._fallback(query, collection, image, top_k, trace, steps, str(e))

    def _run_agentic(
        self,
        query: str,
        collection: str | None,
        image: str | bytes | None,
        top_k: int | None,
        trace: "TraceContext | None",
        steps: list[dict[str, Any]],
    ) -> AgentResult:
        """The happy-path agentic pipeline: route → transform → hops → synthesize."""
        cfg = self._settings.agent

        # 1. Route — decide whether to retrieve and which collections apply.
        decision = self._route(query, trace)
        steps.append(
            {
                "stage": "route",
                "need_retrieval": decision.need_retrieval,
                "collections": list(decision.target_collections),
            }
        )

        # 2. Direct-answer path (no retrieval needed and the router supplied one).
        if not decision.need_retrieval and decision.direct_answer:
            steps.append({"stage": "direct_answer"})
            return AgentResult(answer=decision.direct_answer, results=[], steps=steps)

        # 3. Resolve collection filter: explicit param wins; else a single routed
        #    collection is applied (multi-collection routing remains future work).
        target_collection = collection
        if target_collection is None and len(decision.target_collections) == 1:
            target_collection = decision.target_collections[0]

        # 4. Transform — rewrite/decompose into focused sub-queries.
        subqueries = self._transform(query, trace)
        steps.append({"stage": "rewrite", "n_subqueries": len(subqueries)})

        # 5. Multi-hop retrieval loop with fair cross-query context selection.
        selection = self._retrieve_loop(
            query, subqueries, target_collection, image, top_k, trace, steps
        )
        results = selection.results
        audit = selection.to_audit()

        # 6. Synthesize the answer (server-side), grounded in the results.
        synth = self._synthesize(query, results, trace)
        cited_chunk_ids = [
            results[index - 1].chunk_id
            for index in synth.used_citation_ids
            if 1 <= index <= len(results)
        ]
        steps.append(
            {
                "stage": "synthesize",
                "answer_len": len(synth.answer),
                "n_citations": len(synth.used_citation_ids),
            }
        )

        return AgentResult(
            answer=synth.answer,
            results=results,
            steps=steps,
            used_citation_ids=synth.used_citation_ids,
            cited_chunk_ids=cited_chunk_ids,
            audit=audit,
        )

    def _transform(
        self, query: str, trace: "TraceContext | None"
    ) -> list["SubQuery"]:
        """Decompose the query when rewrite is enabled; else single sub-query."""
        from src.core.agent.agent_types import SubQuery

        if not self._settings.agent.rewrite_enabled:
            return [SubQuery(text=query)]
        return self._get_transformer().transform(query, trace=trace)

    def _retrieve_loop(
        self,
        query: str,
        subqueries: list["SubQuery"],
        collection: str | None,
        image: str | bytes | None,
        top_k: int | None,
        trace: "TraceContext | None",
        steps: list[dict[str, Any]],
    ) -> ContextSelection:
        """Retrieve iteratively and select context fairly across requests."""
        cfg = self._settings.agent
        max_hops = cfg.max_hops if cfg.multihop_enabled else 1
        batches: list[RetrievalBatch] = []
        known_candidates: set[str] = set()
        asked: set[str] = set()
        pending = [subquery.text for subquery in subqueries]
        reflect_rounds = 0

        hop = 0
        while pending and hop < max_hops:
            hop += 1
            current, pending = pending, []
            new_hits = 0
            for request_index, subquery in enumerate(current, start=1):
                asked.add(subquery.strip().lower())
                results = self._retrieve(subquery, collection, image, top_k, trace)
                results = self._maybe_rerank(subquery, results, trace)
                request_id = f"hop-{hop}-request-{request_index}"
                batches.append(
                    RetrievalBatch(
                        request_id=request_id,
                        query=subquery,
                        hop=hop,
                        results=results,
                    )
                )
                result_ids = {result.chunk_id for result in results}
                new_hits += len(result_ids - known_candidates)
                known_candidates.update(result_ids)

            selection = select_context(batches, cfg.max_context_chunks)
            context = selection.results
            self._record_hop(trace, hop, current, new_hits, len(context))
            steps.append(
                {
                    "stage": f"hop_{hop}",
                    "subqueries": current,
                    "new_hits": new_hits,
                    "candidate_count": selection.unique_candidate_count,
                    "context_size": len(context),
                }
            )

            if (
                cfg.reflect_enabled
                and context
                and hop < max_hops
                and reflect_rounds < cfg.max_reflect_rounds
            ):
                verdict = self._get_reflector().assess(query, context, trace=trace)
                steps.append(
                    {
                        "stage": f"reflect_{hop}",
                        "sufficient": verdict.sufficient,
                        "n_followup": len(verdict.follow_up_queries),
                    }
                )
                if not verdict.sufficient and verdict.follow_up_queries:
                    reflect_rounds += 1
                    pending = [
                        follow_up
                        for follow_up in verdict.follow_up_queries
                        if follow_up.strip().lower() not in asked
                    ]

        return select_context(batches, cfg.max_context_chunks)

    @staticmethod
    def _record_hop(
        trace: "TraceContext | None",
        hop: int,
        subqueries: list[str],
        new_hits: int,
        total: int,
    ) -> None:
        """Record an ``agent_hop_{n}`` trace stage when a trace is present."""
        if trace is None:
            return
        trace.record_stage(
            f"agent_hop_{hop}",
            method="hybrid_search",
            elapsed_ms=0.0,
            n_subqueries=len(subqueries),
            new_hits=new_hits,
            context_size=total,
        )

    # -------------------------------------------------------------- helpers
    def _route(self, query: str, trace: "TraceContext | None") -> RouteDecision:
        """Run routing when enabled; otherwise default to retrieve-all."""
        if not self._settings.agent.route_enabled:
            return RouteDecision(need_retrieval=True, target_collections=[])
        available = self._get_registry().list_collections()
        return self._get_router().decide(query, available, trace=trace)

    def _retrieve(
        self,
        query: str,
        collection: str | None,
        image: str | bytes | None,
        top_k: int | None,
        trace: "TraceContext | None",
    ) -> list["RetrievalResult"]:
        """Run one hybrid search, optionally filtered by collection."""
        filters = {"collection": collection} if collection else None
        k = top_k or self._settings.agent.retrieval_top_k
        return self._get_hybrid().search(
            query=query, top_k=k, filters=filters, trace=trace, image=image
        )

    def _maybe_rerank(
        self,
        query: str,
        results: list["RetrievalResult"],
        trace: "TraceContext | None",
    ) -> list["RetrievalResult"]:
        """Rerank when enabled and a text query drives it; never blocks on error."""
        if not results or not self._settings.rerank.enabled:
            return results
        reranker = self._get_reranker()
        if reranker is None:
            return results
        return reranker.rerank(query, results, trace=trace)

    def _synthesize(
        self,
        query: str,
        results: list["RetrievalResult"],
        trace: "TraceContext | None",
    ) -> SynthResult:
        """Synthesize a grounded answer and preserve its citation indices."""
        if not self._settings.agent.synthesize_answer:
            return SynthResult()
        return self._get_synthesizer().answer(query, results, trace=trace)

    def _fallback(
        self,
        query: str,
        collection: str | None,
        image: str | bytes | None,
        top_k: int | None,
        trace: "TraceContext | None",
        steps: list[dict[str, Any]],
        reason: str,
    ) -> AgentResult:
        """Degrade to a single hybrid search with best-effort synthesis."""
        results: list["RetrievalResult"] = []
        try:
            results = self._retrieve(query, collection, image, top_k, trace)
        except Exception as e:  # retrieval itself failed — return empty gracefully
            logger.warning(f"Fallback retrieval failed: {e}")

        synth = SynthResult()
        if self._settings.agent.synthesize_answer and results:
            try:
                synth = self._get_synthesizer().answer(query, results, trace=trace)
            except Exception as e:
                logger.warning(f"Fallback synthesis failed, returning results only: {e}")

        cited_chunk_ids = [
            results[index - 1].chunk_id
            for index in synth.used_citation_ids
            if 1 <= index <= len(results)
        ]
        if trace is not None:
            trace.record_stage(
                "agent_fallback", method="agentic_rag", elapsed_ms=0.0, reason=reason[:200]
            )
        steps.append({"stage": "fallback", "reason": reason[:200]})
        return AgentResult(
            answer=synth.answer,
            results=results,
            steps=steps,
            fallback=True,
            used_citation_ids=synth.used_citation_ids,
            cited_chunk_ids=cited_chunk_ids,
        )

    # ----------------------------------------------------------- lazy deps
    def _get_hybrid(self) -> "HybridSearch":
        if self._hybrid is None:
            from src.core.query_engine.hybrid_search import HybridSearch

            self._hybrid = HybridSearch(self._settings)
        return self._hybrid

    def _get_reranker(self) -> "QueryReranker | None":
        if self._reranker is None:
            try:
                from src.core.query_engine.reranker import QueryReranker

                self._reranker = QueryReranker(self._settings)
            except Exception as e:
                logger.warning(f"Reranker unavailable, skipping: {e}")
                return None
        return self._reranker

    def _get_router(self) -> "QueryRouter":
        if self._router is None:
            from src.core.agent.router import QueryRouter

            self._router = QueryRouter(self._settings)
        return self._router

    def _get_transformer(self) -> "QueryTransformer":
        if self._transformer is None:
            from src.core.agent.query_transformer import QueryTransformer

            self._transformer = QueryTransformer(self._settings)
        return self._transformer

    def _get_reflector(self) -> "Reflector":
        if self._reflector is None:
            from src.core.agent.reflector import Reflector

            self._reflector = Reflector(self._settings)
        return self._reflector

    def _get_synthesizer(self) -> "AnswerSynthesizer":
        if self._synthesizer is None:
            from src.core.agent.answer_synthesizer import AnswerSynthesizer

            self._synthesizer = AnswerSynthesizer(self._settings)
        return self._synthesizer

    def _get_registry(self) -> "CollectionRegistry":
        if self._registry is None:
            from src.core.agent.collection_registry import CollectionRegistry

            self._registry = CollectionRegistry()
        return self._registry

"""Tests for AgenticRAG orchestrator — route/retrieve/synthesize + degradation."""

from __future__ import annotations

from src.core.agent.agent_types import (
    AgentResult,
    ReflectVerdict,
    RouteDecision,
    SubQuery,
    SynthResult,
)
from src.core.agent.agentic_rag import AgenticRAG
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult
from src.libs.llm.base_llm import LLMError


class FakeHybrid:
    """Records calls and returns preset results (or raises).

    When ``results_by_query`` is provided, returns per-query results to exercise
    multi-hop accumulation; otherwise returns the same ``results`` for any query.
    """

    def __init__(self, results=None, error: Exception | None = None, results_by_query=None):
        self._results = results or []
        self._error = error
        self._by_query = results_by_query or {}
        self.calls: list[dict] = []

    def search(self, query="", top_k=None, filters=None, trace=None, image=None):
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        if self._error is not None:
            raise self._error
        if query in self._by_query:
            return list(self._by_query[query])
        return list(self._results)


class FakeRouter:
    def __init__(self, decision: RouteDecision):
        self._decision = decision
        self.calls: list[tuple] = []

    def decide(self, query, available_collections, trace=None):
        self.calls.append((query, list(available_collections)))
        return self._decision


class FakeTransformer:
    """Returns preset sub-queries (defaults to the original single query)."""

    def __init__(self, subqueries: list[str] | None = None):
        self._subqueries = subqueries

    def transform(self, query, trace=None):
        texts = self._subqueries if self._subqueries is not None else [query]
        return [SubQuery(text=t) for t in texts]


class FakeSynth:
    def __init__(self, answer="synth answer", error: Exception | None = None):
        self._answer = answer
        self._error = error
        self.calls = 0

    def answer(self, query, context, trace=None):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return SynthResult(
            answer=self._answer, used_citation_ids=list(range(1, len(context) + 1))
        )


class FakeRegistry:
    def __init__(self, names):
        self._names = names

    def list_collections(self):
        return list(self._names)


class FakeReflector:
    """Returns preset verdicts in order; defaults to 'sufficient' afterwards."""

    def __init__(self, verdicts: list[ReflectVerdict] | None = None):
        self._verdicts = list(verdicts or [])
        self.calls = 0

    def assess(self, query, context, trace=None):
        self.calls += 1
        if self._verdicts:
            return self._verdicts.pop(0)
        return ReflectVerdict(sufficient=True)


def _results(n=2):
    return [
        RetrievalResult(chunk_id=f"c{i}", score=1.0, text=f"p{i}") for i in range(1, n + 1)
    ]


def _build(settings=None, **kw):
    kw.setdefault("transformer", FakeTransformer())
    kw.setdefault("reflector", FakeReflector())
    return AgenticRAG(settings or Settings(), **kw)


class TestAgenticRAGHappyPath:
    def test_route_retrieve_synthesize(self):
        hybrid = FakeHybrid(_results(2))
        agent = _build(
            hybrid_search=hybrid,
            router=FakeRouter(RouteDecision(need_retrieval=True, target_collections=["rag"])),
            synthesizer=FakeSynth("the answer"),
            registry=FakeRegistry(["rag", "llm"]),
        )
        out = agent.run("what is rag?")
        assert out.answer == "the answer"
        assert len(out.results) == 2
        assert out.fallback is False
        # single routed collection becomes the search filter
        assert hybrid.calls[0]["filters"] == {"collection": "rag"}
        stages = [s["stage"] for s in out.steps]
        assert stages == ["route", "rewrite", "hop_1", "reflect_1", "synthesize"]

    def test_multi_subquery_dedup_accumulation(self):
        # two sub-queries returning overlapping chunks → deduped by chunk_id
        a = [RetrievalResult(chunk_id="c1", score=1.0, text="p1"),
             RetrievalResult(chunk_id="c2", score=0.9, text="p2")]
        b = [RetrievalResult(chunk_id="c2", score=0.5, text="p2"),
             RetrievalResult(chunk_id="c3", score=0.8, text="p3")]
        hybrid = FakeHybrid(results_by_query={"sub a": a, "sub b": b})
        agent = _build(
            hybrid_search=hybrid,
            router=FakeRouter(RouteDecision(need_retrieval=True)),
            transformer=FakeTransformer(["sub a", "sub b"]),
            synthesizer=FakeSynth(),
            registry=FakeRegistry([]),
        )
        out = agent.run("q")
        ids = sorted(r.chunk_id for r in out.results)
        assert ids == ["c1", "c2", "c3"]  # c2 deduped
        # higher score kept for c2 (1.0 first stays since 0.5 < 0.9? c2 first score 0.9, then 0.5 -> keep 0.9)
        c2 = next(r for r in out.results if r.chunk_id == "c2")
        assert c2.score == 0.9

    def test_context_budget_caps_results(self):
        s = Settings()
        s.agent.max_context_chunks = 2
        hybrid = FakeHybrid(_results(5))
        agent = _build(
            s,
            hybrid_search=hybrid,
            router=FakeRouter(RouteDecision(need_retrieval=True)),
            synthesizer=FakeSynth(),
            registry=FakeRegistry([]),
        )
        out = agent.run("q")
        assert len(out.results) == 2

    def test_explicit_collection_overrides_routing(self):
        hybrid = FakeHybrid(_results(1))
        agent = _build(
            hybrid_search=hybrid,
            router=FakeRouter(RouteDecision(need_retrieval=True, target_collections=["rag"])),
            synthesizer=FakeSynth(),
            registry=FakeRegistry(["rag", "llm"]),
        )
        agent.run("q", collection="llm")
        assert hybrid.calls[0]["filters"] == {"collection": "llm"}

    def test_direct_answer_skips_retrieval(self):
        hybrid = FakeHybrid(_results(2))
        agent = _build(
            hybrid_search=hybrid,
            router=FakeRouter(
                RouteDecision(need_retrieval=False, direct_answer="Hello there")
            ),
            synthesizer=FakeSynth(),
            registry=FakeRegistry(["rag"]),
        )
        out = agent.run("hi")
        assert out.answer == "Hello there"
        assert out.results == []
        assert hybrid.calls == []  # retrieval skipped
        assert [s["stage"] for s in out.steps] == ["route", "direct_answer"]

    def test_route_disabled_retrieves_all(self):
        s = Settings()
        s.agent.route_enabled = False
        hybrid = FakeHybrid(_results(1))
        agent = _build(s, hybrid_search=hybrid, synthesizer=FakeSynth())
        agent.run("q")
        assert hybrid.calls[0]["filters"] is None

    def test_rewrite_disabled_uses_original_query(self):
        s = Settings()
        s.agent.rewrite_enabled = False
        hybrid = FakeHybrid(_results(1))
        # transformer must NOT be consulted when rewrite disabled
        agent = AgenticRAG(
            s,
            hybrid_search=hybrid,
            router=FakeRouter(RouteDecision(need_retrieval=True)),
            synthesizer=FakeSynth(),
            registry=FakeRegistry([]),
        )
        agent.run("original")
        assert hybrid.calls[0]["query"] == "original"

    def test_synthesize_disabled_returns_results_only(self):
        s = Settings()
        s.agent.synthesize_answer = False
        synth = FakeSynth()
        agent = _build(
            s,
            hybrid_search=FakeHybrid(_results(2)),
            router=FakeRouter(RouteDecision(need_retrieval=True)),
            synthesizer=synth,
            registry=FakeRegistry([]),
        )
        out = agent.run("q")
        assert out.answer == ""
        assert len(out.results) == 2
        assert synth.calls == 0
        assert out.fallback is False


class TestAgenticRAGReflect:
    def test_insufficient_triggers_followup_hop(self):
        by_q = {
            "sub1": [RetrievalResult(chunk_id="c1", score=1.0, text="p1")],
            "sub2": [RetrievalResult(chunk_id="c2", score=0.9, text="p2")],
        }
        hybrid = FakeHybrid(results_by_query=by_q)
        agent = _build(
            hybrid_search=hybrid,
            router=FakeRouter(RouteDecision(need_retrieval=True)),
            transformer=FakeTransformer(["sub1"]),
            reflector=FakeReflector(
                [ReflectVerdict(sufficient=False, follow_up_queries=["sub2"])]
            ),
            synthesizer=FakeSynth(),
            registry=FakeRegistry([]),
        )
        out = agent.run("q")
        ids = sorted(r.chunk_id for r in out.results)
        assert ids == ["c1", "c2"]  # follow-up hop fetched c2
        stages = [s["stage"] for s in out.steps]
        assert "hop_1" in stages and "reflect_1" in stages and "hop_2" in stages

    def test_sufficient_stops_after_first_hop(self):
        hybrid = FakeHybrid(_results(1))
        agent = _build(
            hybrid_search=hybrid,
            router=FakeRouter(RouteDecision(need_retrieval=True)),
            transformer=FakeTransformer(["only"]),
            reflector=FakeReflector([ReflectVerdict(sufficient=True)]),
            synthesizer=FakeSynth(),
            registry=FakeRegistry([]),
        )
        out = agent.run("q")
        hops = [s for s in out.steps if s["stage"].startswith("hop_")]
        assert len(hops) == 1

    def test_followup_dedup_against_asked(self):
        # reflector keeps proposing an already-asked query → no extra hop
        hybrid = FakeHybrid(_results(1))
        agent = _build(
            hybrid_search=hybrid,
            router=FakeRouter(RouteDecision(need_retrieval=True)),
            transformer=FakeTransformer(["dup"]),
            reflector=FakeReflector(
                [ReflectVerdict(sufficient=False, follow_up_queries=["dup"])]
            ),
            synthesizer=FakeSynth(),
            registry=FakeRegistry([]),
        )
        out = agent.run("q")
        hops = [s for s in out.steps if s["stage"].startswith("hop_")]
        assert len(hops) == 1  # "dup" already asked → filtered, no second hop

    def test_max_hops_bounds_loop(self):
        s = Settings()
        s.agent.max_hops = 2
        s.agent.max_reflect_rounds = 5
        # always insufficient with a fresh follow-up query each round
        verdicts = [
            ReflectVerdict(sufficient=False, follow_up_queries=[f"q{i}"]) for i in range(5)
        ]
        agent = _build(
            s,
            hybrid_search=FakeHybrid(_results(1)),
            router=FakeRouter(RouteDecision(need_retrieval=True)),
            transformer=FakeTransformer(["start"]),
            reflector=FakeReflector(verdicts),
            synthesizer=FakeSynth(),
            registry=FakeRegistry([]),
        )
        out = agent.run("q")
        hops = [s for s in out.steps if s["stage"].startswith("hop_")]
        assert len(hops) == 2  # capped by max_hops


class TestAgenticRAGDegradation:
    def test_synth_error_degrades_to_fallback(self):
        # retrieval succeeds but synthesis raises → global fallback path
        hybrid = FakeHybrid(_results(2))
        agent = _build(
            hybrid_search=hybrid,
            router=FakeRouter(RouteDecision(need_retrieval=True)),
            synthesizer=FakeSynth(error=LLMError("boom", provider="fake")),
            registry=FakeRegistry([]),
        )
        out = agent.run("q")
        assert out.fallback is True
        assert len(out.results) == 2  # fallback retrieval still returns evidence
        assert out.answer == ""       # fallback synthesis also failed → no answer

    def test_total_retrieval_failure_returns_empty_gracefully(self):
        agent = _build(
            hybrid_search=FakeHybrid(error=RuntimeError("store down")),
            router=FakeRouter(RouteDecision(need_retrieval=True)),
            synthesizer=FakeSynth(),
            registry=FakeRegistry([]),
        )
        out = agent.run("q")
        assert out.fallback is True
        assert out.results == []
        assert out.answer == ""

    def test_fallback_records_trace_stage(self):
        trace = TraceContext("query")
        agent = _build(
            hybrid_search=FakeHybrid(error=RuntimeError("x")),
            router=FakeRouter(RouteDecision(need_retrieval=True)),
            synthesizer=FakeSynth(),
            registry=FakeRegistry([]),
        )
        agent.run("q", trace=trace)
        assert "agent_fallback" in [s["name"] for s in trace.stages]

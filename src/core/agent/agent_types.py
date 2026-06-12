"""Agent data types — contracts shared across the Agentic RAG components.

These lightweight dataclasses carry decisions and intermediate results between
the router, query transformer, multi-hop retrieval loop, reflector, answer
synthesizer, and the main ``AgenticRAG`` orchestrator. They are intentionally
JSON-friendly so they can be recorded into trace stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.types import RetrievalResult


@dataclass
class RouteDecision:
    """Routing decision produced by ``QueryRouter`` (OPTIMIZATION_SPEC §3.1).

    target_collections 已按白名单校验（仅保留 ∈ available 的项）；空列表表示
    不限定 collection（检索全部）。need_retrieval=False 时可由 LLM 给出
    direct_answer 直接回答。
    """

    need_retrieval: bool = True
    target_collections: list[str] = field(default_factory=list)
    reasoning: str = ""
    direct_answer: str = ""


@dataclass
class SubQuery:
    """A single sub-query produced by ``QueryTransformer`` (§3.2)."""

    text: str
    purpose: str = ""


@dataclass
class HopResult:
    """Result accumulated for a single multi-hop retrieval round (§3.3)."""

    hop: int
    subqueries: list[str] = field(default_factory=list)
    results: list[RetrievalResult] = field(default_factory=list)


@dataclass
class ReflectVerdict:
    """Sufficiency verdict produced by ``Reflector`` (§3.4).

    sufficient=True 表示当前上下文足以回答原始查询；否则 follow_up_queries 给出
    补充检索的子查询。
    """

    sufficient: bool = True
    follow_up_queries: list[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class SynthResult:
    """Final answer produced by ``AnswerSynthesizer``.

    used_citation_ids 为答案实际引用的上下文片段序号（1-based，对应合成时给
    LLM 的编号片段）。
    """

    answer: str = ""
    used_citation_ids: list[int] = field(default_factory=list)


@dataclass
class AgentResult:
    """Aggregated output of one ``AgenticRAG.run`` execution.

    answer 为服务端合成的答案文本（synthesize 关闭时可为空）；results 为最终用于
    作答/引用的检索结果；steps 记录每步决策摘要，供 trace/调试。
    """

    answer: str = ""
    results: list[RetrievalResult] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    fallback: bool = False

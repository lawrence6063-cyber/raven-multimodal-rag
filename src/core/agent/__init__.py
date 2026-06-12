"""Agentic RAG package — LLM-driven retrieval orchestration (OPTIMIZATION_SPEC §3).

See ``docs/P1_AGENTIC_RAG_SPEC.md`` for the full design.
"""

from __future__ import annotations

from src.core.agent.agent_types import (
    AgentResult,
    HopResult,
    ReflectVerdict,
    RouteDecision,
    SubQuery,
    SynthResult,
)
from src.core.agent.agentic_rag import AgenticRAG
from src.core.agent.answer_synthesizer import AnswerSynthesizer
from src.core.agent.collection_registry import CollectionRegistry
from src.core.agent.query_transformer import QueryTransformer
from src.core.agent.reflector import Reflector
from src.core.agent.router import QueryRouter

__all__ = [
    "AgentResult",
    "HopResult",
    "ReflectVerdict",
    "RouteDecision",
    "SubQuery",
    "SynthResult",
    "AgenticRAG",
    "AnswerSynthesizer",
    "CollectionRegistry",
    "QueryTransformer",
    "Reflector",
    "QueryRouter",
]

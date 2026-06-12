"""Tests for AgenticQueryTool — result assembly, delegation, validation."""

from __future__ import annotations

import pytest

from src.core.agent.agent_types import AgentResult
from src.core.response.mcp_types import MCPToolResult, TextContent
from src.core.settings import Settings
from src.core.types import RetrievalResult
from src.mcp_server.protocol_handler import INVALID_PARAMS, JsonRpcError
from src.mcp_server.tools.agentic_query import AgenticQueryTool


class FakeAgent:
    def __init__(self, result: AgentResult):
        self._result = result
        self.calls: list[dict] = []

    def run(self, query, collection=None, image=None, top_k=None, trace=None):
        self.calls.append({"query": query, "collection": collection, "top_k": top_k})
        return self._result


class FakeBuilder:
    def __init__(self, result: MCPToolResult):
        self._result = result

    def build(self, results, query):
        return self._result


class FakeDelegate:
    def __init__(self):
        self.calls: list[dict] = []

    def run(self, query="", top_k=None, collection=None, image=None):
        self.calls.append({"query": query, "collection": collection})
        return MCPToolResult(content=[TextContent(text="DELEGATED")])


class FakeCollector:
    def collect(self, trace):
        return None


def _enabled_settings():
    s = Settings()
    s.agent.enabled = True
    return s


def _built_result():
    return MCPToolResult(
        content=[TextContent(text="ORIGINAL")],
        structured_content={
            "query": "q",
            "citations": [
                {"id": 1, "source": "a.pdf", "page": 3, "score": 0.5, "chunk_id": "c1", "text": "t"}
            ],
        },
    )


class TestAgenticQueryToolEnabled:
    def test_synthesized_answer_with_references(self):
        agent_result = AgentResult(
            answer="RAG is X [1].",
            results=[RetrievalResult(chunk_id="c1", score=0.5, text="t")],
        )
        tool = AgenticQueryTool(
            _enabled_settings(),
            agentic_rag=FakeAgent(agent_result),
            response_builder=FakeBuilder(_built_result()),
            trace_collector=FakeCollector(),
        )
        result = tool.run(query="what is rag?")
        text = result.content[0].text
        assert text.startswith("RAG is X [1].")
        assert "### References" in text
        assert "[1] a.pdf (page 3) — score 0.5000" in text
        assert result.structured_content["answer"] == "RAG is X [1]."
        assert result.structured_content["fallback"] is False

    def test_empty_answer_keeps_builder_text(self):
        agent_result = AgentResult(answer="", results=[])
        tool = AgenticQueryTool(
            _enabled_settings(),
            agentic_rag=FakeAgent(agent_result),
            response_builder=FakeBuilder(_built_result()),
            trace_collector=FakeCollector(),
        )
        result = tool.run(query="q")
        assert result.content[0].text == "ORIGINAL"
        assert result.structured_content["answer"] == ""

    def test_fallback_flag_propagates(self):
        agent_result = AgentResult(answer="a", results=[], fallback=True)
        tool = AgenticQueryTool(
            _enabled_settings(),
            agentic_rag=FakeAgent(agent_result),
            response_builder=FakeBuilder(_built_result()),
            trace_collector=FakeCollector(),
        )
        result = tool.run(query="q")
        assert result.structured_content["fallback"] is True

    def test_missing_query_and_image_raises(self):
        tool = AgenticQueryTool(
            _enabled_settings(),
            agentic_rag=FakeAgent(AgentResult()),
            response_builder=FakeBuilder(_built_result()),
            trace_collector=FakeCollector(),
        )
        with pytest.raises(JsonRpcError) as exc:
            tool.run(query="")
        assert exc.value.code == INVALID_PARAMS


class TestAgenticQueryToolDisabled:
    def test_delegates_to_query_knowledge_hub(self):
        delegate = FakeDelegate()
        agent = FakeAgent(AgentResult(answer="should-not-run"))
        tool = AgenticQueryTool(
            Settings(),  # agent.enabled defaults to False
            agentic_rag=agent,
            delegate=delegate,
            trace_collector=FakeCollector(),
        )
        result = tool.run(query="q", collection="rag")
        assert result.content[0].text == "DELEGATED"
        assert delegate.calls == [{"query": "q", "collection": "rag"}]
        assert agent.calls == []  # agent not invoked when disabled

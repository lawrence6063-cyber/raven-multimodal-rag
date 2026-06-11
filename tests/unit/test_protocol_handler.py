"""Unit tests for ProtocolHandler (E2)."""

from __future__ import annotations

import pytest

from src.core.response.mcp_types import MCPToolResult, TextContent
from src.mcp_server.protocol_handler import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    JsonRpcError,
    ProtocolHandler,
)


def _ok_tool(query: str, top_k: int = 5) -> MCPToolResult:
    return MCPToolResult(content=[TextContent(text=f"q={query} k={top_k}")])


def _boom_tool() -> MCPToolResult:
    raise RuntimeError("secret internal stacktrace detail")


@pytest.fixture
def handler() -> ProtocolHandler:
    h = ProtocolHandler()
    h.register(
        name="query_knowledge_hub",
        description="search",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
            "required": ["query"],
        },
        func=_ok_tool,
    )
    h.register(
        name="boom",
        description="always fails",
        input_schema={"type": "object", "properties": {}},
        func=_boom_tool,
    )
    return h


class TestInitialize:
    def test_returns_capabilities_and_server_info(self, handler):
        result = handler.handle_initialize({})
        assert "capabilities" in result and "tools" in result["capabilities"]
        assert result["serverInfo"]["name"]
        assert result["protocolVersion"]


class TestToolsList:
    def test_lists_registered_tool_schemas(self, handler):
        tools = handler.handle_tools_list()["tools"]
        names = {t["name"] for t in tools}
        assert "query_knowledge_hub" in names
        entry = next(t for t in tools if t["name"] == "query_knowledge_hub")
        assert "inputSchema" in entry and "description" in entry


class TestToolsCall:
    def test_routes_and_returns_result(self, handler):
        result = handler.handle_tools_call("query_knowledge_hub", {"query": "fox"})
        assert result["isError"] is False
        assert "q=fox" in result["content"][0]["text"]

    def test_unknown_tool_raises_method_not_found(self, handler):
        with pytest.raises(JsonRpcError) as exc:
            handler.handle_tools_call("does_not_exist", {})
        assert exc.value.code == METHOD_NOT_FOUND

    def test_missing_required_param_raises_invalid_params(self, handler):
        with pytest.raises(JsonRpcError) as exc:
            handler.handle_tools_call("query_knowledge_hub", {"top_k": 3})
        assert exc.value.code == INVALID_PARAMS

    def test_unexpected_argument_raises_invalid_params(self, handler):
        with pytest.raises(JsonRpcError) as exc:
            handler.handle_tools_call(
                "query_knowledge_hub", {"query": "x", "bogus": 1}
            )
        assert exc.value.code == INVALID_PARAMS

    def test_tool_exception_becomes_internal_error_without_stack(self, handler):
        with pytest.raises(JsonRpcError) as exc:
            handler.handle_tools_call("boom", {})
        assert exc.value.code == INTERNAL_ERROR
        assert "stacktrace" not in exc.value.message
        assert "secret" not in exc.value.message

"""ProtocolHandler — SDK-independent JSON-RPC 2.0 method routing (E2).

Encapsulates the three core MCP methods (``initialize``, ``tools/list``,
``tools/call``) over a small tool registry. It is deliberately free of any
``mcp`` SDK import so it can be unit-tested without the SDK; ``server.py`` wires
the SDK's stdio transport callbacks into these methods.

Error handling follows JSON-RPC 2.0 codes:
- -32601 Method/tool not found
- -32602 Invalid params (missing required argument)
- -32603 Internal error (tool raised; stack trace is NOT propagated)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from src.core.response.mcp_types import MCPToolResult
from src.observability.logger import get_logger

logger = get_logger("mcp_server.protocol_handler")

# JSON-RPC 2.0 standard error codes
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Negotiated MCP protocol version and advertised server identity
PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "modular-rag-mcp-server"
SERVER_VERSION = "0.1.0"


class JsonRpcError(Exception):
    """A JSON-RPC error carrying a numeric code and a safe message."""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-RPC error object."""
        return {"code": self.code, "message": self.message}


# ToolFunc the callable backing a tool: accepts kwargs, returns MCPToolResult
ToolFunc = Callable[..., MCPToolResult]


@dataclass
class ToolSpec:
    """A registered tool's metadata and handler."""

    name: str
    description: str
    input_schema: dict[str, Any]
    func: ToolFunc

    def schema_dict(self) -> dict[str, Any]:
        """Serialize to a tools/list entry."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class ProtocolHandler:
    """Routes MCP core methods to a registry of tools."""

    _tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        func: ToolFunc,
    ) -> None:
        """Register a tool. Re-registering the same name overwrites it.

        Args:
            name: Unique tool name.
            description: Human-readable description for tools/list.
            input_schema: JSON Schema for the tool arguments.
            func: Callable invoked with validated keyword arguments.
        """
        self._tools[name] = ToolSpec(name, description, input_schema, func)

    def handle_initialize(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return server capabilities and identity for the initialize handshake."""
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }

    def handle_tools_list(self) -> dict[str, Any]:
        """Return the schema of every registered tool."""
        return {"tools": [spec.schema_dict() for spec in self._tools.values()]}

    def handle_tools_call(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Route to a tool, validate params, and return its result dict.

        Args:
            name: Tool name to invoke.
            arguments: Keyword arguments for the tool.

        Returns:
            The tool's MCPToolResult serialized via ``to_dict()``.

        Raises:
            JsonRpcError: -32601 if unknown, -32602 if a required arg is
                missing, -32603 if the tool raises (no stack trace leaked).
        """
        spec = self._tools.get(name)
        if spec is None:
            raise JsonRpcError(METHOD_NOT_FOUND, f"Unknown tool: {name}")

        args = arguments or {}
        self._validate_required(spec, args)

        try:
            result = spec.func(**args)
        except JsonRpcError:
            raise
        except TypeError as e:
            # Unexpected/typo arguments surface as invalid params
            logger.warning(f"Tool '{name}' received invalid arguments: {e}")
            raise JsonRpcError(INVALID_PARAMS, f"Invalid arguments for tool: {name}")
        except Exception as e:
            logger.error(f"Tool '{name}' failed: {e}")
            raise JsonRpcError(INTERNAL_ERROR, "Internal error while executing tool")

        if isinstance(result, MCPToolResult):
            return result.to_dict()
        return result

    @staticmethod
    def _validate_required(spec: ToolSpec, args: dict[str, Any]) -> None:
        """Ensure all schema-required arguments are present and non-null."""
        required = spec.input_schema.get("required", [])
        missing = [key for key in required if args.get(key) in (None, "")]
        if missing:
            raise JsonRpcError(
                INVALID_PARAMS,
                f"Missing required argument(s): {', '.join(missing)}",
            )

    @property
    def tool_names(self) -> list[str]:
        """Return the names of all registered tools."""
        return list(self._tools.keys())

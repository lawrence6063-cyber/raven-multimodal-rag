"""MCP response data types — SDK-independent content/citation/result models.

These dataclasses model the subset of the MCP tool-result shape this project
returns (text + image content, structured citations). They are intentionally
free of any ``mcp`` SDK dependency so that tools and response builders remain
unit-testable without the SDK installed. The thin ``server.py`` adapter is the
only place that converts these into SDK objects.

See DEV_SPEC.md 3.2.5 (Response & Citation Design).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextContent:
    """A text content item (Markdown), the default tool return type."""

    text: str
    type: str = "text"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to MCP content dict."""
        return {"type": self.type, "text": self.text}


@dataclass
class ImageContent:
    """An image content item carrying Base64-encoded binary data."""

    data: str  # Base64-encoded image bytes
    mime_type: str = "image/png"
    type: str = "image"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to MCP content dict (mimeType per MCP spec)."""
        return {"type": self.type, "data": self.data, "mimeType": self.mime_type}


@dataclass
class Citation:
    """A structured citation pointing back to a retrieved chunk."""

    id: int
    source: str
    chunk_id: str
    score: float
    page: int | None = None
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a citation dict (DEV_SPEC 3.2.5 format)."""
        return {
            "id": self.id,
            "source": self.source,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "score": self.score,
            "text": self.text,
        }


@dataclass
class MCPToolResult:
    """The result of a tool invocation, mapped to MCP CallToolResult shape.

    Attributes:
        content: Ordered content items; item[0] is always human-readable text.
        structured_content: Optional structured payload (e.g. citations).
        is_error: Whether this result represents a tool-level error.
    """

    content: list[TextContent | ImageContent] = field(default_factory=list)
    structured_content: dict[str, Any] | None = None
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to an MCP CallToolResult-compatible dict."""
        result: dict[str, Any] = {
            "content": [c.to_dict() for c in self.content],
            "isError": self.is_error,
        }
        if self.structured_content is not None:
            result["structuredContent"] = self.structured_content
        return result

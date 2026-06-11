"""list_collections tool (E4) — enumerate available document collections.

Scans a documents root directory where each immediate subdirectory represents a
collection, and reports the file count per collection. The root is configurable
so tests can point at a fixtures directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.response.mcp_types import MCPToolResult, TextContent
from src.observability.logger import get_logger

logger = get_logger("mcp_server.tools.list_collections")

# TOOL_NAME registered MCP tool name
TOOL_NAME = "list_collections"
# TOOL_DESCRIPTION human-readable description for tools/list
TOOL_DESCRIPTION = (
    "List the document collections available in the knowledge base, "
    "with the number of documents in each."
)
# INPUT_SCHEMA JSON Schema (no arguments)
INPUT_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}

# _DEFAULT_DOCUMENTS_DIR default root scanned for collection subdirectories
_DEFAULT_DOCUMENTS_DIR = "data/documents"


class ListCollectionsTool:
    """Tool object encapsulating the list_collections workflow."""

    NAME = TOOL_NAME
    DESCRIPTION = TOOL_DESCRIPTION
    INPUT_SCHEMA = INPUT_SCHEMA

    def __init__(self, documents_dir: str = _DEFAULT_DOCUMENTS_DIR):
        self._documents_dir = Path(documents_dir)

    def run(self) -> MCPToolResult:
        """List collections and their document counts.

        Returns:
            MCPToolResult with a Markdown summary and structured collections.
        """
        collections = self._scan()

        if not collections:
            text = (
                "No collections found.\n\n"
                "Ingest documents first, e.g. "
                "`python scripts/ingest.py --path <folder> --collection <name>`."
            )
        else:
            lines = ["## Available collections", ""]
            for entry in collections:
                lines.append(
                    f"- **{entry['name']}** — {entry['document_count']} document(s)"
                )
            text = "\n".join(lines)

        return MCPToolResult(
            content=[TextContent(text=text)],
            structured_content={"collections": collections},
            is_error=False,
        )

    def _scan(self) -> list[dict[str, Any]]:
        """Return sorted collection entries from the documents directory."""
        if not self._documents_dir.is_dir():
            logger.info(f"Documents dir not found: {self._documents_dir}")
            return []

        entries: list[dict[str, Any]] = []
        for child in sorted(self._documents_dir.iterdir()):
            if not child.is_dir():
                continue
            document_count = sum(1 for f in child.rglob("*") if f.is_file())
            entries.append({"name": child.name, "document_count": document_count})
        return entries

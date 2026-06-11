"""Modular RAG MCP Server — main entry point.

Delegates to the MCP Server (Stdio Transport) defined in
``src/mcp_server/server.py``. Logs go to stderr; stdout is reserved for MCP
protocol messages.
"""

import sys

from src.mcp_server.server import main as serve_main


def main() -> int:
    """Start the MCP Server over Stdio Transport."""
    return serve_main()


if __name__ == "__main__":
    sys.exit(main())

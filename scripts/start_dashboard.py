#!/usr/bin/env python3
"""Start the RAG MCP Dashboard (Streamlit app).

Usage:
    python scripts/start_dashboard.py [--port PORT]

This script launches the Streamlit dashboard server. By default it runs on
port 8501. Pass --port to override.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_APP_PATH = _PROJECT_ROOT / "src" / "observability" / "dashboard" / "app.py"


def main():
    port = "8501"
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = sys.argv[idx + 1]

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(_APP_PATH),
        "--server.port", port,
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]

    print(f"🚀 Starting Dashboard on http://localhost:{port}")
    print(f"   App: {_APP_PATH}")
    print(f"   Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, cwd=str(_PROJECT_ROOT), check=True)
    except KeyboardInterrupt:
        print("\n✋ Dashboard stopped.")


if __name__ == "__main__":
    main()

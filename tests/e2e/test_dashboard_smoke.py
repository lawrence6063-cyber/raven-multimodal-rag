"""E2E: Dashboard smoke tests (I2).

Uses Streamlit's ``AppTest`` framework to load the multi-page dashboard and
render every page, asserting that no page raises an unhandled Python exception.
The dashboard is designed to degrade gracefully when no data is present (empty
traces file, empty/uninitialised vector store), so these smoke tests run fully
offline without ingested data.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_APP_PATH = _PROJECT_ROOT / "src" / "observability" / "dashboard" / "app.py"

# Page labels as registered in app.py's navigation radio.
_PAGE_LABELS = [
    "📊 系统总览",
    "📂 数据浏览器",
    "📥 Ingestion 管理",
    "🔬 Ingestion 追踪",
    "🔍 Query 追踪",
    "⚙️ 评估面板",
]

# Generous timeout: first run imports heavy deps (chroma, pandas) lazily.
_RUN_TIMEOUT = 60


def _new_app() -> AppTest:
    return AppTest.from_file(str(_APP_PATH), default_timeout=_RUN_TIMEOUT)


@pytest.mark.e2e
class TestDashboardSmoke:
    """Render each dashboard page and assert no unhandled exceptions."""

    def test_app_loads_without_exception(self):
        at = _new_app().run()
        assert not at.exception, f"app failed to load: {at.exception}"
        # The navigation radio with all six pages must be present.
        assert at.radio, "navigation radio not rendered"
        options = at.radio[0].options
        for label in _PAGE_LABELS:
            assert label in options

    def test_all_pages_render(self):
        for label in _PAGE_LABELS:
            at = _new_app().run()
            assert not at.exception, f"initial load failed before selecting {label}"
            at.radio[0].set_value(label).run()
            assert not at.exception, (
                f"page '{label}' raised an unhandled exception: {at.exception}"
            )
            # A non-empty page must render at least one title element.
            assert at.title, f"page '{label}' rendered no title"

    def test_evaluation_panel_lists_backends(self):
        at = _new_app().run()
        at.radio[0].set_value("⚙️ 评估面板").run()
        assert not at.exception
        # The custom evaluator backend should always be available offline.
        multiselects = at.multiselect
        assert multiselects, "evaluation panel should expose a backend multiselect"
        assert "custom" in multiselects[0].options

"""Dashboard App — Streamlit multi-page application entry point (G1).

Launch with: streamlit run src/observability/dashboard/app.py
or: python scripts/start_dashboard.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="RAG MCP Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _placeholder(title: str):
    """Render a placeholder page for pages not yet implemented."""
    st.title(title)
    st.info("🚧 此页面正在建设中，敬请期待。")


def main():
    """Main entry: register pages and run navigation."""
    from src.observability.dashboard.pages.overview import render as overview_render
    from src.observability.dashboard.pages.data_browser import render as data_browser_render
    from src.observability.dashboard.pages.ingestion_manager import render as ingestion_mgr_render
    from src.observability.dashboard.pages.ingestion_traces import render as ingestion_traces_render
    from src.observability.dashboard.pages.query_traces import render as query_traces_render
    from src.observability.dashboard.pages.evaluation_panel import render as evaluation_panel_render

    pages = {
        "📊 系统总览": overview_render,
        "📂 数据浏览器": data_browser_render,
        "📥 Ingestion 管理": ingestion_mgr_render,
        "🔬 Ingestion 追踪": ingestion_traces_render,
        "🔍 Query 追踪": query_traces_render,
        "⚙️ 评估面板": evaluation_panel_render,
    }

    with st.sidebar:
        st.title("🔬 RAG Dashboard")
        st.markdown("---")
        selection = st.radio("导航", list(pages.keys()), label_visibility="collapsed")

    pages[selection]()


if __name__ == "__main__":
    main()
else:
    # When streamlit imports this module directly
    main()

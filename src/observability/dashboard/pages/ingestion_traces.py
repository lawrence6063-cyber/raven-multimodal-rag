"""Ingestion Traces page — ingestion history and stage waterfall chart (G5)."""

from __future__ import annotations

import streamlit as st

from src.observability.dashboard.services.trace_service import TraceService


def render():
    """Render the ingestion traces page."""
    st.title("🔬 Ingestion 追踪")
    st.markdown("查看摄取历史记录和各阶段耗时分布。")

    trace_service = TraceService()

    traces = trace_service.list_traces(trace_type="ingestion", limit=50)

    if not traces:
        st.info("暂无 Ingestion 追踪记录。请先执行文档摄取操作。")
        return

    # History list
    st.subheader(f"📋 历史记录（最近 {len(traces)} 条）")

    for trace in traces:
        trace_id = trace.get("trace_id", "?")[:8]
        started = trace.get("started_at", "")[:19]
        total_ms = trace.get("total_elapsed_ms", 0)
        stages_count = len(trace.get("stages", []))

        with st.expander(
            f"🕐 {started}  |  ID: {trace_id}...  |  "
            f"耗时: {total_ms:.0f}ms  |  阶段: {stages_count}"
        ):
            _render_trace_detail(trace, trace_service)


def _render_trace_detail(trace: dict, trace_service: TraceService):
    """Render detail for a single ingestion trace."""
    stages = trace_service.get_stage_breakdown(trace)

    if not stages:
        st.warning("此 trace 无阶段记录。")
        return

    # Waterfall chart data
    st.markdown("#### 阶段耗时瀑布图")

    import pandas as pd

    df = pd.DataFrame([
        {"阶段": s["name"], "耗时(ms)": s.get("elapsed_ms", 0), "方法": s.get("method", "")}
        for s in stages
    ])

    st.bar_chart(df.set_index("阶段")["耗时(ms)"])

    # Detail table
    st.markdown("#### 阶段详情")
    st.dataframe(df, use_container_width=True)

    # Trace metadata
    with st.expander("原始 Trace 数据"):
        st.json(trace)

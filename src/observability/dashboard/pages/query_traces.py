"""Query Traces page — query history, stage waterfall, and comparisons (G6)."""

from __future__ import annotations

import streamlit as st

from src.observability.dashboard.services.trace_service import TraceService


def render():
    """Render the query traces page."""
    st.title("🔍 Query 追踪")
    st.markdown("查看查询历史、各阶段耗时、Dense vs Sparse 对比、Rerank 前后变化。")

    trace_service = TraceService()

    traces = trace_service.list_traces(trace_type="query", limit=50)

    if not traces:
        st.info("暂无 Query 追踪记录。请先通过 MCP Server 执行查询操作。")
        return

    # History list
    st.subheader(f"📋 查询历史（最近 {len(traces)} 条）")

    for trace in traces:
        trace_id = trace.get("trace_id", "?")[:8]
        started = trace.get("started_at", "")[:19]
        total_ms = trace.get("total_elapsed_ms", 0)
        stages = trace.get("stages", [])
        stages_count = len(stages)

        with st.expander(
            f"🕐 {started}  |  ID: {trace_id}...  |  "
            f"耗时: {total_ms:.0f}ms  |  阶段: {stages_count}"
        ):
            _render_query_detail(trace, trace_service)


def _render_query_detail(trace: dict, trace_service: TraceService):
    """Render detail for a single query trace."""
    stages = trace_service.get_stage_breakdown(trace)

    if not stages:
        st.warning("此 trace 无阶段记录。")
        return

    # Waterfall chart
    st.markdown("#### 阶段耗时瀑布图")

    import pandas as pd

    df = pd.DataFrame([
        {"阶段": s["name"], "耗时(ms)": s.get("elapsed_ms", 0), "方法": s.get("method", "")}
        for s in stages
    ])

    st.bar_chart(df.set_index("阶段")["耗时(ms)"])

    # Dense vs Sparse comparison
    st.markdown("#### Dense vs Sparse 对比")
    dense_stage = next((s for s in stages if s["name"] == "dense_retrieval"), None)
    sparse_stage = next((s for s in stages if s["name"] == "sparse_retrieval"), None)

    col1, col2 = st.columns(2)
    with col1:
        if dense_stage:
            st.metric("Dense 耗时", f"{dense_stage.get('elapsed_ms', 0):.1f}ms")
            st.metric("Dense 结果数", dense_stage.get("results", "?"))
        else:
            st.info("无 Dense 阶段数据")
    with col2:
        if sparse_stage:
            st.metric("Sparse 耗时", f"{sparse_stage.get('elapsed_ms', 0):.1f}ms")
            st.metric("Sparse 结果数", sparse_stage.get("results", "?"))
        else:
            st.info("无 Sparse 阶段数据")

    # Rerank info
    rerank_stage = next((s for s in stages if s["name"] == "rerank"), None)
    if rerank_stage:
        st.markdown("#### Rerank 信息")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rerank 耗时", f"{rerank_stage.get('elapsed_ms', 0):.1f}ms")
        col2.metric("Rerank 方法", rerank_stage.get("method", "?"))
        col3.metric("降级?", "是" if rerank_stage.get("fallback") else "否")

    # Stage detail table
    st.markdown("#### 阶段详情")
    st.dataframe(df, use_container_width=True)

    # Raw trace
    with st.expander("原始 Trace 数据"):
        st.json(trace)

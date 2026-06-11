"""Evaluation Panel page — run evaluation and view metrics (H4)."""

from __future__ import annotations

import streamlit as st

from src.observability.dashboard.services.eval_service import EvalService


def render():
    """Render the evaluation panel page."""
    st.title("⚙️ 评估面板")
    st.markdown("选择评估后端与黄金测试集，运行评估并查看检索质量指标（hit_rate / mrr 等）。")

    service = EvalService()

    try:
        backends = service.available_backends()
        default_test_set = service.default_test_set()
    except Exception as exc:  # noqa: BLE001 - dashboard must not crash on config errors
        st.error(f"无法初始化评估服务：{exc}")
        return

    if not backends:
        st.warning("未注册任何评估后端。")
        return

    col1, col2 = st.columns([2, 3])
    with col1:
        selected = st.multiselect(
            "评估后端",
            options=backends,
            default=[b for b in ("custom",) if b in backends] or backends[:1],
            help="可同时选择多个后端并行评估（如 custom + ragas）。",
        )
    with col2:
        test_set_path = st.text_input("黄金测试集路径", value=default_test_set)

    run_clicked = st.button("▶️ 运行评估", type="primary", disabled=not selected)

    if not run_clicked:
        st.info("配置后端与测试集后点击「运行评估」。")
        return

    with st.spinner("正在运行评估…"):
        try:
            report = service.run(selected, test_set_path)
        except FileNotFoundError as exc:
            st.error(f"{exc}")
            return
        except Exception as exc:  # noqa: BLE001 - surface a friendly message
            st.error(f"评估失败：{exc}")
            return

    _render_report(report)


def _render_report(report):
    """Render an EvalReport: summary metrics + per-query details."""
    st.success(
        f"评估完成：{report.total_queries} 条查询，后端 {', '.join(report.backends)}。"
    )

    st.subheader("📊 汇总指标")
    if report.metrics:
        metric_items = list(report.metrics.items())
        cols = st.columns(min(4, len(metric_items)))
        for i, (name, value) in enumerate(metric_items):
            cols[i % len(cols)].metric(name, f"{value:.4f}")
    else:
        st.info("无指标输出。")

    st.subheader("🔎 各查询明细")
    import pandas as pd

    rows = [
        {
            "命中": "✅" if item["hit"] else "❌",
            "查询": item["query"],
            "检索数": item["num_retrieved"],
            "期望 chunk": ", ".join(item.get("expected_chunk_ids", [])) or "-",
            "期望来源": ", ".join(item.get("expected_sources", [])) or "-",
        }
        for item in report.per_query
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with st.expander("原始报告数据"):
        st.json(report.to_dict())

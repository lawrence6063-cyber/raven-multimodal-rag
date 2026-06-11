"""Overview page — system status and component configuration cards (G1)."""

from __future__ import annotations

import streamlit as st

from src.observability.dashboard.services.config_service import ConfigService


def render():
    """Render the system overview page."""
    st.title("📊 系统总览")
    st.markdown("显示当前 RAG 系统的组件配置与数据统计。")

    config_service = ConfigService()

    # Component cards
    st.subheader("🧩 组件配置")
    cards = config_service.get_component_cards()

    cols = st.columns(3)
    for i, card in enumerate(cards):
        with cols[i % 3]:
            st.markdown(f"### {card['icon']} {card['name']}")
            for key, value in card.items():
                if key not in ("name", "icon"):
                    st.markdown(f"- **{key}**: `{value}`")

    # Observability info
    st.subheader("📡 可观测性")
    obs = config_service.get_observability_info()
    col1, col2, col3 = st.columns(3)
    col1.metric("Trace", "✅ 开启" if obs["trace_enabled"] else "❌ 关闭")
    col2.metric("日志文件", obs["log_file"])
    col3.metric("日志级别", obs["log_level"])

    # Data statistics (try to load from ChromaStore)
    st.subheader("📈 数据统计")
    try:
        from src.core.settings import load_settings
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        settings = config_service.settings
        store = VectorStoreFactory.create(settings.vector_store)
        stats = store.get_collection_stats()
        col1, col2 = st.columns(2)
        col1.metric("总 Chunk 数", stats.get("total_chunks", "N/A"))
        col2.metric("集合名称", stats.get("collection_name", "N/A"))
    except Exception as e:
        st.info(f"暂无数据统计（向量库未初始化或为空）：{e}")

    # Raw config expander
    with st.expander("🔧 完整配置（YAML 视图）"):
        import yaml

        raw = config_service.get_raw_config()
        st.code(yaml.dump(raw, default_flow_style=False, allow_unicode=True), language="yaml")

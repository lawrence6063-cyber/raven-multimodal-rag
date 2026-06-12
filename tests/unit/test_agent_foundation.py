"""Tests for Agentic RAG foundation: AgentSettings, types, CollectionRegistry."""

from __future__ import annotations

from src.core.agent import CollectionRegistry, RouteDecision, SubQuery, SynthResult
from src.core.settings import AgentSettings, Settings, _build_dataclass


class TestAgentSettings:
    """AgentSettings defaults and config wiring."""

    def test_defaults_are_zero_intrusion(self):
        s = AgentSettings()
        assert s.enabled is False  # off by default: behavior unchanged
        assert s.route_enabled is True
        assert s.max_hops == 3
        assert s.max_subqueries == 3
        assert s.max_reflect_rounds == 2
        assert s.max_context_chunks == 20

    def test_settings_has_agent_field(self):
        assert isinstance(Settings().agent, AgentSettings)
        assert Settings().agent.enabled is False

    def test_build_dataclass_ignores_unknown_fields(self):
        built = _build_dataclass(
            AgentSettings, {"enabled": True, "max_hops": 5, "unknown_key": 1}
        )
        assert built.enabled is True
        assert built.max_hops == 5

    def test_build_dataclass_none_returns_defaults(self):
        built = _build_dataclass(AgentSettings, None)
        assert built.enabled is False


class TestAgentTypes:
    """Lightweight dataclass contracts."""

    def test_route_decision_defaults(self):
        d = RouteDecision()
        assert d.need_retrieval is True
        assert d.target_collections == []

    def test_subquery_and_synth_result(self):
        assert SubQuery(text="q").purpose == ""
        assert SynthResult().used_citation_ids == []


class TestCollectionRegistry:
    """CollectionRegistry directory scanning and degradation."""

    def test_lists_subdirectories(self, tmp_path):
        (tmp_path / "alpha").mkdir()
        (tmp_path / "beta").mkdir()
        (tmp_path / "alpha" / "doc.pdf").write_text("x")
        names = CollectionRegistry(str(tmp_path)).list_collections()
        assert names == ["alpha", "beta"]

    def test_missing_dir_returns_empty(self, tmp_path):
        names = CollectionRegistry(str(tmp_path / "nope")).list_collections()
        assert names == []

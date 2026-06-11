"""Tests for Reranker factory and NoneReranker."""

import pytest

from src.libs.reranker.base_reranker import BaseReranker, NoneReranker, RerankCandidate, RerankerError
from src.libs.reranker.reranker_factory import RerankerFactory, register_reranker, _RERANKER_REGISTRY
from src.core.settings import RerankSettings


class TestNoneReranker:
    """Test NoneReranker preserves original order."""

    def test_preserves_order(self):
        reranker = NoneReranker()
        candidates = [
            RerankCandidate(id="c1", text="first", score=0.5),
            RerankCandidate(id="c2", text="second", score=0.8),
            RerankCandidate(id="c3", text="third", score=0.3),
        ]
        result = reranker.rerank("query", candidates)
        assert [c.id for c in result] == ["c1", "c2", "c3"]

    def test_empty_candidates(self):
        reranker = NoneReranker()
        result = reranker.rerank("query", [])
        assert result == []

    def test_provider_name(self):
        assert NoneReranker().provider_name == "none"


class TestRerankerFactory:
    """Test RerankerFactory routing logic."""

    def test_disabled_returns_none_reranker(self):
        settings = RerankSettings(enabled=False, provider="llm")
        reranker = RerankerFactory.create(settings)
        assert isinstance(reranker, NoneReranker)

    def test_enabled_none_provider(self):
        settings = RerankSettings(enabled=True, provider="none")
        reranker = RerankerFactory.create(settings)
        assert isinstance(reranker, NoneReranker)

    def test_unknown_provider_raises(self):
        settings = RerankSettings(enabled=True, provider="unknown_backend")
        with pytest.raises(RerankerError, match="Unknown reranker provider"):
            RerankerFactory.create(settings)

    def test_available_providers_includes_none(self):
        providers = RerankerFactory.available_providers()
        assert "none" in providers

    def test_register_custom_reranker(self):
        @register_reranker("test_reranker")
        class TestReranker(BaseReranker):
            def __init__(self, settings=None): pass
            def rerank(self, query, candidates): return list(reversed(candidates))
            @property
            def provider_name(self): return "test_reranker"

        settings = RerankSettings(enabled=True, provider="test_reranker")
        reranker = RerankerFactory.create(settings)
        assert reranker.provider_name == "test_reranker"
        _RERANKER_REGISTRY.pop("test_reranker", None)

    def test_disabled_overrides_unknown_provider(self):
        # When disabled, provider value is irrelevant and must not raise.
        settings = RerankSettings(enabled=False, provider="does_not_exist")
        reranker = RerankerFactory.create(settings)
        assert isinstance(reranker, NoneReranker)

    def test_available_providers_is_sorted_unique(self):
        providers = RerankerFactory.available_providers()
        assert providers == sorted(providers)
        assert len(providers) == len(set(providers))

    def test_provider_lookup_is_case_insensitive(self):
        @register_reranker("CaseTest")
        class CaseReranker(BaseReranker):
            def __init__(self, settings=None): pass
            def rerank(self, query, candidates): return candidates
            @property
            def provider_name(self): return "casetest"

        try:
            settings = RerankSettings(enabled=True, provider="CASETEST")
            reranker = RerankerFactory.create(settings)
            assert reranker.provider_name == "casetest"
        finally:
            _RERANKER_REGISTRY.pop("casetest", None)

    def test_unknown_provider_error_lists_available(self):
        settings = RerankSettings(enabled=True, provider="zzz_unknown")
        with pytest.raises(RerankerError, match="none"):
            RerankerFactory.create(settings)

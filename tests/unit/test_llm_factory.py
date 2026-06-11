"""Tests for LLM factory and base interface."""

import pytest

from src.libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse, LLMError
from src.libs.llm.llm_factory import LLMFactory, register_llm, _LLM_REGISTRY
from src.core.settings import LLMSettings

class FakeLLM(BaseLLM):
    """Fake LLM for testing."""

    def __init__(self, settings=None):
        self.settings = settings

    def chat(self, messages, **kwargs):
        return ChatResponse(content="fake response", model="fake-model")

    @property
    def provider_name(self):
        return "fake"


class TestLLMFactory:
    """Test LLMFactory routing logic."""

    def setup_method(self):
        self._original = _LLM_REGISTRY.copy()
        _LLM_REGISTRY["fake"] = FakeLLM

    def teardown_method(self):
        _LLM_REGISTRY.clear()
        _LLM_REGISTRY.update(self._original)

    def test_create_known_provider(self):
        settings = LLMSettings(provider="fake", model="test")
        llm = LLMFactory.create(settings)
        assert isinstance(llm, FakeLLM)
        assert llm.provider_name == "fake"

    def test_create_unknown_provider_raises(self):
        settings = LLMSettings(provider="nonexistent", model="test")
        with pytest.raises(LLMError, match="Unknown LLM provider"):
            LLMFactory.create(settings)

    def test_case_insensitive_provider(self):
        settings = LLMSettings(provider="FAKE", model="test")
        llm = LLMFactory.create(settings)
        assert isinstance(llm, FakeLLM)

    def test_available_providers(self):
        providers = LLMFactory.available_providers()
        assert "fake" in providers

    def test_register_decorator(self):
        @register_llm("test_provider")
        class TestLLM(BaseLLM):
            def __init__(self, settings=None): pass
            def chat(self, messages, **kwargs): return ChatResponse(content="")
            @property
            def provider_name(self): return "test_provider"

        assert "test_provider" in _LLM_REGISTRY
        _LLM_REGISTRY.pop("test_provider", None)


class TestBaseLLM:
    """Test BaseLLM interface contracts."""

    def test_chat_returns_response(self):
        llm = FakeLLM()
        resp = llm.chat([ChatMessage(role="user", content="hello")])
        assert isinstance(resp, ChatResponse)
        assert resp.content == "fake response"

    def test_chat_message_structure(self):
        msg = ChatMessage(role="system", content="you are helpful")
        assert msg.role == "system"
        assert msg.content == "you are helpful"

    def test_llm_error_includes_provider(self):
        err = LLMError("timeout", provider="openai")
        assert "openai" in str(err)
        assert "timeout" in str(err)


class TestBuiltinRegistration:
    """Verify built-in LLM providers are lazily registered by the factory."""

    def test_ensure_builtins_registers_all(self):
        from src.libs.llm import llm_factory

        # Force a fresh registration pass regardless of prior state.
        llm_factory._builtins_registered = False
        llm_factory._ensure_builtins_registered()
        for provider in ("openai", "azure", "ollama", "deepseek", "qwen"):
            assert provider in llm_factory._LLM_REGISTRY

    def test_available_providers_includes_builtins(self):
        providers = LLMFactory.available_providers()
        assert {"openai", "deepseek", "qwen"} <= set(providers)


class TestQwenLLM:
    """Qwen LLM provider (DashScope, OpenAI-compatible)."""

    def test_registered_and_creatable(self):
        import src.libs.llm.qwen_llm  # noqa: F401 - ensure decorator registration

        settings = LLMSettings(provider="qwen", model="qwen-plus")
        llm = LLMFactory.create(settings)
        assert llm.provider_name == "qwen"

    def test_defaults_to_dashscope_base_url(self):
        import src.libs.llm.qwen_llm  # noqa: F401

        llm = LLMFactory.create(LLMSettings(provider="qwen", model="qwen-plus"))
        assert "dashscope.aliyuncs.com" in llm._base_url

    def test_respects_explicit_base_url(self):
        import src.libs.llm.qwen_llm  # noqa: F401

        custom = "https://example.com/v1"
        llm = LLMFactory.create(
            LLMSettings(provider="qwen", model="qwen-plus", base_url=custom)
        )
        assert llm._base_url == custom

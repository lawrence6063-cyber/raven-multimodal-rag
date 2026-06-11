"""Tests for OpenAI/Azure/DeepSeek LLM providers (mock HTTP)."""

import pytest
from unittest.mock import patch, MagicMock

from src.libs.llm.base_llm import ChatMessage, ChatResponse, LLMError
from src.libs.llm.llm_factory import LLMFactory, _LLM_REGISTRY
from src.core.settings import LLMSettings

# Import to trigger registration
from src.libs.llm.openai_llm import OpenAILLM
from src.libs.llm.azure_llm import AzureLLM
from src.libs.llm.deepseek_llm import DeepSeekLLM


class TestOpenAILLM:
    """Test OpenAI LLM with mocked API."""

    def test_factory_creates_openai(self):
        settings = LLMSettings(provider="openai", model="gpt-4o", api_key="sk-test")
        llm = LLMFactory.create(settings)
        assert isinstance(llm, OpenAILLM)
        assert llm.provider_name == "openai"

    @patch("openai.OpenAI")
    def test_chat_success(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_client.chat.completions.create.return_value = mock_response

        settings = LLMSettings(provider="openai", model="gpt-4o", api_key="sk-test")
        llm = OpenAILLM(settings)
        resp = llm.chat([ChatMessage(role="user", content="hi")])

        assert resp.content == "Hello!"
        assert resp.usage["total_tokens"] == 15

    @patch("openai.OpenAI")
    def test_auth_error(self, mock_openai_cls):
        import openai as real_openai
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = real_openai.AuthenticationError(
            "bad key", response=MagicMock(status_code=401), body=None
        )

        settings = LLMSettings(provider="openai", model="gpt-4o", api_key="bad")
        llm = OpenAILLM(settings)
        with pytest.raises(LLMError, match="Authentication"):
            llm.chat([ChatMessage(role="user", content="hi")])


class TestAzureLLM:
    """Test Azure LLM with mocked API."""

    def test_factory_creates_azure(self):
        settings = LLMSettings(provider="azure", model="gpt-4o", api_key="key", azure_endpoint="https://test.openai.azure.com")
        llm = LLMFactory.create(settings)
        assert isinstance(llm, AzureLLM)
        assert llm.provider_name == "azure"

    @patch("openai.AzureOpenAI")
    def test_chat_success(self, mock_azure_cls):
        mock_client = MagicMock()
        mock_azure_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Azure response"))]
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock(prompt_tokens=8, completion_tokens=3, total_tokens=11)
        mock_client.chat.completions.create.return_value = mock_response

        settings = LLMSettings(provider="azure", model="gpt-4o", api_key="key", azure_endpoint="https://x.openai.azure.com")
        llm = AzureLLM(settings)
        resp = llm.chat([ChatMessage(role="user", content="test")])

        assert resp.content == "Azure response"


class TestDeepSeekLLM:
    """Test DeepSeek LLM."""

    def test_factory_creates_deepseek(self):
        settings = LLMSettings(provider="deepseek", model="deepseek-chat", api_key="key")
        llm = LLMFactory.create(settings)
        assert isinstance(llm, DeepSeekLLM)
        assert llm.provider_name == "deepseek"
        assert llm._base_url == "https://api.deepseek.com/v1"


class TestRegistration:
    """Test all LLM providers are registered."""

    def test_all_providers_registered(self):
        assert "openai" in _LLM_REGISTRY
        assert "azure" in _LLM_REGISTRY
        assert "deepseek" in _LLM_REGISTRY
        assert "ollama" in _LLM_REGISTRY

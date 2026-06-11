"""Tests for Ollama LLM (mock HTTP)."""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.libs.llm.base_llm import ChatMessage, LLMError
from src.libs.llm.ollama_llm import OllamaLLM
from src.libs.llm.llm_factory import LLMFactory
from src.core.settings import LLMSettings


class TestOllamaLLM:
    """Test Ollama LLM with mocked HTTP."""

    def test_factory_creates_ollama(self):
        settings = LLMSettings(provider="ollama", model="llama3", base_url="http://localhost:11434")
        llm = LLMFactory.create(settings)
        assert isinstance(llm, OllamaLLM)
        assert llm.provider_name == "ollama"

    @patch("urllib.request.urlopen")
    def test_chat_success(self, mock_urlopen):
        response_body = json.dumps({
            "model": "llama3",
            "message": {"role": "assistant", "content": "Ollama response"},
            "prompt_eval_count": 10,
            "eval_count": 20,
        }).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        settings = LLMSettings(provider="ollama", model="llama3")
        llm = OllamaLLM(settings)
        resp = llm.chat([ChatMessage(role="user", content="hello")])

        assert resp.content == "Ollama response"
        assert resp.usage["prompt_tokens"] == 10
        assert resp.usage["completion_tokens"] == 20

    @patch("urllib.request.urlopen")
    def test_connection_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        settings = LLMSettings(provider="ollama", model="llama3")
        llm = OllamaLLM(settings)
        with pytest.raises(LLMError, match="Cannot connect to Ollama"):
            llm.chat([ChatMessage(role="user", content="hi")])

    def test_default_base_url(self):
        settings = LLMSettings(provider="ollama", model="llama3")
        llm = OllamaLLM(settings)
        assert llm._base_url == "http://localhost:11434"

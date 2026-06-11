"""Tests for Vision LLM factory and Azure Vision LLM."""

import pytest
from unittest.mock import patch, MagicMock

from src.libs.llm.base_vision_llm import BaseVisionLLM
from src.libs.llm.base_llm import ChatResponse, LLMError
from src.libs.llm.azure_vision_llm import AzureVisionLLM
from src.core.settings import VisionLLMSettings


class TestBaseVisionLLM:
    """Test BaseVisionLLM interface."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseVisionLLM()


class TestAzureVisionLLM:
    """Test Azure Vision LLM with mocked API."""

    def test_instantiation(self):
        settings = VisionLLMSettings(
            enabled=True, provider="azure", model="gpt-4o",
            api_key="key", azure_endpoint="https://test.openai.azure.com"
        )
        vision = AzureVisionLLM(settings)
        assert vision.provider_name == "azure_vision"

    @patch("openai.AzureOpenAI")
    def test_chat_with_image_success(self, mock_azure_cls):
        mock_client = MagicMock()
        mock_azure_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="A diagram showing..."))]
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        mock_client.chat.completions.create.return_value = mock_response

        settings = VisionLLMSettings(
            enabled=True, provider="azure", model="gpt-4o",
            api_key="key", azure_endpoint="https://test.openai.azure.com"
        )
        vision = AzureVisionLLM(settings)

        # Test with bytes input — use a real minimal 1x1 PNG
        import base64
        _MINIMAL_PNG = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
            "nGNgYPgPAAEDAQAIicLsAAAAAElFTkSuQmCC"
        )
        resp = vision.chat_with_image("Describe this image", _MINIMAL_PNG)
        assert resp.content == "A diagram showing..."
        assert resp.usage["total_tokens"] == 150

    def test_image_not_found(self):
        settings = VisionLLMSettings(
            enabled=True, provider="azure", model="gpt-4o",
            api_key="key", azure_endpoint="https://test.openai.azure.com"
        )
        vision = AzureVisionLLM(settings)
        with pytest.raises(LLMError, match="Image not found"):
            vision.chat_with_image("describe", "/nonexistent/image.png")

"""
Unit tests for OpenAI LLM service implementation.
Tests written first following TDD approach.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import load_config
from src.models.llm_models import LLMRequest, LLMResponse

# Import the service we're about to create
from src.services.openai import OpenAILLMService


class TestOpenAILLMService:
    """Test OpenAI LLM service implementation."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for testing."""
        mock_client = AsyncMock()

        # Mock successful response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated response from OpenAI"
        mock_response.model = "gpt-4o-mini"
        mock_response.usage.total_tokens = 25
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 10
        mock_response.usage.prompt_tokens_details = None
        mock_response.usage.completion_tokens_details = None

        mock_client.chat.completions.create.return_value = mock_response

        return mock_client

    @pytest.fixture
    def sample_config(self):
        """Sample OpenAI configuration."""
        return {
            "default_model": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://api.openai.com/v1",
            "timeout_seconds": 30,
            "max_retries": 3,
        }

    @pytest.fixture
    def openai_service(self, sample_config, mock_openai_client):
        """OpenAI service instance with mocked client."""
        with patch("src.services.openai.AsyncOpenAI") as mock_openai:
            mock_openai.return_value = mock_openai_client
            service = OpenAILLMService(api_key="test-key", config=sample_config)
            service.client = mock_openai_client  # Direct assignment for testing
            return service

    @pytest.mark.asyncio
    async def test_generate_success(self, openai_service, mock_openai_client):
        """Test successful generation with OpenAI."""
        request = LLMRequest(
            model="gpt-4o-mini",
            prompt="Generate a simple math question",
            temperature=0.7,
            max_tokens=100,
            stream=False,  # Explicitly disable streaming for this test
        )

        response = await openai_service.generate(request)

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.content == "Generated response from OpenAI"
        assert response.model_used == "gpt-4o-mini"
        assert response.tokens_used == 25
        assert response.provider == "openai"
        assert response.cost_estimate > 0
        assert response.latency_ms >= 0  # Could be 0 in fast mock

        # Verify OpenAI client was called correctly
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args[1]

        assert call_args["model"] == "gpt-4o-mini"
        assert call_args["temperature"] == 0.7
        assert call_args["max_tokens"] == 100
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "Generate a simple math question"

    @pytest.mark.asyncio
    async def test_generate_with_system_message(self, openai_service, mock_openai_client):
        """Test generation with system message."""
        request = LLMRequest(
            model="gpt-4o-mini",
            prompt="Create a question about fractions",
            system_message="You are a mathematics teacher",
            temperature=0.5,
            stream=False,
        )

        await openai_service.generate(request)

        # Verify system message handling
        call_args = mock_openai_client.chat.completions.create.call_args[1]
        messages = call_args["messages"]

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a mathematics teacher"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Create a question about fractions"

    @pytest.mark.asyncio
    async def test_generate_with_extra_params(self, openai_service, mock_openai_client):
        """Test generation with OpenAI-specific extra parameters."""
        request = LLMRequest(
            model="gpt-4o-mini",
            prompt="Generate content",
            extra_params={
                "frequency_penalty": 0.1,
                "presence_penalty": 0.2,
                "stop": ["END", "STOP"],
                "n": 1,
            },
            stream=False,
        )

        await openai_service.generate(request)

        # Verify extra params were passed
        call_args = mock_openai_client.chat.completions.create.call_args[1]

        assert call_args["frequency_penalty"] == 0.1
        assert call_args["presence_penalty"] == 0.2
        assert call_args["stop"] == ["END", "STOP"]
        assert call_args["n"] == 1

    @pytest.mark.asyncio
    async def test_generate_api_error_handling(self, openai_service, mock_openai_client):
        """Test handling of OpenAI API errors."""
        from openai import APIError

        # Mock API error
        mock_openai_client.chat.completions.create.side_effect = APIError(
            message="Rate limit exceeded", response=MagicMock(), body=None
        )

        request = LLMRequest(model="gpt-4o-mini", prompt="Test prompt", stream=False)

        with pytest.raises(Exception) as exc_info:
            await openai_service.generate(request)

        assert "OpenAI API error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_timeout_handling(self, openai_service, mock_openai_client):
        """Test handling of request timeouts."""
        # Mock timeout
        mock_openai_client.chat.completions.create.side_effect = asyncio.TimeoutError()

        request = LLMRequest(model="gpt-4o-mini", prompt="Test prompt", stream=False)

        with pytest.raises(Exception) as exc_info:
            await openai_service.generate(request)

        assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_cost_calculation(self, openai_service, mock_openai_client):
        """Test cost calculation for different models."""
        # Test with gpt-4o-mini (cheapest)
        request = LLMRequest(model="gpt-4o-mini", prompt="Short prompt", stream=False)

        response = await openai_service.generate(request)

        # Verify cost calculation
        expected_cost = (15 * 0.00000015) + (10 * 0.0000006)  # gpt-4o-mini pricing
        assert abs(response.cost_estimate - expected_cost) < 0.000001

    def test_parameter_mapping(self, openai_service):
        """Test mapping of LLM parameters to OpenAI format."""
        request = LLMRequest(
            model="gpt-4o-mini",
            prompt="Test prompt",
            system_message="System message",
            temperature=0.8,
            max_tokens=150,
            top_p=0.9,
            stop_sequences=["STOP"],
            extra_params={"frequency_penalty": 0.1, "presence_penalty": 0.2},
        )

        mapped_params = openai_service._map_params(request)

        # Verify parameter mapping
        assert mapped_params["model"] == "gpt-4o-mini"
        assert mapped_params["temperature"] == 0.8
        assert mapped_params["max_tokens"] == 150
        assert mapped_params["top_p"] == 0.9
        assert mapped_params["stop"] == ["STOP"]
        assert mapped_params["frequency_penalty"] == 0.1
        assert mapped_params["presence_penalty"] == 0.2

        # Verify message structure
        messages = mapped_params["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System message"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Test prompt"

    def test_parameter_mapping_no_system_message(self, openai_service):
        """Test parameter mapping without system message."""
        request = LLMRequest(model="gpt-4o-mini", prompt="Test prompt")

        mapped_params = openai_service._map_params(request)

        # Should only have user message
        messages = mapped_params["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Test prompt"

    def test_response_normalization(self, openai_service):
        """Test normalization of OpenAI response to common format."""
        # Mock OpenAI response
        openai_response = MagicMock()
        openai_response.choices = [MagicMock()]
        openai_response.choices[0].message.content = "Test response content"
        openai_response.model = "gpt-4o-mini"
        openai_response.usage.total_tokens = 50
        openai_response.usage.prompt_tokens = 20
        openai_response.usage.completion_tokens = 30
        openai_response.usage.prompt_tokens_details = None
        openai_response.usage.completion_tokens_details = None

        request = LLMRequest(model="gpt-4o-mini", prompt="test", stream=False)

        normalized = openai_service._normalize_response(openai_response, request, 1500)

        assert isinstance(normalized, LLMResponse)
        assert normalized.content == "Test response content"
        assert normalized.model_used == "gpt-4o-mini"
        assert normalized.tokens_used == 50
        assert normalized.provider == "openai"
        assert normalized.latency_ms == 1500
        assert normalized.cost_estimate > 0
        assert "prompt_tokens" in normalized.metadata
        assert "completion_tokens" in normalized.metadata

    def test_model_fallback(self, openai_service):
        """Test model fallback when requested model is not available."""
        request = LLMRequest(
            model="gpt-5-ultra",  # Non-existent model
            prompt="Test prompt",
            stream=False,
        )

        # Should fall back to default model from config
        mapped_params = openai_service._map_params(request)
        # In real implementation, this might fall back to default model
        # For now, just verify it doesn't crash
        assert "model" in mapped_params

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, openai_service, mock_openai_client):
        """Test handling of concurrent requests."""
        requests = [LLMRequest(model="gpt-4o-mini", prompt=f"Request {i}", stream=False) for i in range(3)]

        # Execute concurrent requests
        responses = await asyncio.gather(
            *[openai_service.generate(request) for request in requests]
        )

        # Verify all responses
        assert len(responses) == 3
        for response in responses:
            assert isinstance(response, LLMResponse)
            assert response.provider == "openai"

        # Verify client was called for each request
        assert mock_openai_client.chat.completions.create.call_count == 3

    def test_configuration_validation(self, sample_config):
        """Test service configuration validation."""
        # Test missing API key
        with pytest.raises(ValueError) as exc_info:
            OpenAILLMService(api_key="", config=sample_config)
        assert "API key" in str(exc_info.value)

        # Test invalid config
        invalid_config = sample_config.copy()
        invalid_config["timeout_seconds"] = -1

        with pytest.raises(ValueError) as exc_info:
            OpenAILLMService(api_key="test-key", config=invalid_config)
        assert "timeout" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rate_limiting_behavior(self, openai_service, mock_openai_client):
        """Test behavior under rate limiting."""
        from openai import RateLimitError

        # First call succeeds, second hits rate limit, third succeeds
        mock_openai_client.chat.completions.create.side_effect = [
            mock_openai_client.chat.completions.create.return_value,
            RateLimitError(message="Rate limit exceeded", response=MagicMock(), body=None),
            mock_openai_client.chat.completions.create.return_value,
        ]

        request = LLMRequest(model="gpt-4o-mini", prompt="Test", stream=False)

        # First request should succeed
        response1 = await openai_service.generate(request)
        assert response1.content == "Generated response from OpenAI"

        # Second request should fail with rate limit
        with pytest.raises(Exception) as exc_info:
            await openai_service.generate(request)
        assert "rate limit" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_streaming_support_preparation(self, openai_service):
        """Test preparation for streaming support (future feature)."""
        # This test verifies the service structure supports streaming
        request = LLMRequest(
            model="gpt-4o-mini", prompt="Test streaming", extra_params={"stream": True}
        )

        mapped_params = openai_service._map_params(request)

        # Verify streaming parameter is preserved
        assert mapped_params.get("stream") is True

        # Note: Actual streaming implementation would be added later
        # This test just ensures the structure supports it


class TestOpenAIServiceIntegration:
    """Integration tests for OpenAI service with configuration."""

    def test_service_creation_from_config(self):
        """Test creating service from application configuration."""
        config = load_config()
        openai_config = config.llm_providers.openai

        # Test service creation (without actual API key)
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            service = OpenAILLMService.from_config(openai_config)

            assert service.config["default_model"] == openai_config.default_model
            assert service.config["base_url"] == openai_config.base_url
            assert service.config["timeout_seconds"] == openai_config.timeout_seconds

    @pytest.mark.asyncio
    async def test_agent_integration(self):
        """Test integration with agent configuration."""
        config = load_config()

        # Create request for question generator agent
        request = config.create_llm_request_for_agent(
            agent_name="question_generator", prompt="Generate a question about algebra"
        )

        # Verify agent-specific configuration
        assert request.model == "gpt-4.1-nano"  # From agent config
        assert request.temperature == 0.8  # From agent config
        assert request.max_tokens == 2000  # From agent config
        assert "frequency_penalty" in request.extra_params

    def test_cost_tracking_integration(self):
        """Test cost tracking integration with configuration."""
        config = load_config()

        # Verify cost tracking is enabled
        assert config.llm_providers.cost_tracking is True

        # Test cost calculation helper
        from src.services.openai import calculate_openai_cost

        cost = calculate_openai_cost("gpt-4o-mini", 100, 50)
        expected = (100 * 0.00000015) + (50 * 0.0000006)
        assert abs(cost - expected) < 0.000001

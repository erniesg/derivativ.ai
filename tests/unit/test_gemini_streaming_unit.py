"""
Unit tests for Google Gemini streaming functionality.
Tests streaming responses and chunk processing without making actual API calls.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.models.llm_models import LLMRequest
from src.models.streaming_models import StreamingChunkType
from src.services.gemini import GeminiLLMService


class MockGeminiStreamChunk:
    """Mock Gemini stream chunk for testing."""

    def __init__(self, text: str = None, usage_metadata: dict = None, safety_ratings: list = None):
        self.text = text
        self.usage_metadata = usage_metadata
        self.safety_ratings = safety_ratings or []

        if usage_metadata:
            self.usage_metadata = Mock()
            self.usage_metadata.prompt_token_count = usage_metadata.get("prompt_token_count", 0)
            self.usage_metadata.candidates_token_count = usage_metadata.get(
                "candidates_token_count", 0
            )


class MockGeminiStreamResponse:
    """Mock Gemini streaming response for testing."""

    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.chunks) + 1:  # +1 for final usage chunk
            raise StopAsyncIteration

        if self.index < len(self.chunks):
            # Content chunks
            text = self.chunks[self.index]
            chunk = MockGeminiStreamChunk(text=text)
        else:
            # Final usage chunk
            chunk = MockGeminiStreamChunk(
                usage_metadata={"prompt_token_count": 10, "candidates_token_count": 15}
            )

        self.index += 1
        return chunk


@pytest.fixture
def gemini_service():
    """Create Gemini service for testing."""
    with (
        patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
        patch("google.generativeai.configure"),
        patch("google.generativeai.GenerativeModel") as mock_model,
    ):
        config = {
            "default_model": "gemini-2.0-flash-exp",
            "timeout_seconds": 30,
            "max_retries": 3,
        }
        service = GeminiLLMService(api_key="test-key", config=config)
        service.model = mock_model.return_value
        return service


@pytest.fixture
def streaming_request():
    """Create a test streaming request."""
    return LLMRequest(
        model="gemini-2.0-flash-exp",
        prompt="Write a short poem about AI",
        max_tokens=50,
        temperature=0.7,
        stream=True,
    )


class TestGeminiStreamingModels:
    """Test Gemini-specific streaming model behavior."""

    def test_gemini_pricing_calculation(self):
        """Test Gemini cost calculation."""
        from src.services.gemini import calculate_gemini_cost

        # Test gemini-2.0-flash-exp pricing (free tier)
        cost = calculate_gemini_cost("gemini-2.0-flash-exp", 1000, 500)
        assert cost == 0.0  # Free tier

        # Test gemini-1.5-flash pricing
        cost = calculate_gemini_cost("gemini-1.5-flash", 1000, 500)
        expected = (1000 / 1000 * 0.075) + (500 / 1000 * 0.30)
        assert abs(cost - expected) < 0.000001

        # Test default fallback
        cost = calculate_gemini_cost("unknown-model", 1000, 500)
        expected_fallback = (1000 / 1000 * 0.075) + (500 / 1000 * 0.30)  # flash pricing
        assert abs(cost - expected_fallback) < 0.000001

    def test_gemini_service_initialization(self, gemini_service):
        """Test Gemini service initialization."""
        assert gemini_service.config["default_model"] == "gemini-2.0-flash-exp"
        assert gemini_service.config["timeout_seconds"] == 30
        assert str(gemini_service) == "GeminiLLMService(model=gemini-2.0-flash-exp)"

    def test_get_available_models(self, gemini_service):
        """Test available models list."""
        models = gemini_service.get_available_models()
        assert "gemini-2.0-flash-exp" in models
        assert "gemini-1.5-flash" in models
        assert "gemini-1.5-pro" in models
        assert len(models) >= 3


class TestGeminiStreaming:
    """Test Gemini streaming functionality."""

    @pytest.mark.asyncio
    async def test_generate_stream_success(self, gemini_service, streaming_request):
        """Test successful streaming generation."""
        # Mock the Gemini model streaming response
        mock_chunks = ["AI ", "creates ", "and ", "innovates ", "constantly."]
        mock_stream = MockGeminiStreamResponse(mock_chunks)

        async def mock_generate_content_async(*args, **kwargs):
            return mock_stream

        gemini_service.model.generate_content_async = mock_generate_content_async

        chunks = []
        async for chunk in gemini_service.generate_stream(streaming_request):
            chunks.append(chunk)

        # Should have content chunks + done chunk
        content_chunks = [c for c in chunks if c.chunk_type == StreamingChunkType.CONTENT]
        done_chunks = [c for c in chunks if c.chunk_type == StreamingChunkType.DONE]

        assert len(content_chunks) == len(mock_chunks)
        assert len(done_chunks) == 1

        # Verify accumulated content
        final_content = "".join(c.delta for c in content_chunks)
        assert final_content == "".join(mock_chunks)

        # Check cost tracking in done chunk
        done_chunk = done_chunks[0]
        assert "cost_estimate" in done_chunk.metadata
        assert done_chunk.metadata["cost_estimate"] >= 0  # Free tier, so 0

    @pytest.mark.asyncio
    async def test_generate_stream_api_error(self, gemini_service, streaming_request):
        """Test streaming with API error."""

        async def mock_generate_error(*args, **kwargs):
            raise Exception("API Error: Rate limit exceeded")

        gemini_service.model.generate_content_async = mock_generate_error

        chunks = []
        async for chunk in gemini_service.generate_stream(streaming_request):
            chunks.append(chunk)

        # Should have error chunk
        error_chunks = [c for c in chunks if c.chunk_type == StreamingChunkType.ERROR]
        assert len(error_chunks) == 1
        assert "Rate limit exceeded" in error_chunks[0].metadata["error"]

    @pytest.mark.asyncio
    async def test_generate_routing_to_stream(self, gemini_service):
        """Test that generate() routes to streaming when stream=True."""
        request = LLMRequest(
            model="gemini-2.0-flash-exp",
            prompt="Test",
            stream=True,
        )

        # Mock generate_stream method
        mock_stream_response = AsyncMock()
        with patch.object(gemini_service, "generate_stream", return_value=mock_stream_response):
            result = await gemini_service.generate(request)

            # Should return the streaming generator
            assert result == mock_stream_response
            gemini_service.generate_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_routing_to_non_stream(self, gemini_service):
        """Test that generate() routes to non-streaming when stream=False."""
        request = LLMRequest(
            model="gemini-2.0-flash-exp",
            prompt="Test",
            stream=False,
        )

        # Mock generate_non_stream method
        mock_response = Mock()
        with patch.object(gemini_service, "generate_non_stream", return_value=mock_response):
            result = await gemini_service.generate(request)

            # Should return the non-streaming response
            assert result == mock_response
            gemini_service.generate_non_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_with_overrides(self, gemini_service):
        """Test streaming with runtime parameter overrides."""
        base_request = LLMRequest(
            model="gemini-2.0-flash-exp",
            prompt="Test prompt",
            temperature=0.5,
            stream=True,
        )

        mock_stream = MockGeminiStreamResponse(["Test"])

        async def mock_generate_content_async(*args, **kwargs):
            # Store the call arguments for verification
            mock_generate_content_async.call_kwargs = kwargs
            return mock_stream

        gemini_service.model.generate_content_async = mock_generate_content_async

        chunks = []
        async for chunk in gemini_service.generate_stream(
            base_request, temperature=0.9, max_tokens=100
        ):
            chunks.append(chunk)
            break  # Just get first chunk

        # Verify the API was called with overridden parameters
        call_args = mock_generate_content_async.call_kwargs
        generation_config = call_args["generation_config"]
        assert generation_config.temperature == 0.9  # Overridden
        assert generation_config.max_output_tokens == 100  # Overridden

    @pytest.mark.asyncio
    async def test_build_prompt_with_system_message(self, gemini_service):
        """Test prompt building with system message."""
        request = LLMRequest(
            model="gemini-2.0-flash-exp",
            prompt="Write a poem",
            system_message="You are a creative poet",
            stream=True,
        )

        prompt = gemini_service._build_prompt(request)
        assert "You are a creative poet" in prompt
        assert "Write a poem" in prompt
        assert prompt.startswith("You are a creative poet")

    @pytest.mark.asyncio
    async def test_build_prompt_without_system_message(self, gemini_service):
        """Test prompt building without system message."""
        request = LLMRequest(
            model="gemini-2.0-flash-exp",
            prompt="Write a poem",
            stream=True,
        )

        prompt = gemini_service._build_prompt(request)
        assert prompt == "Write a poem"


class TestGeminiNonStreaming:
    """Test Gemini non-streaming functionality."""

    @pytest.mark.asyncio
    async def test_generate_non_stream_success(self, gemini_service):
        """Test successful non-streaming generation."""
        request = LLMRequest(
            model="gemini-2.0-flash-exp",
            prompt="Say hello",
            max_tokens=10,
            temperature=0.1,
            stream=False,
        )

        # Mock Gemini response
        mock_response = Mock()
        mock_response.text = "Hello there!"
        mock_response.finish_reason = "stop"
        mock_response.safety_ratings = []
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 5
        mock_response.usage_metadata.candidates_token_count = 8

        async def mock_generate_content_async(*args, **kwargs):
            return mock_response

        gemini_service.model.generate_content_async = mock_generate_content_async

        response = await gemini_service.generate_non_stream(request)

        assert response.content == "Hello there!"
        assert response.model_used == "gemini-2.0-flash-exp"
        assert response.tokens_used == 13  # 5 + 8
        assert response.cost_estimate == 0.0  # Free tier
        assert response.provider == "google"
        assert response.metadata["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_batch_generate(self, gemini_service):
        """Test batch generation functionality."""
        requests = [
            LLMRequest(model="gemini-2.0-flash-exp", prompt="Hello", stream=False),
            LLMRequest(model="gemini-2.0-flash-exp", prompt="Goodbye", stream=False),
        ]

        # Mock individual generate calls
        mock_responses = [Mock(), Mock()]

        with patch.object(gemini_service, "generate", side_effect=mock_responses):
            responses = await gemini_service.batch_generate(requests)

            assert len(responses) == 2
            assert responses == mock_responses

    @pytest.mark.asyncio
    async def test_validate_connection(self, gemini_service):
        """Test connection validation."""
        # Mock successful response
        mock_response = Mock()
        mock_response.content = "Hi"

        with patch.object(gemini_service, "generate", return_value=mock_response):
            result = await gemini_service.validate_connection()
            assert result is True

        # Mock failed response
        with patch.object(gemini_service, "generate", side_effect=Exception("Connection failed")):
            result = await gemini_service.validate_connection()
            assert result is False


class TestGeminiServiceConfig:
    """Test Gemini service configuration and factory methods."""

    def test_from_config_success(self):
        """Test creating service from config."""
        from src.models.llm_models import LLMProviderConfig

        config = LLMProviderConfig(
            default_model="gemini-2.0-flash-exp",
            api_key_env="GOOGLE_API_KEY",
            timeout_seconds=30,
            max_retries=3,
        )

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
            patch("google.generativeai.configure"),
            patch("google.generativeai.GenerativeModel"),
        ):
            service = GeminiLLMService.from_config(config)
            assert service.config["default_model"] == "gemini-2.0-flash-exp"
            assert service.config["timeout_seconds"] == 30

    def test_from_config_missing_api_key(self):
        """Test error when API key is missing."""
        from src.models.llm_models import LLMProviderConfig

        config = LLMProviderConfig(
            default_model="gemini-2.0-flash-exp",
            api_key_env="MISSING_KEY",
            timeout_seconds=30,
            max_retries=3,
        )

        with pytest.raises(ValueError, match="API key not found"):
            GeminiLLMService.from_config(config)

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid timeout
        with (
            pytest.raises(ValueError, match="timeout_seconds must be positive"),
            patch("google.generativeai.configure"),
            patch("google.generativeai.GenerativeModel"),
        ):
            GeminiLLMService(api_key="test", config={"timeout_seconds": 0})

        # Test invalid retries
        with (
            pytest.raises(ValueError, match="max_retries must be non-negative"),
            patch("google.generativeai.configure"),
            patch("google.generativeai.GenerativeModel"),
        ):
            GeminiLLMService(api_key="test", config={"max_retries": -1})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit tests for Anthropic streaming functionality.
Tests streaming responses and chunk processing without making actual API calls.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.models.llm_models import LLMRequest
from src.models.streaming_models import StreamingChunkType
from src.services.anthropic import AnthropicLLMService


class MockAnthropicStreamChunk:
    """Mock Anthropic stream chunk for testing."""

    def __init__(
        self,
        chunk_type: str,
        content: str = None,
        model: str = "claude-3-5-haiku-20241022",
        usage: dict = None,
        index: int = 0,
    ):
        self.type = chunk_type
        self.model = model
        self.index = index

        if chunk_type == "content_block_delta" and content is not None:
            self.delta = Mock()
            self.delta.text = content

        if chunk_type == "message_start" and usage:
            self.message = Mock()
            self.message.usage = Mock()
            self.message.usage.input_tokens = usage.get("input_tokens", 0)

        if chunk_type == "message_delta" and usage:
            self.usage = Mock()
            self.usage.output_tokens = usage.get("output_tokens", 0)


class MockAnthropicStreamResponse:
    """Mock Anthropic streaming response for testing."""

    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.chunks) + 2:  # +2 for start and end chunks
            raise StopAsyncIteration

        if self.index == 0:
            # Message start chunk
            chunk = MockAnthropicStreamChunk("message_start", usage={"input_tokens": 10})
        elif self.index <= len(self.chunks):
            # Content chunks
            content = self.chunks[self.index - 1] if self.index <= len(self.chunks) else ""
            chunk = MockAnthropicStreamChunk("content_block_delta", content=content)
        else:
            # Message end chunk
            chunk = MockAnthropicStreamChunk("message_delta", usage={"output_tokens": 15})

        self.index += 1
        return chunk


@pytest.fixture
def anthropic_service():
    """Create Anthropic service for testing."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        config = {
            "default_model": "claude-3-5-haiku-20241022",
            "timeout_seconds": 30,
            "max_retries": 3,
        }
        return AnthropicLLMService(api_key="test-key", config=config)


@pytest.fixture
def streaming_request():
    """Create a test streaming request."""
    return LLMRequest(
        model="claude-3-5-haiku-20241022",
        prompt="Write a short haiku about Claude",
        max_tokens=50,
        temperature=0.7,
        stream=True,
    )


class TestAnthropicStreamingModels:
    """Test Anthropic-specific streaming model behavior."""

    def test_anthropic_pricing_calculation(self):
        """Test Anthropic cost calculation."""
        from src.services.anthropic import calculate_anthropic_cost

        # Test claude-3-5-haiku-20241022 pricing
        cost = calculate_anthropic_cost("claude-3-5-haiku-20241022", 1000, 500)
        expected = (1000 / 1000 * 0.0008) + (500 / 1000 * 0.004)
        assert abs(cost - expected) < 0.000001

        # Test default fallback
        cost = calculate_anthropic_cost("unknown-model", 1000, 500)
        assert cost == expected  # Should fallback to haiku pricing

    def test_anthropic_service_initialization(self, anthropic_service):
        """Test Anthropic service initialization."""
        assert anthropic_service.config["default_model"] == "claude-3-5-haiku-20241022"
        assert anthropic_service.config["timeout_seconds"] == 30
        assert str(anthropic_service) == "AnthropicLLMService(model=claude-3-5-haiku-20241022)"

    def test_get_available_models(self, anthropic_service):
        """Test available models list."""
        models = anthropic_service.get_available_models()
        assert "claude-3-5-haiku-20241022" in models
        assert "claude-3-5-sonnet-20241022" in models
        assert len(models) >= 3


class TestAnthropicStreaming:
    """Test Anthropic streaming functionality."""

    @pytest.mark.asyncio
    async def test_generate_stream_success(self, anthropic_service, streaming_request):
        """Test successful streaming generation."""
        # Mock the Anthropic client streaming response
        mock_chunks = ["Claude ", "thinks ", "deeply ", "about ", "words."]
        mock_stream = MockAnthropicStreamResponse(mock_chunks)

        async def mock_create(**kwargs):
            return mock_stream

        with patch.object(anthropic_service.client.messages, "create", side_effect=mock_create):
            chunks = []
            async for chunk in anthropic_service.generate_stream(streaming_request):
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
            assert done_chunk.metadata["cost_estimate"] > 0

    @pytest.mark.asyncio
    async def test_generate_stream_api_error(self, anthropic_service, streaming_request):
        """Test streaming with API error."""
        import anthropic
        from httpx import Request

        # Mock API error with proper arguments
        mock_request = Mock(spec=Request)
        mock_error = anthropic.APIError("Rate limit exceeded", request=mock_request, body=None)

        async def mock_create_error(**kwargs):
            raise mock_error

        with patch.object(
            anthropic_service.client.messages, "create", side_effect=mock_create_error
        ):
            chunks = []
            async for chunk in anthropic_service.generate_stream(streaming_request):
                chunks.append(chunk)

            # Should have error chunk
            error_chunks = [c for c in chunks if c.chunk_type == StreamingChunkType.ERROR]
            assert len(error_chunks) == 1
            assert "Rate limit exceeded" in error_chunks[0].metadata["error"]

    @pytest.mark.asyncio
    async def test_generate_routing_to_stream(self, anthropic_service):
        """Test that generate() routes to streaming when stream=True."""
        request = LLMRequest(model="claude-3-5-haiku-20241022", prompt="Test", stream=True)

        # Mock generate_stream method
        mock_stream_response = AsyncMock()
        with patch.object(anthropic_service, "generate_stream", return_value=mock_stream_response):
            result = await anthropic_service.generate(request)

            # Should return the streaming generator
            assert result == mock_stream_response
            anthropic_service.generate_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_routing_to_non_stream(self, anthropic_service):
        """Test that generate() routes to non-streaming when stream=False."""
        request = LLMRequest(model="claude-3-5-haiku-20241022", prompt="Test", stream=False)

        # Mock generate_non_stream method
        mock_response = Mock()
        with patch.object(anthropic_service, "generate_non_stream", return_value=mock_response):
            result = await anthropic_service.generate(request)

            # Should return the non-streaming response
            assert result == mock_response
            anthropic_service.generate_non_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_with_overrides(self, anthropic_service):
        """Test streaming with runtime parameter overrides."""
        base_request = LLMRequest(
            model="claude-3-5-haiku-20241022", prompt="Test prompt", temperature=0.5, stream=True
        )

        mock_stream = MockAnthropicStreamResponse(["Test"])

        async def mock_create(**kwargs):
            # Store the call arguments for verification
            mock_create.call_kwargs = kwargs
            return mock_stream

        with patch.object(anthropic_service.client.messages, "create", side_effect=mock_create):
            chunks = []
            async for chunk in anthropic_service.generate_stream(
                base_request, temperature=0.9, max_tokens=100
            ):
                chunks.append(chunk)
                break  # Just get first chunk

            # Verify the API was called with overridden parameters
            call_args = mock_create.call_kwargs
            assert call_args["temperature"] == 0.9  # Overridden
            assert call_args["max_tokens"] == 100  # Overridden
            assert call_args["stream"] is True  # From request

    @pytest.mark.asyncio
    async def test_parameter_mapping_anthropic_format(self, anthropic_service, streaming_request):
        """Test that LLM parameters are correctly mapped to Anthropic format."""
        mock_stream = MockAnthropicStreamResponse(["Test"])

        async def mock_create(**kwargs):
            mock_create.call_kwargs = kwargs
            return mock_stream

        with patch.object(anthropic_service.client.messages, "create", side_effect=mock_create):
            async for chunk in anthropic_service.generate_stream(streaming_request):
                break  # Just test the first chunk

            # Verify API call parameters
            call_args = mock_create.call_kwargs
            assert call_args["model"] == "claude-3-5-haiku-20241022"
            assert call_args["temperature"] == 0.7
            assert call_args["max_tokens"] == 50
            assert call_args["stream"] is True
            assert "messages" in call_args
            assert len(call_args["messages"]) == 1
            assert call_args["messages"][0]["role"] == "user"
            assert call_args["messages"][0]["content"] == "Write a short haiku about Claude"

    @pytest.mark.asyncio
    async def test_system_message_mapping(self, anthropic_service):
        """Test that system messages are properly mapped to Anthropic format."""
        request = LLMRequest(
            model="claude-3-5-haiku-20241022",
            prompt="Write a poem",
            system_message="You are a creative poet",
            stream=True,
        )

        mock_stream = MockAnthropicStreamResponse(["Test"])

        async def mock_create(**kwargs):
            mock_create.call_kwargs = kwargs
            return mock_stream

        with patch.object(anthropic_service.client.messages, "create", side_effect=mock_create):
            async for chunk in anthropic_service.generate_stream(request):
                break

            # Verify system message is in correct Anthropic format
            call_args = mock_create.call_kwargs
            assert call_args["system"] == "You are a creative poet"
            assert "messages" in call_args
            assert call_args["messages"][0]["role"] == "user"
            assert call_args["messages"][0]["content"] == "Write a poem"


class TestAnthropicNonStreaming:
    """Test Anthropic non-streaming functionality."""

    @pytest.mark.asyncio
    async def test_generate_non_stream_success(self, anthropic_service):
        """Test successful non-streaming generation."""
        request = LLMRequest(
            model="claude-3-5-haiku-20241022",
            prompt="Say hello",
            max_tokens=10,
            temperature=0.1,
            stream=False,
        )

        # Mock Anthropic response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Hello there!"
        mock_response.model = "claude-3-5-haiku-20241022"
        mock_response.stop_reason = "end_turn"
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 8

        async def mock_create(**kwargs):
            return mock_response

        with patch.object(anthropic_service.client.messages, "create", side_effect=mock_create):
            response = await anthropic_service.generate_non_stream(request)

            assert response.content == "Hello there!"
            assert response.model_used == "claude-3-5-haiku-20241022"
            assert response.tokens_used == 13  # 5 + 8
            assert response.cost_estimate > 0
            assert response.provider == "anthropic"
            assert response.metadata["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_batch_generate(self, anthropic_service):
        """Test batch generation functionality."""
        requests = [
            LLMRequest(model="claude-3-5-haiku-20241022", prompt="Hello", stream=False),
            LLMRequest(model="claude-3-5-haiku-20241022", prompt="Goodbye", stream=False),
        ]

        # Mock individual generate calls
        mock_responses = [Mock(), Mock()]

        with patch.object(anthropic_service, "generate", side_effect=mock_responses):
            responses = await anthropic_service.batch_generate(requests)

            assert len(responses) == 2
            assert responses == mock_responses

    @pytest.mark.asyncio
    async def test_validate_connection(self, anthropic_service):
        """Test connection validation."""
        # Mock successful response
        mock_response = Mock()
        mock_response.content = "Hi"

        with patch.object(anthropic_service, "generate", return_value=mock_response):
            result = await anthropic_service.validate_connection()
            assert result is True

        # Mock failed response
        with patch.object(
            anthropic_service, "generate", side_effect=Exception("Connection failed")
        ):
            result = await anthropic_service.validate_connection()
            assert result is False


class TestAnthropicServiceConfig:
    """Test Anthropic service configuration and factory methods."""

    def test_from_config_success(self):
        """Test creating service from config."""
        from src.models.llm_models import LLMProviderConfig

        config = LLMProviderConfig(
            default_model="claude-3-5-haiku-20241022",
            api_key_env="ANTHROPIC_API_KEY",
            timeout_seconds=30,
            max_retries=3,
        )

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            service = AnthropicLLMService.from_config(config)
            assert service.config["default_model"] == "claude-3-5-haiku-20241022"
            assert service.config["timeout_seconds"] == 30

    def test_from_config_missing_api_key(self):
        """Test error when API key is missing."""
        from src.models.llm_models import LLMProviderConfig

        config = LLMProviderConfig(
            default_model="claude-3-5-haiku-20241022",
            api_key_env="MISSING_KEY",
            timeout_seconds=30,
            max_retries=3,
        )

        with pytest.raises(ValueError, match="API key not found"):
            AnthropicLLMService.from_config(config)

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid timeout
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            AnthropicLLMService(api_key="test", config={"timeout_seconds": 0})

        # Test invalid retries
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            AnthropicLLMService(api_key="test", config={"max_retries": -1})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit tests for OpenAI streaming functionality.
Tests streaming responses and chunk processing without making actual API calls.
"""

import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.models.llm_models import LLMRequest
from src.models.streaming_models import StreamHandler, StreamingChunk, StreamingChunkType
from src.services.openai import OpenAILLMService


class MockOpenAIStreamChunk:
    """Mock OpenAI stream chunk for testing."""

    def __init__(self, content: str = None, finish_reason: str = None, model: str = "gpt-4o-mini"):
        self.choices = [Mock()]
        self.choices[0].delta = Mock()
        self.choices[0].finish_reason = finish_reason
        self.model = model

        if content is not None:
            self.choices[0].delta.content = content
        else:
            # Simulate no content (e.g., first chunk)
            self.choices[0].delta = Mock(spec=[])  # Empty delta


class MockOpenAIStreamResponse:
    """Mock OpenAI streaming response for testing."""

    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.chunks):
            raise StopAsyncIteration

        chunk_content = self.chunks[self.index]
        self.index += 1

        # Last chunk has no content but finish_reason
        if self.index == len(self.chunks):
            return MockOpenAIStreamChunk(content=chunk_content, finish_reason="stop")
        else:
            return MockOpenAIStreamChunk(content=chunk_content)


async def create_mock_stream(chunks: list[str]) -> MockOpenAIStreamResponse:
    """Create an awaitable mock stream."""
    return MockOpenAIStreamResponse(chunks)


@pytest.fixture
def openai_service():
    """Create OpenAI service for testing."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        config = {"default_model": "gpt-4o-mini", "timeout_seconds": 30, "max_retries": 3}
        return OpenAILLMService(api_key="test-key", config=config)


@pytest.fixture
def streaming_request():
    """Create a test streaming request."""
    return LLMRequest(
        model="gpt-4o-mini",
        prompt="Write a short haiku about AI",
        max_tokens=50,
        temperature=0.7,
        stream=True,
    )


class TestStreamingModels:
    """Test streaming model classes."""

    def test_streaming_chunk_creation(self):
        """Test StreamingChunk creation and validation."""
        chunk = StreamingChunk(
            chunk_type=StreamingChunkType.CONTENT,
            content="Hello",
            delta=" world",
            total_content="Hello world",
            chunk_index=1,
            metadata={"model": "gpt-4o-mini"},
        )

        assert chunk.chunk_type == StreamingChunkType.CONTENT
        assert chunk.content == "Hello"
        assert chunk.delta == " world"
        assert chunk.total_content == "Hello world"
        assert chunk.chunk_index == 1
        assert chunk.metadata["model"] == "gpt-4o-mini"
        assert isinstance(chunk.timestamp, float)

    def test_stream_handler_initialization(self):
        """Test StreamHandler initialization."""
        handler = StreamHandler("gpt-4o-mini", "openai", "stream-123")

        assert handler.response.model_used == "gpt-4o-mini"
        assert handler.response.provider == "openai"
        assert handler.response.stream_id == "stream-123"
        assert handler.accumulated_content == ""
        assert handler.chunk_index == 0

    def test_stream_handler_process_content_chunk(self):
        """Test processing content chunks."""
        handler = StreamHandler("gpt-4o-mini", "openai", "stream-123")

        # Process first chunk
        chunk1 = handler.process_chunk(
            StreamingChunkType.CONTENT, delta="Hello", metadata={"model": "gpt-4o-mini"}
        )

        assert chunk1.chunk_type == StreamingChunkType.CONTENT
        assert chunk1.delta == "Hello"
        assert chunk1.total_content == "Hello"
        assert chunk1.chunk_index == 0
        assert handler.accumulated_content == "Hello"

        # Process second chunk
        chunk2 = handler.process_chunk(
            StreamingChunkType.CONTENT, delta=" world", metadata={"model": "gpt-4o-mini"}
        )

        assert chunk2.delta == " world"
        assert chunk2.total_content == "Hello world"
        assert chunk2.chunk_index == 1
        assert handler.accumulated_content == "Hello world"

    def test_stream_handler_finalize(self):
        """Test stream handler finalization."""
        handler = StreamHandler("gpt-4o-mini", "openai", "stream-123")

        # Add some content
        handler.process_chunk(StreamingChunkType.CONTENT, delta="Test content")

        # Finalize
        final_response = handler.finalize(total_tokens=25, cost_estimate=0.000015)

        assert final_response.final_content == "Test content"
        assert final_response.total_tokens == 25
        assert final_response.cost_estimate == 0.000015
        assert final_response.completed is True
        assert final_response.error_message is None
        assert isinstance(final_response.end_time, float)

    def test_stream_handler_finalize_with_error(self):
        """Test stream handler finalization with error."""
        handler = StreamHandler("gpt-4o-mini", "openai", "stream-123")

        # Finalize with error
        final_response = handler.finalize(error_message="API Error")

        assert final_response.completed is False
        assert final_response.error_message == "API Error"


class TestOpenAIStreaming:
    """Test OpenAI streaming functionality."""

    @pytest.mark.asyncio
    async def test_generate_stream_success(self, openai_service, streaming_request):
        """Test successful streaming generation."""
        # Mock the OpenAI client streaming response
        mock_chunks = ["AI ", "thinks ", "and ", "learns ", "fast."]
        mock_stream = MockOpenAIStreamResponse(mock_chunks)

        async def mock_create(**kwargs):
            return mock_stream

        with patch.object(
            openai_service.client.chat.completions, "create", side_effect=mock_create
        ):
            chunks = []
            async for chunk in openai_service.generate_stream(streaming_request):
                chunks.append(chunk)

            # Should have content chunks + done chunk
            assert len(chunks) >= len(mock_chunks)

            # Check content chunks
            content_chunks = [c for c in chunks if c.chunk_type == StreamingChunkType.CONTENT]
            assert len(content_chunks) == len(mock_chunks)

            # Check done chunk
            done_chunks = [c for c in chunks if c.chunk_type == StreamingChunkType.DONE]
            assert len(done_chunks) == 1

            # Verify accumulated content
            final_content = "".join(c.delta for c in content_chunks)
            assert final_content == "".join(mock_chunks)

    @pytest.mark.asyncio
    async def test_generate_stream_api_error(self, openai_service, streaming_request):
        """Test streaming with API error."""
        from httpx import Request, Response
        from openai import APIError

        # Mock API error with proper arguments
        mock_request = Mock(spec=Request)
        mock_response = Mock(spec=Response)
        mock_response.status_code = 429
        mock_error = APIError("Rate limit exceeded", request=mock_request, body=None)

        async def mock_create_error(**kwargs):
            raise mock_error

        with patch.object(
            openai_service.client.chat.completions, "create", side_effect=mock_create_error
        ):
            chunks = []
            async for chunk in openai_service.generate_stream(streaming_request):
                chunks.append(chunk)

            # Should have error chunk
            error_chunks = [c for c in chunks if c.chunk_type == StreamingChunkType.ERROR]
            assert len(error_chunks) == 1
            assert "Rate limit exceeded" in error_chunks[0].metadata["error"]

    @pytest.mark.asyncio
    async def test_generate_stream_empty_response(self, openai_service, streaming_request):
        """Test streaming with empty response."""
        # Mock empty stream
        mock_stream = MockOpenAIStreamResponse([])

        async def mock_create(**kwargs):
            return mock_stream

        with patch.object(
            openai_service.client.chat.completions, "create", side_effect=mock_create
        ):
            chunks = []
            async for chunk in openai_service.generate_stream(streaming_request):
                chunks.append(chunk)

            # Should have at least done chunk
            done_chunks = [c for c in chunks if c.chunk_type == StreamingChunkType.DONE]
            assert len(done_chunks) == 1

    @pytest.mark.asyncio
    async def test_generate_routing_to_stream(self, openai_service):
        """Test that generate() routes to streaming when stream=True."""
        request = LLMRequest(model="gpt-4o-mini", prompt="Test", stream=True)

        # Mock generate_stream method
        mock_stream_response = AsyncMock()
        with patch.object(openai_service, "generate_stream", return_value=mock_stream_response):
            result = await openai_service.generate(request)

            # Should return the streaming generator
            assert result == mock_stream_response
            openai_service.generate_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_routing_to_non_stream(self, openai_service):
        """Test that generate() routes to non-streaming when stream=False."""
        request = LLMRequest(model="gpt-4o-mini", prompt="Test", stream=False)

        # Mock generate_non_stream method
        mock_response = Mock()
        with patch.object(openai_service, "generate_non_stream", return_value=mock_response):
            result = await openai_service.generate(request)

            # Should return the non-streaming response
            assert result == mock_response
            openai_service.generate_non_stream.assert_called_once()

    def test_streaming_chunk_types(self):
        """Test all streaming chunk types are available."""
        assert StreamingChunkType.CONTENT == "content"
        assert StreamingChunkType.FUNCTION_CALL == "function_call"
        assert StreamingChunkType.METADATA == "metadata"
        assert StreamingChunkType.ERROR == "error"
        assert StreamingChunkType.DONE == "done"

    @pytest.mark.asyncio
    async def test_stream_timing_tracking(self, openai_service, streaming_request):
        """Test that streaming tracks timing correctly."""
        mock_chunks = ["Fast ", "response"]
        mock_stream = MockOpenAIStreamResponse(mock_chunks)

        with patch.object(
            openai_service.client.chat.completions, "create", return_value=mock_stream
        ):
            start_time = time.time()

            chunks = []
            async for chunk in openai_service.generate_stream(streaming_request):
                chunks.append(chunk)

            end_time = time.time()

            # Check timing is reasonable
            content_chunks = [c for c in chunks if c.chunk_type == StreamingChunkType.CONTENT]
            if content_chunks:
                first_chunk_time = content_chunks[0].timestamp
                assert start_time <= first_chunk_time <= end_time

            done_chunks = [c for c in chunks if c.chunk_type == StreamingChunkType.DONE]
            if done_chunks:
                final_response = done_chunks[0].metadata.get("final_response", {})
                latency = final_response.get("total_latency_ms", 0)
                assert latency >= 0  # Should have some latency


class TestStreamingIntegration:
    """Integration tests for streaming functionality."""

    @pytest.mark.asyncio
    async def test_streaming_with_overrides(self, openai_service):
        """Test streaming with runtime parameter overrides."""
        base_request = LLMRequest(
            model="gpt-4o-mini", prompt="Test prompt", temperature=0.5, stream=True
        )

        mock_stream = MockOpenAIStreamResponse(["Test"])

        async def mock_create(**kwargs):
            # Store the call arguments for verification
            mock_create.call_kwargs = kwargs
            return mock_stream

        with patch.object(
            openai_service.client.chat.completions, "create", side_effect=mock_create
        ):
            chunks = []
            async for chunk in openai_service.generate_stream(
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
    async def test_streaming_parameter_mapping(self, openai_service, streaming_request):
        """Test that LLM parameters are correctly mapped to OpenAI format."""
        mock_stream = MockOpenAIStreamResponse(["Test"])

        with patch.object(
            openai_service.client.chat.completions, "create", return_value=mock_stream
        ) as mock_create:
            async for chunk in openai_service.generate_stream(streaming_request):
                break  # Just test the first chunk

            # Verify API call parameters
            call_args = mock_create.call_args[1]
            assert call_args["model"] == "gpt-4o-mini"
            assert call_args["temperature"] == 0.7
            assert call_args["max_tokens"] == 50
            assert call_args["stream"] is True
            assert "messages" in call_args
            assert len(call_args["messages"]) == 1
            assert call_args["messages"][0]["role"] == "user"
            assert call_args["messages"][0]["content"] == "Write a short haiku about AI"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
LLM Service Interface and Implementations.

Provides a clean abstraction for all LLM interactions with support for
multiple providers, streaming responses, retry logic, and cost tracking.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from src.models.llm_models import LLMProviderConfig, LLMRequest, LLMResponse
from src.models.streaming_models import StreamingGenerator


class LLMError(Exception):
    """Base exception for LLM service errors"""

    def __init__(self, message: str, provider: str = None, model: str = None):
        self.provider = provider
        self.model = model
        super().__init__(message)


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out"""

    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limit is exceeded"""

    pass


class LLMService(ABC):
    """Abstract base class for LLM service implementations with streaming support."""

    @abstractmethod
    async def generate(
        self, request: LLMRequest, **overrides
    ) -> Union[LLMResponse, StreamingGenerator]:
        """
        Generate completion using LLM API.

        Args:
            request: LLM request with parameters
            **overrides: Runtime parameter overrides

        Returns:
            StreamingGenerator if streaming=True, LLMResponse if streaming=False

        Raises:
            LLMError: For general LLM-related errors
            LLMTimeoutError: For timeout errors
            LLMRateLimitError: For rate limit errors
        """
        pass

    @abstractmethod
    async def generate_stream(self, request: LLMRequest, **overrides) -> StreamingGenerator:
        """
        Generate streaming completion using LLM API.

        Args:
            request: LLM request with parameters
            **overrides: Runtime parameter overrides

        Returns:
            AsyncIterator of StreamingChunk objects
        """
        pass

    @abstractmethod
    async def generate_non_stream(self, request: LLMRequest, **overrides) -> LLMResponse:
        """
        Generate non-streaming completion using LLM API.

        Args:
            request: LLM request with parameters
            **overrides: Runtime parameter overrides

        Returns:
            Complete LLM response
        """
        pass

    @abstractmethod
    async def batch_generate(
        self, requests: list[LLMRequest], **overrides
    ) -> list[LLMResponse]:
        """
        Generate multiple completions concurrently.

        Args:
            requests: List of LLM requests
            **overrides: Runtime parameter overrides applied to all requests

        Returns:
            List of LLM responses in the same order as requests
        """
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
        """
        Validate connection to LLM API.

        Returns:
            True if connection is successful
        """
        pass

    @abstractmethod
    def get_available_models(self) -> list[str]:
        """Get list of available models for this service."""
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: LLMProviderConfig) -> "LLMService":
        """
        Create service from configuration.

        Args:
            config: Provider configuration

        Returns:
            Configured LLM service
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """String representation of the service."""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """Detailed string representation."""
        pass


# Error classes for LLM service exceptions
class LLMError(Exception):
    """Base exception for LLM service errors"""

    def __init__(self, message: str, provider: str = None, model: str = None):
        self.provider = provider
        self.model = model
        super().__init__(message)


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out"""
    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limit is exceeded"""
    pass


# Simple mock service for backward compatibility with tests
class MockLLMService(LLMService):
    """
    Simple mock LLM service for testing.
    
    Provides minimal implementation for backward compatibility.
    """

    def __init__(self, response_delay: float = 0.1, failure_rate: float = 0.0):
        self.response_delay = response_delay
        self.failure_rate = failure_rate
        self._call_count = 0

    async def generate(self, request: LLMRequest, **overrides) -> Union[LLMResponse, StreamingGenerator]:
        """Generate mock response."""
        if request.stream:
            return self.generate_stream(request, **overrides)
        else:
            return await self.generate_non_stream(request, **overrides)

    async def generate_stream(self, request: LLMRequest, **overrides) -> StreamingGenerator:
        """Generate mock streaming response."""
        # Simple mock streaming that yields a single chunk
        from src.models.streaming_models import StreamHandler, StreamingChunkType
        
        handler = StreamHandler(request.model, "mock", "test-stream")
        
        # Simulate delay
        await asyncio.sleep(self.response_delay)
        
        # Yield content chunk
        chunk = handler.process_chunk(
            StreamingChunkType.CONTENT,
            delta="Mock response content",
            metadata={"model": request.model}
        )
        yield chunk
        
        # Yield done chunk
        final_response = handler.finalize(total_tokens=10, cost_estimate=0.001)
        done_chunk = handler.process_chunk(
            StreamingChunkType.DONE,
            metadata={"final_response": final_response.model_dump()}
        )
        yield done_chunk

    async def generate_non_stream(self, request: LLMRequest, **overrides) -> LLMResponse:
        """Generate mock non-streaming response."""
        await asyncio.sleep(self.response_delay)
        
        return LLMResponse(
            content="Mock response content",
            model_used=request.model,
            tokens_used=10,
            cost_estimate=0.001,
            latency_ms=int(self.response_delay * 1000),
            provider="mock",
            metadata={"test": True}
        )

    async def batch_generate(self, requests: list[LLMRequest], **overrides) -> list[LLMResponse]:
        """Generate mock batch responses."""
        tasks = [self.generate_non_stream(req, **overrides) for req in requests]
        return await asyncio.gather(*tasks)

    async def validate_connection(self) -> bool:
        """Mock connection validation."""
        return True

    def get_available_models(self) -> list[str]:
        """Return mock available models."""
        return ["mock-model"]

    @classmethod
    def from_config(cls, config: LLMProviderConfig) -> "MockLLMService":
        """Create mock service from config."""
        return cls()

    def __str__(self) -> str:
        return "MockLLMService"

    def __repr__(self) -> str:
        return "MockLLMService()"


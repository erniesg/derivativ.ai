"""
LLM Service Interface and Implementations.

Provides a clean abstraction for all LLM interactions with support for
multiple providers, streaming responses, retry logic, and cost tracking.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Union

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
    async def batch_generate(self, requests: list[LLMRequest], **overrides) -> list[LLMResponse]:
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


# Removed duplicate LLMTimeoutError and LLMRateLimitError definitions
# They are already defined at the top of the file


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

    async def generate(
        self, request: LLMRequest, **overrides
    ) -> Union[LLMResponse, StreamingGenerator]:
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
            metadata={"model": request.model},
        )
        yield chunk

        # Yield done chunk
        final_response = handler.finalize(total_tokens=10, cost_estimate=0.001)
        done_chunk = handler.process_chunk(
            StreamingChunkType.DONE, metadata={"final_response": final_response.model_dump()}
        )
        yield done_chunk

    async def generate_non_stream(self, request: LLMRequest, **overrides) -> LLMResponse:
        """Generate mock non-streaming response with proper JSON for agents."""
        await asyncio.sleep(self.response_delay)

        # Check if this is for a specific agent and return appropriate JSON
        content = "Mock response content"

        # QuestionGenerator expects JSON with question structure
        if "question" in request.prompt.lower() and "generate" in request.prompt.lower():
            # Extract marks from prompt if available
            marks = 3  # default
            if "marks" in request.prompt:
                import re

                marks_match = re.search(r"marks[:\s]+(\d+)", request.prompt.lower())
                if marks_match:
                    marks = int(marks_match.group(1))

            content = f"""{{
                "question_text": "Calculate the value of 3x + 2 when x = 5",
                "marks": {marks},
                "command_word": "Calculate",
                "solution_steps": ["Substitute x = 5 into 3x + 2", "Calculate 3(5) + 2 = 15 + 2", "Final answer is 17"],
                "final_answer": "17"
            }}"""
        # MarkerAgent expects marking scheme JSON
        elif "marking" in request.prompt.lower() or "mark" in request.prompt.lower():
            # Extract marks from prompt if available
            marks = 3  # default
            if "marks" in request.prompt:
                import re

                marks_match = re.search(r"marks[:\s]+(\d+)", request.prompt.lower())
                if marks_match:
                    marks = int(marks_match.group(1))

            # Create appropriate mark allocation based on total marks
            if marks == 1:
                content = f"""{{
                "total_marks": {marks},
                "mark_allocation_criteria": [
                    {{"criterion_text": "Correct answer", "marks_value": 1, "mark_type": "A"}}
                ],
                "final_answers": [{{"answer_text": "17", "value_numeric": 17}}]
            }}"""
            elif marks == 2:
                content = f"""{{
                "total_marks": {marks},
                "mark_allocation_criteria": [
                    {{"criterion_text": "Method shown", "marks_value": 1, "mark_type": "M"}},
                    {{"criterion_text": "Correct answer", "marks_value": 1, "mark_type": "A"}}
                ],
                "final_answers": [{{"answer_text": "17", "value_numeric": 17}}]
            }}"""
            else:
                # For 3+ marks, distribute appropriately
                working_marks = marks - 2
                content = f"""{{
                "total_marks": {marks},
                "mark_allocation_criteria": [
                    {{"criterion_text": "Correct method", "marks_value": 1, "mark_type": "M"}},
                    {{"criterion_text": "Correct working", "marks_value": {working_marks}, "mark_type": "M"}},
                    {{"criterion_text": "Final answer", "marks_value": 1, "mark_type": "A"}}
                ],
                "final_answers": [{{"answer_text": "17", "value_numeric": 17}}]
            }}"""
        # ReviewAgent expects quality assessment JSON
        elif "review" in request.prompt.lower() or "quality" in request.prompt.lower():
            content = """{
                "overall_quality_score": 0.85,
                "mathematical_accuracy": 0.9,
                "cambridge_compliance": 0.8,
                "grade_appropriateness": 0.85,
                "question_clarity": 0.85,
                "marking_accuracy": 0.85,
                "feedback_summary": "Good quality question with clear structure",
                "specific_issues": [],
                "suggested_improvements": [],
                "decision": "approve"
            }"""

        return LLMResponse(
            content=content,
            model_used=request.model,
            tokens_used=len(content.split()),
            cost_estimate=0.001,
            latency_ms=int(self.response_delay * 1000),
            provider="mock",
            metadata={"test": True},
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

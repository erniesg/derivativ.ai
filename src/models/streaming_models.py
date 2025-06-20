"""
Streaming models for real-time LLM responses.
Provides async iteration over streaming content chunks.
"""

import time
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class StreamingChunkType(str, Enum):
    """Types of streaming chunks."""

    CONTENT = "content"  # Text content chunk
    FUNCTION_CALL = "function_call"  # Function call chunk
    METADATA = "metadata"  # Metadata/usage info
    ERROR = "error"  # Error chunk
    DONE = "done"  # Stream completion


class StreamingChunk(BaseModel):
    """Individual chunk in a streaming response."""

    chunk_type: StreamingChunkType = Field(..., description="Type of chunk")
    content: str = Field("", description="Text content (for content chunks)")
    delta: str = Field("", description="New text since last chunk")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional chunk metadata")
    timestamp: float = Field(default_factory=time.time, description="Chunk timestamp")

    # Cumulative tracking
    total_content: str = Field("", description="All content accumulated so far")
    chunk_index: int = Field(0, description="Index of this chunk in stream")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk_type": "content",
                "content": "The answer to",
                "delta": " to",
                "total_content": "The answer to",
                "chunk_index": 3,
                "metadata": {"model": "gpt-4o-mini"},
                "timestamp": 1640995200.0,
            }
        }
    )


class StreamingResponse(BaseModel):
    """Complete streaming response with metadata."""

    model_used: str = Field(..., description="Model that generated the response")
    provider: str = Field(..., description="Provider that generated the response")
    stream_id: str = Field(..., description="Unique identifier for this stream")

    # Will be populated when stream completes
    final_content: str = Field("", description="Complete response content")
    total_tokens: int = Field(0, description="Total tokens used")
    cost_estimate: float = Field(0.0, description="Estimated cost")
    total_latency_ms: int = Field(0, description="Total response time")

    # Streaming metadata
    start_time: float = Field(default_factory=time.time, description="Stream start time")
    first_chunk_time: Optional[float] = Field(None, description="Time of first content chunk")
    end_time: Optional[float] = Field(None, description="Stream completion time")
    chunk_count: int = Field(0, description="Total number of chunks")

    # Error handling
    error_message: Optional[str] = Field(None, description="Error message if stream failed")
    completed: bool = Field(False, description="Whether stream completed successfully")

    def get_time_to_first_chunk_ms(self) -> Optional[int]:
        """Get latency to first chunk in milliseconds."""
        if self.first_chunk_time:
            return int((self.first_chunk_time - self.start_time) * 1000)
        return None

    def get_total_latency_ms(self) -> int:
        """Get total latency in milliseconds."""
        if self.end_time:
            return int((self.end_time - self.start_time) * 1000)
        return int((time.time() - self.start_time) * 1000)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_used": "gpt-4o-mini",
                "provider": "openai",
                "stream_id": "stream_123",
                "final_content": "The answer is 42.",
                "total_tokens": 25,
                "cost_estimate": 0.000015,
                "chunk_count": 8,
                "completed": True,
            }
        }
    )


class StreamHandler:
    """
    Handler for processing streaming responses.
    Accumulates chunks and tracks streaming metadata.
    """

    def __init__(self, model: str, provider: str, stream_id: str):
        self.response = StreamingResponse(model_used=model, provider=provider, stream_id=stream_id)
        self.accumulated_content = ""
        self.chunk_index = 0

    def process_chunk(
        self,
        chunk_type: StreamingChunkType,
        content: str = "",
        delta: str = "",
        metadata: dict[str, Any] = None,
    ) -> StreamingChunk:
        """Process a new chunk and return the structured chunk."""

        # Update accumulated content
        if chunk_type == StreamingChunkType.CONTENT:
            if delta:
                self.accumulated_content += delta
            elif content and not self.accumulated_content:
                self.accumulated_content = content

        # Track first chunk time
        if chunk_type == StreamingChunkType.CONTENT and not self.response.first_chunk_time:
            self.response.first_chunk_time = time.time()

        # Create chunk
        chunk = StreamingChunk(
            chunk_type=chunk_type,
            content=content,
            delta=delta,
            total_content=self.accumulated_content,
            chunk_index=self.chunk_index,
            metadata=metadata or {},
        )

        self.chunk_index += 1
        self.response.chunk_count += 1

        return chunk

    def finalize(
        self, total_tokens: int = 0, cost_estimate: float = 0.0, error_message: Optional[str] = None
    ) -> StreamingResponse:
        """Finalize the streaming response."""
        self.response.final_content = self.accumulated_content
        self.response.total_tokens = total_tokens
        self.response.cost_estimate = cost_estimate
        self.response.end_time = time.time()
        self.response.total_latency_ms = self.response.get_total_latency_ms()
        self.response.error_message = error_message
        self.response.completed = error_message is None

        return self.response


# Type alias for streaming generators
StreamingGenerator = AsyncIterator[StreamingChunk]

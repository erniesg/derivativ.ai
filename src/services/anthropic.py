"""
Anthropic LLM service implementation.
Provides integration with Anthropic's Claude API using the latest models.
"""

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Optional, Union

import anthropic
from anthropic import AsyncAnthropic

from src.models.llm_models import LLMProviderConfig, LLMRequest, LLMResponse
from src.models.streaming_models import (
    StreamHandler,
    StreamingChunkType,
    StreamingGenerator,
)
from src.services.llm_service import LLMService

logger = logging.getLogger(__name__)

# Anthropic model pricing (per 1K tokens) - updated for latest models
ANTHROPIC_PRICING = {
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},  # Cheapest
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
}


def calculate_anthropic_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for Anthropic API usage."""
    pricing = ANTHROPIC_PRICING.get(
        model, ANTHROPIC_PRICING["claude-3-5-haiku-20241022"]
    )  # Default to haiku

    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]

    return input_cost + output_cost


class AnthropicLLMService(LLMService):
    """
    Anthropic LLM service implementation.

    Supports all Anthropic Claude models with proper parameter mapping,
    error handling, and cost tracking.
    """

    def __init__(self, api_key: str, config: Optional[dict[str, Any]] = None):
        """
        Initialize Anthropic service.

        Args:
            api_key: Anthropic API key
            config: Service configuration
        """
        if not api_key or not api_key.strip():
            raise ValueError("Anthropic API key is required")

        self.config = config or {}
        self._validate_config()

        # Initialize Anthropic client
        self.client = AsyncAnthropic(
            api_key=api_key,
            base_url=self.config.get("base_url"),
            timeout=self.config.get("timeout_seconds", 30),
            max_retries=self.config.get("max_retries", 3),
        )

        logger.info(
            f"Initialized Anthropic service with model: {self.config.get('default_model', 'claude-3-5-haiku-20241022')}"
        )

    def _validate_config(self):
        """Validate service configuration."""
        timeout = self.config.get("timeout_seconds", 30)
        if timeout <= 0:
            raise ValueError("timeout_seconds must be positive")

        max_retries = self.config.get("max_retries", 3)
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

    async def generate(
        self, request: LLMRequest, **overrides
    ) -> Union[LLMResponse, StreamingGenerator]:
        """
        Generate completion using Anthropic API.

        Args:
            request: LLM request with parameters
            **overrides: Runtime parameter overrides

        Returns:
            StreamingGenerator if streaming=True, LLMResponse if streaming=False
        """
        # Apply runtime overrides
        if overrides:
            request_dict = request.model_dump()
            request_dict.update(overrides)
            request = LLMRequest(**request_dict)

        # Route to streaming or non-streaming
        if request.stream:
            return self.generate_stream(request, **overrides)
        else:
            return await self.generate_non_stream(request, **overrides)

    async def generate_stream(self, request: LLMRequest, **overrides) -> StreamingGenerator:
        """
        Generate streaming completion using Anthropic API.

        Args:
            request: LLM request with parameters
            **overrides: Runtime parameter overrides

        Returns:
            AsyncIterator of StreamingChunk objects
        """
        # Apply runtime overrides
        if overrides:
            request_dict = request.model_dump()
            request_dict.update(overrides)
            request = LLMRequest(**request_dict)

        stream_id = str(uuid.uuid4())
        handler = StreamHandler(request.model, "anthropic", stream_id)

        try:
            # Map parameters to Anthropic format
            anthropic_params = self._map_params(request)
            anthropic_params["stream"] = True  # Force streaming

            logger.debug(f"Starting Anthropic streaming with model: {anthropic_params['model']}")

            # Make streaming API call
            stream = await self.client.messages.create(**anthropic_params)

            total_input_tokens = 0
            total_output_tokens = 0

            async for chunk in stream:
                if chunk.type == "message_start":
                    # Track input tokens from message start
                    if hasattr(chunk, "message") and hasattr(chunk.message, "usage"):
                        total_input_tokens = chunk.message.usage.input_tokens

                elif chunk.type == "content_block_delta":
                    # Process content delta
                    if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                        streaming_chunk = handler.process_chunk(
                            StreamingChunkType.CONTENT,
                            delta=chunk.delta.text,
                            metadata={
                                "model": request.model,
                                "block_index": getattr(chunk, "index", 0),
                            },
                        )
                        yield streaming_chunk

                elif chunk.type == "message_delta" and hasattr(chunk, "usage"):
                    # Track final usage
                    total_output_tokens = chunk.usage.output_tokens

            # Calculate final cost
            cost_estimate = calculate_anthropic_cost(
                request.model, total_input_tokens, total_output_tokens
            )

            # Emit completion chunk
            final_response = handler.finalize(
                total_tokens=total_input_tokens + total_output_tokens, cost_estimate=cost_estimate
            )

            completion_chunk = handler.process_chunk(
                StreamingChunkType.DONE,
                metadata={
                    "final_response": final_response.model_dump(),
                    "total_tokens": total_input_tokens + total_output_tokens,
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "cost_estimate": cost_estimate,
                },
            )
            yield completion_chunk

        except anthropic.APIError as e:
            logger.error(f"Anthropic streaming API error: {e}")
            error_chunk = handler.process_chunk(
                StreamingChunkType.ERROR, metadata={"error": str(e)}
            )
            handler.finalize(error_message=str(e))
            yield error_chunk

        except Exception as e:
            logger.error(f"Unexpected error in Anthropic streaming: {e}")
            error_chunk = handler.process_chunk(
                StreamingChunkType.ERROR, metadata={"error": str(e)}
            )
            handler.finalize(error_message=str(e))
            yield error_chunk

    async def generate_non_stream(self, request: LLMRequest, **overrides) -> LLMResponse:
        """
        Generate non-streaming completion using Anthropic API.

        Args:
            request: LLM request with parameters
            **overrides: Runtime parameter overrides

        Returns:
            Complete LLM response
        """
        start_time = time.time()

        try:
            # Map parameters to Anthropic format
            anthropic_params = self._map_params(request)
            anthropic_params["stream"] = False  # Force non-streaming

            logger.debug(
                f"Calling Anthropic non-streaming API with model: {anthropic_params['model']}"
            )

            # Make non-streaming API call
            response = await self.client.messages.create(**anthropic_params)

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)

            # Normalize response
            return self._normalize_response(response, request, latency_ms)

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise Exception(f"Anthropic API error: {e}")

        except asyncio.TimeoutError:
            logger.error("Anthropic API request timeout")
            raise Exception("Anthropic API request timeout")

        except Exception as e:
            logger.error(f"Unexpected error in Anthropic service: {e}")
            raise Exception(f"Anthropic service error: {e}")

    def _map_params(self, request: LLMRequest) -> dict[str, Any]:
        """Map LLM request parameters to Anthropic API format."""
        # Build messages array
        messages = []

        # Add user message
        messages.append({"role": "user", "content": request.prompt})

        # Base parameters
        params = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
        }

        # Add system message if provided
        if request.system_message:
            params["system"] = request.system_message

        # Add stop sequences if provided
        if request.stop_sequences:
            params["stop_sequences"] = request.stop_sequences

        # Add Anthropic-specific extra parameters
        anthropic_extras = {
            "top_k": None,
            "metadata": None,
            "tools": None,
            "tool_choice": None,
        }

        for key, default in anthropic_extras.items():
            if key in request.extra_params:
                params[key] = request.extra_params[key]

        return params

    def _normalize_response(
        self, response: Any, request: LLMRequest, latency_ms: int
    ) -> LLMResponse:
        """Normalize Anthropic response to common LLM response format."""
        # Extract content
        content = ""
        if response.content:
            # Anthropic returns a list of content blocks
            content = "".join(block.text for block in response.content if hasattr(block, "text"))

        # Extract token usage
        usage = response.usage
        total_tokens = usage.input_tokens + usage.output_tokens
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens

        # Calculate cost
        cost_estimate = calculate_anthropic_cost(response.model, input_tokens, output_tokens)

        # Build metadata
        metadata = {
            "stop_reason": response.stop_reason,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model_version": response.model,
        }

        return LLMResponse(
            content=content,
            model_used=response.model,
            tokens_used=total_tokens,
            cost_estimate=cost_estimate,
            latency_ms=latency_ms,
            provider="anthropic",
            metadata=metadata,
        )

    def get_available_models(self) -> list[str]:
        """Get list of available Anthropic models."""
        return list(ANTHROPIC_PRICING.keys())

    @classmethod
    def from_config(cls, config: LLMProviderConfig) -> "AnthropicLLMService":
        """
        Create Anthropic service from configuration.

        Args:
            config: Provider configuration

        Returns:
            Configured Anthropic service
        """
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {config.api_key_env}")

        service_config = {
            "default_model": config.default_model,
            "base_url": config.base_url,
            "timeout_seconds": config.timeout_seconds,
            "max_retries": config.max_retries,
        }

        return cls(api_key=api_key, config=service_config)

    async def batch_generate(self, requests: list[LLMRequest], **overrides) -> list[LLMResponse]:
        """
        Generate multiple completions concurrently.

        Args:
            requests: List of LLM requests
            **overrides: Runtime parameter overrides applied to all requests

        Returns:
            List of LLM responses in the same order as requests
        """
        # Execute all requests concurrently
        tasks = [self.generate(request, **overrides) for request in requests]

        return await asyncio.gather(*tasks)

    async def validate_connection(self) -> bool:
        """
        Validate connection to Anthropic API.

        Returns:
            True if connection is successful
        """
        try:
            test_request = LLMRequest(
                model=self.config.get("default_model", "claude-3-5-haiku-20241022"),
                prompt="Hello",
                max_tokens=5,
                temperature=0.1,
                stream=False,
            )

            response = await self.generate(test_request)
            return response.content is not None and len(response.content.strip()) > 0

        except Exception as e:
            logger.error(f"Anthropic connection validation failed: {e}")
            return False

    def __str__(self) -> str:
        """String representation of the service."""
        model = self.config.get("default_model", "claude-3-5-haiku-20241022")
        return f"AnthropicLLMService(model={model})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"AnthropicLLMService("
            f"model={self.config.get('default_model')}, "
            f"base_url={self.config.get('base_url')}, "
            f"timeout={self.config.get('timeout_seconds')}s"
            f")"
        )

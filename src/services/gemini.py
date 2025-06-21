"""
Google Gemini LLM service implementation.
Provides integration with Google's Gemini API using the latest models.
"""

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Optional, Union

import google.generativeai as genai
from google.generativeai.types import AsyncGenerateContentResponse

from src.models.llm_models import LLMProviderConfig, LLMRequest, LLMResponse
from src.models.streaming_models import (
    StreamHandler,
    StreamingChunkType,
    StreamingGenerator,
)
from src.services.llm_service import LLMService

logger = logging.getLogger(__name__)

# Google Gemini model pricing (per 1K tokens) - updated for latest models
GEMINI_PRICING = {
    "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},  # Free tier
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-8b": {"input": 0.0375, "output": 0.15},
    "gemini-1.5-pro": {"input": 3.5, "output": 10.5},
    "gemini-1.0-pro": {"input": 0.5, "output": 1.5},
}


def calculate_gemini_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for Google Gemini API usage."""
    pricing = GEMINI_PRICING.get(model, GEMINI_PRICING["gemini-1.5-flash"])  # Default to flash

    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]

    return input_cost + output_cost


class GeminiLLMService(LLMService):
    """
    Google Gemini LLM service implementation.

    Supports all Gemini models with proper parameter mapping,
    error handling, and cost tracking.
    """

    def __init__(self, api_key: str, config: Optional[dict[str, Any]] = None):
        """
        Initialize Gemini service.

        Args:
            api_key: Google API key
            config: Service configuration
        """
        if not api_key or not api_key.strip():
            raise ValueError("Google API key is required")

        self.config = config or {}
        self._validate_config()

        # Configure Gemini API
        genai.configure(api_key=api_key)

        # Initialize the model
        model_name = self.config.get("default_model", "gemini-2.0-flash-exp")

        # Create generation config
        generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=1000,
            top_p=0.9,
        )

        # Create safety settings (minimal restrictions for educational content)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        logger.info(f"Initialized Gemini service with model: {model_name}")

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
        Generate completion using Gemini API.

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
        Generate streaming completion using Gemini API.

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
        handler = StreamHandler(request.model, "google", stream_id)

        try:
            # Prepare the prompt
            prompt = self._build_prompt(request)

            # Create generation config with request parameters
            generation_config = genai.types.GenerationConfig(
                temperature=request.temperature,
                max_output_tokens=request.max_tokens,
                top_p=request.top_p,
                stop_sequences=request.stop_sequences if request.stop_sequences else None,
            )

            logger.debug(f"Starting Gemini streaming with model: {request.model}")

            # Make streaming API call
            response = await self.model.generate_content_async(
                prompt, generation_config=generation_config, stream=True
            )

            total_tokens = 0
            input_tokens = 0
            output_tokens = 0

            async for chunk in response:
                if chunk.text:
                    # Process content chunk
                    streaming_chunk = handler.process_chunk(
                        StreamingChunkType.CONTENT,
                        delta=chunk.text,
                        metadata={
                            "model": request.model,
                            "safety_ratings": getattr(chunk, "safety_ratings", []),
                        },
                    )
                    yield streaming_chunk

                # Track usage if available
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    input_tokens = chunk.usage_metadata.prompt_token_count or 0
                    output_tokens = chunk.usage_metadata.candidates_token_count or 0
                    total_tokens = input_tokens + output_tokens

            # Calculate final cost
            cost_estimate = calculate_gemini_cost(request.model, input_tokens, output_tokens)

            # Emit completion chunk
            final_response = handler.finalize(
                total_tokens=total_tokens, cost_estimate=cost_estimate
            )

            completion_chunk = handler.process_chunk(
                StreamingChunkType.DONE,
                metadata={
                    "final_response": final_response.model_dump(),
                    "total_tokens": total_tokens,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_estimate": cost_estimate,
                },
            )
            yield completion_chunk

        except Exception as e:
            logger.error(f"Gemini streaming API error: {e}")
            error_chunk = handler.process_chunk(
                StreamingChunkType.ERROR, metadata={"error": str(e)}
            )
            handler.finalize(error_message=str(e))
            yield error_chunk

    async def generate_non_stream(self, request: LLMRequest, **overrides) -> LLMResponse:
        """
        Generate non-streaming completion using Gemini API.

        Args:
            request: LLM request with parameters
            **overrides: Runtime parameter overrides

        Returns:
            Complete LLM response
        """
        start_time = time.time()

        try:
            # Prepare the prompt
            prompt = self._build_prompt(request)

            # Create generation config with request parameters
            generation_config = genai.types.GenerationConfig(
                temperature=request.temperature,
                max_output_tokens=request.max_tokens,
                top_p=request.top_p,
                stop_sequences=request.stop_sequences if request.stop_sequences else None,
            )

            logger.debug(f"Calling Gemini non-streaming API with model: {request.model}")

            # Make non-streaming API call
            response = await self.model.generate_content_async(
                prompt, generation_config=generation_config, stream=False
            )

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)

            # Normalize response
            return self._normalize_response(response, request, latency_ms)

        except Exception as e:
            logger.error(f"Unexpected error in Gemini service: {e}")
            raise Exception(f"Gemini service error: {e}")

    def _build_prompt(self, request: LLMRequest) -> str:
        """Build the prompt for Gemini API."""
        if request.system_message:
            return f"{request.system_message}\n\n{request.prompt}"
        return request.prompt

    def _normalize_response(
        self, response: AsyncGenerateContentResponse, request: LLMRequest, latency_ms: int
    ) -> LLMResponse:
        """Normalize Gemini response to common LLM response format."""
        # Extract content
        content = response.text or ""

        # Extract token usage if available
        usage = getattr(response, "usage_metadata", None)
        if usage:
            input_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0
            total_tokens = input_tokens + output_tokens
        else:
            # Rough estimate if usage not available
            total_tokens = len(content.split()) * 1.3  # Rough token estimate
            input_tokens = len(request.prompt.split()) * 1.3
            output_tokens = total_tokens - input_tokens

        # Calculate cost
        cost_estimate = calculate_gemini_cost(request.model, int(input_tokens), int(output_tokens))

        # Build metadata
        metadata = {
            "finish_reason": getattr(response, "finish_reason", "unknown"),
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "model_version": request.model,
            "safety_ratings": getattr(response, "safety_ratings", []),
        }

        return LLMResponse(
            content=content,
            model_used=request.model,
            tokens_used=int(total_tokens),
            cost_estimate=cost_estimate,
            latency_ms=latency_ms,
            provider="google",
            metadata=metadata,
        )

    def get_available_models(self) -> list[str]:
        """Get list of available Gemini models."""
        return list(GEMINI_PRICING.keys())

    @classmethod
    def from_config(cls, config: LLMProviderConfig) -> "GeminiLLMService":
        """
        Create Gemini service from configuration.

        Args:
            config: Provider configuration

        Returns:
            Configured Gemini service
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
        Validate connection to Gemini API.

        Returns:
            True if connection is successful
        """
        try:
            test_request = LLMRequest(
                model=self.config.get("default_model", "gemini-2.0-flash-exp"),
                prompt="Hello",
                max_tokens=5,
                temperature=0.1,
                stream=False,
            )

            response = await self.generate(test_request)
            return response.content is not None and len(response.content.strip()) > 0

        except Exception as e:
            logger.error(f"Gemini connection validation failed: {e}")
            return False

    def __str__(self) -> str:
        """String representation of the service."""
        model = self.config.get("default_model", "gemini-2.0-flash-exp")
        return f"GeminiLLMService(model={model})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"GeminiLLMService("
            f"model={self.config.get('default_model')}, "
            f"base_url={self.config.get('base_url')}, "
            f"timeout={self.config.get('timeout_seconds')}s"
            f")"
        )

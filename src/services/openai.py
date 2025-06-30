"""
OpenAI LLM service implementation.
Provides integration with OpenAI's chat completion API using the latest models.
"""

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Optional, Union

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel

from src.models.llm_models import LLMProviderConfig, LLMRequest, LLMResponse
from src.models.streaming_models import (
    StreamHandler,
    StreamingChunkType,
    StreamingGenerator,
)
from src.services.llm_service import LLMService

logger = logging.getLogger(__name__)

# OpenAI model pricing (per 1K tokens) - updated for latest models
OPENAI_PRICING = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4.1-nano": {"input": 0.0001, "output": 0.0004},  # Estimated for new model
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}


def calculate_openai_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for OpenAI API usage."""
    pricing = OPENAI_PRICING.get(model, OPENAI_PRICING["gpt-4o-mini"])  # Default to mini

    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]

    return input_cost + output_cost


class OpenAILLMService(LLMService):
    """
    OpenAI LLM service implementation.

    Supports all OpenAI chat models with proper parameter mapping,
    error handling, and cost tracking.
    """

    def __init__(self, api_key: str, config: Optional[dict[str, Any]] = None):
        """
        Initialize OpenAI service.

        Args:
            api_key: OpenAI API key
            config: Service configuration
        """
        if not api_key or not api_key.strip():
            raise ValueError("OpenAI API key is required")

        self.config = config or {}
        self._validate_config()

        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.config.get("base_url"),
            timeout=self.config.get("timeout_seconds", 30),
            max_retries=self.config.get("max_retries", 3),
        )

        logger.info(
            f"Initialized OpenAI service with model: {self.config.get('default_model', 'gpt-4o-mini')}"
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
        Generate completion using OpenAI API.

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
        Generate streaming completion using OpenAI API.

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
        handler = StreamHandler(request.model, "openai", stream_id)

        try:
            # Map parameters to OpenAI format
            openai_params = self._map_params(request)
            openai_params["stream"] = True  # Force streaming

            logger.debug(f"Starting OpenAI streaming with model: {openai_params['model']}")

            # Make streaming API call
            stream = await self.client.chat.completions.create(**openai_params)

            total_tokens = 0

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta

                    # Process content delta
                    if hasattr(delta, "content") and delta.content:
                        streaming_chunk = handler.process_chunk(
                            StreamingChunkType.CONTENT,
                            delta=delta.content,
                            metadata={
                                "model": chunk.model,
                                "finish_reason": chunk.choices[0].finish_reason,
                            },
                        )
                        yield streaming_chunk

                # Track usage if available
                if hasattr(chunk, "usage") and chunk.usage:
                    total_tokens = chunk.usage.total_tokens

            # Calculate final cost
            cost_estimate = calculate_openai_cost(
                request.model,
                total_tokens // 2,  # Rough split
                total_tokens // 2,
            )

            # Emit completion chunk
            final_response = handler.finalize(
                total_tokens=total_tokens, cost_estimate=cost_estimate
            )

            completion_chunk = handler.process_chunk(
                StreamingChunkType.DONE,
                metadata={
                    "final_response": final_response.model_dump(),
                    "total_tokens": total_tokens,
                    "cost_estimate": cost_estimate,
                },
            )
            yield completion_chunk

        except openai.APIError as e:
            logger.error(f"OpenAI streaming API error: {e}")
            error_chunk = handler.process_chunk(
                StreamingChunkType.ERROR, metadata={"error": str(e)}
            )
            handler.finalize(error_message=str(e))
            yield error_chunk

        except Exception as e:
            logger.error(f"Unexpected error in OpenAI streaming: {e}")
            error_chunk = handler.process_chunk(
                StreamingChunkType.ERROR, metadata={"error": str(e)}
            )
            handler.finalize(error_message=str(e))
            yield error_chunk

    async def parse_structured(
        self, request: LLMRequest, response_format: type[BaseModel], **overrides
    ) -> BaseModel:
        """
        Generate structured output using OpenAI's parse method.

        Args:
            request: LLM request with parameters
            response_format: Pydantic model class for structured output
            **overrides: Runtime parameter overrides

        Returns:
            Parsed Pydantic model instance
        """
        try:
            # Map parameters to OpenAI format
            openai_params = self._map_params(request)
            openai_params["stream"] = False

            # Apply runtime overrides (but don't let them override response_format)
            if overrides:
                filtered_overrides = {
                    k: v
                    for k, v in overrides.items()
                    if k not in {"response_format", "text_format"}
                }
                openai_params.update(filtered_overrides)

            logger.info("OpenAI parse_structured called with:")
            logger.info(f"  - response_format: {response_format}")
            logger.info(f"  - response_format type: {type(response_format)}")
            logger.info(f"  - model: {openai_params['model']}")

            # Check for new 2025 responses.parse method
            if hasattr(self.client, "responses") and hasattr(self.client.responses, "parse"):
                return await self._parse_with_new_api(openai_params, response_format)

            # Check if old parse method is available
            elif hasattr(self.client.chat.completions, "parse"):
                return await self._parse_with_old_api(openai_params, response_format)
            else:
                return await self._parse_with_json_object(openai_params, response_format)

        except openai.APIError as e:
            logger.error(f"OpenAI structured parsing API error: {e}")
            raise Exception(f"OpenAI structured parsing error: {e.message}")

        except Exception as e:
            logger.error(f"Unexpected error in OpenAI structured parsing: {e}")
            raise Exception(f"OpenAI structured parsing error: {e}")

    async def _parse_with_new_api(
        self, openai_params: dict[str, Any], response_format: type[BaseModel]
    ) -> BaseModel:
        """Parse using new 2025 OpenAI responses.parse() API."""
        logger.info(
            f"Using new OpenAI 2025 responses.parse method with text_format={response_format.__name__}"
        )

        # Convert messages format for new API
        input_messages = openai_params.pop("messages")

        # Build parameters for new API
        new_params = {
            "model": openai_params["model"],
            "input": input_messages,
            "text_format": response_format,
        }

        # Add optional parameters if present
        for param in ["temperature", "max_tokens", "top_p"]:
            if param in openai_params:
                new_params[param] = openai_params[param]

        response = await self.client.responses.parse(**new_params)
        logger.info(f"Parse successful, got parsed object: {type(response.output_parsed)}")
        return response.output_parsed

    async def _parse_with_old_api(
        self, openai_params: dict[str, Any], response_format: type[BaseModel]
    ) -> BaseModel:
        """Parse using older OpenAI chat.completions.parse() API."""
        logger.info(
            f"Using older OpenAI parse method with response_format={response_format.__name__}"
        )
        completion = await self.client.chat.completions.parse(
            **openai_params, response_format=response_format
        )
        logger.info(
            f"Parse successful, got parsed object: {type(completion.choices[0].message.parsed)}"
        )
        return completion.choices[0].message.parsed

    async def _parse_with_json_object(
        self, openai_params: dict[str, Any], response_format: type[BaseModel]
    ) -> BaseModel:
        """Parse using json_object mode fallback."""
        logger.warning("OpenAI parse method not available, falling back to json_object mode")
        openai_params["response_format"] = {"type": "json_object"}

        completion = await self.client.chat.completions.create(**openai_params)
        raw_content = completion.choices[0].message.content
        logger.info(f"Raw OpenAI response (first 200 chars): {raw_content[:200]}...")

        # Parse JSON manually and validate against schema
        import json

        try:
            parsed_data = json.loads(raw_content)
            logger.info(f"Parsed JSON keys: {list(parsed_data.keys())}")

            # Handle wrapped responses (e.g., {'worksheet': {...}})
            if len(parsed_data) == 1 and isinstance(next(iter(parsed_data.values())), dict):
                # Unwrap the nested data
                wrapped_key = next(iter(parsed_data.keys()))
                logger.info(f"Unwrapping OpenAI response from '{wrapped_key}' key")
                parsed_data = parsed_data[wrapped_key]
                logger.info(f"Unwrapped data keys: {list(parsed_data.keys())}")

            # Validate against schema
            logger.info(
                f"Attempting to create {response_format.__name__} with data keys: {list(parsed_data.keys())}"
            )
            return response_format(**parsed_data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"OpenAI JSON validation failed: {e}")
            logger.error(
                f"Raw parsed data: {parsed_data if 'parsed_data' in locals() else 'Failed to parse JSON'}"
            )
            raise Exception(f"Failed to parse OpenAI JSON response: {e}")

    async def generate_non_stream(self, request: LLMRequest, **overrides) -> LLMResponse:
        """
        Generate non-streaming completion using OpenAI API.

        Args:
            request: LLM request with parameters
            **overrides: Runtime parameter overrides

        Returns:
            Complete LLM response
        """
        start_time = time.time()

        try:
            # Map parameters to OpenAI format
            openai_params = self._map_params(request)
            openai_params["stream"] = False  # Force non-streaming

            logger.debug(f"Calling OpenAI non-streaming API with model: {openai_params['model']}")

            # Make non-streaming API call
            response = await self.client.chat.completions.create(**openai_params)

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)

            # Normalize response
            return self._normalize_response(response, request, latency_ms)

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise Exception(f"OpenAI API error: {e.message}")

        except asyncio.TimeoutError:
            logger.error("OpenAI API request timeout")
            raise Exception("OpenAI API request timeout")

        except Exception as e:
            logger.error(f"Unexpected error in OpenAI service: {e}")
            raise Exception(f"OpenAI service error: {e}")

    def _map_params(self, request: LLMRequest) -> dict[str, Any]:
        """Map LLM request parameters to OpenAI API format."""
        # Build messages array
        messages = []

        # Add system message if provided
        if request.system_message:
            messages.append({"role": "system", "content": request.system_message})

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

        # Add stop sequences if provided
        if request.stop_sequences:
            params["stop"] = request.stop_sequences

        # Add streaming parameter
        if hasattr(request, "stream"):
            params["stream"] = request.stream

        # Add OpenAI-specific extra parameters
        openai_extras = {
            "frequency_penalty": None,
            "presence_penalty": None,
            "n": None,
            "stream": None,
            "logprobs": None,
            "top_logprobs": None,
            "response_format": None,
            "seed": None,
            "tools": None,
            "tool_choice": None,
            "user": None,
            "stop": None,  # Add stop parameter support
        }

        for key, default in openai_extras.items():
            if key in request.extra_params:
                params[key] = request.extra_params[key]

        return params

    def _normalize_response(
        self, response: Any, request: LLMRequest, latency_ms: int
    ) -> LLMResponse:
        """Normalize OpenAI response to common LLM response format."""
        # Extract content
        content = response.choices[0].message.content or ""

        # Extract token usage
        usage = response.usage
        total_tokens = usage.total_tokens
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

        # Calculate cost
        cost_estimate = calculate_openai_cost(response.model, input_tokens, output_tokens)

        # Build metadata
        metadata = {
            "finish_reason": response.choices[0].finish_reason,
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "model_version": response.model,
        }

        # Add usage details if available
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            metadata["prompt_tokens_details"] = usage.prompt_tokens_details

        if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            metadata["completion_tokens_details"] = usage.completion_tokens_details

        return LLMResponse(
            content=content,
            model_used=response.model,
            tokens_used=total_tokens,
            cost_estimate=cost_estimate,
            latency_ms=latency_ms,
            provider="openai",
            metadata=metadata,
        )

    def get_available_models(self) -> list[str]:
        """Get list of available OpenAI models."""
        return list(OPENAI_PRICING.keys())

    @classmethod
    def from_config(cls, config: LLMProviderConfig) -> "OpenAILLMService":
        """
        Create OpenAI service from configuration.

        Args:
            config: Provider configuration

        Returns:
            Configured OpenAI service
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
        Validate connection to OpenAI API.

        Returns:
            True if connection is successful
        """
        try:
            test_request = LLMRequest(
                model=self.config.get("default_model", "gpt-4o-mini"),
                prompt="Hello",
                max_tokens=5,
                temperature=0.1,
            )

            response = await self.generate(test_request)
            return response.content is not None and len(response.content.strip()) > 0

        except Exception as e:
            logger.error(f"OpenAI connection validation failed: {e}")
            return False

    def __str__(self) -> str:
        """String representation of the service."""
        model = self.config.get("default_model", "gpt-4o-mini")
        return f"OpenAILLMService(model={model})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"OpenAILLMService("
            f"model={self.config.get('default_model')}, "
            f"base_url={self.config.get('base_url')}, "
            f"timeout={self.config.get('timeout_seconds')}s"
            f")"
        )

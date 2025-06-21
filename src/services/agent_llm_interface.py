"""
Agent-compatible LLM interface.
Provides a simpler interface for agents while using our LLMService underneath.
"""

from typing import Optional, Union

from ..models.enums import LLMModel
from ..models.llm_models import LLMRequest, LLMResponse
from .llm_service import LLMService
from .mock_llm_service import MockLLMService


class AgentLLMInterface:
    """
    Simple LLM interface for agents.
    Wraps our LLMService to provide the interface agents expect.
    """

    def __init__(self, llm_service: LLMService):
        """
        Initialize with an LLM service.

        Args:
            llm_service: The underlying LLM service to use
        """
        self.llm_service = llm_service

    async def generate(
        self,
        prompt: str,
        model: Union[str, LLMModel] = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response using the LLM service.

        Args:
            prompt: The prompt to send to the LLM
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Timeout in seconds
            **kwargs: Additional parameters

        Returns:
            LLM response
        """
        # Convert model enum to string if needed
        if isinstance(model, LLMModel):
            model_str = model.value
        else:
            model_str = str(model)

        # Create LLMRequest
        request = LLMRequest(
            model=model_str,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens or 1000,
            stream=False,
            timeout=timeout,
            **kwargs
        )

        # Call the underlying service
        response = await self.llm_service.generate(request)

        # Ensure we return LLMResponse, not StreamingGenerator
        if hasattr(response, 'content'):
            return response
        else:
            # If it's a streaming response, collect it
            chunks = []
            async for chunk in response:
                if hasattr(chunk, 'content'):
                    chunks.append(chunk.content)

            # Create a response from collected chunks
            return LLMResponse(
                content=''.join(chunks),
                model=model_str,
                provider=getattr(self.llm_service, 'provider_name', 'unknown'),
                model_used=model_str,
                tokens_used=100,  # Approximate
                cost_estimate=0.001,
                generation_time=1.0,
                latency_ms=1000,  # Add missing field
                finish_reason="stop"
            )


def create_agent_llm_interface(llm_service: Optional[LLMService] = None) -> AgentLLMInterface:
    """
    Create an agent-compatible LLM interface.

    Args:
        llm_service: LLM service to use (creates mock if None)

    Returns:
        AgentLLMInterface instance
    """
    if llm_service is None:
        llm_service = MockLLMService()

    return AgentLLMInterface(llm_service)

"""
Mock LLM Service for testing and development.
Provides realistic responses without requiring API keys.
"""

import asyncio
from typing import Any

from src.models.llm_models import LLMRequest, LLMResponse
from src.models.streaming_models import StreamingChunk, StreamingGenerator
from src.services.llm_service import LLMService


class MockLLMService(LLMService):
    """
    Mock LLM service that returns predefined responses.
    Useful for testing and development without API costs.
    """

    def __init__(self, provider_name: str = "mock"):
        self.provider_name = provider_name
        self.model_name = "mock-model"
        self.total_tokens_used = 0
        self.total_cost = 0.0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a mock response based on the request."""
        # Simulate API delay
        await asyncio.sleep(0.1)

        # Generate realistic mock response based on prompt content
        mock_content = self._generate_mock_content(request.prompt)

        # Mock token usage
        prompt_tokens = len(request.prompt.split()) * 1.3  # Rough estimate
        completion_tokens = len(mock_content.split()) * 1.3
        total_tokens = int(prompt_tokens + completion_tokens)

        self.total_tokens_used += total_tokens
        mock_cost = total_tokens * 0.00001  # Mock cost
        self.total_cost += mock_cost

        return LLMResponse(
            content=mock_content,
            model=request.model,
            provider=self.provider_name,
            model_used=request.model,
            tokens_used=total_tokens,
            cost_estimate=mock_cost,
            generation_time=0.1,
            latency_ms=100,  # Add missing field
            finish_reason="stop",
        )

    async def generate_stream(self, request: LLMRequest) -> StreamingGenerator:
        """Generate a mock streaming response."""
        mock_content = self._generate_mock_content(request.prompt)
        words = mock_content.split()

        async def mock_stream():
            for i, word in enumerate(words):
                chunk_content = word + (" " if i < len(words) - 1 else "")
                chunk = StreamingChunk(
                    content=chunk_content,
                    is_final=i == len(words) - 1,
                    total_tokens=len(words),
                    model=request.model,
                )
                yield chunk
                await asyncio.sleep(0.05)  # Simulate streaming delay

        return mock_stream()

    def _generate_mock_content(self, prompt: str) -> str:
        """Generate realistic mock content based on the prompt."""
        prompt_lower = prompt.lower()

        # Question generation mock
        if "generate" in prompt_lower and ("question" in prompt_lower or "math" in prompt_lower):
            return """{
                "question_text": "Calculate the value of 3x + 5 when x = 4",
                "marks": 2,
                "command_word": "Calculate",
                "subject_content_references": ["C2.1"],
                "solution_steps": [
                    "Substitute x = 4 into the expression",
                    "Calculate 3(4) + 5 = 12 + 5 = 17"
                ],
                "final_answer": "17",
                "marking_criteria": [
                    {
                        "criterion": "Correct substitution of x = 4",
                        "marks": 1,
                        "mark_type": "M"
                    },
                    {
                        "criterion": "Correct final answer",
                        "marks": 1,
                        "mark_type": "A"
                    }
                ]
            }"""

        # Marking scheme mock
        if "marking" in prompt_lower and "scheme" in prompt_lower:
            return """{
                "total_marks": 3,
                "marking_criteria": [
                    {
                        "criterion_text": "Shows correct method for solving equation",
                        "marks_value": 1,
                        "mark_type": "M",
                        "notes": "Award for any valid algebraic approach"
                    },
                    {
                        "criterion_text": "Correct calculation steps",
                        "marks_value": 1,
                        "mark_type": "A",
                        "notes": "Must show working clearly"
                    },
                    {
                        "criterion_text": "Correct final answer",
                        "marks_value": 1,
                        "mark_type": "A",
                        "notes": "Accept equivalent forms"
                    }
                ],
                "final_answers": ["17"],
                "alternative_methods": ["Direct substitution", "Algebraic expansion"],
                "common_errors": ["Forgetting to substitute", "Arithmetic errors"]
            }"""

        # Quality review mock
        if "quality" in prompt_lower and ("review" in prompt_lower or "assess" in prompt_lower):
            return """{
                "overall_quality_score": 0.85,
                "mathematical_accuracy": 0.95,
                "cambridge_compliance": 0.80,
                "grade_appropriateness": 0.85,
                "question_clarity": 0.90,
                "marking_accuracy": 0.85,
                "feedback_summary": "High-quality question with clear mathematical content",
                "specific_issues": [],
                "suggested_improvements": [
                    "Consider adding a diagram for visual learners"
                ],
                "decision": "approve"
            }"""

        # Refinement mock
        if "refine" in prompt_lower or "improve" in prompt_lower:
            return """{
                "refined_question": {
                    "question_text": "Calculate the value of 3x + 5 when x = 4. Show your working clearly.",
                    "marks": 2,
                    "command_word": "Calculate",
                    "subject_content_references": ["C2.1"],
                    "solution_steps": [
                        "Substitute x = 4 into the expression 3x + 5",
                        "Calculate 3(4) + 5 = 12 + 5 = 17"
                    ],
                    "final_answer": "17"
                },
                "improvements_made": [
                    "Added instruction to show working clearly",
                    "Made solution steps more explicit"
                ],
                "justification": "Improved clarity and explicit working requirements",
                "quality_impact": {
                    "mathematical_accuracy": 0.95,
                    "cambridge_compliance": 0.90,
                    "grade_appropriateness": 0.90
                }
            }"""

        # Generic response
        return f"This is a mock response to: {prompt[:100]}..."

    async def get_available_models(self) -> list[str]:
        """Return available mock models."""
        return ["mock-gpt-4", "mock-claude-3", "mock-gemini-2"]

    def get_cost_per_token(self, model: str) -> tuple[float, float]:
        """Return mock costs."""
        return 0.00001, 0.00001  # input, output

    async def validate_connection(self) -> bool:
        """Mock connection is always valid."""
        return True

    def get_usage_stats(self) -> dict[str, Any]:
        """Return mock usage statistics."""
        return {
            "total_tokens_used": self.total_tokens_used,
            "total_cost": self.total_cost,
            "provider": self.provider_name,
            "model": self.model_name,
        }

    async def generate_non_stream(self, request: LLMRequest) -> LLMResponse:
        """Generate non-streaming response (same as generate for mock)."""
        return await self.generate(request)

    async def batch_generate(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Generate batch responses."""
        responses = []
        for request in requests:
            response = await self.generate(request)
            responses.append(response)
        return responses

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MockLLMService":
        """Create mock service from config."""
        return cls(provider_name=config.get("provider", "mock"))

    def __str__(self) -> str:
        """String representation."""
        return f"MockLLMService(provider={self.provider_name}, model={self.model_name})"

    def __repr__(self) -> str:
        """String representation."""
        return self.__str__()

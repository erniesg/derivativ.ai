"""
End-to-End tests for LLM services with agent workflows.
Tests the complete multi-agent pipeline using real agent implementations with mocked LLM calls.
"""

from unittest.mock import AsyncMock

import pytest

from src.agents.marker_agent import MarkerAgent
from src.agents.question_generator import QuestionGeneratorAgent
from src.agents.refinement_agent import RefinementAgent
from src.agents.review_agent import ReviewAgent
from src.models.enums import CalculatorPolicy, CommandWord, LLMModel, Tier
from src.models.llm_models import LLMResponse
from src.models.question_models import GenerationRequest


class TestLLMServicesWithQuestionGeneratorAgent:
    """Test LLM services integration with QuestionGeneratorAgent."""

    @pytest.fixture
    def generation_request(self):
        """Create a sample generation request."""
        return GenerationRequest(
            topic="algebra",
            tier=Tier.CORE,
            grade_level=8,
            marks=3,
            count=1,
            calculator_policy=CalculatorPolicy.NOT_ALLOWED,
            command_word=CommandWord.CALCULATE,
            temperature=0.7,
            max_retries=2,
        )

    @pytest.fixture
    def mock_llm_response_question(self):
        """Mock LLM response for question generation."""
        return LLMResponse(
            content="""{
                "question_text": "Calculate the value of 3x + 2 when x = 5",
                "marks": 3,
                "command_word": "Calculate",
                "subject_content_references": ["C2.1", "C2.2"],
                "solution_steps": [
                    "Substitute x = 5 into the expression",
                    "Calculate 3(5) + 2 = 15 + 2 = 17"
                ],
                "final_answer": "17"
            }""",
            model_used="gpt-4o-mini",
            tokens_used=150,
            cost_estimate=0.001,
            latency_ms=800,
            provider="openai",
            metadata={"temperature": 0.7},
        )

    @pytest.mark.asyncio
    async def test_question_generator_with_openai_service(
        self, generation_request, mock_llm_response_question
    ):
        """Test QuestionGeneratorAgent with OpenAI service."""
        # Mock LLM service that matches the agent's expected interface
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=mock_llm_response_question)

        # Create agent with mocked LLM service
        agent = QuestionGeneratorAgent(
            name="test_generator",
            llm_service=mock_llm,
            config={"max_retries": 2, "generation_timeout": 30},
        )

        # Test question generation
        input_data = generation_request.model_dump()

        result = await agent.process(input_data)

        # Verify result
        assert result.success is True
        assert result.agent_name == "test_generator"
        assert "question" in result.output
        assert len(result.reasoning_steps) > 0

        # Verify LLM service was called
        # The AgentLLMInterface wraps the llm_service and calls its generate method
        # with an LLMRequest object
        mock_llm.generate.assert_called()

        # Check that the LLMRequest was created properly
        call_args = mock_llm.generate.call_args
        if call_args and len(call_args) > 0:
            # First argument should be an LLMRequest
            llm_request = call_args[0][0] if call_args[0] else None
            if llm_request and hasattr(llm_request, "prompt"):
                assert llm_request.prompt is not None
                assert llm_request.model is not None
                assert llm_request.temperature == 0.7

    @pytest.mark.asyncio
    async def test_question_generator_with_anthropic_fallback(self, generation_request):
        """Test QuestionGeneratorAgent with fallback to Anthropic when OpenAI fails."""
        # Mock LLM service that fails first, then succeeds
        mock_llm = AsyncMock()

        fallback_response = LLMResponse(
            content='{"question_text": "Find the value of y when 2y + 3 = 11", "marks": 2}',
            model_used="claude-3-5-haiku-20241022",
            tokens_used=120,
            cost_estimate=0.0008,
            latency_ms=600,
            provider="anthropic",
        )

        # First call fails, fallback succeeds
        mock_llm.generate.side_effect = [Exception("API Error"), fallback_response]

        agent = QuestionGeneratorAgent(
            name="fallback_generator",
            llm_service=mock_llm,
            config={"max_retries": 2, "enable_fallback": True},
        )

        input_data = generation_request.model_dump()
        result = await agent.process(input_data)

        # Should succeed with fallback
        assert result.success is True
        assert len(result.reasoning_steps) > 0

        # Should show retry/attempt reasoning
        reasoning_text = " ".join(result.reasoning_steps).lower()
        assert "attempt 1" in reasoning_text and "attempt 2" in reasoning_text

    @pytest.mark.asyncio
    async def test_question_generator_with_different_llm_models(self, generation_request):
        """Test QuestionGeneratorAgent works with different LLM models."""
        mock_llm = AsyncMock()

        # Mock response for a Gemini model
        gemini_response = LLMResponse(
            content='{"question_text": "Calculate the area of a circle with radius 4cm", "marks": 3}',
            model_used="gemini-2.0-flash-exp",
            tokens_used=100,
            cost_estimate=0.0,
            latency_ms=500,
            provider="google",
        )

        mock_llm.generate = AsyncMock(return_value=gemini_response)

        agent = QuestionGeneratorAgent(name="gemini_generator", llm_service=mock_llm)

        # Use Gemini model in request
        gemini_request = generation_request.model_copy()
        gemini_request.llm_model = LLMModel.GEMINI_1_5_FLASH

        input_data = gemini_request.model_dump()
        result = await agent.process(input_data)

        assert result.success is True
        assert "question" in result.output

        # Verify model was passed correctly
        call_kwargs = mock_llm.generate.call_args[1]
        assert call_kwargs["model"] == LLMModel.GEMINI_1_5_FLASH


class TestLLMServicesWithMarkerAgent:
    """Test LLM services integration with MarkerAgent."""

    @pytest.fixture
    def sample_question(self):
        """Sample question for marking scheme generation."""
        return {
            "question_text": "Calculate the value of 3x + 2 when x = 5",
            "marks": 3,
            "command_word": "Calculate",
            "subject_content_references": ["C2.1", "C2.2"],
        }

    @pytest.fixture
    def mock_marking_response(self):
        """Mock LLM response for marking scheme."""
        return LLMResponse(
            content="""{
                "total_marks": 3,
                "mark_allocation_criteria": [
                    {
                        "criterion_text": "Correct substitution of x = 5",
                        "marks_value": 1,
                        "mark_type": "M"
                    },
                    {
                        "criterion_text": "Correct calculation: 3(5) + 2",
                        "marks_value": 1,
                        "mark_type": "M"
                    },
                    {
                        "criterion_text": "Correct final answer: 17",
                        "marks_value": 1,
                        "mark_type": "A"
                    }
                ],
                "final_answers": [
                    {
                        "answer_text": "17",
                        "value_numeric": 17.0,
                        "unit": null
                    }
                ]
            }""",
            model_used="claude-3-5-haiku-20241022",
            tokens_used=200,
            cost_estimate=0.0016,
            latency_ms=1200,
            provider="anthropic",
        )

    @pytest.mark.asyncio
    async def test_marker_agent_with_anthropic_service(
        self, sample_question, mock_marking_response
    ):
        """Test MarkerAgent with Anthropic service (preferred for marking)."""
        # Mock LLM service
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=mock_marking_response)

        agent = MarkerAgent(
            name="test_marker",
            llm_service=mock_llm,
            config={"preferred_model": "claude-3-5-haiku-20241022", "temperature": 0.3},
        )

        input_data = {
            "question": sample_question,
            "config": {
                "marks": 3,
                "grade_level": 8,
                "subject_content_refs": ["C2.1", "C2.2"],
                "topic": "algebra",
                "tier": "Core",
            },
        }

        result = await agent.process(input_data)

        assert result.success is True
        assert "marking_scheme" in result.output

        # Verify marking scheme content
        marking_scheme = result.output["marking_scheme"]
        assert marking_scheme["total_marks_for_part"] == 3
        assert len(marking_scheme["mark_allocation_criteria"]) >= 1

        # Verify LLM was called
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_marker_agent_error_handling(self, sample_question):
        """Test that MarkerAgent handles errors gracefully."""
        # Mock LLM service that fails
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM Error"))

        agent = MarkerAgent(
            name="error_marker",
            llm_service=mock_llm,
            config={"max_retries": 1, "enable_fallback": False},
        )

        input_data = {
            "question": sample_question,
            "config": {"marks": 3, "topic": "algebra", "tier": "Core"},
        }

        result = await agent.process(input_data)

        assert result.success is False
        assert result.error is not None
        assert len(result.reasoning_steps) > 0


class TestLLMServicesWithReviewAgent:
    """Test LLM services integration with ReviewAgent."""

    @pytest.fixture
    def sample_question_data(self):
        """Sample question with marking scheme for review."""
        return {
            "question": {
                "question_text": "Calculate the value of 3x + 2 when x = 5",
                "marks": 3,
                "command_word": "Calculate",
                "subject_content_references": ["C2.1"],
                "grade_level": 8,
                "tier": "Core",
            },
            "marking_scheme": {
                "total_marks_for_part": 3,
                "mark_allocation_criteria": [
                    {"criterion_text": "Correct substitution", "marks_value": 1, "mark_type": "M"}
                ],
            },
        }

    @pytest.fixture
    def mock_review_response(self):
        """Mock LLM response for quality review."""
        return LLMResponse(
            content="""{
                "overall_quality_score": 0.85,
                "quality_dimensions": {
                    "mathematical_accuracy": 0.9,
                    "cambridge_compliance": 0.8,
                    "grade_appropriateness": 0.85,
                    "clarity": 0.85
                },
                "decision": "APPROVE",
                "confidence": 0.88,
                "feedback": "Well-structured question with clear command word and appropriate difficulty.",
                "suggested_improvements": []
            }""",
            model_used="claude-3-5-sonnet-20241022",
            tokens_used=180,
            cost_estimate=0.0027,
            latency_ms=1000,
            provider="anthropic",
        )

    @pytest.mark.asyncio
    async def test_review_agent_with_multiple_providers(
        self, sample_question_data, mock_review_response
    ):
        """Test ReviewAgent can work with different LLM providers."""
        # Mock LLM service
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=mock_review_response)

        agent = ReviewAgent(llm_service=mock_llm, name="test_reviewer")

        input_data = {"question_data": sample_question_data}

        result = await agent.process(input_data)

        assert result.success is True
        assert "quality_decision" in result.output

        quality_decision = result.output["quality_decision"]
        assert quality_decision["action"] == "approve"
        assert quality_decision["quality_score"] >= 0.85

    @pytest.mark.asyncio
    async def test_review_agent_low_quality_rejection(self, sample_question_data):
        """Test ReviewAgent rejects low quality questions."""
        # Mock LLM service returning low quality score
        mock_llm = AsyncMock()

        reject_response = LLMResponse(
            content='{"overall_quality_score": 0.25, "decision": "reject", "feedback": "Low quality"}',
            model_used="gpt-4o",
            tokens_used=100,
            cost_estimate=0.001,
            latency_ms=500,
            provider="openai",
        )

        mock_llm.generate = AsyncMock(return_value=reject_response)

        agent = ReviewAgent(llm_service=mock_llm, name="reject_reviewer")

        result = await agent.process({"question_data": sample_question_data})

        assert result.success is True
        quality_decision = result.output["quality_decision"]
        assert quality_decision["action"] == "reject"
        assert quality_decision["quality_score"] < 0.5


class TestLLMServicesWithRefinementAgent:
    """Test LLM services integration with RefinementAgent."""

    @pytest.fixture
    def refinement_input(self):
        """Sample input for refinement agent."""
        return {
            "original_question": {
                "question_text": "Calculate x + 2 when x = 5",
                "marks": 2,
                "command_word": "Calculate",
                "grade_level": 8,
            },
            "quality_decision": {
                "action": "refine",
                "quality_score": 0.6,
                "decision": "REFINE",
                "feedback": "Question is too simple for grade level",
                "suggested_improvements": [
                    "Add more steps to the calculation",
                    "Include units in the final answer",
                ],
            },
        }

    @pytest.fixture
    def mock_refinement_response(self):
        """Mock LLM response for question refinement."""
        return LLMResponse(
            content="""{
                "refined_question": {
                    "question_text": "Calculate the value of 3x + 2y when x = 5 and y = 3. Give your answer with appropriate units.",
                    "marks": 3,
                    "command_word": "Calculate",
                    "subject_content_references": ["C2.1", "C2.3"]
                },
                "refinement_metadata": {
                    "improvements_made": [
                        "Added second variable to increase complexity",
                        "Specified units requirement",
                        "Increased mark allocation to reflect difficulty"
                    ],
                    "quality_improvements": {
                        "complexity": 0.3,
                        "clarity": 0.1,
                        "grade_appropriateness": 0.25
                    }
                }
            }""",
            model_used="gpt-4o",
            tokens_used=220,
            cost_estimate=0.0015,
            latency_ms=1100,
            provider="openai",
        )

    @pytest.mark.asyncio
    async def test_refinement_agent_with_gemini_service(
        self, refinement_input, mock_refinement_response
    ):
        """Test RefinementAgent with Google Gemini service."""
        # Mock LLM service
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=mock_refinement_response)

        agent = RefinementAgent(llm_service=mock_llm, name="test_refiner")

        result = await agent.process(refinement_input)

        assert result.success is True
        assert "refined_question" in result.output

        # Verify refinement improvements
        refined = result.output["refined_question"]
        original = refinement_input["original_question"]

        assert refined["marks"] >= original["marks"]  # Should maintain or increase complexity
        assert len(refined["question_text"]) > len(original["question_text"])

    @pytest.mark.asyncio
    async def test_refinement_agent_error_handling(self, refinement_input):
        """Test RefinementAgent handles errors gracefully."""
        # Mock LLM service that fails
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=Exception("Refinement Error"))

        agent = RefinementAgent(llm_service=mock_llm, name="error_refiner")

        result = await agent.process(refinement_input)

        assert result.success is False
        assert result.error is not None
        assert len(result.reasoning_steps) > 0


class TestEndToEndMultiAgentWorkflow:
    """Test complete multi-agent workflow with all LLM services."""

    @pytest.mark.asyncio
    async def test_complete_question_generation_workflow(self):
        """Test complete workflow: Generate → Mark → Review."""
        # Mock responses for each step
        generation_response = LLMResponse(
            content='{"question_text": "Calculate 2x + 3 when x = 4", "marks": 2}',
            model_used="gpt-4o-mini",
            tokens_used=100,
            cost_estimate=0.001,
            latency_ms=500,
            provider="openai",
        )

        marking_response = LLMResponse(
            content='{"total_marks": 2, "mark_allocation_criteria": [{"criterion_text": "Correct method", "marks_value": 1}], "final_answers": [{"answer_text": "11"}]}',
            model_used="claude-3-5-haiku-20241022",
            tokens_used=120,
            cost_estimate=0.0016,
            latency_ms=600,
            provider="anthropic",
        )

        review_response = LLMResponse(
            content='{"overall_quality_score": 0.85, "decision": "approve", "feedback": "Good quality"}',
            model_used="claude-3-5-sonnet-20241022",
            tokens_used=80,
            cost_estimate=0.0024,
            latency_ms=400,
            provider="anthropic",
        )

        # Create mock LLM services for each agent
        gen_llm = AsyncMock()
        gen_llm.generate = AsyncMock(return_value=generation_response)

        mark_llm = AsyncMock()
        mark_llm.generate = AsyncMock(return_value=marking_response)

        review_llm = AsyncMock()
        review_llm.generate = AsyncMock(return_value=review_response)

        # Create agents
        generator = QuestionGeneratorAgent("generator", llm_service=gen_llm)
        marker = MarkerAgent("marker", llm_service=mark_llm)
        reviewer = ReviewAgent(llm_service=review_llm, name="reviewer")

        # Step 1: Generate question
        gen_request = GenerationRequest(
            topic="algebra",
            tier=Tier.CORE,
            grade_level=8,
            marks=2,
            count=1,
            calculator_policy=CalculatorPolicy.NOT_ALLOWED,
            command_word=CommandWord.CALCULATE,
        )

        gen_result = await generator.process(gen_request.model_dump())
        assert gen_result.success
        assert "question" in gen_result.output

        # Step 2: Create marking scheme
        mark_result = await marker.process(
            {
                "question": {"question_text": "Calculate 2x + 3 when x = 4", "marks": 2},
                "config": {"marks": 2, "grade_level": 8, "topic": "algebra", "tier": "Core"},
            }
        )
        assert mark_result.success
        assert "marking_scheme" in mark_result.output

        # Step 3: Review quality
        review_result = await reviewer.process(
            {
                "question_data": {
                    "question": {
                        "question_text": "Calculate 2x + 3 when x = 4",
                        "marks": 2,
                        "command_word": "Calculate",
                        "subject_content_references": ["C2.1"],
                        "grade_level": 8,
                        "tier": "Core",
                    },
                    "marking_scheme": {"total_marks_for_part": 2, "mark_allocation_criteria": []},
                }
            }
        )
        assert review_result.success
        assert "quality_decision" in review_result.output

        # Verify all agents were called
        gen_llm.generate.assert_called_once()
        mark_llm.generate.assert_called_once()
        review_llm.generate.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

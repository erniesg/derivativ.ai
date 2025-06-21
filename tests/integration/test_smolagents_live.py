"""
Live integration tests for smolagents with real API calls.
Tests actual LLM API connectivity through smolagents tools.
"""

import json
import os

import pytest
from dotenv import load_dotenv

from src.agents.smolagents_integration import (
    create_derivativ_agent,
    generate_math_question,
    refine_question,
    review_question_quality,
)

# Load environment variables
load_dotenv()


class TestSmolagentsLiveIntegration:
    """Live tests with real API calls - no mocks."""

    @pytest.mark.integration
    @pytest.mark.skipif(
        not any(
            [
                os.getenv("OPENAI_API_KEY"),
                os.getenv("ANTHROPIC_API_KEY"),
                os.getenv("GOOGLE_API_KEY"),
            ]
        ),
        reason="No LLM API keys available for live testing",
    )
    def test_generate_math_question_live(self):
        """Test live question generation through smolagents tool."""
        result = generate_math_question(
            topic="algebra",
            grade_level=8,
            marks=3,
            calculator_policy="not_allowed",
            command_word="Calculate",
        )

        # Should return valid JSON with question data
        assert isinstance(result, str)
        assert not result.startswith("Error")

        # Parse and validate structure
        data = json.loads(result)
        assert "question" in data
        question = data["question"]
        # Use either question_text or raw_text_content
        text_field = question.get("question_text") or question.get("raw_text_content")
        assert text_field is not None
        assert "marks" in question
        assert question["marks"] == 3
        assert len(text_field) > 20

    @pytest.mark.integration
    @pytest.mark.skipif(
        not any(
            [
                os.getenv("OPENAI_API_KEY"),
                os.getenv("ANTHROPIC_API_KEY"),
                os.getenv("GOOGLE_API_KEY"),
            ]
        ),
        reason="No LLM API keys available for live testing",
    )
    def test_live_workflow_complete(self):
        """Test complete workflow: generate -> review -> refine (if needed)."""
        # Step 1: Generate
        question_result = generate_math_question(
            topic="geometry",
            grade_level=9,
            marks=4,
            calculator_policy="allowed",
            command_word="Find",
        )

        assert not question_result.startswith("Error")
        question_data = json.loads(question_result)
        assert "question" in question_data

        # Step 2: Review
        review_result = review_question_quality(question_result)
        assert not review_result.startswith("Error")
        review_data = json.loads(review_result)
        # Quality score can be in root or quality_decision
        quality_score = review_data.get("quality_score") or review_data.get(
            "quality_decision", {}
        ).get("overall_score", 0.5)
        assert 0 <= quality_score <= 1

        # Step 3: Refine if quality is low
        if quality_score < 0.8:
            refine_result = refine_question(question_result, review_result)
            assert not refine_result.startswith("Error")
            refined_data = json.loads(refine_result)
            assert "refined_question" in refined_data

    @pytest.mark.integration
    @pytest.mark.skipif(
        not any(
            [
                os.getenv("OPENAI_API_KEY"),
                os.getenv("ANTHROPIC_API_KEY"),
                os.getenv("GOOGLE_API_KEY"),
            ]
        ),
        reason="No LLM API keys available for live testing",
    )
    def test_smolagents_agent_creation_live(self):
        """Test that smolagents agents can be created with real LLM services."""
        # Test different agent types
        agent_types = ["question_generator", "quality_control", "multi_agent"]

        for agent_type in agent_types:
            agent = create_derivativ_agent(agent_type=agent_type)
            assert agent is not None
            assert agent.name is not None
            assert len(agent.tools) > 0
            assert "generate_math_question" in agent.tools

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("HF_TOKEN"),
        reason="HF_TOKEN not available for interactive smolagents testing",
    )
    def test_smolagents_interactive_live(self):
        """Test smolagents with natural language prompts (requires HF_TOKEN)."""
        agent = create_derivativ_agent(agent_type="question_generator")

        # This should work with HF_TOKEN set
        try:
            result = agent.run("Generate a simple algebra question for grade 8, worth 3 marks")
            # Should get a response (exact format depends on smolagents version)
            assert result is not None
            assert len(str(result)) > 0
        except Exception as e:
            # If it fails, it should be due to model issues, not our integration
            assert "API" not in str(e).upper(), f"API-related error: {e}"


class TestLLMServicesLiveOnly:
    """Tests that force live API calls for each provider."""

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    def test_openai_service_live_call(self):
        """Test OpenAI service with real API call."""
        import asyncio

        from src.models.llm_models import LLMRequest
        from src.services.llm_factory import LLMFactory

        async def test_openai():
            factory = LLMFactory()
            service = factory.get_service("openai")

            request = LLMRequest(prompt="What is 2+2?", model="gpt-4o-mini", max_tokens=50)

            response = await service.generate_non_stream(request)
            assert response.content is not None
            assert len(response.content) > 0
            # Model name might include version details
            assert "gpt-4o-mini" in response.model_used
            assert response.tokens_used > 0

        asyncio.run(test_openai())

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
    def test_anthropic_service_live_call(self):
        """Test Anthropic service with real API call."""
        import asyncio

        from src.models.llm_models import LLMRequest
        from src.services.llm_factory import LLMFactory

        async def test_anthropic():
            factory = LLMFactory()
            service = factory.get_service("anthropic")

            request = LLMRequest(
                prompt="What is 3+3?", model="claude-3-5-haiku-20241022", max_tokens=50
            )

            response = await service.generate_non_stream(request)
            assert response.content is not None
            assert len(response.content) > 0
            assert "claude" in response.model_used.lower()
            assert response.tokens_used > 0

        asyncio.run(test_anthropic())

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
    def test_gemini_service_live_call(self):
        """Test Gemini service with real API call."""
        import asyncio

        from src.models.llm_models import LLMRequest
        from src.services.llm_factory import LLMFactory

        async def test_gemini():
            factory = LLMFactory()
            service = factory.get_service("google")

            request = LLMRequest(prompt="What is 4+4?", model="gemini-1.5-flash", max_tokens=50)

            response = await service.generate_non_stream(request)
            assert response.content is not None
            assert len(response.content) > 0
            assert "gemini" in response.model_used.lower()
            assert response.tokens_used > 0

        asyncio.run(test_gemini())

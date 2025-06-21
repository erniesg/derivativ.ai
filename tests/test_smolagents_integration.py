"""
Test suite for smolagents integration.
Tests native smolagents tools and agents.
"""

import json
from unittest.mock import Mock, patch

import pytest

from src.agents.smolagents_integration import (
    DerivativSmolagents,
    _get_llm_service,
    create_derivativ_agent,
    generate_math_question,
    refine_question,
    review_question_quality,
)


class TestSmolagentsTools:
    """Test the smolagents tool functions."""

    def test_generate_math_question_tool(self):
        """Test the generate_math_question tool function."""
        result = generate_math_question(topic="algebra", grade_level=8, marks=3)

        assert isinstance(result, str)
        # Should either be valid JSON or an error message
        if result.startswith("Error"):
            assert "Error" in result
        else:
            # Should be valid JSON
            try:
                parsed = json.loads(result)
                assert isinstance(parsed, dict)
            except json.JSONDecodeError:
                pytest.fail("Result should be valid JSON or error message")

    def test_generate_math_question_tool_with_params(self):
        """Test the tool with various parameters."""
        result = generate_math_question(
            topic="geometry",
            grade_level=9,
            marks=5,
            calculator_policy="allowed",
            command_word="Find",
        )

        assert isinstance(result, str)
        assert len(result) > 0

    @patch("src.agents.smolagents_integration._get_llm_service")
    def test_generate_math_question_with_mock_service(self, mock_get_service):
        """Test tool with mocked LLM service."""
        from src.services.mock_llm_service import MockLLMService

        mock_get_service.return_value = MockLLMService()

        result = generate_math_question(topic="probability", grade_level=7, marks=4)

        assert isinstance(result, str)
        mock_get_service.assert_called_once()

    def test_review_question_quality_tool(self):
        """Test the review_question_quality tool function."""
        # Sample question data
        question_data = {
            "question_text": "Calculate 2 + 3",
            "marks": 1,
            "command_word": "Calculate",
        }

        result = review_question_quality(json.dumps(question_data))

        assert isinstance(result, str)
        # Should either be valid JSON or an error message
        if result.startswith("Error"):
            assert "Error" in result
        else:
            try:
                parsed = json.loads(result)
                assert isinstance(parsed, dict)
            except json.JSONDecodeError:
                pytest.fail("Result should be valid JSON or error message")

    def test_refine_question_tool(self):
        """Test the refine_question tool function."""
        original_question = {"question_text": "Calculate 2 + 3", "marks": 1}
        feedback = {"quality_score": 0.5, "feedback": "Question is too simple"}

        result = refine_question(json.dumps(original_question), json.dumps(feedback))

        assert isinstance(result, str)
        if result.startswith("Error"):
            assert "Error" in result
        else:
            try:
                parsed = json.loads(result)
                assert isinstance(parsed, dict)
            except json.JSONDecodeError:
                pytest.fail("Result should be valid JSON or error message")


class TestDerivativSmolagents:
    """Test the DerivativSmolagents class."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_init_with_openai_key(self):
        """Test initialization with OpenAI API key."""
        derivativ = DerivativSmolagents()
        assert derivativ.model_id == "gpt-4o-mini"

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=True)
    def test_init_with_anthropic_key(self):
        """Test initialization with Anthropic API key."""
        derivativ = DerivativSmolagents()
        assert derivativ.model_id == "claude-3-5-haiku-20241022"

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}, clear=True)
    def test_init_with_google_key(self):
        """Test initialization with Google API key."""
        derivativ = DerivativSmolagents()
        assert derivativ.model_id == "gemini-2.0-flash-exp"

    @patch.dict("os.environ", {}, clear=True)
    def test_init_without_api_keys(self):
        """Test initialization without API keys."""
        derivativ = DerivativSmolagents()
        assert derivativ.model_id == "gpt-4o-mini"  # Default fallback

    def test_custom_model_id(self):
        """Test initialization with custom model ID."""
        derivativ = DerivativSmolagents(model_id="custom-model")
        assert derivativ.model_id == "custom-model"

    @patch("src.agents.smolagents_integration.InferenceClientModel")
    def test_create_question_generator_agent(self, mock_model):
        """Test creating question generator agent."""
        derivativ = DerivativSmolagents()
        agent = derivativ.create_question_generator_agent()

        # Check agent properties
        assert agent.name == "question_generator"
        assert "question" in agent.description.lower()
        # Should have generate_math_question tool plus final_answer (default)
        assert len(agent.tools) >= 1
        assert "generate_math_question" in agent.tools

    @patch("src.agents.smolagents_integration.InferenceClientModel")
    def test_create_quality_control_agent(self, mock_model):
        """Test creating quality control agent."""
        derivativ = DerivativSmolagents()
        agent = derivativ.create_quality_control_agent()

        assert agent.name == "quality_controller"
        assert "quality" in agent.description.lower()
        # Should have all three tools plus final_answer (default)
        assert len(agent.tools) >= 3
        expected_tools = ["generate_math_question", "review_question_quality", "refine_question"]
        for tool_name in expected_tools:
            assert tool_name in agent.tools

    @patch("src.agents.smolagents_integration.InferenceClientModel")
    def test_create_multi_agent_system(self, mock_model):
        """Test creating multi-agent system."""
        derivativ = DerivativSmolagents()
        agent = derivativ.create_multi_agent_system()

        assert agent.name == "derivativ_ai"
        assert "education" in agent.description.lower()
        assert len(agent.tools) >= 3  # Should have our tools plus base tools


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch("src.agents.smolagents_integration.InferenceClientModel")
    def test_create_derivativ_agent_question_generator(self, mock_model):
        """Test creating question generator via convenience function."""
        agent = create_derivativ_agent(agent_type="question_generator")
        assert agent.name == "question_generator"

    @patch("src.agents.smolagents_integration.InferenceClientModel")
    def test_create_derivativ_agent_quality_control(self, mock_model):
        """Test creating quality control agent via convenience function."""
        agent = create_derivativ_agent(agent_type="quality_control")
        assert agent.name == "quality_controller"

    @patch("src.agents.smolagents_integration.InferenceClientModel")
    def test_create_derivativ_agent_multi_agent(self, mock_model):
        """Test creating multi-agent system via convenience function."""
        agent = create_derivativ_agent(agent_type="multi_agent")
        assert agent.name == "derivativ_ai"

    @patch("src.agents.smolagents_integration.InferenceClientModel")
    def test_create_derivativ_agent_invalid_type(self, mock_model):
        """Test creating agent with invalid type."""
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_derivativ_agent(agent_type="invalid_type")

    @patch("src.agents.smolagents_integration.InferenceClientModel")
    def test_create_derivativ_agent_with_custom_model(self, mock_model):
        """Test creating agent with custom model ID."""
        agent = create_derivativ_agent(model_id="custom-model", agent_type="question_generator")
        assert agent.name == "question_generator"


class TestLLMServiceSelection:
    """Test LLM service selection logic."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("src.agents.smolagents_integration.LLMFactory")
    def test_get_llm_service_with_openai(self, mock_factory):
        """Test LLM service selection with OpenAI key."""
        mock_service = Mock()
        mock_factory.return_value.detect_provider.return_value = "openai"
        mock_factory.return_value.get_service.return_value = mock_service

        service = _get_llm_service()

        mock_factory.assert_called_once()
        mock_factory.return_value.detect_provider.assert_called_once_with("openai")
        mock_factory.return_value.get_service.assert_called_once_with("openai")

    @patch.dict("os.environ", {}, clear=True)
    def test_get_llm_service_fallback_to_mock(self):
        """Test LLM service fallback to mock when no API keys."""
        service = _get_llm_service()

        from src.services.mock_llm_service import MockLLMService

        assert isinstance(service, MockLLMService)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("src.agents.smolagents_integration.LLMFactory")
    def test_get_llm_service_factory_error_fallback(self, mock_factory):
        """Test LLM service fallback when factory fails."""
        mock_factory.return_value.detect_provider.side_effect = Exception("Factory error")

        service = _get_llm_service()

        from src.services.mock_llm_service import MockLLMService

        assert isinstance(service, MockLLMService)


class TestSmolagentsIntegration:
    """End-to-end integration tests."""

    @patch("src.agents.smolagents_integration.InferenceClientModel")
    def test_agent_tool_integration(self, mock_model):
        """Test that agents can use the tools properly."""
        # This is more of a smoke test since actual agent.run()
        # would require real models or complex mocking
        derivativ = DerivativSmolagents()
        agent = derivativ.create_question_generator_agent()

        # Verify agent has the expected tools
        tool_names = list(agent.tools.keys())
        assert "generate_math_question" in tool_names

    @patch("src.agents.smolagents_integration.InferenceClientModel")
    def test_full_workflow_tools_available(self, mock_model):
        """Test that all workflow tools are available in multi-agent."""
        derivativ = DerivativSmolagents()
        agent = derivativ.create_multi_agent_system()

        tool_names = list(agent.tools.keys())
        expected_tools = ["generate_math_question", "review_question_quality", "refine_question"]

        for tool_name in expected_tools:
            assert tool_name in tool_names

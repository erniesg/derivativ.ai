"""
Unit tests for Question Generator Agent.
Fast, isolated tests focusing on individual methods and core logic.
"""

import pytest

from src.agents.question_generator import QuestionGeneratorAgent, QuestionGeneratorError
from src.models.enums import CommandWord
from src.models.question_models import GenerationRequest, Question


class TestQuestionGeneratorUnit:
    """Unit tests for Question Generator core functionality"""

    def test_agent_initialization_with_defaults(self):
        """Test agent initializes with default services"""
        agent = QuestionGeneratorAgent()
        assert agent.name == "QuestionGenerator"
        assert agent.llm_service is not None
        assert agent.prompt_manager is not None
        assert agent.json_parser is not None

    def test_parse_generation_request_valid_input(self):
        """Test parsing valid generation request"""
        agent = QuestionGeneratorAgent()
        input_data = {"topic": "algebra", "marks": 4, "tier": "Extended", "grade_level": 9}

        request = agent._parse_generation_request(input_data)

        assert isinstance(request, GenerationRequest)
        assert request.topic == "algebra"
        assert request.marks == 4
        assert request.tier.value == "Extended"
        assert request.grade_level == 9

    def test_parse_generation_request_invalid_input(self):
        """Test parsing invalid generation request raises error"""
        agent = QuestionGeneratorAgent()
        invalid_data = {"invalid": "data"}

        with pytest.raises(QuestionGeneratorError):
            agent._parse_generation_request(invalid_data)

    def test_parse_generation_request_invalid_marks(self):
        """Test parsing request with invalid mark values"""
        agent = QuestionGeneratorAgent()
        invalid_data = {
            "topic": "algebra",
            "marks": 25,  # Invalid: too high
            "tier": "Core",
        }

        with pytest.raises(QuestionGeneratorError, match="Marks must be between 1 and 20"):
            agent._parse_generation_request(invalid_data)

    @pytest.mark.asyncio
    async def test_question_object_conversion_from_json(self):
        """Test converting JSON response to Question object"""
        agent = QuestionGeneratorAgent()
        json_data = {
            "question_text": "Calculate 2 + 3",
            "marks": 2,
            "command_word": "Calculate",
            "solution_steps": ["Add the numbers"],
            "final_answer": "5",
        }
        request = GenerationRequest(topic="arithmetic", marks=2, tier="Core")

        question = await agent._convert_to_question_object(json_data, request)

        assert isinstance(question, Question)
        assert question.raw_text_content == "Calculate 2 + 3"
        assert question.marks == 2
        assert question.command_word == CommandWord.CALCULATE

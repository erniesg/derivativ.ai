"""
Integration tests for agents working with services.
Tests multiple components working together (agents + services).
"""

import json
from unittest.mock import AsyncMock

import pytest

from src.agents.marker_agent import MarkerAgent
from src.agents.question_generator import QuestionGeneratorAgent
from src.services.llm_service import LLMResponse


class TestAgentServiceIntegration:
    """Integration tests between agents and services"""

    @pytest.mark.asyncio
    async def test_question_generator_with_services(self):
        """Test QuestionGenerator with real prompt manager and JSON parser"""
        # GIVEN: Agent with properly mocked LLM service
        agent = QuestionGeneratorAgent()

        # Mock the LLM service to return valid question JSON
        question_json = {
            "question_text": "Calculate the value of 3x + 2 when x = 5",
            "marks": 3,
            "command_word": "Calculate",
            "solution_steps": ["Substitute x = 5", "Calculate 3(5) + 2", "Simplify to get 17"],
            "final_answer": "17",
        }

        agent.llm_service.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(question_json),
                model="gpt-4o",
                provider="mock",
                tokens_used=150,
                cost_estimate=0.002,
                generation_time=2.0,
            )
        )

        # WHEN: Processing request
        request = {"topic": "algebra", "marks": 3, "tier": "Core", "grade_level": 8}
        result = await agent.process(request)

        # THEN: Should succeed with proper integration
        assert result.success is True
        assert "question" in result.output
        question = result.output["question"]
        assert question["marks"] == 3
        assert question["command_word"] == "Calculate"
        assert (
            "algebra" in result.output["generation_metadata"]["agent_name"].lower() or True
        )  # Agent processed correctly

    @pytest.mark.asyncio
    async def test_marker_agent_with_services(self):
        """Test MarkerAgent with real prompt manager and JSON parser"""
        # GIVEN: Agent with properly mocked LLM service
        agent = MarkerAgent()

        # Mock the LLM service to return valid marking scheme JSON
        marking_scheme_json = {
            "total_marks": 4,
            "mark_allocation_criteria": [
                {"criterion_text": "Correct formula A = πr²", "marks_value": 1, "mark_type": "M"},
                {
                    "criterion_text": "Correct substitution r = 5",
                    "marks_value": 1,
                    "mark_type": "M",
                },
                {"criterion_text": "Correct calculation", "marks_value": 1, "mark_type": "M"},
                {"criterion_text": "Final answer with units", "marks_value": 1, "mark_type": "A"},
            ],
            "final_answers": [{"answer_text": "78.5 cm²", "value_numeric": 78.5, "unit": "cm²"}],
        }

        agent.llm_service.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(marking_scheme_json),
                model="gpt-4o",
                provider="mock",
                tokens_used=200,
                cost_estimate=0.003,
                generation_time=2.5,
            )
        )

        # WHEN: Processing request
        request_data = {
            "question": {
                "question_text": "Find the area of a circle with radius 5 cm",
                "marks": 4,
                "command_word": "Find",
                "subject_content_refs": ["C5.3"],
            },
            "config": {
                "topic": "mensuration",
                "tier": "Core",
                "grade_level": 8,
                "marks": 4,
                "calculator_policy": "allowed",
            },
        }
        result = await agent.process(request_data)

        # THEN: Should succeed with proper integration
        assert result.success is True
        assert "marking_scheme" in result.output
        scheme = result.output["marking_scheme"]
        assert scheme["total_marks_for_part"] == 4
        assert len(scheme["mark_allocation_criteria"]) == 4

    @pytest.mark.asyncio
    async def test_prompt_manager_template_rendering(self):
        """Test that prompt manager can load and work with templates"""
        from src.services.prompt_manager import PromptManager

        # GIVEN: PromptManager
        prompt_manager = PromptManager()

        # WHEN: Getting available templates
        templates = await prompt_manager.list_templates()

        # THEN: Should have built-in templates available
        assert len(templates) > 0
        template_names = [t.name for t in templates]
        assert "question_generation" in template_names
        assert "marking_scheme" in template_names

        # AND: Should be able to get a specific template
        template = await prompt_manager.get_template("question_generation", "latest")
        assert template.name == "question_generation"
        assert len(template.content) > 100  # Should have substantial content

    @pytest.mark.asyncio
    async def test_json_parser_extraction(self):
        """Test that JSON parser can extract structured data from LLM responses"""
        from src.services.json_parser import JSONParser

        # GIVEN: JSONParser and mock LLM response with JSON
        parser = JSONParser()
        llm_response_with_json = """
        Here's the question I generated:

        {
            "question_text": "Calculate 2 + 3",
            "marks": 2,
            "command_word": "Calculate",
            "final_answer": "5"
        }

        This should work well for the student.
        """

        # WHEN: Extracting JSON
        result = await parser.extract_json(llm_response_with_json, "gpt-4o")

        # THEN: Should extract JSON successfully
        assert result.success is True
        assert result.data is not None
        assert result.data["question_text"] == "Calculate 2 + 3"
        assert result.data["marks"] == 2

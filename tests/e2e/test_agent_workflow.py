"""
End-to-end tests for complete agent workflows.
Tests full user scenarios from request to final output.
"""


import pytest

from src.agents.marker_agent import MarkerAgent
from src.agents.question_generator import QuestionGeneratorAgent


class TestAgentWorkflowE2E:
    """End-to-end tests for complete agent workflows"""

    @pytest.mark.asyncio
    async def test_question_to_marking_scheme_workflow(self):
        """Test complete workflow: generate question → create marking scheme"""
        # STEP 1: Generate a question
        question_agent = QuestionGeneratorAgent()
        question_request = {"topic": "algebra", "marks": 3, "tier": "Core", "grade_level": 8}

        question_result = await question_agent.process(question_request)

        # Verify question generation succeeded
        assert question_result.success is True
        question = question_result.output["question"]
        assert question["marks"] == 3
        assert len(question["raw_text_content"]) > 10

        # STEP 2: Generate marking scheme for the question
        marker_agent = MarkerAgent()
        marking_request = {
            "question": {
                "question_text": question["raw_text_content"],
                "marks": question["marks"],
                "command_word": question["command_word"],
            },
            "config": {
                "topic": "algebra",
                "marks": question["marks"],
                "tier": "Core",
                "grade_level": 8,
            },
        }

        marking_result = await marker_agent.process(marking_request)

        # Verify marking scheme generation succeeded
        assert marking_result.success is True
        scheme = marking_result.output["marking_scheme"]
        assert scheme["total_marks_for_part"] == question["marks"]
        assert len(scheme["mark_allocation_criteria"]) > 0

        # Verify consistency between question and marking scheme
        assert scheme["total_marks_for_part"] == question["marks"]

    @pytest.mark.asyncio
    async def test_multiple_question_generation_workflow(self):
        """Test generating multiple questions in sequence"""
        # GIVEN: Agent and request for multiple questions
        agent = QuestionGeneratorAgent()

        topics = ["algebra", "geometry", "statistics"]
        questions = []

        # WHEN: Generating questions for different topics
        for topic in topics:
            request = {"topic": topic, "marks": 2, "tier": "Core", "grade_level": 7}
            result = await agent.process(request)

            # THEN: Each generation should succeed
            assert result.success is True
            question = result.output["question"]
            questions.append(question)

        # Verify all questions were generated successfully
        assert len(questions) == 3
        assert all(q["marks"] == 2 for q in questions)
        assert all(len(q["raw_text_content"]) > 10 for q in questions)

    @pytest.mark.asyncio
    async def test_different_mark_values_workflow(self):
        """Test generating questions and marking schemes with different mark values"""
        mark_values = [1, 2, 4, 5]

        for marks in mark_values:
            # Generate question
            question_agent = QuestionGeneratorAgent()
            question_request = {"topic": "arithmetic", "marks": marks, "tier": "Core"}

            question_result = await question_agent.process(question_request)
            assert question_result.success is True
            question = question_result.output["question"]
            assert question["marks"] == marks

            # Generate corresponding marking scheme
            marker_agent = MarkerAgent()
            marking_request = {
                "question": {
                    "question_text": question["raw_text_content"],
                    "marks": marks,
                    "command_word": "Calculate",
                },
                "config": {"topic": "arithmetic", "marks": marks, "tier": "Core"},
            }

            marking_result = await marker_agent.process(marking_request)
            assert marking_result.success is True
            scheme = marking_result.output["marking_scheme"]

            # Verify marking scheme matches question marks
            assert scheme["total_marks_for_part"] == marks
            # Should have appropriate number of criteria
            assert len(scheme["mark_allocation_criteria"]) >= 1

            # For multi-mark questions, should have multiple criteria
            if marks > 1:
                assert len(scheme["mark_allocation_criteria"]) > 1

    @pytest.mark.asyncio
    async def test_core_vs_extended_tier_workflow(self):
        """Test workflow with different IGCSE tiers"""
        tiers = ["Core", "Extended"]

        for tier in tiers:
            # Generate question for tier
            question_agent = QuestionGeneratorAgent()
            request = {
                "topic": "algebra",
                "marks": 3,
                "tier": tier,
                "grade_level": 9 if tier == "Extended" else 7,
            }

            result = await question_agent.process(request)
            assert result.success is True
            question = result.output["question"]

            # Verify tier is properly handled
            request_data = result.output["request"]
            assert request_data["tier"] == tier

            # Generate marking scheme
            marker_agent = MarkerAgent()
            marking_request = {
                "question": {
                    "question_text": question["raw_text_content"],
                    "marks": 3,
                    "command_word": "Calculate",
                },
                "config": {
                    "topic": "algebra",
                    "marks": 3,
                    "tier": tier,
                    "grade_level": 9 if tier == "Extended" else 7,
                },
            }

            marking_result = await marker_agent.process(marking_request)
            assert marking_result.success is True

            # Both should work regardless of tier
            assert marking_result.output["marking_scheme"]["total_marks_for_part"] == 3

"""
Comprehensive tests for Review Agent.
Tests end-to-end functionality with mock services.
"""

import json
from unittest.mock import AsyncMock

import pytest

from src.agents.review_agent import ReviewAgent
from src.services.llm_service import LLMResponse


class TestReviewAgent:
    """Integration tests for ReviewAgent"""

    @pytest.mark.asyncio
    async def test_review_agent_high_quality_question(self):
        """Test ReviewAgent correctly identifies high quality questions"""
        # GIVEN: ReviewAgent with mocked LLM response
        agent = ReviewAgent()

        # Mock high quality assessment response
        quality_response = {
            "overall_quality_score": 0.9,
            "mathematical_accuracy": 0.95,
            "cambridge_compliance": 0.9,
            "grade_appropriateness": 0.9,
            "question_clarity": 0.85,
            "marking_accuracy": 0.9,
            "feedback_summary": "Excellent question with clear requirements and appropriate difficulty",
            "specific_issues": [],
            "suggested_improvements": ["Consider adding a diagram for visual learners"],
            "decision": "approve",
        }

        agent.llm_service.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(quality_response),
                model_used="gpt-4o",
                provider="mock",
                tokens_used=200,
                cost_estimate=0.003,
                latency_ms=2000,
            )
        )

        # WHEN: Processing quality review request
        input_data = {
            "question_data": {
                "question": {
                    "question_text": "Calculate the area of a circle with radius 5 cm. Give your answer to 1 decimal place.",
                    "marks": 3,
                    "command_word": "Calculate",
                    "subject_content_references": ["C5.3"],
                    "grade_level": 8,
                    "tier": "Core",
                },
                "marking_scheme": {
                    "total_marks_for_part": 3,
                    "mark_allocation_criteria": [
                        {
                            "criterion_text": "Uses formula A = πr²",
                            "marks_value": 1,
                            "mark_type": "M",
                        },
                        {"criterion_text": "Substitutes r = 5", "marks_value": 1, "mark_type": "M"},
                        {
                            "criterion_text": "Calculates and rounds to 1 d.p.",
                            "marks_value": 1,
                            "mark_type": "A",
                        },
                    ],
                    "final_answers": [
                        {"answer_text": "78.5", "value_numeric": 78.5, "unit": "cm²"}
                    ],
                },
            }
        }

        result = await agent.process(input_data)

        # THEN: Should approve the question
        assert result.success is True
        assert "quality_decision" in result.output

        quality_decision = result.output["quality_decision"]
        assert quality_decision["quality_score"] == 0.9
        assert quality_decision["action"] == "approve"
        assert quality_decision["mathematical_accuracy"] == 0.95
        assert quality_decision["cambridge_compliance"] == 0.9

    @pytest.mark.asyncio
    async def test_review_agent_medium_quality_question(self):
        """Test ReviewAgent correctly identifies medium quality questions needing review"""
        # GIVEN: ReviewAgent with mocked LLM response
        agent = ReviewAgent()

        # Mock medium quality assessment response
        quality_response = {
            "overall_quality_score": 0.75,
            "mathematical_accuracy": 0.8,
            "cambridge_compliance": 0.7,
            "grade_appropriateness": 0.75,
            "question_clarity": 0.7,
            "marking_accuracy": 0.8,
            "feedback_summary": "Adequate question but needs some improvements for clarity",
            "specific_issues": [
                "Command word could be more specific",
                "Missing unit specification in question",
            ],
            "suggested_improvements": [
                "Use 'Find' instead of 'Calculate' for better clarity",
                "Specify units required in the answer",
            ],
            "decision": "needs_revision",
        }

        agent.llm_service.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(quality_response),
                model_used="gpt-4o",
                provider="mock",
                tokens_used=250,
                cost_estimate=0.004,
                latency_ms=2500,
            )
        )

        # WHEN: Processing quality review request
        input_data = {
            "question_data": {
                "question": {
                    "question_text": "Work out the value of x in the equation 2x + 5 = 13",
                    "marks": 2,
                    "command_word": "Work out",
                    "subject_content_references": ["A2.1"],
                    "grade_level": 7,
                    "tier": "Core",
                },
                "marking_scheme": {
                    "total_marks_for_part": 2,
                    "mark_allocation_criteria": [
                        {
                            "criterion_text": "Rearranges equation",
                            "marks_value": 1,
                            "mark_type": "M",
                        },
                        {"criterion_text": "Finds x = 4", "marks_value": 1, "mark_type": "A"},
                    ],
                },
            }
        }

        result = await agent.process(input_data)

        # THEN: Should require manual review or refinement
        assert result.success is True
        assert "quality_decision" in result.output

        quality_decision = result.output["quality_decision"]
        assert quality_decision["quality_score"] == 0.75
        assert quality_decision["action"] in ["manual_review", "refine"]
        assert len(quality_decision["suggested_improvements"]) == 2

    @pytest.mark.asyncio
    async def test_review_agent_low_quality_question(self):
        """Test ReviewAgent correctly identifies low quality questions for rejection"""
        # GIVEN: ReviewAgent with mocked LLM response
        agent = ReviewAgent()

        # Mock low quality assessment response
        quality_response = {
            "overall_quality_score": 0.4,
            "mathematical_accuracy": 0.3,
            "cambridge_compliance": 0.2,
            "grade_appropriateness": 0.5,
            "question_clarity": 0.4,
            "marking_accuracy": 0.6,
            "feedback_summary": "Poor quality question with multiple serious issues",
            "specific_issues": [
                "Mathematical error in the problem setup",
                "Does not align with Cambridge syllabus requirements",
                "Unclear wording and ambiguous requirements",
                "Marking scheme does not match question complexity",
            ],
            "suggested_improvements": [
                "Completely revise the mathematical content",
                "Align with proper Cambridge content references",
                "Rewrite for clarity and precision",
            ],
            "decision": "reject",
        }

        agent.llm_service.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(quality_response),
                model_used="gpt-4o",
                provider="mock",
                tokens_used=300,
                cost_estimate=0.005,
                latency_ms=3000,
            )
        )

        # WHEN: Processing quality review request
        input_data = {
            "question_data": {
                "question": {
                    "question_text": "Solve the equation x² = -4",  # Problematic for IGCSE level
                    "marks": 5,
                    "command_word": "Solve",
                    "subject_content_references": ["A1.1"],  # Wrong reference
                    "grade_level": 7,  # Too advanced for grade 7
                    "tier": "Core",
                },
                "marking_scheme": {
                    "total_marks_for_part": 5,
                    "mark_allocation_criteria": [
                        {
                            "criterion_text": "Recognizes no real solutions",
                            "marks_value": 5,
                            "mark_type": "A",
                        }
                    ],
                },
            }
        }

        result = await agent.process(input_data)

        # THEN: Should reject the question
        assert result.success is True
        assert "quality_decision" in result.output

        quality_decision = result.output["quality_decision"]
        assert quality_decision["quality_score"] == 0.4
        assert quality_decision["action"] in ["reject", "regenerate"]
        assert len(quality_decision["suggested_improvements"]) >= 3
        assert quality_decision["mathematical_accuracy"] < 0.5
        assert quality_decision["cambridge_compliance"] < 0.5

    @pytest.mark.asyncio
    async def test_review_agent_invalid_input_handling(self):
        """Test ReviewAgent handles invalid input gracefully"""
        # GIVEN: ReviewAgent
        agent = ReviewAgent()

        # Invalid input (missing required fields)
        invalid_input = {"invalid_structure": "missing question_data"}

        # WHEN: Processing invalid input
        result = await agent.process(invalid_input)

        # THEN: Should handle error gracefully
        assert result.success is False
        assert result.error is not None
        assert "validation" in result.error.lower() or "question_data" in result.error.lower()

    @pytest.mark.asyncio
    async def test_review_agent_llm_service_failure(self):
        """Test ReviewAgent handles LLM service failures"""
        # GIVEN: ReviewAgent with failing LLM service
        agent = ReviewAgent()

        # Mock LLM service failure
        agent.llm_service.generate = AsyncMock(side_effect=Exception("LLM service unavailable"))

        # Valid input data
        input_data = {
            "question_data": {
                "question": {
                    "question_text": "Calculate 2 + 3",
                    "marks": 1,
                    "command_word": "Calculate",
                    "subject_content_references": ["C1.1"],
                    "grade_level": 7,
                },
                "marking_scheme": {
                    "total_marks_for_part": 1,
                    "mark_allocation_criteria": [
                        {"criterion_text": "Correct answer", "marks_value": 1, "mark_type": "A"}
                    ],
                },
            }
        }

        # WHEN: Processing with failed LLM service
        result = await agent.process(input_data)

        # THEN: Should handle error gracefully
        assert result.success is False
        assert result.error is not None
        assert "Review failed" in result.error or "LLM" in result.error

    @pytest.mark.asyncio
    async def test_review_agent_malformed_json_response(self):
        """Test ReviewAgent handles malformed JSON responses from LLM"""
        # GIVEN: ReviewAgent with malformed JSON response
        agent = ReviewAgent()

        # Mock malformed JSON response
        agent.llm_service.generate = AsyncMock(
            return_value=LLMResponse(
                content="This is not valid JSON at all { malformed",
                model_used="gpt-4o",
                provider="mock",
                tokens_used=100,
                cost_estimate=0.002,
                latency_ms=1500,
            )
        )

        # Valid input data
        input_data = {
            "question_data": {
                "question": {
                    "question_text": "Calculate 2 + 3",
                    "marks": 1,
                    "command_word": "Calculate",
                    "subject_content_references": ["C1.1"],
                    "grade_level": 7,
                },
                "marking_scheme": {
                    "total_marks_for_part": 1,
                    "mark_allocation_criteria": [
                        {"criterion_text": "Correct answer", "marks_value": 1, "mark_type": "A"}
                    ],
                },
            }
        }

        # WHEN: Processing with malformed JSON
        result = await agent.process(input_data)

        # THEN: Should handle parsing error gracefully
        assert result.success is False
        assert result.error is not None
        assert any(keyword in result.error.lower() for keyword in ["json", "parse", "invalid"])

    @pytest.mark.asyncio
    async def test_review_agent_reasoning_steps_tracking(self):
        """Test ReviewAgent tracks reasoning steps for transparency"""
        # GIVEN: ReviewAgent with mocked LLM response
        agent = ReviewAgent()

        # Mock quality assessment response
        quality_response = {
            "overall_quality_score": 0.85,
            "mathematical_accuracy": 0.9,
            "cambridge_compliance": 0.85,
            "grade_appropriateness": 0.8,
            "question_clarity": 0.85,
            "marking_accuracy": 0.85,
            "feedback_summary": "Good quality question",
            "specific_issues": [],
            "suggested_improvements": [],
            "decision": "approve",
        }

        agent.llm_service.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(quality_response),
                model_used="gpt-4o",
                provider="mock",
                tokens_used=150,
                cost_estimate=0.002,
                latency_ms=2000,
            )
        )

        # WHEN: Processing review request
        input_data = {
            "question_data": {
                "question": {
                    "question_text": "Calculate 5 × 7",
                    "marks": 1,
                    "command_word": "Calculate",
                    "subject_content_references": ["C1.2"],
                    "grade_level": 6,
                },
                "marking_scheme": {
                    "total_marks_for_part": 1,
                    "mark_allocation_criteria": [
                        {"criterion_text": "35", "marks_value": 1, "mark_type": "A"}
                    ],
                },
            }
        }

        result = await agent.process(input_data)

        # THEN: Should track reasoning steps
        assert result.success is True
        assert len(result.reasoning_steps) > 0

        # Check for quality assessment specific reasoning
        reasoning_content = " ".join(result.reasoning_steps)
        assert any(
            keyword in reasoning_content.lower()
            for keyword in ["quality", "assess", "review", "score"]
        )

    @pytest.mark.asyncio
    async def test_review_agent_performance_timing(self):
        """Test ReviewAgent completes within reasonable time limits"""
        # GIVEN: ReviewAgent with mocked LLM response
        agent = ReviewAgent()

        # Mock quick quality assessment response
        quality_response = {
            "overall_quality_score": 0.8,
            "mathematical_accuracy": 0.85,
            "cambridge_compliance": 0.8,
            "grade_appropriateness": 0.75,
            "question_clarity": 0.8,
            "marking_accuracy": 0.8,
            "feedback_summary": "Good question",
            "specific_issues": [],
            "suggested_improvements": [],
            "decision": "approve",
        }

        agent.llm_service.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(quality_response),
                model_used="gpt-4o",
                provider="mock",
                tokens_used=120,
                cost_estimate=0.002,
                latency_ms=1000,
            )
        )

        # Simple input data
        input_data = {
            "question_data": {
                "question": {
                    "question_text": "What is 8 + 7?",
                    "marks": 1,
                    "command_word": "Calculate",
                    "subject_content_references": ["C1.1"],
                    "grade_level": 5,
                },
                "marking_scheme": {
                    "total_marks_for_part": 1,
                    "mark_allocation_criteria": [
                        {"criterion_text": "15", "marks_value": 1, "mark_type": "A"}
                    ],
                },
            }
        }

        # WHEN: Processing review request
        result = await agent.process(input_data)

        # THEN: Should complete successfully and track timing
        assert result.success is True
        assert hasattr(result, "processing_time")
        assert result.processing_time is not None
        assert result.processing_time > 0
        # Should complete in reasonable time (allowing for test overhead)
        assert result.processing_time < 30.0  # 30 seconds max for tests

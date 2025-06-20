"""
End-to-end tests for Review Agent.
Tests complete workflows and integration scenarios.
"""

import json
from unittest.mock import AsyncMock

import pytest

from src.agents.review_agent import ReviewAgent


class TestReviewAgentE2E:
    """End-to-end tests for ReviewAgent workflows"""

    @pytest.mark.asyncio
    async def test_complete_quality_assessment_workflow(self):
        """Test complete quality assessment workflow from input to decision"""
        # GIVEN: ReviewAgent with realistic scenarios
        agent = ReviewAgent()

        # Mock realistic quality assessment responses for different scenarios
        test_scenarios = [
            {
                "name": "High Quality - Approve",
                "input": {
                    "question_data": {
                        "question": {
                            "question_text": "A rectangle has length 12 cm and width 8 cm. Calculate its perimeter.",
                            "marks": 3,
                            "command_word": "Calculate",
                            "subject_content_references": ["C5.1"],
                            "grade_level": 7,
                            "tier": "Core",
                        },
                        "marking_scheme": {
                            "total_marks_for_part": 3,
                            "mark_allocation_criteria": [
                                {
                                    "criterion_text": "Uses formula P = 2(l + w)",
                                    "marks_value": 1,
                                    "mark_type": "M",
                                },
                                {
                                    "criterion_text": "Substitutes values correctly",
                                    "marks_value": 1,
                                    "mark_type": "M",
                                },
                                {
                                    "criterion_text": "Calculates P = 40 cm",
                                    "marks_value": 1,
                                    "mark_type": "A",
                                },
                            ],
                            "final_answers": [
                                {"answer_text": "40", "value_numeric": 40, "unit": "cm"}
                            ],
                        },
                    }
                },
                "response": {
                    "overall_quality_score": 0.9,
                    "mathematical_accuracy": 0.95,
                    "cambridge_compliance": 0.9,
                    "grade_appropriateness": 0.9,
                    "question_clarity": 0.85,
                    "marking_accuracy": 0.9,
                    "feedback_summary": "Excellent question with clear instructions and appropriate difficulty",
                    "specific_issues": [],
                    "suggested_improvements": ["Consider adding a diagram for visual support"],
                    "decision": "approve",
                },
                "expected_action": "approve",
            },
            {
                "name": "Medium Quality - Manual Review",
                "input": {
                    "question_data": {
                        "question": {
                            "question_text": "Find x when 3x - 7 = 20",
                            "marks": 2,
                            "command_word": "Find",
                            "subject_content_references": ["A2.1"],
                            "grade_level": 8,
                            "tier": "Core",
                        },
                        "marking_scheme": {
                            "total_marks_for_part": 2,
                            "mark_allocation_criteria": [
                                {
                                    "criterion_text": "Adds 7 to both sides",
                                    "marks_value": 1,
                                    "mark_type": "M",
                                },
                                {
                                    "criterion_text": "Divides by 3 to get x = 9",
                                    "marks_value": 1,
                                    "mark_type": "A",
                                },
                            ],
                        },
                    }
                },
                "response": {
                    "overall_quality_score": 0.75,
                    "mathematical_accuracy": 0.85,
                    "cambridge_compliance": 0.7,
                    "grade_appropriateness": 0.8,
                    "question_clarity": 0.7,
                    "marking_accuracy": 0.7,
                    "feedback_summary": "Adequate question but could benefit from clearer instructions",
                    "specific_issues": [
                        "Could specify to show working",
                        "Marking scheme could be more detailed",
                    ],
                    "suggested_improvements": [
                        "Add 'Show your working' to question text",
                        "Include intermediate steps in marking scheme",
                    ],
                    "decision": "needs_revision",
                },
                "expected_action": "manual_review",
            },
            {
                "name": "Low Quality - Reject",
                "input": {
                    "question_data": {
                        "question": {
                            "question_text": "Solve x² + 1 = 0",  # Complex numbers not in IGCSE
                            "marks": 3,
                            "command_word": "Solve",
                            "subject_content_references": ["A1.1"],  # Wrong reference
                            "grade_level": 9,
                            "tier": "Core",
                        },
                        "marking_scheme": {
                            "total_marks_for_part": 3,
                            "mark_allocation_criteria": [
                                {
                                    "criterion_text": "No real solutions",
                                    "marks_value": 3,
                                    "mark_type": "A",
                                }
                            ],
                        },
                    }
                },
                "response": {
                    "overall_quality_score": 0.3,
                    "mathematical_accuracy": 0.4,
                    "cambridge_compliance": 0.1,
                    "grade_appropriateness": 0.2,
                    "question_clarity": 0.6,
                    "marking_accuracy": 0.5,
                    "feedback_summary": "Question is inappropriate for Cambridge IGCSE level",
                    "specific_issues": [
                        "Complex numbers not covered in IGCSE syllabus",
                        "Too advanced for target grade level",
                        "Incorrect syllabus reference",
                    ],
                    "suggested_improvements": [
                        "Use simpler quadratic with real solutions",
                        "Align with appropriate syllabus content",
                        "Reduce complexity for grade level",
                    ],
                    "decision": "reject",
                },
                "expected_action": "reject",
            },
        ]

        # Test each scenario
        for scenario in test_scenarios:
            # Mock LLM response for this scenario
            agent.llm_service.generate = AsyncMock(
                return_value=type(
                    "MockResponse",
                    (),
                    {
                        "content": json.dumps(scenario["response"]),
                        "model": "gpt-4o",
                        "provider": "mock",
                        "tokens_used": 200,
                        "cost_estimate": 0.003,
                        "generation_time": 2.0,
                    },
                )()
            )

            # WHEN: Processing the scenario
            result = await agent.process(scenario["input"])

            # THEN: Should produce expected results
            assert result.success is True, f"Failed scenario: {scenario['name']}"
            assert "quality_decision" in result.output

            quality_decision = result.output["quality_decision"]
            assert (
                quality_decision["action"] == scenario["expected_action"]
            ), f"Wrong action for {scenario['name']}: expected {scenario['expected_action']}, got {quality_decision['action']}"

            # Verify quality score matches expected range
            expected_score = scenario["response"]["overall_quality_score"]
            assert (
                abs(quality_decision["quality_score"] - expected_score) < 0.01
            ), f"Quality score mismatch for {scenario['name']}"

            # Verify reasoning steps are tracked
            assert len(result.reasoning_steps) > 0, f"No reasoning steps for {scenario['name']}"

    @pytest.mark.asyncio
    async def test_quality_assessment_with_different_grade_levels(self):
        """Test quality assessment across different Cambridge IGCSE grade levels"""
        # GIVEN: ReviewAgent
        agent = ReviewAgent()

        # Test scenarios for different grade levels
        grade_scenarios = [
            {
                "grade": 5,
                "question": "Calculate 15 + 23",
                "marks": 1,
                "expected_appropriateness": 0.95,  # Very appropriate for grade 5
            },
            {
                "grade": 7,
                "question": "Solve the equation 2x + 5 = 13",
                "marks": 2,
                "expected_appropriateness": 0.9,  # Appropriate for grade 7
            },
            {
                "grade": 10,
                "question": "Find the gradient of the line passing through (2, 3) and (5, 9)",
                "marks": 3,
                "expected_appropriateness": 0.85,  # Appropriate for grade 10
            },
        ]

        for scenario in grade_scenarios:
            # Mock response with grade-appropriate assessment
            grade_response = {
                "overall_quality_score": 0.85,
                "mathematical_accuracy": 0.9,
                "cambridge_compliance": 0.85,
                "grade_appropriateness": scenario["expected_appropriateness"],
                "question_clarity": 0.8,
                "marking_accuracy": 0.8,
                "feedback_summary": f"Appropriate for grade {scenario['grade']}",
                "specific_issues": [],
                "suggested_improvements": [],
                "decision": "approve",
            }

            agent.llm_service.generate = AsyncMock(
                return_value=type(
                    "MockResponse",
                    (),
                    {
                        "content": json.dumps(grade_response),
                        "model": "gpt-4o",
                        "provider": "mock",
                        "tokens_used": 180,
                        "cost_estimate": 0.003,
                        "generation_time": 2.0,
                    },
                )()
            )

            # Input data for this grade level
            input_data = {
                "question_data": {
                    "question": {
                        "question_text": scenario["question"],
                        "marks": scenario["marks"],
                        "command_word": "Calculate" if scenario["grade"] <= 6 else "Solve",
                        "subject_content_references": ["C1.1"],
                        "grade_level": scenario["grade"],
                        "tier": "Core",
                    },
                    "marking_scheme": {
                        "total_marks_for_part": scenario["marks"],
                        "mark_allocation_criteria": [
                            {
                                "criterion_text": "Correct answer",
                                "marks_value": scenario["marks"],
                                "mark_type": "A",
                            }
                        ],
                    },
                }
            }

            # WHEN: Processing grade-specific review
            result = await agent.process(input_data)

            # THEN: Should assess grade appropriateness correctly
            assert result.success is True
            quality_decision = result.output["quality_decision"]

            # Check grade appropriateness score
            assert (
                abs(
                    quality_decision["grade_appropriateness"] - scenario["expected_appropriateness"]
                )
                < 0.1
            ), f"Grade appropriateness mismatch for grade {scenario['grade']}"

            # Should approve appropriate questions
            assert quality_decision["action"] == "approve"

    @pytest.mark.asyncio
    async def test_quality_assessment_with_cambridge_compliance_issues(self):
        """Test quality assessment with various Cambridge compliance issues"""
        # GIVEN: ReviewAgent
        agent = ReviewAgent()

        # Test scenarios with compliance issues
        compliance_scenarios = [
            {
                "name": "Wrong Subject Content Reference",
                "question_data": {
                    "question": {
                        "question_text": "Calculate the area of a circle with radius 3 cm",
                        "marks": 2,
                        "command_word": "Calculate",
                        "subject_content_references": ["A1.1"],  # Wrong - should be geometry
                        "grade_level": 8,
                        "tier": "Core",
                    }
                },
                "expected_compliance": 0.4,  # Poor compliance due to wrong reference
            },
            {
                "name": "Inappropriate Command Word",
                "question_data": {
                    "question": {
                        "question_text": "Determine the solution to x + 5 = 12",  # 'Determine' not standard
                        "marks": 1,
                        "command_word": "Determine",  # Non-Cambridge command word
                        "subject_content_references": ["A2.1"],
                        "grade_level": 7,
                        "tier": "Core",
                    }
                },
                "expected_compliance": 0.6,  # Reduced compliance due to command word
            },
            {
                "name": "Good Cambridge Compliance",
                "question_data": {
                    "question": {
                        "question_text": "Find the value of x when 3x = 15",
                        "marks": 2,
                        "command_word": "Find",  # Standard Cambridge command word
                        "subject_content_references": ["A2.1"],  # Correct reference
                        "grade_level": 7,
                        "tier": "Core",
                    }
                },
                "expected_compliance": 0.9,  # Good compliance
            },
        ]

        for scenario in compliance_scenarios:
            # Mock response with appropriate compliance score
            # Overall score reflects compliance level
            overall_score = 0.9 if scenario["expected_compliance"] > 0.8 else 0.7
            compliance_response = {
                "overall_quality_score": overall_score,
                "mathematical_accuracy": 0.85,
                "cambridge_compliance": scenario["expected_compliance"],
                "grade_appropriateness": 0.8,
                "question_clarity": 0.75,
                "marking_accuracy": 0.8,
                "feedback_summary": f"Compliance assessment: {scenario['name']}",
                "specific_issues": []
                if scenario["expected_compliance"] > 0.8
                else ["Cambridge compliance issues detected"],
                "suggested_improvements": []
                if scenario["expected_compliance"] > 0.8
                else ["Align with Cambridge standards"],
                "decision": "approve"
                if scenario["expected_compliance"] > 0.8
                else "needs_revision",
            }

            agent.llm_service.generate = AsyncMock(
                return_value=type(
                    "MockResponse",
                    (),
                    {
                        "content": json.dumps(compliance_response),
                        "model": "gpt-4o",
                        "provider": "mock",
                        "tokens_used": 220,
                        "cost_estimate": 0.004,
                        "generation_time": 2.5,
                    },
                )()
            )

            # Input with marking scheme
            input_data = {
                "question_data": {
                    "question": scenario["question_data"]["question"],
                    "marking_scheme": {
                        "total_marks_for_part": scenario["question_data"]["question"]["marks"],
                        "mark_allocation_criteria": [
                            {
                                "criterion_text": "Correct solution",
                                "marks_value": scenario["question_data"]["question"]["marks"],
                                "mark_type": "A",
                            }
                        ],
                    },
                }
            }

            # WHEN: Processing compliance review
            result = await agent.process(input_data)

            # THEN: Should assess compliance correctly
            assert result.success is True
            quality_decision = result.output["quality_decision"]

            # Check compliance score
            assert (
                abs(quality_decision["cambridge_compliance"] - scenario["expected_compliance"])
                < 0.1
            ), f"Compliance score mismatch for {scenario['name']}"

            # Check appropriate action based on compliance
            if scenario["expected_compliance"] > 0.8:
                # High compliance should result in approve action
                assert (
                    str(quality_decision["action"]) == "approve"
                    or quality_decision["action"].value == "approve"
                )
            else:
                # Low compliance should result in review/refine actions
                action_value = (
                    str(quality_decision["action"])
                    if hasattr(quality_decision["action"], "value")
                    else quality_decision["action"]
                )
                assert action_value in ["manual_review", "refine", "needs_revision"] or (
                    hasattr(quality_decision["action"], "value")
                    and quality_decision["action"].value in ["manual_review", "refine"]
                )

    @pytest.mark.asyncio
    async def test_quality_assessment_error_recovery(self):
        """Test Review Agent error recovery in realistic failure scenarios"""
        # GIVEN: ReviewAgent
        agent = ReviewAgent()

        # Test error recovery scenarios
        error_scenarios = [
            {
                "name": "LLM Timeout",
                "error": Exception("Request timeout after 30 seconds"),
                "expected_recovery": "graceful_failure",
            },
            {
                "name": "Invalid JSON Response",
                "mock_response": "This is not JSON { invalid format",
                "expected_recovery": "json_parse_error",
            },
            {
                "name": "Missing Quality Scores",
                "mock_response": json.dumps({"feedback_summary": "incomplete response"}),
                "expected_recovery": "validation_error",
            },
        ]

        valid_input = {
            "question_data": {
                "question": {
                    "question_text": "Calculate 5 + 3",
                    "marks": 1,
                    "command_word": "Calculate",
                    "subject_content_references": ["C1.1"],
                    "grade_level": 6,
                    "tier": "Core",
                },
                "marking_scheme": {
                    "total_marks_for_part": 1,
                    "mark_allocation_criteria": [
                        {"criterion_text": "8", "marks_value": 1, "mark_type": "A"}
                    ],
                },
            }
        }

        for scenario in error_scenarios:
            if "error" in scenario:
                # Mock service failure
                agent.llm_service.generate = AsyncMock(side_effect=scenario["error"])
            else:
                # Mock invalid response
                agent.llm_service.generate = AsyncMock(
                    return_value=type(
                        "MockResponse",
                        (),
                        {
                            "content": scenario["mock_response"],
                            "model": "gpt-4o",
                            "provider": "mock",
                            "tokens_used": 50,
                            "cost_estimate": 0.001,
                            "generation_time": 1.0,
                        },
                    )()
                )

            # WHEN: Processing with error conditions
            result = await agent.process(valid_input)

            # THEN: Should handle errors gracefully
            assert result.success is False, f"Should fail gracefully for {scenario['name']}"
            assert result.error is not None
            assert len(result.reasoning_steps) > 0  # Should still track reasoning

            # Check error message is informative
            error_message = result.error.lower()
            if scenario["expected_recovery"] == "graceful_failure":
                assert any(
                    keyword in error_message for keyword in ["timeout", "unavailable", "failed"]
                )
            elif scenario["expected_recovery"] == "json_parse_error":
                assert any(keyword in error_message for keyword in ["json", "parse", "format"])
            elif scenario["expected_recovery"] == "validation_error":
                assert any(
                    keyword in error_message for keyword in ["validation", "missing", "incomplete"]
                )

    @pytest.mark.asyncio
    async def test_quality_assessment_performance_e2e(self):
        """Test end-to-end performance of quality assessment workflow"""
        # GIVEN: ReviewAgent with performance monitoring
        agent = ReviewAgent()

        # Mock fast quality assessment
        quick_response = {
            "overall_quality_score": 0.8,
            "mathematical_accuracy": 0.85,
            "cambridge_compliance": 0.8,
            "grade_appropriateness": 0.8,
            "question_clarity": 0.75,
            "marking_accuracy": 0.8,
            "feedback_summary": "Good quality question",
            "specific_issues": [],
            "suggested_improvements": [],
            "decision": "approve",
        }

        agent.llm_service.generate = AsyncMock(
            return_value=type(
                "MockResponse",
                (),
                {
                    "content": json.dumps(quick_response),
                    "model": "gpt-4o",
                    "provider": "mock",
                    "tokens_used": 150,
                    "cost_estimate": 0.002,
                    "generation_time": 1.5,
                },
            )()
        )

        # Batch of questions to test performance
        test_questions = [
            "Calculate 7 × 8",
            "Find the area of a square with side 6 cm",
            "Solve x + 12 = 20",
            "What is 15% of 80?",
            "Find the perimeter of a rectangle 5 cm by 3 cm",
        ]

        performance_results = []

        for i, question_text in enumerate(test_questions):
            input_data = {
                "question_data": {
                    "question": {
                        "question_text": question_text,
                        "marks": 2,
                        "command_word": "Calculate" if "Calculate" in question_text else "Find",
                        "subject_content_references": ["C1.1"],
                        "grade_level": 7,
                        "tier": "Core",
                    },
                    "marking_scheme": {
                        "total_marks_for_part": 2,
                        "mark_allocation_criteria": [
                            {
                                "criterion_text": "Correct method",
                                "marks_value": 1,
                                "mark_type": "M",
                            },
                            {
                                "criterion_text": "Correct answer",
                                "marks_value": 1,
                                "mark_type": "A",
                            },
                        ],
                    },
                }
            }

            # WHEN: Processing question
            result = await agent.process(input_data)

            # THEN: Should complete successfully and quickly
            assert result.success is True
            assert result.processing_time is not None
            assert result.processing_time < 10.0  # Should complete within 10 seconds

            performance_results.append(
                {
                    "question_number": i + 1,
                    "processing_time": result.processing_time,
                    "quality_score": result.output["quality_decision"]["quality_score"],
                }
            )

        # Check overall performance
        average_time = sum(r["processing_time"] for r in performance_results) / len(
            performance_results
        )
        assert average_time < 5.0, f"Average processing time too slow: {average_time}s"

        # All questions should be processed successfully
        assert len(performance_results) == len(test_questions)
        assert all(r["quality_score"] > 0 for r in performance_results)

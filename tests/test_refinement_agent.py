"""
Integration tests for Refinement Agent.
Tests end-to-end functionality with mock services.
"""

import json
from unittest.mock import AsyncMock

import pytest

from src.agents.refinement_agent import RefinementAgent
from src.services.llm_service import LLMResponse


class TestRefinementAgent:
    """Integration tests for RefinementAgent"""

    @pytest.mark.asyncio
    async def test_refinement_agent_improves_low_quality_question(self):
        """Test RefinementAgent successfully improves a low quality question"""
        # GIVEN: RefinementAgent with mocked LLM response
        agent = RefinementAgent()

        # Mock successful refinement response
        refinement_response = {
            "question_text": "Calculate the total cost when buying 3 notebooks at £2.50 each and 2 pens at £1.25 each. Show your working and give your answer in pounds.",
            "marks": 4,
            "command_word": "Calculate",
            "grade_level": 7,
            "improvements_made": [
                "Added real-world context with multiple items",
                "Increased complexity with mixed calculations",
                "Specified format for answer (pounds)",
                "Added instruction to show working",
                "Increased marks to reflect complexity",
            ],
            "marking_scheme": {
                "total_marks_for_part": 4,
                "mark_allocation_criteria": [
                    {
                        "criterion_text": "Calculates cost of notebooks: 3 × £2.50 = £7.50",
                        "marks_value": 1,
                        "mark_type": "M",
                    },
                    {
                        "criterion_text": "Calculates cost of pens: 2 × £1.25 = £2.50",
                        "marks_value": 1,
                        "mark_type": "M",
                    },
                    {
                        "criterion_text": "Adds totals: £7.50 + £2.50",
                        "marks_value": 1,
                        "mark_type": "M",
                    },
                    {"criterion_text": "Final answer: £10.00", "marks_value": 1, "mark_type": "A"},
                ],
            },
        }

        agent.llm_service.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(refinement_response),
                model_used="gpt-4o",
                provider="mock",
                tokens_used=350,
                cost_estimate=0.005,
                latency_ms=2500,
            )
        )

        # Original question data with quality issues
        input_data = {
            "original_question": {
                "question_text": "Calculate 3 + 2",
                "marks": 1,
                "command_word": "Calculate",
                "grade_level": 7,
                "subject_content_references": ["C1.1"],
            },
            "quality_decision": {
                "action": "refine",
                "quality_score": 0.55,
                "suggested_improvements": [
                    "Question too simple for grade 7",
                    "Add real-world context",
                    "Increase complexity to match grade level",
                ],
                "reasoning": "Question is too basic for the target grade level",
            },
        }

        # WHEN: Processing refinement request
        result = await agent.process(input_data)

        # THEN: Should successfully refine the question
        assert result.success is True
        assert "refined_question" in result.output
        assert "refinement_metadata" in result.output

        refined_question = result.output["refined_question"]
        assert refined_question["marks"] > 1  # Should increase complexity
        assert "notebook" in refined_question["question_text"].lower()  # Real-world context
        assert refined_question["grade_level"] == 7  # Preserved

        metadata = result.output["refinement_metadata"]
        assert metadata["original_quality_score"] == 0.55
        assert len(metadata["improvements_made"]) >= 3

    @pytest.mark.asyncio
    async def test_refinement_agent_handles_cambridge_compliance_issues(self):
        """Test RefinementAgent addresses Cambridge compliance issues"""
        # GIVEN: RefinementAgent with mocked LLM response
        agent = RefinementAgent()

        # Mock Cambridge compliance improvement response
        compliance_response = {
            "question_text": "Find the value of x when 2x + 5 = 13. Show all your working.",
            "marks": 3,
            "command_word": "Find",  # Changed from non-standard "Determine"
            "grade_level": 8,
            "subject_content_references": ["A2.1"],  # Correct reference
            "improvements_made": [
                "Changed command word from 'Determine' to 'Find' (Cambridge standard)",
                "Corrected subject content reference to A2.1 (Linear equations)",
                "Added 'Show all your working' instruction",
                "Increased marks to 3 to reflect working requirement",
            ],
        }

        agent.llm_service.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(compliance_response),
                model_used="gpt-4o",
                provider="mock",
                tokens_used=280,
                cost_estimate=0.004,
                latency_ms=2000,
            )
        )

        # Original question with Cambridge compliance issues
        input_data = {
            "original_question": {
                "question_text": "Determine x when 2x + 5 = 13",
                "marks": 2,
                "command_word": "Determine",  # Non-Cambridge command word
                "grade_level": 8,
                "subject_content_references": ["G1.1"],  # Wrong reference (geometry)
            },
            "quality_decision": {
                "action": "refine",
                "quality_score": 0.65,
                "cambridge_compliance": 0.4,  # Low compliance
                "suggested_improvements": [
                    "Use standard Cambridge command words",
                    "Correct subject content reference for algebra",
                    "Add working instruction for marks allocation",
                ],
            },
        }

        # WHEN: Processing Cambridge compliance refinement
        result = await agent.process(input_data)

        # THEN: Should fix Cambridge compliance issues
        assert result.success is True

        refined_question = result.output["refined_question"]
        assert refined_question["command_word"] == "Find"  # Fixed command word
        assert "A2.1" in refined_question["subject_content_references"]  # Fixed reference
        assert "working" in refined_question["question_text"].lower()  # Added instruction

    @pytest.mark.asyncio
    async def test_refinement_agent_fallback_strategy(self):
        """Test RefinementAgent uses fallback when primary approach fails"""
        # GIVEN: RefinementAgent with failing primary and successful fallback
        agent = RefinementAgent()

        # Mock fallback response (simpler format)
        fallback_response = {
            "question_text": "Calculate the perimeter of a rectangle with length 8 cm and width 5 cm. Show your method.",
            "marks": 2,
            "improvements_made": [
                "Added real-world geometric context",
                "Specified measurement units",
                "Added instruction to show method",
            ],
        }

        # First call fails (primary), second succeeds (fallback)
        call_count = 0

        def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Primary attempt - malformed JSON
                return LLMResponse(
                    content="This is not valid JSON { malformed response",
                    model_used="gpt-4o",
                    provider="mock",
                    tokens_used=100,
                    cost_estimate=0.002,
                    latency_ms=1500,
                )
            else:
                # Fallback attempt - valid JSON
                return LLMResponse(
                    content=json.dumps(fallback_response),
                    model_used="gpt-4o",
                    provider="mock",
                    tokens_used=200,
                    cost_estimate=0.003,
                    latency_ms=2000,
                )

        agent.llm_service.generate = AsyncMock(side_effect=mock_generate)

        # Input data requiring refinement
        input_data = {
            "original_question": {
                "question_text": "Calculate perimeter",
                "marks": 1,
                "command_word": "Calculate",
                "grade_level": 6,
            },
            "quality_decision": {
                "action": "refine",
                "quality_score": 0.45,
                "suggested_improvements": ["Add context", "Specify units", "Increase clarity"],
            },
        }

        # WHEN: Processing with fallback needed
        result = await agent.process(input_data)

        # THEN: Should succeed using fallback strategy
        assert result.success is True
        assert "refined_question" in result.output

        refined_question = result.output["refined_question"]
        assert "rectangle" in refined_question["question_text"]
        assert refined_question["marks"] == 2

        # Should track fallback usage in metadata
        metadata = result.output["refinement_metadata"]
        assert metadata["strategy_used"] == "fallback"

    @pytest.mark.asyncio
    async def test_refinement_agent_preserves_essential_structure(self):
        """Test RefinementAgent preserves essential question structure during refinement"""
        # GIVEN: RefinementAgent with mocked LLM response
        agent = RefinementAgent()

        # Mock refinement response that preserves structure
        structure_response = {
            "question_text": "A triangle has sides of length 5 cm, 12 cm, and 13 cm. Prove that this is a right-angled triangle using Pythagoras' theorem. Show all your working.",
            "marks": 4,
            "command_word": "Prove",  # Preserved from original
            "grade_level": 9,  # Preserved from original
            "subject_content_references": ["G3.2"],  # Preserved original geometry reference
            "improvements_made": [
                "Added specific measurements for concrete calculation",
                "Specified method to use (Pythagoras' theorem)",
                "Added working instruction",
                "Increased marks to reflect proof requirement",
            ],
        }

        agent.llm_service.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(structure_response),
                model_used="gpt-4o",
                provider="mock",
                tokens_used=320,
                cost_estimate=0.004,
                latency_ms=2200,
            )
        )

        # Original question with specific constraints that should be preserved
        input_data = {
            "original_question": {
                "question_text": "Prove a triangle is right-angled",
                "marks": 2,
                "command_word": "Prove",  # Should be preserved
                "grade_level": 9,  # Should be preserved
                "tier": "Extended",  # Should be preserved
                "subject_content_references": ["G3.2"],  # Should be preserved
            },
            "quality_decision": {
                "action": "refine",
                "quality_score": 0.6,
                "suggested_improvements": [
                    "Add specific triangle measurements",
                    "Specify proof method",
                    "Add working instruction",
                ],
            },
        }

        # WHEN: Processing refinement with structure preservation
        result = await agent.process(input_data)

        # THEN: Should preserve essential structure while improving content
        assert result.success is True

        refined_question = result.output["refined_question"]
        assert refined_question["command_word"] == "Prove"  # Preserved
        assert refined_question["grade_level"] == 9  # Preserved
        assert refined_question["tier"] == "Extended"  # Preserved
        assert "G3.2" in refined_question["subject_content_references"]  # Preserved

        # But should have improvements
        assert "5 cm" in refined_question["question_text"]  # Added specifics
        assert "Pythagoras" in refined_question["question_text"]  # Added method
        assert refined_question["marks"] > 2  # Increased complexity

    @pytest.mark.asyncio
    async def test_refinement_agent_handles_multiple_improvement_areas(self):
        """Test RefinementAgent addresses multiple improvement areas simultaneously"""
        # GIVEN: RefinementAgent with comprehensive refinement response
        agent = RefinementAgent()

        # Mock comprehensive improvement response
        comprehensive_response = {
            "question_text": "Sarah is planning a rectangular garden. She wants the length to be 3 metres more than the width. If the perimeter must be exactly 26 metres, find the dimensions of the garden. Show your working clearly and check your answer.",
            "marks": 6,
            "command_word": "Find",
            "grade_level": 9,
            "subject_content_references": ["A4.1", "A2.3"],
            "improvements_made": [
                "Added real-world context (garden planning)",
                "Incorporated proper algebraic setup with constraints",
                "Increased marks to reflect multi-step problem",
                "Added checking requirement for mathematical verification",
                "Improved Cambridge compliance with proper command word",
                "Aligned difficulty with grade 9 standards",
                "Added clear working instruction",
            ],
        }

        agent.llm_service.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(comprehensive_response),
                model_used="gpt-4o",
                provider="mock",
                tokens_used=400,
                cost_estimate=0.006,
                latency_ms=3000,
            )
        )

        # Original question with multiple issues
        input_data = {
            "original_question": {
                "question_text": "Solve x + 3 = 10",
                "marks": 1,
                "command_word": "Solve",
                "grade_level": 9,
                "subject_content_references": ["A1.1"],
            },
            "quality_decision": {
                "action": "refine",
                "quality_score": 0.5,
                "mathematical_accuracy": 0.9,  # Good
                "cambridge_compliance": 0.7,  # Okay
                "grade_appropriateness": 0.2,  # Very poor - too easy
                "question_clarity": 0.6,  # Needs improvement
                "suggested_improvements": [
                    "Increase complexity for grade 9",
                    "Add real-world application",
                    "Create multi-step problem",
                    "Improve question clarity and context",
                    "Add verification requirement",
                ],
            },
        }

        # WHEN: Processing comprehensive refinement
        result = await agent.process(input_data)

        # THEN: Should address all improvement areas
        assert result.success is True

        refined_question = result.output["refined_question"]

        # Grade appropriateness improvement
        assert refined_question["marks"] > 3  # Much more complex
        assert "perimeter" in refined_question["question_text"]  # Multi-step algebra

        # Real-world context improvement
        assert "garden" in refined_question["question_text"]  # Real context

        # Clarity improvement
        assert "show your working" in refined_question["question_text"].lower()
        assert "check" in refined_question["question_text"].lower()

        # Cambridge compliance
        assert refined_question["command_word"] == "Find"

        metadata = result.output["refinement_metadata"]
        assert len(metadata["improvements_made"]) >= 5  # Multiple improvements

    @pytest.mark.asyncio
    async def test_refinement_agent_invalid_input_handling(self):
        """Test RefinementAgent handles invalid input gracefully"""
        # GIVEN: RefinementAgent
        agent = RefinementAgent()

        # Invalid input (missing required fields)
        invalid_input = {"invalid_structure": "missing original_question and quality_decision"}

        # WHEN: Processing invalid input
        result = await agent.process(invalid_input)

        # THEN: Should handle error gracefully
        assert result.success is False
        assert result.error is not None
        assert "validation" in result.error.lower() or "original_question" in result.error.lower()

    @pytest.mark.asyncio
    async def test_refinement_agent_llm_service_failure(self):
        """Test RefinementAgent handles LLM service failures"""
        # GIVEN: RefinementAgent with failing LLM service
        agent = RefinementAgent()

        # Mock LLM service failure
        agent.llm_service.generate = AsyncMock(side_effect=Exception("LLM service unavailable"))

        # Valid input data
        input_data = {
            "original_question": {
                "question_text": "Calculate 2 + 3",
                "marks": 1,
                "command_word": "Calculate",
                "grade_level": 6,
            },
            "quality_decision": {
                "action": "refine",
                "quality_score": 0.5,
                "suggested_improvements": ["Add context"],
            },
        }

        # WHEN: Processing with failed LLM service
        result = await agent.process(input_data)

        # THEN: Should handle error gracefully
        assert result.success is False
        assert result.error is not None
        assert "Refinement failed" in result.error or "LLM" in result.error

    @pytest.mark.asyncio
    async def test_refinement_agent_reasoning_steps_tracking(self):
        """Test RefinementAgent tracks reasoning steps for transparency"""
        # GIVEN: RefinementAgent with mocked LLM response
        agent = RefinementAgent()

        # Mock refinement response
        refinement_response = {
            "question_text": "Calculate the cost of 4 apples at 25p each. Give your answer in pence.",
            "marks": 2,
            "command_word": "Calculate",
            "grade_level": 5,
            "improvements_made": ["Added real-world context", "Specified units"],
        }

        agent.llm_service.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(refinement_response),
                model_used="gpt-4o",
                provider="mock",
                tokens_used=200,
                cost_estimate=0.003,
                latency_ms=1800,
            )
        )

        # Valid input data
        input_data = {
            "original_question": {
                "question_text": "Calculate 4 × 25",
                "marks": 1,
                "command_word": "Calculate",
                "grade_level": 5,
            },
            "quality_decision": {
                "action": "refine",
                "quality_score": 0.6,
                "suggested_improvements": ["Add context", "Specify units"],
            },
        }

        # WHEN: Processing refinement request
        result = await agent.process(input_data)

        # THEN: Should track reasoning steps
        assert result.success is True
        assert len(result.reasoning_steps) > 0

        # Check for refinement-specific reasoning
        reasoning_content = " ".join(result.reasoning_steps)
        assert any(
            keyword in reasoning_content.lower()
            for keyword in ["refine", "improve", "enhance", "strategy"]
        )

    @pytest.mark.asyncio
    async def test_refinement_agent_performance_timing(self):
        """Test RefinementAgent completes within reasonable time limits"""
        # GIVEN: RefinementAgent with mocked LLM response
        agent = RefinementAgent()

        # Mock quick refinement response
        quick_response = {
            "question_text": "Calculate 6 × 8 showing your multiplication method",
            "marks": 2,
            "command_word": "Calculate",
            "grade_level": 6,
            "improvements_made": ["Added method instruction", "Increased marks"],
        }

        agent.llm_service.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(quick_response),
                model_used="gpt-4o",
                provider="mock",
                tokens_used=150,
                cost_estimate=0.002,
                latency_ms=1200,
            )
        )

        # Simple input data
        input_data = {
            "original_question": {
                "question_text": "Calculate 6 × 8",
                "marks": 1,
                "command_word": "Calculate",
                "grade_level": 6,
            },
            "quality_decision": {
                "action": "refine",
                "quality_score": 0.65,
                "suggested_improvements": ["Add method instruction"],
            },
        }

        # WHEN: Processing refinement request
        result = await agent.process(input_data)

        # THEN: Should complete successfully and track timing
        assert result.success is True
        assert hasattr(result, "processing_time")
        assert result.processing_time is not None
        assert result.processing_time > 0
        # Should complete in reasonable time (allowing for test overhead)
        assert result.processing_time < 30.0  # 30 seconds max for tests

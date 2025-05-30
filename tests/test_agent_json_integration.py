#!/usr/bin/env python3
"""
Test Agent JSON Integration - Testing robust JSON parser integration with agents.

Tests that agents can handle various LLM response formats using the robust parser.
"""

import pytest
import json
import sys
import os
from unittest.mock import Mock, MagicMock

# Add project root to Python path for clean imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.refinement_agent import RefinementAgent
from src.agents.review_agent import ReviewAgent
from src.agents.question_generator import QuestionGeneratorAgent
from src.agents.marker_agent import MarkerAgent
from src.models.question_models import CandidateQuestion, CommandWord, LLMModel
from src.models import QuestionTaxonomy, SolutionAndMarkingScheme, SolverAlgorithm, AnswerSummary, MarkAllocationCriterion, SolverStep
import uuid


class TestAgentJSONIntegration:
    """Test that agents properly handle various JSON response formats."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        mock = Mock()
        mock.__str__ = Mock(return_value="mock-model")
        return mock

    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock config manager."""
        return Mock()

    @pytest.fixture
    def sample_question(self):
        """Create a sample question for testing."""
        # Create minimal required objects
        taxonomy = QuestionTaxonomy(
            topic_path=["Mathematics", "Geometry", "Circles"],
            subject_content_references=["C5.3"],  # Actual syllabus code: Circles, arcs and sectors
            skill_tags=["AREA_CALCULATION", "CIRCLE_PROPERTIES", "FORMULA_APPLICATION"],
            cognitive_level="Application"
        )

        marking_scheme = SolutionAndMarkingScheme(
            final_answers_summary=[
                AnswerSummary(
                    answer_text="78.54 cm²",
                    value_numeric=78.54,
                    unit="cm²"
                )
            ],
            mark_allocation_criteria=[
                MarkAllocationCriterion(
                    criterion_id="M1",
                    criterion_text="Use correct formula",
                    mark_code_display="M1",
                    marks_value=1.0,
                    mark_type_primary="M",
                    qualifiers_and_notes="oe"
                )
            ],
            total_marks_for_part=3
        )

        solver = SolverAlgorithm(
            steps=[
                SolverStep(
                    step_number=1,
                    description_text="Apply formula πr²",
                    mathematical_expression_latex="A = \\pi r^2",
                    skill_applied_tag="FORMULA_APPLICATION"
                ),
                SolverStep(
                    step_number=2,
                    description_text="Substitute r=5",
                    mathematical_expression_latex="A = \\pi \\times 5^2",
                    skill_applied_tag="SUBSTITUTION"
                ),
                SolverStep(
                    step_number=3,
                    description_text="Calculate result",
                    mathematical_expression_latex="A = 78.54 \\text{ cm}^2",
                    skill_applied_tag="NUMERICAL_CALCULATION"
                )
            ]
        )

        return CandidateQuestion(
            question_id_local=str(uuid.uuid4()),
            question_id_global=str(uuid.uuid4()),
            question_number_display="1",
            marks=3,
            command_word=CommandWord.CALCULATE,
            raw_text_content="Calculate area of circle radius 5",
            formatted_text_latex=None,
            taxonomy=taxonomy,
            solution_and_marking_scheme=marking_scheme,
            solver_algorithm=solver,
            generation_id=uuid.uuid4(),
            target_grade_input=8,
            llm_model_used_generation=LLMModel.GPT_4O.value,
            llm_model_used_marking_scheme=LLMModel.GPT_4O.value,
            prompt_template_version_generation="v1.0",
            prompt_template_version_marking_scheme="v1.0"
        )

    def test_refinement_agent_json_formats(self, mock_model, mock_config_manager, sample_question):
        """Test refinement agent with various JSON formats."""

        agent = RefinementAgent(mock_model, mock_config_manager)
        agent.prompt_loader = Mock()
        agent.prompt_loader.format_refinement_prompt = Mock(return_value="test prompt")

        # Test data
        simple_data = {
            "question_text": "Improved question text",
            "marks": 3
        }

        # Test formats
        test_formats = [
            # JSON in code block
            f"```json\n{json.dumps(simple_data)}\n```",
            # Raw JSON
            json.dumps(simple_data),
            # JSON with explanation
            f"Here's the improved question:\n{json.dumps(simple_data)}\nLooks good!",
            # JSON with trailing comma (malformed)
            '{"question_text": "Fixed question", "marks": 3,}',
            # JSON with single quotes
            "{'question_text': 'Fixed question', 'marks': 3}",
        ]

        for i, test_format in enumerate(test_formats):
            # Test extraction
            json_str = agent._extract_json_from_response(test_format)

            assert json_str is not None, f"Format {i+1} failed to extract JSON"

            # Verify we can parse the extracted JSON
            try:
                data = json.loads(json_str)
                assert "question_text" in data or "raw_text_content" in data, f"Format {i+1} missing expected fields"
            except json.JSONDecodeError:
                pytest.fail(f"Format {i+1} extracted invalid JSON: {json_str}")

    def test_review_agent_json_formats(self, mock_model, mock_config_manager):
        """Test review agent with various JSON formats."""

        agent = ReviewAgent(mock_model, mock_config_manager, debug=False)

        # Test data
        review_data = {
            "outcome": "approve",
            "overall_score": 0.85,
            "feedback_summary": "Good question",
            "specific_feedback": {},
            "suggested_improvements": [],
            "syllabus_compliance": 0.8,
            "difficulty_alignment": 0.9,
            "marking_quality": 0.8
        }

        # Test formats
        test_formats = [
            # JSON in code block
            f"```json\n{json.dumps(review_data)}\n```",
            # Raw JSON
            json.dumps(review_data),
            # JSON with explanation
            f"Review completed:\n{json.dumps(review_data)}\nEnd of review.",
            # Malformed JSON with trailing comma
            '''{
                "outcome": "approve",
                "overall_score": 0.85,
                "feedback_summary": "Good question",
            }''',
        ]

        for i, test_format in enumerate(test_formats):
            # Test extraction
            json_str = agent._extract_json_from_response(test_format)

            assert json_str is not None, f"Format {i+1} failed to extract JSON"

            # Verify we can parse the extracted JSON
            try:
                data = json.loads(json_str)
                # Should have at least some expected fields
                assert any(key in data for key in ["outcome", "overall_score", "feedback_summary"]), \
                    f"Format {i+1} missing expected review fields"
            except json.JSONDecodeError:
                pytest.fail(f"Format {i+1} extracted invalid JSON: {json_str}")

    def test_marker_agent_json_formats(self, mock_model, mock_config_manager):
        """Test marker agent with various JSON formats."""

        agent = MarkerAgent(mock_model, mock_config_manager, debug=False)

        # Test data
        marking_data = {
            "final_answers_summary": [
                {
                    "answer_text": "78.54 cm²",
                    "value_numeric": 78.54,
                    "unit": "cm²"
                }
            ],
            "mark_allocation_criteria": [
                {
                    "criterion_id": "M1",
                    "criterion_text": "Correct formula",
                    "mark_code_display": "M1",
                    "marks_value": 1.0,
                    "mark_type_primary": "M"
                }
            ],
            "total_marks_for_part": 3
        }

        # Test formats
        test_formats = [
            # JSON in code block
            f"```json\n{json.dumps(marking_data)}\n```",
            # Raw JSON
            json.dumps(marking_data),
            # JSON with explanation
            f"Here's the marking scheme:\n{json.dumps(marking_data)}\nEnd of scheme.",
            # Malformed JSON with trailing comma
            '''{
                "final_answers_summary": [],
                "mark_allocation_criteria": [],
                "total_marks_for_part": 3,
            }''',
        ]

        for i, test_format in enumerate(test_formats):
            # Test extraction
            json_str = agent._extract_json_from_response(test_format)

            assert json_str is not None, f"Format {i+1} failed to extract JSON"

            # Verify we can parse the extracted JSON
            try:
                data = json.loads(json_str)
                # Should have expected marking scheme fields
                assert any(key in data for key in ["final_answers_summary", "mark_allocation_criteria", "total_marks_for_part"]), \
                    f"Format {i+1} missing expected marking fields"
            except json.JSONDecodeError:
                pytest.fail(f"Format {i+1} extracted invalid JSON: {json_str}")

    def test_question_generator_json_formats(self, mock_model, mock_config_manager):
        """Test question generator with various JSON formats."""

        agent = QuestionGeneratorAgent(mock_model, mock_config_manager, debug=False)

        # Test data
        question_data = {
            "question_id_local": "Test_Q123",
            "raw_text_content": "What is 2+2?",
            "marks": 1,
            "command_word": "Calculate"
        }

        # Test formats
        test_formats = [
            # JSON in code block
            f"```json\n{json.dumps(question_data)}\n```",
            # Raw JSON
            json.dumps(question_data),
            # JSON with thinking tokens
            f"<think>Let me create a question</think>\n{json.dumps(question_data)}",
            # JSON with trailing comma
            '''{
                "question_id_local": "Test_Q123",
                "raw_text_content": "What is 2+2?",
                "marks": 1,
            }''',
        ]

        for i, test_format in enumerate(test_formats):
            # Test extraction
            result = agent._parse_llm_response(test_format)

            assert result is not None, f"Format {i+1} failed to parse response"
            assert "raw_text_content" in result or "question_text" in result, \
                f"Format {i+1} missing question content"

    def test_robust_parser_edge_cases(self):
        """Test edge cases that might occur in production."""

        from src.utils.json_parser import extract_json_robust

        # Edge cases
        edge_cases = [
            # Empty response
            "",
            # No JSON
            "This is just text with no JSON at all",
            # Multiple JSON objects
            '{"first": "object"} some text {"second": "object"}',
            # Nested JSON as string
            '{"data": "{\\"nested\\": \\"json\\"}"}',
            # JSON with unicode
            '{"text": "Calculate π × r² = area"}',
            # Broken JSON with unmatched braces
            '{"incomplete": "json"',
            # JSON with comments (invalid)
            '''{
                // This is a comment
                "key": "value"
            }''',
        ]

        for i, case in enumerate(edge_cases):
            result = extract_json_robust(case)
            # Should not crash and should return a proper result
            assert hasattr(result, 'success'), f"Edge case {i+1} didn't return proper result"
            assert hasattr(result, 'error'), f"Edge case {i+1} missing error field"

    def test_json_extraction_performance(self):
        """Test that JSON extraction doesn't take too long on large responses."""

        from src.utils.json_parser import extract_json_robust
        import time

        # Create a large response with JSON at the end
        large_text = "This is a very long response. " * 1000
        json_data = {"question": "test", "marks": 5}
        large_response = large_text + f"```json\n{json.dumps(json_data)}\n```"

        start_time = time.time()
        result = extract_json_robust(large_response)
        end_time = time.time()

        # Should complete in reasonable time (< 1 second)
        assert (end_time - start_time) < 1.0, "JSON extraction took too long"
        assert result.success == True, "Failed to extract from large response"
        assert result.data == json_data, "Extracted wrong data"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

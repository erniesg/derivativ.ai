"""
Unit tests for Review Agent.
Tests individual methods and core business logic in isolation.
"""


import pytest

from src.agents.review_agent import ReviewAgent
from src.models.enums import QualityAction
from src.models.question_models import QualityDecision


class TestReviewAgentUnit:
    """Unit tests for ReviewAgent business logic"""

    def test_agent_initialization_with_defaults(self):
        """Test ReviewAgent initializes with default services"""
        # WHEN: Creating agent with defaults
        agent = ReviewAgent()

        # THEN: Should initialize with proper services
        assert agent.name == "Review"
        assert agent.llm_service is not None
        assert agent.prompt_manager is not None
        assert agent.json_parser is not None

    def test_parse_review_request_valid_input(self):
        """Test parsing valid review request data"""
        # GIVEN: ReviewAgent
        agent = ReviewAgent()

        # Sample input data
        input_data = {
            "question_data": {
                "question": {
                    "question_text": "Calculate 2 + 3",
                    "marks": 2,
                    "command_word": "Calculate",
                    "subject_content_references": ["C1.1"],
                    "grade_level": 7,
                },
                "marking_scheme": {
                    "total_marks_for_part": 2,
                    "mark_allocation_criteria": [
                        {
                            "criterion_text": "Correct calculation",
                            "marks_value": 2,
                            "mark_type": "A",
                        }
                    ],
                },
            }
        }

        # WHEN: Parsing request
        question_data = agent._parse_review_request(input_data)

        # THEN: Should extract question data correctly
        assert question_data is not None
        assert "question" in question_data
        assert "marking_scheme" in question_data
        assert question_data["question"]["marks"] == 2

    def test_parse_review_request_invalid_input(self):
        """Test parsing invalid review request raises appropriate error"""
        # GIVEN: ReviewAgent
        agent = ReviewAgent()

        # Invalid input data (missing question_data)
        input_data = {"invalid": "data"}

        # WHEN/THEN: Should raise validation error
        with pytest.raises(ValueError, match="question_data"):
            agent._parse_review_request(input_data)

    def test_calculate_overall_quality_score(self):
        """Test overall quality score calculation from dimension scores"""
        # GIVEN: ReviewAgent
        agent = ReviewAgent()

        # Sample dimension scores
        dimension_scores = {
            "mathematical_accuracy": 0.9,
            "cambridge_compliance": 0.8,
            "grade_appropriateness": 0.85,
            "question_clarity": 0.9,
            "marking_accuracy": 0.8,
        }

        # WHEN: Calculating overall score
        overall_score = agent._calculate_overall_quality_score(dimension_scores)

        # THEN: Should return weighted average
        assert 0.0 <= overall_score <= 1.0
        assert abs(overall_score - 0.85) < 0.05  # Expected weighted average

    def test_determine_quality_action_high_score(self):
        """Test quality action determination for high scores"""
        # GIVEN: ReviewAgent
        agent = ReviewAgent()

        # High quality score
        quality_score = 0.90

        # WHEN: Determining action
        action = agent._determine_quality_action(quality_score)

        # THEN: Should approve
        assert action == QualityAction.APPROVE

    def test_determine_quality_action_medium_score(self):
        """Test quality action determination for medium scores"""
        # GIVEN: ReviewAgent
        agent = ReviewAgent()

        # Medium quality score
        quality_score = 0.75

        # WHEN: Determining action
        action = agent._determine_quality_action(quality_score)

        # THEN: Should require manual review
        assert action == QualityAction.MANUAL_REVIEW

    def test_determine_quality_action_low_score(self):
        """Test quality action determination for low scores"""
        # GIVEN: ReviewAgent
        agent = ReviewAgent()

        # Low quality score
        quality_score = 0.45

        # WHEN: Determining action
        action = agent._determine_quality_action(quality_score)

        # THEN: Should reject or require regeneration
        assert action in [QualityAction.REJECT, QualityAction.REGENERATE]

    def test_convert_to_quality_decision_object(self):
        """Test conversion of LLM response to QualityDecision object"""
        # GIVEN: ReviewAgent
        agent = ReviewAgent()

        # Sample LLM response data
        llm_response_data = {
            "overall_quality_score": 0.85,
            "mathematical_accuracy": 0.9,
            "cambridge_compliance": 0.8,
            "grade_appropriateness": 0.9,
            "question_clarity": 0.85,
            "marking_accuracy": 0.8,
            "feedback_summary": "Good quality question with minor improvements needed",
            "specific_issues": ["Consider adding more challenging sub-parts"],
            "suggested_improvements": ["Add part (b) for extension"],
            "decision": "approve",
        }

        # WHEN: Converting to QualityDecision object
        quality_decision = agent._convert_to_quality_decision_object(llm_response_data)

        # THEN: Should create valid QualityDecision
        assert isinstance(quality_decision, QualityDecision)
        assert quality_decision.quality_score == 0.85
        assert quality_decision.action == QualityAction.APPROVE
        assert quality_decision.mathematical_accuracy == 0.9
        assert quality_decision.cambridge_compliance == 0.8
        assert len(quality_decision.suggested_improvements) == 1

    def test_validate_quality_scores_valid_range(self):
        """Test validation of quality scores within valid range"""
        # GIVEN: ReviewAgent
        agent = ReviewAgent()

        # Valid quality scores
        scores = {
            "overall_quality_score": 0.85,
            "mathematical_accuracy": 0.9,
            "cambridge_compliance": 0.8,
        }

        # WHEN: Validating scores
        is_valid = agent._validate_quality_scores(scores)

        # THEN: Should be valid
        assert is_valid is True

    def test_validate_quality_scores_invalid_range(self):
        """Test validation rejects scores outside valid range"""
        # GIVEN: ReviewAgent
        agent = ReviewAgent()

        # Invalid quality scores (outside 0-1 range)
        scores = {
            "overall_quality_score": 1.5,  # Invalid
            "mathematical_accuracy": -0.1,  # Invalid
            "cambridge_compliance": 0.8,
        }

        # WHEN: Validating scores
        is_valid = agent._validate_quality_scores(scores)

        # THEN: Should be invalid
        assert is_valid is False

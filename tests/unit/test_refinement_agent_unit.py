"""
Unit tests for Refinement Agent.
Tests individual methods and core business logic in isolation.
"""


import pytest

from src.agents.refinement_agent import RefinementAgent


class TestRefinementAgentUnit:
    """Unit tests for RefinementAgent business logic"""

    def test_agent_initialization_with_defaults(self):
        """Test RefinementAgent initializes with default services"""
        # WHEN: Creating agent with defaults
        agent = RefinementAgent()

        # THEN: Should initialize with proper services
        assert agent.name == "Refinement"
        assert agent.llm_service is not None
        assert agent.prompt_manager is not None
        assert agent.json_parser is not None

    def test_parse_refinement_request_valid_input(self):
        """Test parsing valid refinement request data"""
        # GIVEN: RefinementAgent
        agent = RefinementAgent()

        # Sample input data
        input_data = {
            "original_question": {
                "question_text": "Calculate 2 + 3",
                "marks": 2,
                "command_word": "Calculate",
                "grade_level": 7,
            },
            "quality_decision": {
                "action": "refine",
                "quality_score": 0.65,
                "suggested_improvements": [
                    "Make question more challenging",
                    "Add context to problem",
                ],
                "reasoning": "Question is too simple for grade level",
            },
        }

        # WHEN: Parsing request
        refinement_data = agent._parse_refinement_request(input_data)

        # THEN: Should extract refinement data correctly
        assert refinement_data is not None
        assert "original_question" in refinement_data
        assert "quality_decision" in refinement_data
        assert len(refinement_data["quality_decision"]["suggested_improvements"]) == 2

    def test_parse_refinement_request_invalid_input(self):
        """Test parsing invalid refinement request raises appropriate error"""
        # GIVEN: RefinementAgent
        agent = RefinementAgent()

        # Invalid input data (missing original_question)
        input_data = {"quality_decision": {"action": "refine"}}

        # WHEN/THEN: Should raise validation error
        with pytest.raises(ValueError, match="original_question"):
            agent._parse_refinement_request(input_data)

    def test_determine_refinement_strategy_high_quality(self):
        """Test refinement strategy determination for high quality scores"""
        # GIVEN: RefinementAgent
        agent = RefinementAgent()

        # Quality decision with specific issues
        quality_decision = {
            "quality_score": 0.75,
            "mathematical_accuracy": 0.9,
            "cambridge_compliance": 0.6,  # Low - needs improvement
            "grade_appropriateness": 0.8,
            "suggested_improvements": ["Align with Cambridge syllabus references"],
        }

        # WHEN: Determining strategy
        strategy = agent._determine_refinement_strategy(quality_decision)

        # THEN: Should use targeted approach for specific issues
        assert strategy["approach"] == "targeted"
        assert "cambridge_compliance" in strategy["focus_areas"]

    def test_determine_refinement_strategy_low_quality(self):
        """Test refinement strategy determination for low quality scores"""
        # GIVEN: RefinementAgent
        agent = RefinementAgent()

        # Quality decision with multiple issues
        quality_decision = {
            "quality_score": 0.45,
            "mathematical_accuracy": 0.5,
            "cambridge_compliance": 0.4,
            "grade_appropriateness": 0.4,
            "suggested_improvements": [
                "Fix mathematical errors",
                "Improve clarity",
                "Adjust difficulty level",
            ],
        }

        # WHEN: Determining strategy
        strategy = agent._determine_refinement_strategy(quality_decision)

        # THEN: Should use comprehensive approach
        assert strategy["approach"] == "comprehensive"
        assert len(strategy["focus_areas"]) >= 2

    def test_create_refinement_prompt_targeted(self):
        """Test creation of targeted refinement prompt"""
        # GIVEN: RefinementAgent
        agent = RefinementAgent()

        # Original question and targeted strategy
        original_question = {
            "question_text": "Calculate 5 × 7",
            "marks": 1,
            "command_word": "Calculate",
            "grade_level": 6,
        }

        strategy = {
            "approach": "targeted",
            "focus_areas": ["cambridge_compliance"],
            "specific_issues": ["Use proper command word"],
        }

        # WHEN: Creating prompt
        prompt = agent._create_refinement_prompt(original_question, strategy)

        # THEN: Should include targeted instructions
        assert "cambridge_compliance" in prompt.lower()
        assert "command word" in prompt.lower()
        assert original_question["question_text"] in prompt

    def test_create_refinement_prompt_comprehensive(self):
        """Test creation of comprehensive refinement prompt"""
        # GIVEN: RefinementAgent
        agent = RefinementAgent()

        # Original question and comprehensive strategy
        original_question = {
            "question_text": "Solve x + 2 = 10",
            "marks": 2,
            "command_word": "Solve",
            "grade_level": 8,
        }

        strategy = {
            "approach": "comprehensive",
            "focus_areas": ["mathematical_accuracy", "grade_appropriateness"],
            "specific_issues": ["Too easy for grade level", "Needs more steps"],
        }

        # WHEN: Creating prompt
        prompt = agent._create_refinement_prompt(original_question, strategy)

        # THEN: Should include comprehensive instructions
        assert "comprehensive" in prompt.lower() or "overall" in prompt.lower()
        assert "mathematical_accuracy" in prompt.lower()
        assert "grade_appropriateness" in prompt.lower()

    def test_validate_refinement_response_valid(self):
        """Test validation of valid refinement response"""
        # GIVEN: RefinementAgent
        agent = RefinementAgent()

        # Valid refinement response
        response = {
            "question_text": "Calculate the area of a rectangle with length 8m and width 5m. Show your working.",
            "marks": 3,
            "command_word": "Calculate",
            "grade_level": 7,
            "improvements_made": [
                "Added context and real-world application",
                "Increased marks to match complexity",
                "Added instruction to show working",
            ],
        }

        # WHEN: Validating response
        is_valid = agent._validate_refinement_response(response)

        # THEN: Should be valid
        assert is_valid is True

    def test_validate_refinement_response_invalid(self):
        """Test validation rejects invalid refinement response"""
        # GIVEN: RefinementAgent
        agent = RefinementAgent()

        # Invalid refinement response (missing required fields)
        response = {
            "question_text": "Some question",
            # Missing marks, command_word, grade_level
        }

        # WHEN: Validating response
        is_valid = agent._validate_refinement_response(response)

        # THEN: Should be invalid
        assert is_valid is False

    def test_apply_refinement_improvements(self):
        """Test applying refinement improvements to original question"""
        # GIVEN: RefinementAgent
        agent = RefinementAgent()

        # Original question data
        original_question = {
            "question_text": "Calculate 2 + 3",
            "marks": 1,
            "command_word": "Calculate",
            "grade_level": 6,
            "subject_content_references": ["C1.1"],
        }

        # Refinement improvements
        improvements = {
            "question_text": "Calculate the total cost when buying 2 books at £3 each. Show your working.",
            "marks": 2,
            "improvements_made": [
                "Added real-world context",
                "Increased complexity and marks",
                "Added instruction to show working",
            ],
        }

        # WHEN: Applying improvements
        refined_question = agent._apply_refinement_improvements(original_question, improvements)

        # THEN: Should create improved question
        assert refined_question["question_text"] == improvements["question_text"]
        assert refined_question["marks"] == 2
        assert refined_question["command_word"] == "Calculate"  # Preserved
        assert refined_question["grade_level"] == 6  # Preserved
        assert "subject_content_references" in refined_question  # Preserved

    def test_generate_refinement_metadata(self):
        """Test generation of refinement metadata"""
        # GIVEN: RefinementAgent
        agent = RefinementAgent()

        # Original quality decision and improvements
        quality_decision = {
            "action": "refine",
            "quality_score": 0.65,
            "suggested_improvements": ["Add context", "Increase difficulty"],
        }

        improvements_made = [
            "Added real-world context with money calculation",
            "Increased marks from 1 to 2 to match complexity",
        ]

        # WHEN: Generating metadata
        metadata = agent._generate_refinement_metadata(quality_decision, improvements_made)

        # THEN: Should include refinement tracking information
        assert metadata["agent_name"] == "Refinement"
        assert metadata["original_quality_score"] == 0.65
        assert len(metadata["improvements_made"]) == 2
        assert "refinement_timestamp" in metadata

    def test_calculate_expected_quality_improvement(self):
        """Test calculation of expected quality improvement after refinement"""
        # GIVEN: RefinementAgent
        agent = RefinementAgent()

        # Quality scores and improvement areas
        original_scores = {
            "mathematical_accuracy": 0.8,
            "cambridge_compliance": 0.5,  # Low
            "grade_appropriateness": 0.6,  # Medium
            "question_clarity": 0.9,
        }

        improvements_made = [
            "Fixed Cambridge syllabus references",
            "Adjusted difficulty for grade level",
        ]

        # WHEN: Calculating expected improvement
        expected_scores = agent._calculate_expected_quality_improvement(
            original_scores, improvements_made
        )

        # THEN: Should show improvement in relevant areas
        assert expected_scores["mathematical_accuracy"] >= 0.8  # Unchanged
        assert expected_scores["cambridge_compliance"] > 0.5  # Should improve
        assert expected_scores["grade_appropriateness"] > 0.6  # Should improve
        assert expected_scores["question_clarity"] >= 0.9  # Unchanged or slight improvement

    def test_determine_refinement_fallback_strategy(self):
        """Test determination of fallback strategy when primary refinement fails"""
        # GIVEN: RefinementAgent
        agent = RefinementAgent()

        # Quality decision with specific patterns
        quality_decision = {
            "quality_score": 0.55,
            "mathematical_accuracy": 0.3,  # Very low
            "suggested_improvements": ["Fix calculation errors", "Verify mathematical correctness"],
        }

        # WHEN: Determining fallback strategy
        fallback = agent._determine_refinement_fallback_strategy(quality_decision)

        # THEN: Should suggest appropriate fallback
        if quality_decision["mathematical_accuracy"] < 0.5:
            assert fallback["strategy"] == "regenerate"
            assert "mathematical_accuracy" in fallback["reason"]
        else:
            assert fallback["strategy"] == "manual_review"

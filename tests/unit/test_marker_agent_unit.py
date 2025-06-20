"""
Unit tests for Marker Agent.
Fast, isolated tests focusing on individual methods and core logic.
"""

import pytest

from src.agents.marker_agent import MarkerAgent, MarkerAgentError
from src.models.question_models import GenerationRequest, MarkType, SolutionAndMarkingScheme


class TestMarkerAgentUnit:
    """Unit tests for Marker Agent core functionality"""

    def test_agent_initialization_with_defaults(self):
        """Test agent initializes with default services"""
        agent = MarkerAgent()
        assert agent.name == "Marker"
        assert agent.llm_service is not None
        assert agent.prompt_manager is not None
        assert agent.json_parser is not None

    def test_parse_marking_request_valid_input(self):
        """Test parsing valid marking request"""
        agent = MarkerAgent()
        question = {"question_text": "Calculate 2 + 3", "marks": 2, "command_word": "Calculate"}
        config_data = {"topic": "arithmetic", "marks": 2, "tier": "Core", "grade_level": 7}
        input_data = {"question": question, "config": config_data}

        parsed_question, parsed_config = agent._parse_marking_request(input_data)

        assert parsed_question["question_text"] == "Calculate 2 + 3"
        assert isinstance(parsed_config, GenerationRequest)
        assert parsed_config.marks == 2

    def test_parse_marking_request_invalid_input(self):
        """Test parsing invalid marking request raises error"""
        agent = MarkerAgent()
        invalid_data = {"invalid": "data"}

        with pytest.raises(MarkerAgentError):
            agent._parse_marking_request(invalid_data)

    def test_convert_to_marking_scheme_object(self):
        """Test converting JSON to SolutionAndMarkingScheme object"""
        agent = MarkerAgent()
        json_data = {
            "total_marks": 3,
            "mark_allocation_criteria": [
                {"criterion_text": "Correct method", "marks_value": 1, "mark_type": "M"},
                {"criterion_text": "Correct calculation", "marks_value": 1, "mark_type": "M"},
                {"criterion_text": "Final answer", "marks_value": 1, "mark_type": "A"},
            ],
            "final_answers": [{"answer_text": "5", "value_numeric": 5.0, "unit": None}],
        }

        scheme = agent._convert_to_marking_scheme_object(json_data, question_marks=3)

        assert isinstance(scheme, SolutionAndMarkingScheme)
        assert scheme.total_marks_for_part == 3
        assert len(scheme.mark_allocation_criteria) == 3
        assert len(scheme.final_answers_summary) == 1

        # Check mark types are properly assigned
        mark_types = [criterion.mark_type_primary for criterion in scheme.mark_allocation_criteria]
        assert MarkType.M in mark_types
        assert MarkType.A in mark_types

    def test_mark_type_assignment_logic(self):
        """Test that mark types are assigned correctly according to Cambridge standards"""
        agent = MarkerAgent()
        json_data = {
            "total_marks": 2,
            "mark_allocation_criteria": [
                {"criterion_text": "Method", "marks_value": 1, "mark_type": "M"},
                {"criterion_text": "Answer", "marks_value": 1, "mark_type": "A"},
            ],
            "final_answers": [{"answer_text": "10"}],
        }

        scheme = agent._convert_to_marking_scheme_object(json_data, question_marks=2)

        criteria = scheme.mark_allocation_criteria
        assert len(criteria) == 2
        assert criteria[0].mark_type_primary == MarkType.M  # Method mark
        assert criteria[1].mark_type_primary == MarkType.A  # Accuracy mark

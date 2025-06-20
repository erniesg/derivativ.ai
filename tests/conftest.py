"""
Pytest configuration and shared fixtures for Derivativ AI tests.
"""

import asyncio
from typing import Any

import pytest

from src.models.enums import CalculatorPolicy, CommandWord, LLMModel, SubjectContentReference, Tier
from src.models.question_models import GenerationRequest
from src.services import JSONParser, MockLLMService, PromptManager


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_llm_service():
    """Provide a mock LLM service for testing"""
    return MockLLMService(response_delay=0.1, failure_rate=0.0)


@pytest.fixture
def prompt_manager():
    """Provide a prompt manager instance for testing"""
    return PromptManager(enable_cache=False)  # Disable cache for consistent tests


@pytest.fixture
def json_parser():
    """Provide a JSON parser instance for testing"""
    return JSONParser(enable_cache=False)  # Disable cache for consistent tests


@pytest.fixture
def sample_generation_request():
    """Provide a sample generation request for testing"""
    return GenerationRequest(
        topic="algebra",
        tier=Tier.CORE,
        grade_level=7,
        marks=3,
        count=1,
        calculator_policy=CalculatorPolicy.NOT_ALLOWED,
        subject_content_refs=[SubjectContentReference.C2_1],
        command_word=CommandWord.CALCULATE,
        llm_model=LLMModel.GPT_4O,
        temperature=0.7,
        max_retries=2,
    )


@pytest.fixture
def sample_question_json():
    """Provide sample question JSON data for testing"""
    return {
        "question_text": "Calculate the value of 3x + 2 when x = 5",
        "marks": 3,
        "command_word": "Calculate",
        "subject_content_references": ["C2.1"],
        "solution_steps": [
            "Substitute x = 5 into the expression",
            "Calculate 3(5) + 2 = 15 + 2 = 17",
        ],
        "final_answer": "17",
        "marking_criteria": [
            {"criterion": "Correct substitution", "marks": 1, "mark_type": "M"},
            {"criterion": "Correct calculation", "marks": 2, "mark_type": "A"},
        ],
    }


@pytest.fixture
def sample_question_object(sample_question_json):
    """Provide a sample Question object for testing"""
    from src.models.enums import CommandWord, SubjectContentReference
    from src.models.question_models import (
        FinalAnswer,
        MarkingCriterion,
        Question,
        QuestionTaxonomy,
        SolutionAndMarkingScheme,
        SolverAlgorithm,
        SolverStep,
    )

    return Question(
        question_id_local="test_q1",
        question_id_global="derivativ_test_q1",
        question_number_display="1",
        marks=2,
        command_word=CommandWord.CALCULATE,
        raw_text_content=sample_question_json["question_text"],
        taxonomy=QuestionTaxonomy(
            topic_path=["algebra"],
            subject_content_references=[SubjectContentReference.C2_1],
            skill_tags=["substitution", "calculation"],
        ),
        solution_and_marking_scheme=SolutionAndMarkingScheme(
            final_answers_summary=[FinalAnswer(answer_text="17")],
            mark_allocation_criteria=[
                MarkingCriterion(
                    criterion_id="crit_1",
                    criterion_text="Correct substitution",
                    mark_code_display="M1",
                    marks_value=1,
                ),
                MarkingCriterion(
                    criterion_id="crit_2",
                    criterion_text="Correct calculation",
                    mark_code_display="A1",
                    marks_value=1,
                ),
            ],
            total_marks_for_part=2,
        ),
        solver_algorithm=SolverAlgorithm(
            steps=[
                SolverStep(
                    step_number=1,
                    description_text="Substitute x = 5",
                    skill_applied_tag="substitution",
                ),
                SolverStep(
                    step_number=2,
                    description_text="Calculate result",
                    skill_applied_tag="arithmetic",
                ),
            ]
        ),
    )


@pytest.fixture
def failing_llm_service():
    """Provide a mock LLM service that always fails for testing error handling"""
    service = MockLLMService(response_delay=0.1, failure_rate=1.0)
    return service


@pytest.fixture
def agent_config():
    """Provide default agent configuration for testing"""
    return {
        "max_retries": 2,
        "generation_timeout": 30,
        "quality_threshold": 0.7,
        "enable_fallback": True,
    }


# Test utilities
class MockAsyncContext:
    """Mock async context manager for testing"""

    def __init__(self, return_value=None):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


def create_mock_llm_response(content: str, model: str = "gpt-4o", tokens: int = 100):
    """Create a mock LLM response for testing"""
    from src.services.llm_service import LLMResponse

    return LLMResponse(
        content=content,
        model=model,
        provider="mock",
        tokens_used=tokens,
        cost_estimate=0.001,
        generation_time=1.0,
        metadata={"test": True},
    )


def create_mock_json_extraction_result(
    success: bool = True, data: dict[str, Any] = None, method: str = "test_extraction"
):
    """Create a mock JSON extraction result for testing"""
    from src.services.json_parser import JSONExtractionResult

    return JSONExtractionResult(
        success=success,
        data=data or {},
        raw_json='{"test": true}',
        extraction_method=method,
        confidence_score=0.9 if success else 0.0,
    )

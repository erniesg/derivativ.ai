"""
Pytest configuration and shared fixtures for Derivativ AI tests.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

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


# Centralized Mock Service Fixtures
@pytest.fixture
def mock_openai_client():
    """Centralized mock OpenAI client for all tests."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Generated response from OpenAI"
    mock_response.model = "gpt-4o-mini"
    mock_response.usage.total_tokens = 25
    mock_response.usage.prompt_tokens = 15
    mock_response.usage.completion_tokens = 10
    mock_response.usage.prompt_tokens_details = None
    mock_response.usage.completion_tokens_details = None
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_anthropic_client():
    """Centralized mock Anthropic client for all tests."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "Generated response from Anthropic"
    mock_response.model = "claude-3-5-sonnet-20241022"
    mock_response.usage.input_tokens = 15
    mock_response.usage.output_tokens = 10
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_gemini_client():
    """Centralized mock Gemini client for all tests."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Generated response from Gemini"
    mock_response.usage_metadata.prompt_token_count = 15
    mock_response.usage_metadata.candidates_token_count = 10
    mock_response.usage_metadata.total_token_count = 25
    mock_client.generate_content_async.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_supabase_client():
    """Centralized mock Supabase client for all tests."""
    mock_client = Mock()
    mock_table = Mock()
    mock_client.table.return_value = mock_table
    # Set up common return values
    mock_table.insert.return_value = mock_table
    mock_table.select.return_value = mock_table
    mock_table.update.return_value = mock_table
    mock_table.delete.return_value = mock_table
    mock_table.execute.return_value = mock_table
    return mock_client, mock_table


@pytest.fixture
def mock_llm_factory():
    """Centralized mock LLM factory for all tests."""
    factory = MagicMock()
    llm_service = AsyncMock()
    llm_service.generate.return_value = create_mock_llm_response("Test response")
    factory.get_service.return_value = llm_service
    return factory, llm_service


@pytest.fixture
def mock_openai_streaming_response():
    """Mock OpenAI streaming response for streaming tests."""
    mock_stream = AsyncMock()
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "streaming content"
    mock_stream.__aiter__.return_value = [mock_chunk]
    return mock_stream


@pytest.fixture
def mock_anthropic_streaming_response():
    """Mock Anthropic streaming response for streaming tests."""
    mock_stream = AsyncMock()
    mock_chunk = MagicMock()
    mock_chunk.delta.text = "streaming content"
    mock_stream.__aiter__.return_value = [mock_chunk]
    return mock_stream


@pytest.fixture
def mock_gemini_streaming_response():
    """Mock Gemini streaming response for streaming tests."""
    mock_stream = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.text = "streaming content"
    mock_stream.__iter__.return_value = [mock_chunk]
    return mock_stream


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
    from src.models.llm_models import LLMResponse

    return LLMResponse(
        content=content,
        model_used=model,
        provider="mock",
        tokens_used=tokens,
        cost_estimate=0.001,
        latency_ms=1000,
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

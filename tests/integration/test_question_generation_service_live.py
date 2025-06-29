
import pytest
from uuid import uuid4
from unittest.mock import MagicMock

from src.models.question_models import GenerationRequest
from src.models.enums import Tier, CommandWord
from src.services.question_generation_service import QuestionGenerationService
from src.database.supabase_repository import QuestionRepository, GenerationSessionRepository
from src.agents.question_generator import QuestionGeneratorAgent
from src.services.llm_factory import LLMFactory

@pytest.mark.asyncio
async def test_generate_questions_live_succeeds():
    """
    Tests that the QuestionGenerationService can successfully generate questions
    using a live QuestionGeneratorAgent and a real LLM.
    """
    # 1. Setup - using a REAL agent
    live_agent = QuestionGeneratorAgent(llm_service=LLMFactory().get_service("openai"))

    # Mock the database repositories to isolate the service-agent interaction
    mock_question_repo = MagicMock(spec=QuestionRepository)
    mock_session_repo = MagicMock(spec=GenerationSessionRepository)
    mock_question_repo.save_question.return_value = "fake_question_id"
    mock_session_repo.save_session.return_value = "fake_session_id"

    # 2. Instantiate the service with the live agent.
    service = QuestionGenerationService(
        question_repository=mock_question_repo,
        session_repository=mock_session_repo,
        agent=live_agent
    )

    request = GenerationRequest(
        topic="differentiation",
        marks=6,
        tier=Tier.EXTENDED,
        command_word=CommandWord.SOLVE,
    )

    # 3. Execute and Assert
    # We expect this to fail or not behave as expected because the live agent isn't used.
    # For the purpose of TDD, we'll start by just running it and seeing it pass vacuously
    # because the mock works, then we'll refactor the service to accept the agent.
    
    # This test's true purpose is to fail after we refactor the service constructor,
    # proving the live connection is being attempted.
    
    session = await service.generate_questions(request)

    assert session is not None
    assert len(session.questions) > 0
    # This assertion will pass with the mock, but fail if the live agent has an issue.
    assert session.questions[0].raw_text_content is not None

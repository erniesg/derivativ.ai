"""
FastAPI dependency injection for repositories and services.
Provides singleton instances of database clients and services.
"""

import os
from functools import lru_cache

from fastapi import Depends, HTTPException

from src.agents.document_formatter_agent import DocumentFormatterAgent
from src.database.supabase_repository import (
    GenerationSessionRepository,
    QuestionRepository,
    get_supabase_client,
)
from src.services.document_generation_service import DocumentGenerationService
from src.services.llm_factory import LLMFactory
from src.services.prompt_manager import PromptManager
from src.services.question_generation_service import QuestionGenerationService
from src.supabase_realtime.supabase_realtime import get_realtime_client


@lru_cache
def get_supabase_credentials() -> tuple[str, str]:
    """
    Get Supabase credentials from environment.

    Returns:
        Tuple of (url, key)

    Raises:
        HTTPException: If credentials are not configured
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")

    if not url or not key:
        raise HTTPException(
            status_code=503,
            detail="Supabase not configured. Please set SUPABASE_URL and SUPABASE_ANON_KEY environment variables.",
        )

    return url, key


@lru_cache
def is_demo_mode() -> bool:
    """Check if running in demo mode (no database required)."""
    return os.getenv("DEMO_MODE", "false").lower() in ("true", "1", "yes")


@lru_cache
def get_database_client():
    """
    Get singleton Supabase client instance.

    Returns:
        Configured Supabase client
    """
    url, key = get_supabase_credentials()
    return get_supabase_client(url, key)


def get_question_repository(client=Depends(get_database_client)) -> QuestionRepository:
    """
    Get QuestionRepository instance.

    Args:
        client: Supabase client (injected)

    Returns:
        Configured QuestionRepository
    """
    return QuestionRepository(client)


def get_session_repository(client=Depends(get_database_client)) -> GenerationSessionRepository:
    """
    Get GenerationSessionRepository instance.

    Args:
        client: Supabase client (injected)

    Returns:
        Configured GenerationSessionRepository
    """
    return GenerationSessionRepository(client)


def get_question_generation_service(
    question_repo: QuestionRepository = Depends(get_question_repository),
    session_repo: GenerationSessionRepository = Depends(get_session_repository),
) -> QuestionGenerationService:
    """
    Get QuestionGenerationService instance.

    Args:
        question_repo: Question repository (injected)
        session_repo: Session repository (injected)

    Returns:
        Configured QuestionGenerationService
    """
    return QuestionGenerationService(question_repo, session_repo)


@lru_cache
def get_realtime_client_instance():
    """
    Get singleton Realtime client instance.

    Returns:
        Configured Realtime client, or None if not configured
    """
    try:
        url, key = get_supabase_credentials()
        return get_realtime_client(url, key)
    except HTTPException:
        # Realtime is optional - return None if not configured
        return None


def get_optional_realtime_client():
    """
    Get optional Realtime client instance.

    Returns:
        Configured Realtime client, or None if not available
    """
    return get_realtime_client_instance()


@lru_cache
def get_llm_factory() -> LLMFactory:
    """
    Get singleton LLMFactory instance.

    Returns:
        Configured LLMFactory
    """
    return LLMFactory()


@lru_cache
def get_prompt_manager() -> PromptManager:
    """
    Get singleton PromptManager instance.

    Returns:
        Configured PromptManager
    """
    return PromptManager()


def get_document_generation_service(
    llm_factory: LLMFactory = Depends(get_llm_factory),
    prompt_manager: PromptManager = Depends(get_prompt_manager),
) -> DocumentGenerationService:
    """
    Get DocumentGenerationService instance.

    Args:
        llm_factory: LLM factory (injected)
        prompt_manager: Prompt manager (injected)

    Returns:
        Configured DocumentGenerationService
    """
    if is_demo_mode():
        # Use mock repository for demo mode
        from unittest.mock import AsyncMock, MagicMock

        from src.models.enums import SubjectContentReference
        from src.models.question_models import (
            FinalAnswer,
            MarkingCriterion,
            Question,
            QuestionTaxonomy,
            SolutionAndMarkingScheme,
            SolverAlgorithm,
            SolverStep,
        )

        mock_repo = MagicMock()

        # Create sample question for demo
        sample_question = Question(
            question_id_local="1a",
            question_id_global="demo_q1",
            question_number_display="1 (a)",
            marks=3,
            command_word="Calculate",
            raw_text_content="Calculate the area of a triangle with base 6cm and height 4cm.",
            taxonomy=QuestionTaxonomy(
                topic_path=["Geometry", "Area"],
                subject_content_references=[SubjectContentReference.C5_2],
                skill_tags=["area_calculation"],
            ),
            solution_and_marking_scheme=SolutionAndMarkingScheme(
                final_answers_summary=[
                    FinalAnswer(answer_text="12 cm²", value_numeric=12.0, unit="cm²")
                ],
                mark_allocation_criteria=[
                    MarkingCriterion(
                        criterion_id="1",
                        criterion_text="Correct method",
                        mark_code_display="M1",
                        marks_value=1,
                    ),
                    MarkingCriterion(
                        criterion_id="2",
                        criterion_text="Correct answer",
                        mark_code_display="A1",
                        marks_value=2,
                    ),
                ],
                total_marks_for_part=3,
            ),
            solver_algorithm=SolverAlgorithm(
                steps=[
                    SolverStep(
                        step_number=1, description_text="Use formula Area = (1/2) × base × height"
                    ),
                    SolverStep(
                        step_number=2,
                        description_text="Substitute values: Area = (1/2) × 6 × 4 = 12 cm²",
                    ),
                ]
            ),
        )

        mock_repo.list_questions = AsyncMock(
            return_value=[
                {
                    "content_json": sample_question.model_dump(),
                    "quality_score": 0.85,
                    "tier": "Core",
                    "marks": 3,
                }
            ]
        )

        return DocumentGenerationService(mock_repo, llm_factory, prompt_manager)
    else:
        # Get real repository - this will fail if Supabase not configured
        try:
            client = get_database_client()
            question_repo = QuestionRepository(client)
            return DocumentGenerationService(question_repo, llm_factory, prompt_manager)
        except HTTPException:
            # Fallback to mock for demo if database not available
            os.environ["DEMO_MODE"] = "true"
            # Clear cache and retry with demo mode
            is_demo_mode.cache_clear()
            return get_document_generation_service(llm_factory, prompt_manager)


def get_document_formatter_agent(
    llm_factory: LLMFactory = Depends(get_llm_factory),
) -> DocumentFormatterAgent:
    """
    Get DocumentFormatterAgent instance.

    Args:
        llm_factory: LLM factory (injected)

    Returns:
        Configured DocumentFormatterAgent
    """
    # Create LLM service for the agent
    llm_service = llm_factory.get_service("openai")
    return DocumentFormatterAgent(llm_service=llm_service)


# Global service instances for backwards compatibility
# These will be replaced by dependency injection


def initialize_global_services():
    """
    Initialize global service instances for endpoints that haven't been updated
    to use dependency injection yet.

    This is a transitional function while we migrate to full dependency injection.
    """
    try:
        # Initialize repositories
        client = get_database_client()
        question_repo = QuestionRepository(client)
        session_repo = GenerationSessionRepository(client)

        # Initialize services
        question_service = QuestionGenerationService(question_repo, session_repo)

        # Initialize realtime client
        realtime_client = get_realtime_client_instance()

        # Update global references in endpoint modules
        from src.api.endpoints import questions, sessions, websocket

        questions.question_repository = question_repo
        questions.question_generation_service = question_service

        sessions.session_repository = session_repo

        if realtime_client:
            websocket.initialize_realtime_client(*get_supabase_credentials())
        websocket.question_generation_service = question_service

        return True

    except Exception as e:
        # Log error but don't fail startup - endpoints will show 503 if needed
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to initialize services: {e}")
        return False


def check_database_health() -> dict:
    """
    Check database connectivity and health.

    Returns:
        Health status dictionary
    """
    try:
        client = get_database_client()

        # Test basic connectivity by querying a simple table
        response = client.table("tiers").select("value").limit(1).execute()

        return {
            "database": "healthy",
            "supabase": "connected",
            "tables_accessible": len(response.data) >= 0,
        }

    except Exception as e:
        return {
            "database": "unhealthy",
            "supabase": "disconnected",
            "error": str(e),
        }


def check_realtime_health() -> dict:
    """
    Check Realtime connectivity and health.

    Returns:
        Realtime health status dictionary
    """
    try:
        realtime_client = get_realtime_client_instance()

        if realtime_client is None:
            return {
                "realtime": "not_configured",
                "websocket": "unavailable",
            }

        return {
            "realtime": "configured",
            "websocket": "available",
        }

    except Exception as e:
        return {
            "realtime": "error",
            "websocket": "unavailable",
            "error": str(e),
        }


def get_system_health() -> dict:
    """
    Get comprehensive system health status.

    Returns:
        Complete health status dictionary
    """
    if is_demo_mode():
        # In demo mode, return healthy status without checking external dependencies
        return {
            "status": "healthy",
            "service": "derivativ-api",
            "database": "demo_mode",
            "realtime": "demo_mode",
            "mode": "demo",
        }

    database_health = check_database_health()
    realtime_health = check_realtime_health()

    overall_status = "healthy"
    if database_health.get("database") != "healthy":
        overall_status = "degraded"
    if "error" in database_health or "error" in realtime_health:
        overall_status = "unhealthy"

    return {
        "status": overall_status,
        "service": "derivativ-api",
        **database_health,
        **realtime_health,
    }

"""
FastAPI dependency injection for repositories and services.
Provides singleton instances of database clients and services.
"""

import os
from functools import lru_cache

from fastapi import Depends, HTTPException

from src.database.supabase_repository import (
    GenerationSessionRepository,
    QuestionRepository,
    get_supabase_client,
)
from src.realtime.supabase_realtime import get_realtime_client
from src.services.question_generation_service import QuestionGenerationService


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

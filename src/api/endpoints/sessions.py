"""
Generation session management endpoints.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.api.dependencies import get_session_repository
from src.database.supabase_repository import GenerationSessionRepository
from src.models.question_models import GenerationStatus

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependencies (will be injected via dependency injection in production)
session_repository: Optional[GenerationSessionRepository] = None


class SessionListResponse(BaseModel):
    """Response model for session listing."""

    sessions: list[dict]
    pagination: dict


@router.get("/sessions/{session_id}", response_model=dict)
async def get_session(
    session_id: str,
    repository: GenerationSessionRepository = Depends(get_session_repository),
):
    """
    Get a specific generation session by ID.

    Args:
        session_id: Session identifier
        repository: Session repository (injected)

    Returns:
        Session data with questions and agent results
    """
    try:
        # Use injected repository, fallback to global for backwards compatibility
        repo = repository or session_repository
        if not repo:
            raise HTTPException(
                status_code=503,
                detail="Session repository not available. Please check database configuration.",
            )

        session = repo.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return session.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session")


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    status: Optional[GenerationStatus] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    repository: GenerationSessionRepository = Depends(get_session_repository),
):
    """
    List generation sessions with filtering and pagination.

    Args:
        status: Filter by session status
        limit: Maximum number of results
        offset: Number of results to skip
        repository: Session repository (injected)

    Returns:
        List of sessions with pagination info
    """
    try:
        # Use injected repository, fallback to global for backwards compatibility
        repo = repository or session_repository
        if not repo:
            raise HTTPException(
                status_code=503,
                detail="Session repository not available. Please check database configuration.",
            )

        sessions = repo.list_sessions(status=status, limit=limit, offset=offset)

        return SessionListResponse(
            sessions=sessions,
            pagination={
                "limit": limit,
                "offset": offset,
                "total": len(sessions),  # This would be calculated differently in production
            },
        )

    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions")

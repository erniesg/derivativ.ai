"""
Question generation and management endpoints.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.database.supabase_repository import QuestionRepository
from src.models.enums import CommandWord, Tier
from src.models.question_models import GenerationRequest
from src.services.question_generation_service import QuestionGenerationService

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependencies (will be injected via dependency injection in production)
question_repository: Optional[QuestionRepository] = None
question_generation_service: Optional[QuestionGenerationService] = None


class QuestionListResponse(BaseModel):
    """Response model for question listing."""

    questions: list[dict]
    pagination: dict


class GenerationResponse(BaseModel):
    """Response model for question generation."""

    session_id: str
    questions: list[dict]
    status: str
    agent_results: list[dict]


@router.post("/questions/generate", status_code=201, response_model=GenerationResponse)
async def generate_questions(request: GenerationRequest):
    """
    Generate new questions using AI agents.

    Args:
        request: Question generation parameters

    Returns:
        Generation session with questions and agent results
    """
    try:
        # Generate questions using the service
        session = await question_generation_service.generate_questions(request)

        return GenerationResponse(
            session_id=str(session.session_id),
            questions=[q.model_dump() for q in session.questions],
            status=session.status.value,
            agent_results=[ar.model_dump() for ar in session.agent_results],
        )

    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e!s}")


@router.get("/questions/{question_id}", response_model=dict)
async def get_question(question_id: str):
    """
    Get a specific question by ID.

    Args:
        question_id: Global question identifier

    Returns:
        Question data
    """
    try:
        question = question_repository.get_question(question_id)

        if not question:
            raise HTTPException(status_code=404, detail="Question not found")

        return question.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get question {question_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve question")


@router.get("/questions", response_model=QuestionListResponse)
async def list_questions(
    tier: Optional[Tier] = Query(None, description="Filter by tier"),
    min_quality_score: Optional[float] = Query(
        None, ge=0, le=1, description="Minimum quality score"
    ),
    command_word: Optional[CommandWord] = Query(None, description="Filter by command word"),
    limit: int = Query(50, ge=1, le=100, description="Number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
):
    """
    List questions with filtering and pagination.

    Args:
        tier: Filter by Core/Extended tier
        min_quality_score: Minimum quality score filter
        command_word: Filter by command word
        limit: Maximum number of results
        offset: Number of results to skip

    Returns:
        List of questions with pagination info
    """
    try:
        questions = question_repository.list_questions(
            tier=tier,
            min_quality_score=min_quality_score,
            command_word=command_word,
            limit=limit,
            offset=offset,
        )

        return QuestionListResponse(
            questions=questions,
            pagination={
                "limit": limit,
                "offset": offset,
                "total": len(questions),  # This would be calculated differently in production
            },
        )

    except Exception as e:
        logger.error(f"Failed to list questions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve questions")


@router.delete("/questions/{question_id}", status_code=204)
async def delete_question(question_id: str):
    """
    Delete a question by ID.

    Args:
        question_id: Global question identifier
    """
    try:
        success = question_repository.delete_question(question_id)

        if not success:
            raise HTTPException(status_code=404, detail="Question not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete question {question_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete question")

"""
Agent API endpoints for direct access to individual AI agents.

Provides endpoints for interacting with individual agents (QuestionGenerator,
ReviewAgent, MarkerAgent, etc.) and the multi-agent orchestrator.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.agents.orchestrator import MultiAgentOrchestrator
from src.models.question_models import GenerationRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["Agents"])


# Agent dependency
def get_orchestrator() -> MultiAgentOrchestrator:
    """Get MultiAgentOrchestrator instance."""
    return MultiAgentOrchestrator()


# Individual Agent Endpoints


@router.post("/question-generator/generate")
async def generate_question_direct(
    request: dict[str, Any],
    orchestrator: MultiAgentOrchestrator = Depends(get_orchestrator),
) -> dict[str, Any]:
    """
    Generate a question using the QuestionGenerator agent directly.

    This bypasses the full multi-agent workflow and only uses the
    question generation agent.
    """
    try:
        # Get the question generator agent
        generator = orchestrator._get_agent("generator", force_async=True)

        # Process the request
        result = await generator.process(request)

        if result.success:
            return {
                "success": True,
                "question": result.output,
                "reasoning_steps": result.reasoning_steps,
                "processing_time": result.processing_time,
            }
        else:
            return {
                "success": False,
                "error": result.error,
                "reasoning_steps": result.reasoning_steps,
            }

    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Question generation failed: {e!s}")


@router.post("/reviewer/assess")
async def assess_question_quality(
    request: dict[str, Any],
    orchestrator: MultiAgentOrchestrator = Depends(get_orchestrator),
) -> dict[str, Any]:
    """
    Assess question quality using the ReviewAgent directly.

    Takes a question and marking scheme, returns quality scores
    and improvement suggestions.
    """
    try:
        # Get the review agent
        reviewer = orchestrator._get_agent("reviewer", force_async=True)

        # Process the request
        result = await reviewer.process(request)

        if result.success:
            return {
                "success": True,
                "quality_assessment": result.output,
                "reasoning_steps": result.reasoning_steps,
                "processing_time": result.processing_time,
            }
        else:
            return {
                "success": False,
                "error": result.error,
                "reasoning_steps": result.reasoning_steps,
            }

    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Quality assessment failed: {e!s}")


@router.post("/marker/create-scheme")
async def create_marking_scheme(
    request: dict[str, Any],
    orchestrator: MultiAgentOrchestrator = Depends(get_orchestrator),
) -> dict[str, Any]:
    """
    Create a marking scheme using the MarkerAgent directly.

    Takes a question and creates a detailed marking scheme following
    Cambridge IGCSE standards.
    """
    try:
        # Get the marker agent
        marker = orchestrator._get_agent("marker", force_async=True)

        # Process the request
        result = await marker.process(request)

        if result.success:
            return {
                "success": True,
                "marking_scheme": result.output,
                "reasoning_steps": result.reasoning_steps,
                "processing_time": result.processing_time,
            }
        else:
            return {
                "success": False,
                "error": result.error,
                "reasoning_steps": result.reasoning_steps,
            }

    except Exception as e:
        logger.error(f"Marking scheme creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Marking scheme creation failed: {e!s}")


@router.post("/refiner/improve")
async def refine_question(
    request: dict[str, Any],
    orchestrator: MultiAgentOrchestrator = Depends(get_orchestrator),
) -> dict[str, Any]:
    """
    Improve a question using the RefinementAgent directly.

    Takes a question and review feedback, returns an improved version.
    """
    try:
        # Get the refinement agent
        refiner = orchestrator._get_agent("refiner", force_async=True)

        # Process the request
        result = await refiner.process(request)

        if result.success:
            return {
                "success": True,
                "refined_question": result.output,
                "reasoning_steps": result.reasoning_steps,
                "processing_time": result.processing_time,
            }
        else:
            return {
                "success": False,
                "error": result.error,
                "reasoning_steps": result.reasoning_steps,
            }

    except Exception as e:
        logger.error(f"Question refinement failed: {e}")
        raise HTTPException(status_code=500, detail=f"Question refinement failed: {e!s}")


# Multi-Agent Orchestrator Endpoints


@router.post("/orchestrator/generate")
async def orchestrate_question_generation(
    request: GenerationRequest,
    orchestrator: MultiAgentOrchestrator = Depends(get_orchestrator),
) -> dict[str, Any]:
    """
    Generate a question using the full multi-agent workflow.

    Coordinates all agents (generator, marker, reviewer, refiner) to
    produce a high-quality question with complete marking scheme.
    """
    try:
        result = await orchestrator.generate_question_async(request)
        return result

    except Exception as e:
        logger.error(f"Orchestrated generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Orchestrated generation failed: {e!s}")


@router.get("/orchestrator/status")
async def get_orchestrator_status(
    orchestrator: MultiAgentOrchestrator = Depends(get_orchestrator),
) -> dict[str, Any]:
    """
    Get the status and configuration of the multi-agent orchestrator.

    Returns information about available agents, quality thresholds,
    and current configuration.
    """
    try:
        return {
            "status": "operational",
            "quality_thresholds": orchestrator.quality_thresholds,
            "available_agents": ["generator", "marker", "reviewer", "refiner"],
            "agent_configurations": {
                "use_sync": orchestrator.use_sync,
                "agents_cached": len(orchestrator._agents),
            },
        }

    except Exception as e:
        logger.error(f"Failed to get orchestrator status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get orchestrator status: {e!s}")


@router.get("/orchestrator/workflow-summary/{workflow_id}")
async def get_workflow_summary(workflow_id: str) -> dict[str, Any]:
    """
    Get a summary of a completed workflow execution.

    Returns human-readable summary of agent interactions and decisions.
    """
    try:
        # In a full implementation, this would retrieve from database
        # For now, return placeholder
        return {
            "workflow_id": workflow_id,
            "summary": "Workflow summary not yet implemented",
            "agents_used": [],
            "total_processing_time": 0,
            "final_quality_score": 0.0,
        }

    except Exception as e:
        logger.error(f"Failed to get workflow summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get workflow summary")


# Agent Configuration Endpoints


@router.get("/config")
async def get_agent_configuration() -> dict[str, Any]:
    """
    Get current agent configuration and settings.

    Returns LLM models, temperature settings, quality thresholds, etc.
    """
    try:
        # In a full implementation, this would read from config
        from src.core.config import load_config

        config = load_config()

        return {
            "agents": config.get("agents", {}),
            "quality_control": config.get("quality_control", {}),
            "llm_settings": config.get("llm", {}),
        }

    except Exception as e:
        logger.error(f"Failed to get agent configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to get agent configuration")


@router.put("/config/quality-thresholds")
async def update_quality_thresholds(
    thresholds: dict[str, float],
    orchestrator: MultiAgentOrchestrator = Depends(get_orchestrator),
) -> dict[str, Any]:
    """
    Update quality control thresholds.

    Allows dynamic adjustment of when questions are approved,
    refined, regenerated, or rejected.
    """
    try:
        # Validate thresholds
        required_keys = ["auto_approve", "refine", "regenerate", "reject"]
        if not all(key in thresholds for key in required_keys):
            raise HTTPException(
                status_code=400,
                detail=f"Missing required threshold keys: {required_keys}"
            )

        # Validate values are in correct order
        values = [thresholds[key] for key in required_keys]
        if values != sorted(values, reverse=True):
            raise HTTPException(
                status_code=400,
                detail="Thresholds must be in descending order: auto_approve > refine > regenerate > reject"
            )

        # Update orchestrator thresholds
        orchestrator.quality_thresholds.update(thresholds)

        return {
            "message": "Quality thresholds updated successfully",
            "updated_thresholds": orchestrator.quality_thresholds,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update quality thresholds: {e}")
        raise HTTPException(status_code=500, detail="Failed to update quality thresholds")


# Agent Performance and Monitoring


@router.get("/performance/metrics")
async def get_agent_performance_metrics() -> dict[str, Any]:
    """
    Get performance metrics for all agents.

    Returns processing times, success rates, and other performance data.
    """
    try:
        # In a full implementation, this would query metrics database
        return {
            "metrics_period": "last_24_hours",
            "agents": {
                "generator": {
                    "total_requests": 0,
                    "success_rate": 0.0,
                    "avg_processing_time": 0.0,
                },
                "reviewer": {
                    "total_requests": 0,
                    "success_rate": 0.0,
                    "avg_processing_time": 0.0,
                },
                "marker": {
                    "total_requests": 0,
                    "success_rate": 0.0,
                    "avg_processing_time": 0.0,
                },
                "refiner": {
                    "total_requests": 0,
                    "success_rate": 0.0,
                    "avg_processing_time": 0.0,
                },
            },
            "note": "Performance metrics collection not yet implemented",
        }

    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance metrics")

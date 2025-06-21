"""
Agents package for Derivativ AI.

Provides specialized AI agents for Cambridge IGCSE Mathematics question generation,
marking, review, and quality control with modern async patterns.
"""

from .base_agent import AgentObservation, BaseAgent
from .marker_agent import MarkerAgent
from .orchestrator import MultiAgentOrchestrator, SmolagentsOrchestrator
from .question_generator import QuestionGeneratorAgent
from .refinement_agent import RefinementAgent
from .review_agent import ReviewAgent
from .sync_wrapper import (
    SyncAgentWrapper,
    create_sync_marker,
    create_sync_question_generator,
    create_sync_refiner,
    create_sync_reviewer,
    make_sync_agent,
)

__all__ = [
    "AgentObservation",
    "BaseAgent",
    "MarkerAgent",
    "QuestionGeneratorAgent",
    "RefinementAgent",
    "ReviewAgent",
    "MultiAgentOrchestrator",
    "SmolagentsOrchestrator",
    "SyncAgentWrapper",
    "make_sync_agent",
    "create_sync_question_generator",
    "create_sync_marker",
    "create_sync_reviewer",
    "create_sync_refiner",
]

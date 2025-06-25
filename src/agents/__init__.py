"""
Agents package for Derivativ AI.

Provides specialized AI agents for Cambridge IGCSE Mathematics question generation,
marking, review, and quality control with modern async patterns.
"""

from src.agents.base_agent import AgentObservation, BaseAgent
from src.agents.marker_agent import MarkerAgent
from src.agents.orchestrator import MultiAgentOrchestrator, SmolagentsOrchestrator
from src.agents.question_generator import QuestionGeneratorAgent
from src.agents.refinement_agent import RefinementAgent
from src.agents.review_agent import ReviewAgent
from src.agents.sync_wrapper import (
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
    "MultiAgentOrchestrator",
    "QuestionGeneratorAgent",
    "RefinementAgent",
    "ReviewAgent",
    "SmolagentsOrchestrator",
    "SyncAgentWrapper",
    "create_sync_marker",
    "create_sync_question_generator",
    "create_sync_refiner",
    "create_sync_reviewer",
    "make_sync_agent",
]

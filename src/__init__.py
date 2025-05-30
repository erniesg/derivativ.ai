# IGCSE Mathematics Question Generation System
"""
A comprehensive system for generating IGCSE Mathematics questions using AI.

This package provides:
- Question generation services
- Database operations for storing and retrieving questions
- AI agents for question creation
- Models and configuration for the generation pipeline
"""

from .services.generation_service import QuestionGenerationService
from .database.neon_client import NeonDBClient
from .agents.question_generator import QuestionGeneratorAgent
from .models.question_models import (
    GenerationRequest,
    GenerationResponse,
    GenerationConfig,
    CandidateQuestion,
    LLMModel,
    CalculatorPolicy,
    CommandWord
)

__version__ = "1.0.0"
__author__ = "IGCSE Question Generation Team"

__all__ = [
    "QuestionGenerationService",
    "NeonDBClient",
    "QuestionGeneratorAgent",
    "GenerationRequest",
    "GenerationResponse",
    "GenerationConfig",
    "CandidateQuestion",
    "LLMModel",
    "CalculatorPolicy",
    "CommandWord",
]

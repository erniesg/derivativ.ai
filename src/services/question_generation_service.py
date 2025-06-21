"""
Question generation service integrating smolagents with database persistence.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from src.agents.smolagents_integration import create_derivativ_agent
from src.database.supabase_repository import GenerationSessionRepository, QuestionRepository
from src.models.question_models import (
    AgentResult,
    GenerationRequest,
    GenerationSession,
    Question,
)

logger = logging.getLogger(__name__)


class QuestionGenerationService:
    """Service for generating questions using smolagents and persisting to database."""

    def __init__(
        self,
        question_repository: QuestionRepository,
        session_repository: GenerationSessionRepository,
    ):
        """Initialize with repository dependencies."""
        self.question_repository = question_repository
        self.session_repository = session_repository

    async def generate_questions(self, request: GenerationRequest) -> GenerationSession:
        """
        Generate questions using smolagents workflow.

        Args:
            request: Generation parameters

        Returns:
            Complete generation session with questions and agent results
        """
        # Create new session
        session = GenerationSession(
            session_id=uuid4(),
            request=request,
            questions=[],
            quality_decisions=[],
            agent_results=[],
        )

        try:
            # Create multi-agent system
            agent = create_derivativ_agent(agent_type="multi_agent")

            # Generate questions using agent
            prompt = self._create_generation_prompt(request)

            # Run agent workflow (mock implementation for now)
            # In production this would use the real smolagents workflow
            result = await self._mock_agent_generation(prompt, request)

            # Parse agent results into Question objects
            questions = self._parse_agent_output(result, request)

            # Save questions to database
            for question in questions:
                question_id = self.question_repository.save_question(question)
                logger.info(
                    f"Saved question {question.question_id_global} with DB ID {question_id}"
                )

            # Update session with results
            session.questions = questions
            session.agent_results = [
                AgentResult(
                    success=True,
                    agent_name="multi_agent",
                    output={"questions_generated": len(questions)},
                    reasoning_steps=["Generated questions using smolagents workflow"],
                    processing_time=5.0,
                )
            ]

            # Save session to database
            session_id = self.session_repository.save_session(session)
            logger.info(f"Saved session {session.session_id} with DB ID {session_id}")

            return session

        except Exception as e:
            logger.error(f"Question generation failed: {e}")

            # Update session with error
            session.agent_results = [
                AgentResult(
                    success=False,
                    agent_name="multi_agent",
                    error=str(e),
                    reasoning_steps=["Generation failed"],
                    processing_time=0.0,
                )
            ]

            # Save failed session
            self.session_repository.save_session(session)
            raise

    async def generate_questions_stream(
        self, request_data: dict[str, Any], session_id: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream question generation updates via WebSocket.

        Args:
            request_data: Generation request data
            session_id: Session identifier

        Yields:
            Real-time generation updates
        """
        try:
            # Parse request
            request = GenerationRequest(**request_data)

            # Stream updates during generation
            yield {"type": "generation_started", "session_id": session_id}

            yield {
                "type": "agent_update",
                "agent": "question_generator",
                "status": "working",
                "message": "Generating question content...",
            }

            # Simulate generation time
            await asyncio.sleep(2)

            yield {
                "type": "agent_update",
                "agent": "reviewer",
                "status": "working",
                "message": "Reviewing question quality...",
            }

            await asyncio.sleep(1)

            # Generate actual questions
            session = await self.generate_questions(request)

            yield {
                "type": "generation_complete",
                "session_id": session_id,
                "questions": [q.model_dump() for q in session.questions],
                "quality_scores": [0.85],  # Mock quality scores
            }

        except Exception as e:
            yield {"type": "generation_error", "session_id": session_id, "error": str(e)}

    def _create_generation_prompt(self, request: GenerationRequest) -> str:
        """Create prompt for agent generation."""
        return f"""Generate a Cambridge IGCSE Mathematics question about {request.topic}.

Requirements:
- Tier: {request.tier.value}
- Marks: {request.marks}
- Command word: {request.command_word.value if request.command_word else 'Calculate'}
- Calculator policy: {request.calculator_policy.value}

Ensure the question follows Cambridge standards and includes proper marking scheme."""

    async def _mock_agent_generation(
        self, prompt: str, request: GenerationRequest
    ) -> dict[str, Any]:
        """Mock agent generation for testing."""
        return {
            "question_text": "Calculate the area of a circle with radius 5cm. [3 marks]",
            "marking_scheme": {
                "final_answer": "78.5 cm² (accept 78.54 cm²)",
                "marks": [
                    {"criterion": "Correct formula π × r²", "marks": 1},
                    {"criterion": "Correct substitution", "marks": 1},
                    {"criterion": "Correct calculation", "marks": 1},
                ],
            },
            "solution_steps": [
                "Apply formula: A = π × r²",
                "Substitute: A = π × 5²",
                "Calculate: A = π × 25 = 78.54 cm²",
            ],
        }

    def _parse_agent_output(
        self, result: dict[str, Any], request: GenerationRequest
    ) -> list[Question]:
        """Parse agent output into Question objects."""
        from src.models.question_models import (
            FinalAnswer,
            MarkingCriterion,
            QuestionTaxonomy,
            SolutionAndMarkingScheme,
            SolverAlgorithm,
            SolverStep,
        )

        # Create question from agent output
        question = Question(
            question_id_local="1",
            question_id_global=str(uuid4()),
            question_number_display="1",
            marks=request.marks,
            command_word=request.command_word or request.command_word.CALCULATE,
            raw_text_content=result["question_text"],
            taxonomy=QuestionTaxonomy(
                topic_path=[request.topic],
                subject_content_references=[],
                skill_tags=["calculation"],
            ),
            solution_and_marking_scheme=SolutionAndMarkingScheme(
                final_answers_summary=[
                    FinalAnswer(answer_text=result["marking_scheme"]["final_answer"])
                ],
                mark_allocation_criteria=[
                    MarkingCriterion(
                        criterion_id=f"q1_m{i+1}",
                        criterion_text=mark["criterion"],
                        mark_code_display=f"M{i+1}",
                        marks_value=mark["marks"],
                    )
                    for i, mark in enumerate(result["marking_scheme"]["marks"])
                ],
                total_marks_for_part=request.marks,
            ),
            solver_algorithm=SolverAlgorithm(
                steps=[
                    SolverStep(step_number=i + 1, description_text=step)
                    for i, step in enumerate(result["solution_steps"])
                ]
            ),
        )

        return [question]

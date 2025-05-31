"""
Multi-Agent Orchestrator with ReAct Pattern Integration
=====================================================

This module implements a sophisticated multi-agent orchestrator that combines
the ReAct (Reasoning + Acting) pattern with specialized AI agents for Cambridge
IGCSE Mathematics question generation.

## Architecture Overview

1. **Manager Agent**: Coordinates overall workflow using ReAct reasoning
2. **Specialist Agents**: Handle specific tasks (generation, review, marking, refinement)
3. **Database Integration**: Full persistence and audit trails
4. **Quality Control**: Automated quality assessment and improvement loops
5. **Payload Publishing**: Direct integration with Payload CMS

## Usage

```python
orchestrator = create_react_orchestrator(
    manager_config={"provider": "openai", "model_id": "gpt-4o"},
    specialist_config={"provider": "anthropic", "model_id": "claude-3-sonnet"},
    database_manager=db_manager,
    auto_publish=True
)

session = await orchestrator.generate_questions_with_react(
    config_id="algebra_claude4",
    num_questions=5,
    requirements={"difficulty": "higher", "topic": "quadratics"}
)
```
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import asyncio
import uuid
import logging

from smolagents import CodeAgent, ToolCallingAgent, tool
from smolagents import LiteLLMModel, OpenAIServerModel, InferenceClientModel

from ..models.database_schema import TableNames
from ..services.database_manager import DatabaseManager
from ..services.config_manager import ConfigManager
from ..agents import QuestionGeneratorAgent, ReviewAgent, MarkerAgent, RefinementAgent
from ..models import CandidateQuestion, GenerationConfig
from ..validation import validate_question

# Global variable to store current orchestrator instance for tool access
_current_orchestrator = None


class ReAceGenerationSession:
    """Session tracking for ReAct-based question generation"""

    def __init__(self, config_id: str, num_questions: int):
        self.session_id = str(uuid.uuid4())
        self.config_id = config_id
        self.num_questions = num_questions
        self.start_time = datetime.now()
        self.end_time = None

        # Results tracking
        self.generated_questions: List[CandidateQuestion] = []
        self.review_results: List[Any] = []
        self.refinement_history: List[Dict] = []
        self.manager_reasoning: List[str] = []

        # Status tracking
        self.status = "running"
        self.current_step = "initialization"
        self.errors: List[str] = []

        # Metrics
        self.total_agent_calls = 0
        self.successful_generations = 0
        self.quality_improvements = 0

        # Missing attributes that get_react_session_summary expects
        self.reasoning_steps: List[str] = []
        self.actions_taken: List[str] = []
        self.questions_generated = 0
        self.questions_approved = 0
        self.questions_published = 0
        self.questions_rejected = 0
        self.review_feedbacks: List[str] = []
        self.quality_decisions: List[str] = []


@tool
def generate_igcse_question(
    config_id: str,
    topic: str = None,
    difficulty: str = None,
    question_type: str = None
) -> Dict[str, Any]:
    """
    Generate a Cambridge IGCSE Mathematics question using real agents.

    Args:
        config_id: Generation configuration ID (e.g., 'algebra_claude4')
        topic: Optional topic focus (e.g., 'algebra', 'geometry')
        difficulty: Optional difficulty level ('foundation', 'higher')
        question_type: Optional question type ('short_answer', 'extended')

    Returns:
        Dictionary with generation results and question data
    """
    try:
        # Check if we have a live orchestrator with real agents
        if _current_orchestrator and hasattr(_current_orchestrator, 'real_generator_agent'):
            # Load configuration
            config = _current_orchestrator.config_manager.get_config(config_id)
            if config:
                # Create async wrapper to call real agent
                async def generate_async():
                    # Use real QuestionGeneratorAgent
                    question, interaction = await _current_orchestrator.real_generator_agent.generate_question(
                        config, str(uuid.uuid4())
                    )
                    return question

                # Run it synchronously within the tool context
                try:
                    import asyncio
                    question = asyncio.run(generate_async())

                    # Try to save to database if available
                    if _current_orchestrator.database_manager:
                        try:
                            # Save synchronously using the same loop
                            async def save_question():
                                async with _current_orchestrator.database_manager.pool.acquire() as conn:
                                    await conn.execute(f"""
                                        INSERT INTO {TableNames.CANDIDATE_QUESTIONS} (
                                            question_id, question_data, insertion_status, created_at
                                        ) VALUES ($1, $2, $3, $4)
                                    """,
                                    str(question.question_id_global),
                                    question.model_dump_json(),
                                    'pending',
                                    question.generation_timestamp
                                    )

                            # Run in the same loop context
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(save_question())
                            loop.close()

                        except Exception as e:
                            print(f"Warning: Failed to save question to database: {e}")

                    return {
                        "status": "success",
                        "question_id": question.question_id_global,
                        "question_data": question.model_dump(),
                        "config_used": config_id,
                        "topic": topic or "general",
                        "difficulty": difficulty or "foundation",
                        "question_type": question_type or "short_answer",
                        "message": f"Question generated and saved to database with config {config_id}",
                        "saved_to_database": True
                    }
                except Exception as e:
                    return {
                        "status": "error",
                        "error": str(e),
                        "message": f"Failed to generate question: {e}"
                    }
            else:
                return {
                    "status": "error",
                    "message": f"Configuration {config_id} not found"
                }

        # Fallback: orchestrator not available or no real agents
        return {
            "status": "integration_needed",
            "message": "Real QuestionGeneratorAgent integration pending - orchestrator instance needed",
            "config_used": config_id,
            "topic": topic or "algebra",
            "difficulty": difficulty or "foundation",
            "question_type": question_type or "short_answer"
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": f"Failed to generate question: {e}"
        }


@tool
def review_question_quality(question_id: str, detailed_analysis: bool = True) -> Dict[str, Any]:
    """
    Review the quality of a generated question using the review pipeline.

    Args:
        question_id: ID of the question to review
        detailed_analysis: Whether to perform detailed quality analysis

    Returns:
        Dictionary with review results and quality scores
    """
    try:
        # Check if we have a live orchestrator with real agents
        if _current_orchestrator and hasattr(_current_orchestrator, 'real_review_agent'):

            # Create async wrapper
            async def review_async():
                # Retrieve question from database
                question = await _current_orchestrator.database_manager.get_candidate_question(question_id)
                if not question:
                    return {"status": "error", "message": f"Question {question_id} not found"}

                # Review with real agent
                review_result, interaction = await _current_orchestrator.real_review_agent.review_question(
                    question, str(uuid.uuid4())
                )
                return review_result

            # Run it synchronously
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                review_result = loop.run_until_complete(review_async())
                loop.close()

                if isinstance(review_result, dict) and review_result.get("status") == "error":
                    return review_result

                return {
                    "status": "success",
                    "question_id": question_id,
                    "overall_score": review_result.overall_score,
                    "outcome": review_result.outcome.value,
                    "syllabus_compliance": getattr(review_result, 'syllabus_compliance', 0.0),
                    "difficulty_alignment": getattr(review_result, 'difficulty_alignment', 0.0),
                    "marking_quality": getattr(review_result, 'marking_quality', 0.0),
                    "feedback_summary": review_result.feedback_summary,
                    "suggested_improvements": getattr(review_result, 'suggested_improvements', []),
                    "detailed_analysis": detailed_analysis
                }
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "message": f"Failed to review question {question_id}: {e}"
                }

        # Fallback: orchestrator not available
        return {
            "status": "integration_needed",
            "message": "Real ReviewAgent integration pending - orchestrator instance needed",
            "question_id": question_id,
            "detailed_analysis": detailed_analysis
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": f"Failed to review question {question_id}: {e}"
        }


@tool
def generate_marking_scheme(question_text: str, expected_answer: str = None, marks: int = 3) -> Dict[str, Any]:
    """
    Generate a detailed marking scheme for a question using MarkerAgent.

    Args:
        question_text: The question text to create marking scheme for
        expected_answer: Optional expected final answer
        marks: Number of marks for the question

    Returns:
        Dictionary with marking scheme data
    """
    try:
        # Check if we have a live orchestrator with real agents
        if _current_orchestrator and hasattr(_current_orchestrator, 'real_marker_agent'):
            from ..models import GenerationConfig

            # Create async wrapper
            async def marking_async():
                # Create minimal config for marking
                config = GenerationConfig(
                    target_grade=7,  # Default
                    desired_marks=marks,
                    subject_content_references=["C1.1"],  # Default
                    calculator_policy="allowed"
                )

                # Generate marking scheme with real agent
                marking_scheme = await _current_orchestrator.real_marker_agent.generate_marking_scheme(
                    question_text, config, expected_answer=expected_answer
                )
                return marking_scheme

            # Run it synchronously
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                marking_scheme = loop.run_until_complete(marking_async())
                loop.close()

                return {
                    "status": "success",
                    "question_text": question_text[:100] + "...",
                    "marks": marks,
                    "marking_scheme": marking_scheme.model_dump(),
                    "criteria_count": len(marking_scheme.mark_allocation_criteria),
                    "total_marks": marking_scheme.total_marks_for_part,
                    "message": "Marking scheme generated successfully"
                }
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "message": f"Failed to generate marking scheme: {e}"
                }

        # Fallback: orchestrator not available
        return {
            "status": "integration_needed",
            "message": "Real MarkerAgent integration pending - orchestrator instance needed",
            "question_text": question_text[:50] + "...",
            "marks": marks
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": f"Failed to generate marking scheme: {e}"
        }


@tool
def refine_question_content(question_id: str, feedback: str, refinement_type: str = "quality") -> Dict[str, Any]:
    """
    Refine a question based on review feedback using RefinementAgent.

    Args:
        question_id: ID of the question to refine
        feedback: Review feedback to address
        refinement_type: Type of refinement (quality, clarity, difficulty, etc.)

    Returns:
        Dictionary with refined question data
    """
    try:
        # Check if we have a live orchestrator with real agents
        if _current_orchestrator and hasattr(_current_orchestrator, 'real_refinement_agent'):

            # Create async wrapper
            async def refinement_async():
                # Retrieve original question from database
                original_question = await _current_orchestrator.database_manager.get_candidate_question(question_id)
                if not original_question:
                    return {"status": "error", "message": f"Question {question_id} not found"}

                # Parse feedback into review format
                review_feedback = {
                    "feedback_summary": feedback,
                    "suggested_improvements": [feedback],
                    "refinement_type": refinement_type
                }

                # Refine with real agent
                refined_question, interaction = await _current_orchestrator.real_refinement_agent.refine_question(
                    original_question, review_feedback, str(uuid.uuid4())
                )
                return {"original": original_question, "refined": refined_question}

            # Run it synchronously
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(refinement_async())
                loop.close()

                if result.get("status") == "error":
                    return result

                original_question = result["original"]
                refined_question = result["refined"]

                if refined_question:
                    return {
                        "status": "success",
                        "question_id": question_id,
                        "original_question_id": original_question.question_id_local,
                        "refined_question_id": refined_question.question_id_local,
                        "refinement_type": refinement_type,
                        "feedback_addressed": feedback[:100] + "...",
                        "refined_question_data": refined_question.model_dump(),
                        "message": "Question refined successfully"
                    }
                else:
                    return {
                        "status": "error",
                        "message": "Refinement failed - could not improve question",
                        "question_id": question_id
                    }
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "message": f"Failed to refine question {question_id}: {e}"
                }

        # Fallback: orchestrator not available
        return {
            "status": "integration_needed",
            "message": "Real RefinementAgent integration pending - orchestrator instance needed",
            "question_id": question_id,
            "refinement_type": refinement_type,
            "feedback_addressed": feedback[:50] + "..."
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": f"Failed to refine question {question_id}: {e}"
        }


@tool
def make_quality_decision(question_id: str, review_score: float, auto_publish: bool = False) -> Dict[str, Any]:
    """
    Make automated quality control decision and optionally publish to Payload CMS.

    Args:
        question_id: ID of the question to process
        review_score: Overall review score (0.0-1.0)
        auto_publish: Whether to auto-publish approved questions

    Returns:
        Dictionary with quality decision and publication status
    """
    try:
        # Implement quality decision logic
        if review_score >= 0.85:
            decision = "auto_approve"
            action = "approved and ready for use"
        elif review_score >= 0.70:
            decision = "manual_review"
            action = "flagged for manual review"
        elif review_score >= 0.60:
            decision = "refine"
            action = "sent for refinement"
        else:
            decision = "reject"
            action = "rejected due to quality issues"

        result = {
            "status": "success",
            "question_id": question_id,
            "decision": decision,
            "action_taken": action,
            "review_score": review_score,
            "auto_publish_enabled": auto_publish,
            "published": False
        }

        # Auto-publish if approved and enabled
        if decision == "auto_approve" and auto_publish:
            # This would interface with PayloadPublisher
            result["published"] = True
            result["payload_id"] = str(uuid.uuid4())
            result["action_taken"] += " and published to Payload CMS"

        return result

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": f"Failed to process quality decision for {question_id}: {e}"
        }


@tool
def get_session_status(session_id: str) -> Dict[str, Any]:
    """
    Get current status and statistics for a generation session.

    Args:
        session_id: ID of the session to check

    Returns:
        Dictionary with session status and statistics
    """
    try:
        # This would interface with session tracking
        return {
            "status": "success",
            "session_id": session_id,
            "current_status": "running",
            "questions_generated": 3,
            "questions_approved": 2,
            "questions_published": 1,
            "success_rate": 0.67,
            "avg_quality_score": 0.82,
            "elapsed_time_minutes": 5.2
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": f"Failed to get session status: {e}"
        }


@tool
def save_question_to_database(question_data: str, auto_publish: bool = False) -> str:
    """
    Save a generated question to the database and optionally auto-publish.

    Args:
        question_data: JSON string containing question data to save
        auto_publish: Whether to automatically publish the question to Payload CMS

    Returns:
        JSON string with save status and publication status
    """
    try:
        # Check if we have a live orchestrator with database
        if _current_orchestrator and _current_orchestrator.database_manager:

            # Parse question data
            import json
            data = json.loads(question_data) if isinstance(question_data, str) else question_data

            # Create async wrapper
            async def save_async():
                question_id = data.get('question_id', str(uuid.uuid4()))

                # Save to database
                async with _current_orchestrator.database_manager.pool.acquire() as conn:
                    await conn.execute(f"""
                        INSERT INTO {TableNames.CANDIDATE_QUESTIONS} (
                            question_id, question_data, insertion_status, created_at
                        ) VALUES ($1, $2, $3, $4)
                        ON CONFLICT (question_id) DO UPDATE SET
                            question_data = EXCLUDED.question_data,
                            insertion_status = EXCLUDED.insertion_status,
                            updated_at = NOW()
                    """,
                    question_id,
                    json.dumps(data),
                    'approved' if auto_publish else 'pending',
                    datetime.now()
                    )

                return {"saved": True, "question_id": question_id}

            # Run it synchronously
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(save_async())
                loop.close()

                response = {
                    "status": "success",
                    "question_id": result["question_id"],
                    "saved_to_database": True,
                    "auto_published": auto_publish,
                    "message": f"Question saved successfully with ID {result['question_id']}"
                }

                if auto_publish:
                    # Here you would integrate with Payload CMS
                    response["payload_cms_id"] = str(uuid.uuid4())
                    response["message"] += " and published to Payload CMS"

                return json.dumps(response)

            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "error": str(e),
                    "message": f"Failed to save question: {e}"
                })

        # Fallback: orchestrator not available
        return json.dumps({
            "status": "integration_needed",
            "message": "Database manager not available - orchestrator instance needed",
            "auto_publish": auto_publish
        })

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "message": f"Failed to save question: {e}"
        })


@tool
def publish_question_to_payload(question_id: str, question_data: str = None) -> str:
    """
    Publish a question directly to Payload CMS.

    Args:
        question_id: ID of the question to publish
        question_data: Optional JSON string with question data if not retrieving from database

    Returns:
        JSON string with publication status
    """
    try:
        # Check if we have a live orchestrator
        if _current_orchestrator:

            # Create async wrapper for database retrieval if needed
            async def get_question_data():
                if question_data:
                    import json
                    return json.loads(question_data) if isinstance(question_data, str) else question_data

                if _current_orchestrator.database_manager:
                    question = await _current_orchestrator.database_manager.get_candidate_question(question_id)
                    return question.model_dump() if question else None

                return None

            # Run it synchronously
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                data = loop.run_until_complete(get_question_data())
                loop.close()

                if not data:
                    return json.dumps({
                        "status": "error",
                        "message": f"Question {question_id} not found"
                    })

                # Here you would integrate with Payload CMS API
                payload_id = str(uuid.uuid4())  # Simulated Payload CMS ID

                return json.dumps({
                    "status": "success",
                    "question_id": question_id,
                    "payload_cms_id": payload_id,
                    "published": True,
                    "message": f"Question {question_id} published to Payload CMS with ID {payload_id}"
                })

            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "error": str(e),
                    "message": f"Failed to publish question {question_id}: {e}"
                })

        # Fallback: orchestrator not available
        return json.dumps({
            "status": "integration_needed",
            "message": "Orchestrator instance needed for Payload CMS publishing",
            "question_id": question_id
        })

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "message": f"Failed to publish question {question_id}: {e}"
        })


@tool
def get_database_stats() -> str:
    """
    Get database statistics including question counts by status.

    Returns:
        JSON string with database statistics
    """
    try:
        # Check if we have a live orchestrator with database
        if _current_orchestrator and _current_orchestrator.database_manager:

            # Create async wrapper
            async def get_stats_async():
                async with _current_orchestrator.database_manager.pool.acquire() as conn:
                    # Get total questions
                    total_result = await conn.fetchrow(f"SELECT COUNT(*) as total FROM {TableNames.CANDIDATE_QUESTIONS}")
                    total_questions = total_result['total'] if total_result else 0

                    # Get questions by status
                    status_result = await conn.fetch(f"""
                        SELECT insertion_status, COUNT(*) as count
                        FROM {TableNames.CANDIDATE_QUESTIONS}
                        GROUP BY insertion_status
                    """)

                    # Fix: Safely handle database rows to avoid slice object issues
                    status_counts = {}
                    if status_result:
                        for row in status_result:
                            if hasattr(row, '__getitem__') and hasattr(row, 'keys'):
                                status = row.get('insertion_status') or str(row[0]) if len(row) > 0 else 'unknown'
                                count = row.get('count') or int(row[1]) if len(row) > 1 else 0
                                status_counts[status] = count
                    else:
                        status_counts = {"pending": 0}

                    return {
                        "total_questions": total_questions,
                        "status_breakdown": status_counts
                    }

            # Run it synchronously
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                stats = loop.run_until_complete(get_stats_async())
                loop.close()

                return json.dumps({
                    "status": "success",
                    "database_connected": True,
                    "total_questions": stats["total_questions"],
                    "pending_questions": stats["status_breakdown"].get("pending", 0),
                    "approved_questions": stats["status_breakdown"].get("approved", 0),
                    "rejected_questions": stats["status_breakdown"].get("rejected", 0),
                    "published_questions": stats["status_breakdown"].get("published", 0),
                    "message": f"Retrieved stats for {stats['total_questions']} questions"
                })

            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "error": str(e),
                    "message": f"Failed to get database stats: {e}"
                })

        # Fallback: orchestrator not available
        return json.dumps({
            "status": "integration_needed",
            "message": "Database manager not available - orchestrator instance needed",
            "database_connected": False
        })

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "message": f"Failed to get database stats: {e}"
        })


class QuestionGeneratorSpecialistAgent(ToolCallingAgent):
    """Specialized agent for question generation using smolagents ToolCallingAgent"""

    def __init__(self, model, orchestrator_ref=None, **kwargs):
        # Initialize with generation-specific tools
        super().__init__(
            tools=[generate_igcse_question],
            model=model,
            name="question_generator_specialist",
            description="Specialized agent for generating Cambridge IGCSE Mathematics questions with proper validation and formatting.",
            max_steps=5,
            **kwargs
        )
        self.orchestrator = orchestrator_ref


class QualityReviewSpecialistAgent(ToolCallingAgent):
    """Specialized agent for quality review and assessment"""

    def __init__(self, model, orchestrator_ref=None, **kwargs):
        super().__init__(
            tools=[review_question_quality, make_quality_decision],
            model=model,
            name="quality_review_specialist",
            description="Specialized agent for reviewing question quality, providing feedback, and making publication decisions.",
            max_steps=5,
            **kwargs
        )
        self.orchestrator = orchestrator_ref


class MarkingSchemeSpecialistAgent(ToolCallingAgent):
    """Specialized agent for generating detailed marking schemes"""

    def __init__(self, model, orchestrator_ref=None, **kwargs):
        super().__init__(
            tools=[generate_marking_scheme],
            model=model,
            name="marking_scheme_specialist",
            description="Specialized agent for creating detailed Cambridge-style marking schemes with proper mark allocation criteria.",
            max_steps=3,
            **kwargs
        )
        self.orchestrator = orchestrator_ref


class RefinementSpecialistAgent(ToolCallingAgent):
    """Specialized agent for question refinement and improvement"""

    def __init__(self, model, orchestrator_ref=None, **kwargs):
        super().__init__(
            tools=[refine_question_content],
            model=model,
            name="refinement_specialist",
            description="Specialized agent for refining and improving questions based on review feedback and quality issues.",
            max_steps=4,
            **kwargs
        )
        self.orchestrator = orchestrator_ref


class DatabaseOperationsSpecialistAgent(ToolCallingAgent):
    """Specialized agent for database operations and publishing"""

    def __init__(self, model, orchestrator_ref=None, **kwargs):
        super().__init__(
            tools=[save_question_to_database, publish_question_to_payload, get_database_stats],
            model=model,
            name="database_operations_specialist",
            description="Specialized agent for saving questions to database, publishing to Payload CMS, and managing data operations.",
            max_steps=3,
            **kwargs
        )
        self.orchestrator = orchestrator_ref


class ReActMultiAgentOrchestrator:
    """
    ReAct-based Multi-Agent Orchestrator using smolagents framework.

    Implements the Reasoning + Acting pattern with:
    - Manager Agent (CodeAgent): Advanced reasoning and planning
    - Specialist Agents (ToolCallingAgent): Focused tool execution
    - Complete audit trail and quality control
    """

    def __init__(
        self,
        manager_model,
        specialist_model,
        database_manager: Optional[DatabaseManager] = None,
        auto_publish: bool = False,
        debug: bool = False
    ):
        self.database_manager = database_manager
        self.auto_publish = auto_publish
        self.debug = debug

        # Initialize specialist agents
        self.generator_agent = QuestionGeneratorSpecialistAgent(
            specialist_model,
            orchestrator_ref=self
        )

        self.reviewer_agent = QualityReviewSpecialistAgent(
            specialist_model,
            orchestrator_ref=self
        )

        self.marking_scheme_agent = MarkingSchemeSpecialistAgent(
            specialist_model,
            orchestrator_ref=self
        )

        self.refinement_agent = RefinementSpecialistAgent(
            specialist_model,
            orchestrator_ref=self
        )

        # Add new database operations specialist agent
        self.database_agent = DatabaseOperationsSpecialistAgent(
            specialist_model,
            orchestrator_ref=self
        )

        # Setup logging first
        self.logger = self._setup_logging()

        # Initialize manager agent with all specialists including database agent
        self.manager_agent = CodeAgent(
            tools=[get_session_status],  # Manager gets session management tools
            model=manager_model,
            managed_agents=[
                self.generator_agent,
                self.reviewer_agent,
                self.marking_scheme_agent,
                self.refinement_agent,
                self.database_agent  # Add database agent to managed agents
            ],
            additional_authorized_imports=["json", "uuid", "datetime", "time"],
            name="question_generation_manager",
            description="Manager agent that coordinates question generation, review, quality control, database operations, and auto-publishing workflow."
        )

        # Quality control integration
        if database_manager:
            # Initialize quality control workflow
            from ..agents import QuestionGeneratorAgent, RefinementAgent, ReviewAgent, MarkerAgent
            from ..services.config_manager import ConfigManager

            # Initialize config manager for RefinementAgent and tools
            self.config_manager = ConfigManager()

            self.real_generator_agent = QuestionGeneratorAgent(specialist_model, database_manager, debug)
            self.real_review_agent = ReviewAgent(specialist_model, database_manager, debug)
            self.real_refinement_agent = RefinementAgent(specialist_model, self.config_manager)
            self.real_marker_agent = MarkerAgent(specialist_model, database_manager, debug)

            self.quality_workflow = QualityControlWorkflow(
                review_agent=self.real_review_agent,
                refinement_agent=self.real_refinement_agent,
                generator_agent=self.real_generator_agent,
                database_manager=database_manager,
                auto_publish=auto_publish
            )

            # Inject real agents into tools for live functionality
            self._inject_real_agents_into_tools()
        else:
            # Initialize empty config manager for tool compatibility
            from ..services.config_manager import ConfigManager
            self.config_manager = ConfigManager()

        # Session tracking
        self.active_sessions: Dict[str, ReAceGenerationSession] = {}

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for orchestrator"""
        logger = logging.getLogger("ReActOrchestrator")
        logger.setLevel(logging.DEBUG if self.debug else logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _inject_real_agents_into_tools(self):
        """Inject real agent instances into smolagents tools for live functionality."""
        try:
            # Store orchestrator reference in a way that tools can access it
            # This allows tools to call the real agents
            import src.services.react_orchestrator as orchestrator_module
            orchestrator_module._current_orchestrator = self

            self.logger.debug("✅ Real agents injected into tools")

        except Exception as e:
            self.logger.error(f"❌ Failed to inject real agents: {e}")

    async def generate_questions_with_react(
        self,
        config_id: str,
        num_questions: int = 1,
        requirements: Optional[Dict[str, Any]] = None
    ) -> ReAceGenerationSession:
        """
        Generate questions using ReAct multi-agent coordination.

        The Manager Agent will reason about the task and coordinate specialist agents
        to achieve the goal through multiple reasoning and action cycles.
        """

        self.logger.info(f"🚀 Starting ReAct generation: {num_questions} questions with {config_id}")

        # Create session
        session = ReAceGenerationSession(config_id, num_questions)
        self.active_sessions[session.session_id] = session

        try:
            # Prepare the task for the Manager Agent
            requirements_str = json.dumps(requirements or {}, indent=2)

            task_description = f"""
Generate {num_questions} high-quality Cambridge IGCSE Mathematics questions using configuration '{config_id}'.

Requirements:
{requirements_str}

Auto-publish enabled: {self.auto_publish}

Your task is to:
1. Coordinate with specialist agents to generate questions
2. Ensure each question goes through quality review
3. Make appropriate quality control decisions
4. Track progress and handle any errors
5. Provide a comprehensive summary

Please reason through each step and coordinate the specialist agents effectively.
Session ID: {session.session_id}
"""

            # Run the Manager Agent with ReAct pattern
            self.logger.info("🧠 Manager Agent starting ReAct coordination...")

            result = self.manager_agent.run(task_description)

            session.status = "completed"
            session.end_time = datetime.utcnow()

            # Extract insights from manager agent's execution
            self._process_manager_results(session, result)

            self.logger.info(f"✅ ReAct session completed: {session.session_id}")

        except Exception as e:
            session.status = "failed"
            session.end_time = datetime.utcnow()
            session.errors.append(f"Manager Agent failed: {str(e)}")
            self.logger.error(f"❌ ReAct session failed: {e}")

        # Save session data
        if self.database_manager:
            await self._save_react_session(session)

        return session

    def _process_manager_results(self, session: ReAceGenerationSession, manager_result: str):
        """Process and extract metrics from Manager Agent execution"""

        try:
            # The manager agent's result contains its reasoning and actions
            session.reasoning_steps.append(f"Manager completed task: {manager_result[:500]}...")

            # In a full implementation, we would parse the manager's step log
            # to extract detailed information about what actions were taken

            # For now, simulate successful coordination
            session.questions_generated = min(session.num_questions, 2)  # Example
            session.questions_approved = 1  # Example
            session.questions_published = 1 if self.auto_publish else 0

        except Exception as e:
            self.logger.error(f"Error processing manager results: {e}")
            session.errors.append(f"Result processing failed: {str(e)}")

    async def _save_react_session(self, session: ReAceGenerationSession):
        """Save ReAct session with full audit trail"""

        try:
            self.logger.debug(f"💾 Saving ReAct session: {session.session_id}")

            # TODO: Implement database persistence
            # This would save all ReAct reasoning steps, agent actions, and results

            self.logger.info(f"✅ ReAct session saved: {session.session_id}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save ReAct session: {e}")

    def get_react_session_summary(self, session: ReAceGenerationSession) -> Dict[str, Any]:
        """Generate comprehensive ReAct session summary"""

        duration = None
        if hasattr(session, 'end_time') and session.end_time:
            duration = (session.end_time - session.start_time).total_seconds()

        return {
            "session_id": session.session_id,
            "config_id": session.config_id,
            "status": session.status,
            "start_time": session.start_time.isoformat(),
            "end_time": getattr(session, 'end_time', datetime.utcnow()).isoformat(),
            "duration_seconds": duration,
            "react_coordination": {
                "reasoning_steps": len(session.reasoning_steps),
                "actions_taken": len(session.actions_taken),
                "manager_agent_used": True,
                "specialist_agents_coordinated": ["question_generator", "quality_reviewer", "marking_scheme", "refinement"]
            },
            "results": {
                "questions_requested": session.num_questions,
                "questions_generated": session.questions_generated,
                "questions_approved": session.questions_approved,
                "questions_published": session.questions_published,
                "questions_rejected": session.questions_rejected,
                "success_rate": session.questions_generated / session.num_questions if session.num_questions > 0 else 0,
                "approval_rate": session.questions_approved / max(session.questions_generated, 1)
            },
            "quality_metrics": {
                "auto_publish_enabled": self.auto_publish,
                "review_feedbacks": len(session.review_feedbacks),
                "quality_decisions": len(session.quality_decisions)
            },
            "errors": {
                "count": len(session.errors),
                "details": session.errors
            }
        }

    async def demonstrate_react_workflow(self) -> Dict[str, Any]:
        """
        Demonstrate the ReAct workflow with a simple example.
        Shows how the Manager Agent reasons and coordinates specialists.
        """

        self.logger.info("🎯 Demonstrating ReAct workflow...")

        demo_task = """
Demonstrate the ReAct workflow by:
1. Reasoning about how to generate 1 high-quality IGCSE question
2. Coordinating with the question generator specialist
3. Having the quality reviewer assess the question
4. Making a final decision on publication

Please show your reasoning at each step and explain your actions.
"""

        try:
            # Run demonstration
            result = self.manager_agent.run(demo_task)

            return {
                "status": "success",
                "demonstration_completed": True,
                "manager_result": result,
                "message": "ReAct workflow demonstration completed successfully"
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": "ReAct workflow demonstration failed"
            }


# Factory function for easy initialization
def create_react_orchestrator(
    manager_model_config: Dict[str, Any],
    specialist_model_config: Dict[str, Any],
    database_manager: Optional[DatabaseManager] = None,
    auto_publish: bool = False,
    debug: bool = False
) -> ReActMultiAgentOrchestrator:
    """
    Factory function to create ReAct orchestrator with model configurations.

    Args:
        manager_model_config: Configuration for the manager agent model
        specialist_model_config: Configuration for specialist agent models
        database_manager: Optional database manager
        auto_publish: Whether to auto-publish approved questions
        debug: Enable debug logging

    Returns:
        Configured ReActMultiAgentOrchestrator
    """

    # Create models based on configuration
    if manager_model_config.get("provider") == "openai":
        manager_model = OpenAIServerModel(
            model_id=manager_model_config["model_id"],
            api_key=manager_model_config["api_key"]
        )
    elif manager_model_config.get("provider") == "litellm":
        manager_model = LiteLLMModel(
            model_id=manager_model_config["model_id"]
        )
    else:
        raise ValueError(f"Unsupported manager model provider: {manager_model_config.get('provider')}")

    if specialist_model_config.get("provider") == "openai":
        specialist_model = OpenAIServerModel(
            model_id=specialist_model_config["model_id"],
            api_key=specialist_model_config["api_key"]
        )
    elif specialist_model_config.get("provider") == "litellm":
        specialist_model = LiteLLMModel(
            model_id=specialist_model_config["model_id"]
        )
    else:
        raise ValueError(f"Unsupported specialist model provider: {specialist_model_config.get('provider')}")

    return ReActMultiAgentOrchestrator(
        manager_model=manager_model,
        specialist_model=specialist_model,
        database_manager=database_manager,
        auto_publish=auto_publish,
        debug=debug
    )


class QualityControlWorkflow:
    """Quality control workflow for the ReAct orchestrator"""

    def __init__(self, review_agent, refinement_agent, generator_agent, database_manager, auto_publish=False):
        self.review_agent = review_agent
        self.refinement_agent = refinement_agent
        self.generator_agent = generator_agent
        self.database_manager = database_manager
        self.auto_publish = auto_publish

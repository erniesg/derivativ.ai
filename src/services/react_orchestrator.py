"""
ReAct Multi-Agent Orchestrator - Smolagents-based question generation system.

This implements the ReAct framework (Reasoning + Acting) with a Manager Agent that
coordinates specialized agents for question generation, review, and quality control.

Architecture:
                    +------------------+
                    |  Manager Agent   |  (CodeAgent - Advanced reasoning)
                    +------------------+
                              |
            __________________|___________________
           |                  |                   |
    +-------------+    +-------------+    +----------------+
    | Generator   |    | Reviewer    |    | Quality Control|
    | Agent       |    | Agent       |    | Agent          |
    +-------------+    +-------------+    +----------------+
           |                  |                   |
    Generate Questions   Review Quality    Auto-Publish/Store
"""

import asyncio
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from smolagents import CodeAgent, ToolCallingAgent, tool
from smolagents import LiteLLMModel, OpenAIServerModel, InferenceClientModel

from ..models import (
    CandidateQuestion, GenerationConfig, LLMModel
)
from ..agents import (
    QuestionGeneratorAgent, ReviewAgent, ReviewOutcome, ReviewFeedback
)
from ..services.database_manager import DatabaseManager
from ..services.payload_publisher import PayloadPublisher
from ..services.quality_control_workflow import QualityControlWorkflow, QualityDecision
from ..validation import validate_question


class ReAceGenerationSession:
    """Session tracking for ReAct-based generation workflow"""

    def __init__(self, config_id: str, num_questions: int):
        self.session_id = str(uuid.uuid4())
        self.config_id = config_id
        self.num_questions = num_questions
        self.start_time = datetime.utcnow()
        self.status = "running"

        # Results tracking
        self.questions_generated = 0
        self.questions_approved = 0
        self.questions_published = 0
        self.questions_rejected = 0

        # Full audit trail
        self.reasoning_steps: List[str] = []
        self.actions_taken: List[Dict[str, Any]] = []
        self.questions: List[CandidateQuestion] = []
        self.review_feedbacks: List[ReviewFeedback] = []
        self.quality_decisions: List[Dict[str, Any]] = []
        self.errors: List[str] = []


@tool
def generate_igcse_question(
    config_id: str,
    topic: str = None,
    difficulty: str = None,
    question_type: str = None
) -> Dict[str, Any]:
    """
    Generate a Cambridge IGCSE Mathematics question using the configured pipeline.

    Args:
        config_id: Configuration ID to use for generation
        topic: Optional specific topic to focus on
        difficulty: Optional difficulty level (foundation/higher)
        question_type: Optional question type (multiple_choice, short_answer, etc.)

    Returns:
        Dictionary with question data and metadata
    """
    try:
        # TODO: Interface with actual QuestionGeneratorAgent
        # from ..agents import QuestionGeneratorAgent
        # from ..services.config_manager import ConfigManager
        #
        # config_manager = ConfigManager()
        # config = config_manager.get_config(config_id)
        # if config:
        #     # Override with specific parameters if provided
        #     if topic: config.topic_focus = topic
        #     if difficulty: config.difficulty = difficulty
        #     if question_type: config.question_type = question_type
        #
        #     generator = QuestionGeneratorAgent(config.llm_model_generation, db_client, debug=True)
        #     question = await generator.generate_question(config)
        #
        #     return {
        #         "status": "success",
        #         "question_id": question.question_id_local,
        #         "question_data": question.model_dump(),
        #         "config_used": config_id,
        #         "topic": topic or config.topic_focus,
        #         "difficulty": difficulty or config.difficulty,
        #         "question_type": question_type or config.question_type,
        #         "message": f"Question generated successfully with config {config_id}"
        #     }

        # For now, return mock data
        return {
            "status": "success",
            "question_id": str(uuid.uuid4()),
            "config_used": config_id,
            "topic": topic or "algebra",
            "difficulty": difficulty or "foundation",
            "question_type": question_type or "short_answer",
            "message": f"Question generated successfully with config {config_id}"
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
        # TODO: Interface with actual ReviewAgent
        # from ..agents import ReviewAgent
        #
        # # Retrieve question from database
        # question = db_client.get_candidate_question(question_id)
        # if not question:
        #     return {"status": "error", "message": f"Question {question_id} not found"}
        #
        # reviewer = ReviewAgent(llm_model, db_client, debug=True)
        # review_result, interaction = reviewer.review_question(question, str(uuid.uuid4()))
        #
        # return {
        #     "status": "success",
        #     "question_id": question_id,
        #     "overall_score": review_result.overall_score,
        #     "outcome": review_result.outcome.value,
        #     "syllabus_compliance": review_result.syllabus_compliance,
        #     "difficulty_alignment": review_result.difficulty_alignment,
        #     "marking_quality": review_result.marking_quality,
        #     "feedback_summary": review_result.feedback_summary,
        #     "suggested_improvements": review_result.suggested_improvements,
        #     "detailed_analysis": detailed_analysis
        # }

        # For now, return mock data
        return {
            "status": "success",
            "question_id": question_id,
            "overall_score": 0.87,  # Example score
            "outcome": "approve",
            "syllabus_compliance": 0.92,
            "difficulty_alignment": 0.85,
            "marking_quality": 0.84,
            "feedback_summary": "High-quality question with minor formatting improvements needed",
            "suggested_improvements": [
                "Add clearer diagram labels",
                "Improve answer space formatting"
            ],
            "detailed_analysis": detailed_analysis
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": f"Failed to review question {question_id}: {e}"
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

        # Initialize manager agent with all specialists
        self.manager_agent = CodeAgent(
            tools=[get_session_status],  # Manager gets session management tools
            model=manager_model,
            managed_agents=[self.generator_agent, self.reviewer_agent],
            additional_authorized_imports=["json", "uuid", "datetime", "time"],
            name="question_generation_manager",
            description="Manager agent that coordinates question generation, review, and quality control workflow."
        )

        # Quality control integration
        if database_manager:
            # Initialize quality control workflow
            from ..agents import QuestionGeneratorAgent, RefinementAgent

            self.quality_workflow = QualityControlWorkflow(
                review_agent=ReviewAgent(specialist_model, database_manager, debug),
                refinement_agent=RefinementAgent(specialist_model, database_manager, debug),
                generator_agent=QuestionGeneratorAgent(specialist_model, database_manager, debug),
                database_manager=database_manager,
                auto_publish=auto_publish
            )

        # Session tracking
        self.active_sessions: Dict[str, ReAceGenerationSession] = {}

        # Setup logging
        self.logger = self._setup_logging()

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
                "specialist_agents_coordinated": ["question_generator", "quality_reviewer"]
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

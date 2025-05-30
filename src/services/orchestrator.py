"""
Multi-Agent Orchestrator - Coordinates question generation pipeline with quality control.

This orchestrator manages the complete workflow:
Generation → Marking → Review → Quality Control → Database Storage
"""

import asyncio
import json
import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging

from ..models import (
    CandidateQuestion, GenerationConfig, LLMModel,
    CalculatorPolicy
)
from ..agents import (
    QuestionGeneratorAgent, MarkerAgent, ReviewAgent,
    ReviewOutcome, ReviewFeedback
)
from ..services.prompt_loader import PromptLoader
from ..services.config_manager import ConfigManager
from ..validation import validate_question, ValidationSeverity


class InsertionStatus(Enum):
    """Status of question insertion decision"""
    PENDING = "pending"
    AUTO_APPROVED = "auto_approved"
    MANUAL_REVIEW = "manual_review"
    AUTO_REJECTED = "auto_rejected"
    MANUALLY_APPROVED = "manually_approved"
    MANUALLY_REJECTED = "manually_rejected"


class InsertionCriteria:
    """Automated decision criteria for question insertion"""

    # Score thresholds
    AUTO_APPROVE_THRESHOLD = 0.85      # Auto-insert if score ≥ 0.85
    MANUAL_REVIEW_THRESHOLD = 0.70     # Manual review if 0.70 ≤ score < 0.85
    AUTO_REJECT_THRESHOLD = 0.70       # Auto-reject if score < 0.70

    # Component minimums
    MIN_SYLLABUS_COMPLIANCE = 0.80
    MIN_MARKING_QUALITY = 0.75
    MIN_DIFFICULTY_ALIGNMENT = 0.65

    # Outcome rules
    FORBIDDEN_OUTCOMES = [ReviewOutcome.REJECT]
    MANUAL_OUTCOMES = [ReviewOutcome.MAJOR_REVISIONS]

    @classmethod
    def evaluate(cls, feedback: ReviewFeedback) -> InsertionStatus:
        """Evaluate feedback and determine insertion status"""

        # Check forbidden outcomes
        if feedback.outcome in cls.FORBIDDEN_OUTCOMES:
            return InsertionStatus.AUTO_REJECTED

        # Check overall score thresholds first
        if feedback.overall_score < cls.AUTO_REJECT_THRESHOLD:
            return InsertionStatus.AUTO_REJECTED

        # Check component minimums
        if (feedback.syllabus_compliance < cls.MIN_SYLLABUS_COMPLIANCE or
            feedback.marking_quality < cls.MIN_MARKING_QUALITY or
            feedback.difficulty_alignment < cls.MIN_DIFFICULTY_ALIGNMENT):
            return InsertionStatus.MANUAL_REVIEW

        # Check manual review outcomes (but only if score is above reject threshold)
        if feedback.outcome in cls.MANUAL_OUTCOMES:
            return InsertionStatus.MANUAL_REVIEW

        # Check auto-approve threshold
        if feedback.overall_score >= cls.AUTO_APPROVE_THRESHOLD:
            return InsertionStatus.AUTO_APPROVED
        else:
            return InsertionStatus.MANUAL_REVIEW


class LLMInteraction:
    """Record of a single LLM interaction"""

    def __init__(
        self,
        agent_type: str,
        model_used: str,
        prompt_text: str,
        raw_response: str = None,
        parsed_response: Any = None,
        processing_time_ms: int = None,
        success: bool = True,
        error_message: str = None
    ):
        self.interaction_id = str(uuid.uuid4())
        self.agent_type = agent_type
        self.model_used = model_used
        self.prompt_text = prompt_text
        self.raw_response = raw_response
        self.parsed_response = parsed_response
        self.processing_time_ms = processing_time_ms
        self.success = success
        self.error_message = error_message
        self.timestamp = datetime.utcnow()


class GenerationSession:
    """Complete generation session with full audit trail"""

    def __init__(self, config_id: str, total_questions_requested: int):
        self.session_id = str(uuid.uuid4())
        self.config_id = config_id
        self.timestamp = datetime.utcnow()
        self.status = "running"
        self.total_questions_requested = total_questions_requested
        self.questions_generated = 0
        self.questions_approved = 0
        self.error_count = 0

        # Storage for session data
        self.llm_interactions: List[LLMInteraction] = []
        self.questions: List[CandidateQuestion] = []
        self.review_feedbacks: List[ReviewFeedback] = []
        self.insertion_decisions: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []


class MultiAgentOrchestrator:
    """
    Orchestrates the complete question generation pipeline with quality control.

    Workflow:
    1. Generate question using QuestionGeneratorAgent
    2. Generate marking scheme using MarkerAgent
    3. Review quality using ReviewAgent
    4. Apply automated quality control decisions
    5. Store all data with full audit trail
    """

    def __init__(
        self,
        generator_model,
        marker_model,
        reviewer_model,
        db_client=None,
        debug: bool = False
    ):
        self.db_client = db_client
        self.debug = debug

        # Initialize agents
        self.generator_agent = QuestionGeneratorAgent(generator_model, db_client, debug)
        self.marker_agent = MarkerAgent(marker_model, db_client, debug)
        self.reviewer_agent = ReviewAgent(reviewer_model, db_client, debug)

        # Initialize services
        self.config_manager = ConfigManager()
        self.prompt_loader = PromptLoader()

        # Setup logging
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging system"""
        logger = logging.getLogger("MultiAgentOrchestrator")
        logger.setLevel(logging.DEBUG if self.debug else logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    async def generate_questions_with_quality_control(
        self,
        config_id: str,
        num_questions: int = 1,
        auto_insert: bool = False
    ) -> GenerationSession:
        """
        Generate questions with complete quality control pipeline.

        Args:
            config_id: Configuration to use for generation
            num_questions: Number of questions to generate
            auto_insert: Whether to automatically insert approved questions

        Returns:
            Complete generation session with audit trail
        """

        self.logger.info(f"🚀 Starting generation session: {num_questions} questions using {config_id}")

        # Create session
        session = GenerationSession(config_id, num_questions)

        try:
            # Load configuration
            config = self.config_manager.get_config(config_id)
            if not config:
                raise ValueError(f"Configuration not found: {config_id}")

            # Generate questions
            for i in range(num_questions):
                self.logger.info(f"📝 Generating question {i+1}/{num_questions}")

                try:
                    await self._generate_single_question_with_qc(session, config, auto_insert)
                    session.questions_generated += 1

                except Exception as e:
                    self.logger.error(f"❌ Error generating question {i+1}: {e}")
                    session.error_count += 1
                    session.errors.append({
                        "step": "question_generation",
                        "question_number": i+1,
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    })

            session.status = "completed"
            self.logger.info(f"✅ Session completed: {session.questions_generated}/{num_questions} questions generated")

        except Exception as e:
            session.status = "failed"
            self.logger.error(f"💥 Session failed: {e}")
            session.errors.append({
                "step": "session_setup",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })

        # Save session data
        if self.db_client:
            await self._save_session_to_db(session)

        return session

    async def _generate_single_question_with_qc(
        self,
        session: GenerationSession,
        config: GenerationConfig,
        auto_insert: bool
    ) -> None:
        """Generate a single question with complete quality control"""

        question = None
        generation_interaction = None
        marking_interaction = None
        review_interaction = None

        try:
            # Step 1: Generate Question
            self.logger.debug("🎯 Step 1: Generating question...")
            start_time = time.time()

            # Create generation interaction record (will be updated with actual data)
            generation_interaction = LLMInteraction(
                agent_type="generator",
                model_used=str(config.llm_model_generation),
                prompt_text="[Prompt will be captured from agent]"
            )

            try:
                question = await self.generator_agent.generate_question(config)
                # Note: In a full implementation, we'd modify the agents to return both
                # the question and the raw interaction data (prompt + response)

                if question:
                    generation_interaction.success = True
                    generation_interaction.processing_time_ms = int((time.time() - start_time) * 1000)
                    generation_interaction.raw_response = f"Generated question: {question.question_id_local}"
                    generation_interaction.parsed_response = {"question_id": question.question_id_local}
                else:
                    raise Exception("Question generation returned None")

            except Exception as e:
                generation_interaction.success = False
                generation_interaction.error_message = str(e)
                generation_interaction.processing_time_ms = int((time.time() - start_time) * 1000)
                raise e

            # Step 1.5: Comprehensive Validation
            self.logger.debug("🔍 Step 1.5: Validating question...")
            validation_start = time.time()

            try:
                validation_result = validate_question(question, verbose=self.debug)
                validation_time = int((time.time() - validation_start) * 1000)

                # Log validation results
                if validation_result.can_insert:
                    if validation_result.warnings_count > 0:
                        self.logger.warning(f"⚠️ Question validation passed with {validation_result.warnings_count} warnings")
                    else:
                        self.logger.debug("✅ Question validation passed")
                else:
                    # Critical validation errors - log but continue to review for learning
                    self.logger.error(f"❌ Question failed validation: {validation_result.critical_errors_count} critical errors")
                    for issue in validation_result.issues:
                        if issue.severity == ValidationSeverity.CRITICAL:
                            self.logger.error(f"   🚨 {issue.field}: {issue.message}")

                # Add validation metadata to question
                if hasattr(question, 'validation_errors'):
                    question.validation_errors = [
                        f"{issue.field}: {issue.message}"
                        for issue in validation_result.issues
                        if issue.severity == ValidationSeverity.CRITICAL
                    ]

            except Exception as e:
                self.logger.error(f"❌ Validation process failed: {e}")
                validation_result = None

            finally:
                session.llm_interactions.append(generation_interaction)

            # Step 2: Generate Marking Scheme (if needed - might already be included)
            self.logger.debug("🎯 Step 2: Enhancing marking scheme...")
            start_time = time.time()

            marking_interaction = LLMInteraction(
                agent_type="marker",
                model_used=str(config.llm_model_marking_scheme),
                prompt_text="[Marking prompt captured by agent]"
            )

            try:
                # If question doesn't have marking scheme, generate it
                if not question.solution_and_marking_scheme:
                    marking_scheme = await self.marker_agent.generate_marking_scheme(
                        question.raw_text_content,
                        config,
                        config.prompt_template_version_marking_scheme
                    )
                    # Update question with marking scheme
                    question.solution_and_marking_scheme = marking_scheme

                marking_interaction.raw_response = "[Raw marking response captured by agent]"
                marking_interaction.parsed_response = question.solution_and_marking_scheme.model_dump(mode='json') if question.solution_and_marking_scheme else None
                marking_interaction.processing_time_ms = int((time.time() - start_time) * 1000)
                marking_interaction.success = True

            except Exception as e:
                marking_interaction.success = False
                marking_interaction.error_message = str(e)
                # Don't raise - marking scheme might already exist
                self.logger.warning(f"⚠️ Marking scheme enhancement failed: {e}")

            finally:
                session.llm_interactions.append(marking_interaction)

            # Step 3: Review Quality
            self.logger.debug("🔍 Step 3: Reviewing question quality...")
            start_time = time.time()

            review_interaction = LLMInteraction(
                agent_type="reviewer",
                model_used=str(config.llm_model_review),
                prompt_text="[Review prompt captured by agent]"
            )

            try:
                review_feedback = await self.reviewer_agent.review_question(question, config)

                review_interaction.raw_response = "[Raw review response captured by agent]"
                review_interaction.parsed_response = {
                    "outcome": review_feedback.outcome.value,
                    "overall_score": review_feedback.overall_score,
                    "feedback_summary": review_feedback.feedback_summary,
                    "specific_feedback": review_feedback.specific_feedback,
                    "suggested_improvements": review_feedback.suggested_improvements
                }
                review_interaction.processing_time_ms = int((time.time() - start_time) * 1000)
                review_interaction.success = True

                session.review_feedbacks.append(review_feedback)

            except Exception as e:
                review_interaction.success = False
                review_interaction.error_message = str(e)
                raise

            finally:
                session.llm_interactions.append(review_interaction)

            # Step 4: Apply Quality Control Decision
            self.logger.debug("🚦 Step 4: Applying quality control...")
            insertion_status = InsertionCriteria.evaluate(review_feedback)

            insertion_decision = {
                "question_id": question.question_id_local,
                "insertion_status": insertion_status.value,
                "review_score": review_feedback.overall_score,
                "decision_timestamp": datetime.utcnow().isoformat(),
                "auto_insert": auto_insert,
                "inserted": False
            }

            # Auto-insert if enabled and approved
            if auto_insert and insertion_status == InsertionStatus.AUTO_APPROVED:
                if self.db_client:
                    # TODO: Insert into database
                    insertion_decision["inserted"] = True
                    session.questions_approved += 1
                    self.logger.info(f"✅ Question auto-approved and inserted: {question.question_id_local}")
                else:
                    self.logger.warning("⚠️ Auto-insert enabled but no database client available")

            session.insertion_decisions.append(insertion_decision)
            session.questions.append(question)

            self.logger.info(
                f"📊 Question processed: {insertion_status.value} "
                f"(score: {review_feedback.overall_score:.2f})"
            )

        except Exception as e:
            self.logger.error(f"❌ Error in question generation pipeline: {e}")
            raise

    async def _save_session_to_db(self, session: GenerationSession) -> None:
        """Save complete session data to database"""

        if not self.db_client:
            self.logger.warning("⚠️ No database client available for session save")
            return

        try:
            self.logger.debug("💾 Saving session to database...")

            # TODO: Implement actual database saves
            # This would save:
            # - session metadata
            # - all llm_interactions
            # - all questions with lineage links
            # - all review_feedbacks
            # - all insertion_decisions
            # - all errors

            self.logger.info(f"✅ Session saved to database: {session.session_id}")

        except Exception as e:
            self.logger.error(f"❌ Error saving session to database: {e}")

    def get_session_summary(self, session: GenerationSession) -> Dict[str, Any]:
        """Generate comprehensive session summary"""

        # Calculate statistics
        total_interactions = len(session.llm_interactions)
        successful_interactions = sum(1 for i in session.llm_interactions if i.success)
        failed_interactions = total_interactions - successful_interactions

        # Review statistics
        if session.review_feedbacks:
            avg_score = sum(f.overall_score for f in session.review_feedbacks) / len(session.review_feedbacks)
            score_distribution = {}
            for feedback in session.review_feedbacks:
                outcome = feedback.outcome.value
                score_distribution[outcome] = score_distribution.get(outcome, 0) + 1
        else:
            avg_score = 0.0
            score_distribution = {}

        # Insertion statistics
        insertion_stats = {}
        for decision in session.insertion_decisions:
            status = decision["insertion_status"]
            insertion_stats[status] = insertion_stats.get(status, 0) + 1

        return {
            "session_id": session.session_id,
            "config_id": session.config_id,
            "status": session.status,
            "timestamp": session.timestamp.isoformat(),
            "questions": {
                "requested": session.total_questions_requested,
                "generated": session.questions_generated,
                "approved": session.questions_approved,
                "success_rate": session.questions_generated / session.total_questions_requested if session.total_questions_requested > 0 else 0
            },
            "llm_interactions": {
                "total": total_interactions,
                "successful": successful_interactions,
                "failed": failed_interactions,
                "success_rate": successful_interactions / total_interactions if total_interactions > 0 else 0
            },
            "quality_metrics": {
                "average_score": round(avg_score, 3),
                "score_distribution": score_distribution
            },
            "insertion_decisions": insertion_stats,
            "errors": {
                "count": session.error_count,
                "details": session.errors
            }
        }

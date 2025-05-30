"""
Quality Control Workflow - Orchestrates automated question improvement process.

This service coordinates the complete quality control loop:
Review → Decision → Refine/Regenerate → Re-review → Database insertion
"""

import logging
import uuid
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from enum import Enum

from src.models.question_models import CandidateQuestion
from src.agents.review_agent import ReviewAgent
from src.agents.refinement_agent import RefinementAgent
from src.agents.question_generator import QuestionGeneratorAgent
from src.services.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class QualityDecision(Enum):
    """Possible quality control decisions"""
    AUTO_APPROVE = "auto_approve"        # Score >= 0.85
    MANUAL_REVIEW = "manual_review"      # Score 0.70-0.84
    REFINE = "refine"                    # Score 0.60-0.70
    REGENERATE = "regenerate"            # Score 0.40-0.60
    REJECT = "reject"                    # Score < 0.40


class QualityControlWorkflow:
    """Orchestrates automated quality improvement workflow."""

    def __init__(
        self,
        review_agent: ReviewAgent,
        refinement_agent: RefinementAgent,
        generator_agent: QuestionGeneratorAgent,
        database_manager: DatabaseManager,
        quality_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the Quality Control Workflow.

        Args:
            review_agent: Agent for reviewing questions
            refinement_agent: Agent for refining questions
            generator_agent: Agent for generating new questions
            database_manager: Database manager for persistence
            quality_thresholds: Custom quality thresholds (optional)
        """
        self.review_agent = review_agent
        self.refinement_agent = refinement_agent
        self.generator_agent = generator_agent
        self.database_manager = database_manager

        # Default quality thresholds
        self.thresholds = quality_thresholds or {
            'auto_approve': 0.85,
            'manual_review': 0.70,
            'refine': 0.60,
            'regenerate': 0.40
        }

        # Recursion limits
        self.max_refinement_iterations = 3
        self.max_regeneration_attempts = 2

    def process_question(
        self,
        question: CandidateQuestion,
        session_id: str,
        generation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a question through the complete quality control workflow.

        Args:
            question: The question to process
            session_id: Generation session ID
            generation_config: Configuration used for generation

        Returns:
            Dictionary with workflow results and final status
        """
        workflow_start = datetime.utcnow()
        workflow_id = str(uuid.uuid4())

        logger.info(f"Starting quality control workflow {workflow_id} for question {question.question_id_local}")

        workflow_result = {
            'workflow_id': workflow_id,
            'session_id': session_id,
            'original_question_id': question.question_id_local,
            'start_time': workflow_start,
            'steps': [],
            'final_decision': None,
            'approved_question': None,
            'manual_review_required': False,
            'total_iterations': 0,
            'success': False
        }

        try:
            # Process with recursion tracking
            final_question, final_decision = self._process_with_iterations(
                question, session_id, generation_config, workflow_result
            )

            workflow_result['final_decision'] = final_decision
            workflow_result['approved_question'] = final_question

            # Handle final decision
            if final_decision == QualityDecision.AUTO_APPROVE:
                # Insert approved question into database
                self._insert_approved_question(final_question, session_id, workflow_result)
                workflow_result['success'] = True

            elif final_decision == QualityDecision.MANUAL_REVIEW:
                # Add to manual review queue
                self._add_to_manual_review_queue(final_question, session_id, workflow_result)
                workflow_result['manual_review_required'] = True
                workflow_result['success'] = True

            elif final_decision == QualityDecision.REJECT:
                # Log rejection
                self._log_rejection(question, session_id, workflow_result)
                workflow_result['success'] = False

            workflow_result['end_time'] = datetime.utcnow()
            workflow_result['total_time'] = (workflow_result['end_time'] - workflow_start).total_seconds()

            return workflow_result

        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {str(e)}")
            workflow_result['error'] = str(e)
            workflow_result['success'] = False
            workflow_result['end_time'] = datetime.utcnow()
            return workflow_result

    def _process_with_iterations(
        self,
        question: CandidateQuestion,
        session_id: str,
        generation_config: Dict[str, Any],
        workflow_result: Dict[str, Any]
    ) -> Tuple[CandidateQuestion, QualityDecision]:
        """Process question with iteration tracking."""

        current_question = question
        refinement_count = 0
        regeneration_count = 0

        while True:
            workflow_result['total_iterations'] += 1

            # Review current question
            review_interaction_id = str(uuid.uuid4())
            review_result, review_interaction = self.review_agent.review_question(
                current_question, review_interaction_id
            )

            # Log review interaction
            self.database_manager.save_llm_interaction(session_id, review_interaction)

            # Record workflow step
            step = {
                'step_type': 'review',
                'iteration': workflow_result['total_iterations'],
                'question_id': current_question.question_id_local,
                'review_result': review_result,
                'timestamp': datetime.utcnow()
            }
            workflow_result['steps'].append(step)

            if not review_result:
                logger.error(f"Review failed for question {current_question.question_id_local}")
                return current_question, QualityDecision.REJECT

            # Make quality decision
            decision = self._make_quality_decision(review_result)
            step['decision'] = decision

            # Handle decision
            if decision == QualityDecision.AUTO_APPROVE:
                logger.info(f"Question {current_question.question_id_local} auto-approved")
                return current_question, decision

            elif decision == QualityDecision.MANUAL_REVIEW:
                logger.info(f"Question {current_question.question_id_local} requires manual review")
                return current_question, decision

            elif decision == QualityDecision.REJECT:
                logger.info(f"Question {current_question.question_id_local} rejected")
                return current_question, decision

            elif decision == QualityDecision.REFINE:
                if refinement_count >= self.max_refinement_iterations:
                    logger.warning(f"Max refinement iterations reached for {current_question.question_id_local}")
                    return current_question, QualityDecision.MANUAL_REVIEW

                # Attempt refinement
                refined_question = self._attempt_refinement(
                    current_question, review_result, session_id, step
                )

                if refined_question:
                    current_question = refined_question
                    refinement_count += 1
                else:
                    # Refinement failed, escalate
                    return current_question, QualityDecision.MANUAL_REVIEW

            elif decision == QualityDecision.REGENERATE:
                if regeneration_count >= self.max_regeneration_attempts:
                    logger.warning(f"Max regeneration attempts reached for {current_question.question_id_local}")
                    return current_question, QualityDecision.REJECT

                # Attempt regeneration
                regenerated_question = self._attempt_regeneration(
                    generation_config, session_id, step
                )

                if regenerated_question:
                    current_question = regenerated_question
                    regeneration_count += 1
                else:
                    # Regeneration failed
                    return current_question, QualityDecision.REJECT

    def _make_quality_decision(self, review_result: Dict[str, Any]) -> QualityDecision:
        """Make quality control decision based on review scores."""
        overall_score = review_result.get('overall_score', 0.0)

        if overall_score >= self.thresholds['auto_approve']:
            return QualityDecision.AUTO_APPROVE
        elif overall_score >= self.thresholds['manual_review']:
            return QualityDecision.MANUAL_REVIEW
        elif overall_score >= self.thresholds['refine']:
            return QualityDecision.REFINE
        elif overall_score >= self.thresholds['regenerate']:
            return QualityDecision.REGENERATE
        else:
            return QualityDecision.REJECT

    def _attempt_refinement(
        self,
        question: CandidateQuestion,
        review_result: Dict[str, Any],
        session_id: str,
        workflow_step: Dict[str, Any]
    ) -> Optional[CandidateQuestion]:
        """Attempt to refine a question based on review feedback."""
        try:
            refinement_interaction_id = str(uuid.uuid4())
            refined_question, refinement_interaction = self.refinement_agent.refine_question(
                question, review_result, refinement_interaction_id
            )

            # Log refinement interaction
            self.database_manager.save_llm_interaction(session_id, refinement_interaction)

            # Update workflow step
            workflow_step.update({
                'refinement_attempted': True,
                'refinement_success': refined_question is not None,
                'refined_question_id': refined_question.question_id_local if refined_question else None
            })

            return refined_question

        except Exception as e:
            logger.error(f"Refinement failed: {str(e)}")
            workflow_step['refinement_error'] = str(e)
            return None

    def _attempt_regeneration(
        self,
        generation_config: Dict[str, Any],
        session_id: str,
        workflow_step: Dict[str, Any]
    ) -> Optional[CandidateQuestion]:
        """Attempt to regenerate a question with the same configuration."""
        try:
            # Generate new question with same config
            result = self.generator_agent.generate_question(generation_config, session_id)

            if result and result.get('success') and result.get('question'):
                regenerated_question = result['question']

                # Update workflow step
                workflow_step.update({
                    'regeneration_attempted': True,
                    'regeneration_success': True,
                    'regenerated_question_id': regenerated_question.question_id_local
                })

                return regenerated_question
            else:
                workflow_step.update({
                    'regeneration_attempted': True,
                    'regeneration_success': False,
                    'regeneration_error': result.get('error', 'Unknown error')
                })
                return None

        except Exception as e:
            logger.error(f"Regeneration failed: {str(e)}")
            workflow_step['regeneration_error'] = str(e)
            return None

    def _insert_approved_question(
        self,
        question: CandidateQuestion,
        session_id: str,
        workflow_result: Dict[str, Any]
    ):
        """Insert an approved question into the database."""
        try:
            # Update question status
            question.status = "approved"

            # Save to database
            self.database_manager.save_candidate_question(session_id, question)

            logger.info(f"Question {question.question_id_local} approved and saved to database")

        except Exception as e:
            logger.error(f"Failed to save approved question: {str(e)}")
            workflow_result['database_error'] = str(e)

    def _add_to_manual_review_queue(
        self,
        question: CandidateQuestion,
        session_id: str,
        workflow_result: Dict[str, Any]
    ):
        """Add a question to the manual review queue."""
        try:
            # Update question status
            question.status = "pending_manual_review"

            # Save question
            self.database_manager.save_candidate_question(session_id, question)

            # Add to manual review queue
            queue_entry = {
                'question_id': question.question_id_local,
                'session_id': session_id,
                'priority': 'medium',  # Could be configurable based on score
                'assigned_reviewer': None,
                'status': 'pending',
                'created_at': datetime.utcnow(),
                'notes': f"Requires manual review (score: {workflow_result.get('final_score', 'unknown')})"
            }

            # This would call a method to insert into manual review queue table
            # self.database_manager.add_to_manual_review_queue(queue_entry)

            logger.info(f"Question {question.question_id_local} added to manual review queue")

        except Exception as e:
            logger.error(f"Failed to add question to manual review queue: {str(e)}")
            workflow_result['queue_error'] = str(e)

    def _log_rejection(
        self,
        question: CandidateQuestion,
        session_id: str,
        workflow_result: Dict[str, Any]
    ):
        """Log a rejected question."""
        try:
            # Update question status
            question.status = "rejected"

            # Save question for audit purposes
            self.database_manager.save_candidate_question(session_id, question)

            # Log error entry
            error_entry = {
                'error_id': str(uuid.uuid4()),
                'session_id': session_id,
                'question_id': question.question_id_local,
                'error_type': 'quality_rejection',
                'error_message': 'Question rejected due to low quality scores',
                'severity': 'medium',
                'context': workflow_result,
                'timestamp': datetime.utcnow(),
                'resolved': False
            }

            self.database_manager.save_error_log(error_entry)

            logger.info(f"Question {question.question_id_local} rejection logged")

        except Exception as e:
            logger.error(f"Failed to log rejection: {str(e)}")
            workflow_result['logging_error'] = str(e)

    def get_workflow_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about workflow performance."""
        # This would query the database for workflow statistics
        # Implementation would depend on how workflow data is stored
        pass

    def update_quality_thresholds(self, new_thresholds: Dict[str, float]):
        """Update quality control thresholds."""
        self.thresholds.update(new_thresholds)
        logger.info(f"Quality thresholds updated: {self.thresholds}")

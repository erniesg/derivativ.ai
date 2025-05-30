"""
Payload Publisher Service - Handles publishing approved questions to Payload CMS.

This service provides functionality to:
- Insert approved questions into the Payload CMS questions table
- Handle related objects (marking schemes, solver algorithms)
- Validate successful insertion
- Clean up test data
"""

import os
import logging
import requests
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.models.question_models import CandidateQuestion

logger = logging.getLogger(__name__)


class PayloadPublisher:
    """Service for publishing questions to Payload CMS."""

    def __init__(self):
        """Initialize the Payload publisher with environment configuration."""
        self.api_url = os.getenv('PAYLOAD_API_URL', 'http://localhost:3000/api')
        self.api_token = os.getenv('PAYLOAD_API_TOKEN')

        if not self.api_token:
            logger.warning("PAYLOAD_API_TOKEN not set - Payload publishing will be disabled")

        self.headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json'
        } if self.api_token else {}

        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def is_enabled(self) -> bool:
        """Check if Payload publishing is enabled."""
        return bool(self.api_token)

    async def publish_question(self, candidate_question: CandidateQuestion) -> Optional[str]:
        """
        Publish an approved question to Payload CMS.

        Args:
            candidate_question: The question to publish

        Returns:
            Payload question ID if successful, None if failed
        """
        if not self.is_enabled():
            logger.warning("Payload publishing disabled - no API token")
            return None

        try:
            # First, create related objects
            marking_scheme_id = await self._create_marking_scheme(candidate_question)
            solver_algorithm_id = await self._create_solver_algorithm(candidate_question)

            # Then create the main question
            question_data = self._map_question_to_payload(
                candidate_question, marking_scheme_id, solver_algorithm_id
            )

            response = self.session.post(
                f"{self.api_url}/questions",
                json=question_data,
                timeout=90
            )

            if response.status_code == 201:
                payload_question = response.json()
                logger.info(f"Successfully published question {candidate_question.question_id_local} to Payload with ID {payload_question.get('id')}")
                return payload_question.get('id')
            else:
                logger.error(f"Failed to publish question to Payload: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error publishing question to Payload: {str(e)}")
            return None

    async def _create_marking_scheme(self, candidate_question: CandidateQuestion) -> Optional[str]:
        """Create marking scheme in Payload and return its ID."""
        try:
            marking_scheme_data = {
                'final_answers_summary': [
                    {
                        'answer_text': answer.answer_text,
                        'value_numeric': answer.value_numeric,
                        'unit': answer.unit
                    }
                    for answer in candidate_question.solution_and_marking_scheme.final_answers_summary
                ],
                'mark_allocation_criteria': [
                    {
                        'criterion_id': criterion.criterion_id,
                        'criterion_text': criterion.criterion_text,
                        'mark_code_display': criterion.mark_code_display,
                        'marks_value': criterion.marks_value,
                        'mark_type_primary': criterion.mark_type_primary,
                        'qualifiers_and_notes': criterion.qualifiers_and_notes
                    }
                    for criterion in candidate_question.solution_and_marking_scheme.mark_allocation_criteria
                ],
                'total_marks_for_part': candidate_question.solution_and_marking_scheme.total_marks_for_part,
                'source': 'ai_generated',
                'created_at': datetime.utcnow().isoformat()
            }

            response = self.session.post(
                f"{self.api_url}/solutionMarkingSchemes",
                json=marking_scheme_data,
                timeout=90
            )

            if response.status_code == 201:
                scheme = response.json()
                return scheme.get('id')
            else:
                logger.error(f"Failed to create marking scheme: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error creating marking scheme: {str(e)}")
            return None

    async def _create_solver_algorithm(self, candidate_question: CandidateQuestion) -> Optional[str]:
        """Create solver algorithm in Payload and return its ID."""
        try:
            solver_data = {
                'steps': [
                    {
                        'step_number': step.step_number,
                        'description_text': step.description_text,
                        'mathematical_expression_latex': step.mathematical_expression_latex,
                        'skill_applied_tag': step.skill_applied_tag,
                        'justification_or_reasoning': step.justification_or_reasoning
                    }
                    for step in candidate_question.solver_algorithm.steps
                ],
                'source': 'ai_generated',
                'created_at': datetime.utcnow().isoformat()
            }

            response = self.session.post(
                f"{self.api_url}/solverAlgorithms",
                json=solver_data,
                timeout=90
            )

            if response.status_code == 201:
                algorithm = response.json()
                return algorithm.get('id')
            else:
                logger.error(f"Failed to create solver algorithm: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error creating solver algorithm: {str(e)}")
            return None

    def _map_question_to_payload(
        self,
        candidate_question: CandidateQuestion,
        marking_scheme_id: Optional[str],
        solver_algorithm_id: Optional[str]
    ) -> Dict[str, Any]:
        """Map CandidateQuestion to Payload questions schema."""
        return {
            'question_id_global': candidate_question.question_id_global,
            'question_number_display': candidate_question.question_number_display,
            'marks': candidate_question.marks,
            'command_word': candidate_question.command_word.value,
            'raw_text_content': candidate_question.raw_text_content,
            'formatted_text_latex': candidate_question.formatted_text_latex,
            'taxonomy': {
                'topic_path': candidate_question.taxonomy.topic_path,
                'subject_content_references': candidate_question.taxonomy.subject_content_references,
                'skill_tags': candidate_question.taxonomy.skill_tags,
                'cognitive_level': candidate_question.taxonomy.cognitive_level,
                'difficulty_estimate_0_to_1': candidate_question.taxonomy.difficulty_estimate_0_to_1
            },
            'solution_and_marking_scheme': marking_scheme_id,
            'solver_algorithm': solver_algorithm_id,
            'source': 'ai_generated',
            'generation_metadata': {
                'original_question_id': candidate_question.question_id_local,
                'generation_id': str(candidate_question.generation_id),
                'target_grade': candidate_question.target_grade_input,
                'llm_model_used': candidate_question.llm_model_used_generation,
                'generation_timestamp': candidate_question.generation_timestamp.isoformat() if candidate_question.generation_timestamp else None,
                'reviewer_notes': candidate_question.reviewer_notes
            },
            'published_at': datetime.utcnow().isoformat()
        }

    async def verify_question_exists(self, question_id_global: str) -> bool:
        """Verify that a question exists in Payload by its global ID."""
        if not self.is_enabled():
            return False

        try:
            response = self.session.get(
                f"{self.api_url}/questions",
                params={'where[question_id_global][equals]': question_id_global},
                timeout=90
            )

            if response.status_code == 200:
                results = response.json().get('docs', [])
                return len(results) > 0
            else:
                logger.error(f"Failed to verify question existence: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error verifying question existence: {str(e)}")
            return False

    async def delete_question(self, payload_question_id: str) -> bool:
        """Delete a question from Payload (for testing/cleanup)."""
        if not self.is_enabled():
            return False

        try:
            response = self.session.delete(
                f"{self.api_url}/questions/{payload_question_id}",
                timeout=90
            )

            if response.status_code in (200, 204):
                logger.info(f"Successfully deleted question {payload_question_id} from Payload")
                return True
            else:
                logger.error(f"Failed to delete question: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error deleting question: {str(e)}")
            return False

    async def find_questions_by_global_id(self, question_id_global: str) -> List[Dict[str, Any]]:
        """Find questions in Payload by global ID (useful for testing)."""
        if not self.is_enabled():
            return []

        try:
            response = self.session.get(
                f"{self.api_url}/questions",
                params={'where[question_id_global][equals]': question_id_global},
                timeout=90
            )

            if response.status_code == 200:
                return response.json().get('docs', [])
            else:
                logger.error(f"Failed to find questions: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error finding questions: {str(e)}")
            return []

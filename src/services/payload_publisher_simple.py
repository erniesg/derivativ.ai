"""
Simplified Payload Publisher - Works with existing production schema.

This version embeds marking schemes and solver algorithms directly in the question
instead of creating separate collections that don't exist in production.
"""

import os
import logging
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.models.question_models import CandidateQuestion

logger = logging.getLogger(__name__)


class SimplePayloadPublisher:
    """Simplified publisher that works with existing production schema."""

    def __init__(self):
        """Initialize the simplified Payload publisher."""
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
        Publish an approved question to Payload CMS (simplified version).

        Args:
            candidate_question: The question to publish

        Returns:
            Payload question ID if successful, None if failed
        """
        if not self.is_enabled():
            logger.warning("Payload publishing disabled - no API token")
            return None

        try:
            # Create a single question object with embedded data
            question_data = self._map_question_to_simple_payload(candidate_question)

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

    def _map_question_to_simple_payload(self, candidate_question: CandidateQuestion) -> Dict[str, Any]:
        """Map CandidateQuestion to simplified Payload schema with embedded data."""
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
            # Embed marking scheme directly
            'solution_and_marking_scheme': {
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
                'total_marks_for_part': candidate_question.solution_and_marking_scheme.total_marks_for_part
            },
            # Embed solver algorithm directly
            'solver_algorithm': {
                'steps': [
                    {
                        'step_number': step.step_number,
                        'description_text': step.description_text,
                        'mathematical_expression_latex': step.mathematical_expression_latex,
                        'skill_applied_tag': step.skill_applied_tag,
                        'justification_or_reasoning': step.justification_or_reasoning
                    }
                    for step in candidate_question.solver_algorithm.steps
                ]
            },
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
        """Delete a question from Payload."""
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
        """Find questions in Payload by global ID."""
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

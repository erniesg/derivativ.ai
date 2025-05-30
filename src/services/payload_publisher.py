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
        self.email = os.getenv('PAYLOAD_EMAIL')
        self.password = os.getenv('PAYLOAD_PASSWORD')

        if not self.api_token and not (self.email and self.password):
            logger.warning("Neither PAYLOAD_API_TOKEN nor PAYLOAD_EMAIL/PAYLOAD_PASSWORD set - Payload publishing will be disabled")

        self.headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json'
        } if self.api_token else {}

        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self._jwt_token = None

    def is_enabled(self) -> bool:
        """Check if Payload publishing is enabled."""
        return bool(self.api_token) or bool(self.email and self.password)

    async def _get_jwt_token(self) -> Optional[str]:
        """Get JWT token by logging in with email/password."""
        if self._jwt_token:
            return self._jwt_token

        if not (self.email and self.password):
            return None

        try:
            response = requests.post(
                f"{self.api_url}/users/login",
                json={"email": self.email, "password": self.password},
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                self._jwt_token = data.get("token")
                logger.info("Successfully obtained JWT token for Payload API")
                return self._jwt_token
            else:
                logger.error(f"Failed to login to Payload: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error obtaining JWT token: {str(e)}")
            return None

    async def _get_headers(self) -> Dict[str, str]:
        """Get headers with authentication token."""
        headers = {'Content-Type': 'application/json'}

        if self.api_token:
            headers['Authorization'] = f'Bearer {self.api_token}'
        else:
            jwt_token = await self._get_jwt_token()
            if jwt_token:
                headers['Authorization'] = f'Bearer {jwt_token}'

        return headers

    async def publish_question(self, candidate_question: CandidateQuestion) -> Optional[str]:
        """
        Publish an approved question to Payload CMS.

        Args:
            candidate_question: The question to publish

        Returns:
            Payload question ID if successful, None if failed
        """
        if not self.is_enabled():
            logger.warning("Payload publishing disabled - no credentials")
            return None

        try:
            headers = await self._get_headers()

            # First, create related objects
            marking_scheme_id = await self._create_marking_scheme(candidate_question, headers)
            solver_algorithm_id = await self._create_solver_algorithm(candidate_question, headers)

            # Then create the main question
            question_data = self._map_question_to_payload(
                candidate_question, marking_scheme_id, solver_algorithm_id
            )

            response = requests.post(
                f"{self.api_url}/questions",
                json=question_data,
                headers=headers,
                timeout=90
            )

            if response.status_code == 201:
                payload_question = response.json()
                logger.info(f"Successfully published question {candidate_question.question_id_local} to Payload with ID {payload_question.get('doc', {}).get('id')}")
                return payload_question.get('doc', {}).get('id')
            else:
                logger.error(f"Failed to publish question to Payload: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error publishing question to Payload: {str(e)}")
            return None

    async def _create_marking_scheme(self, candidate_question: CandidateQuestion, headers: Dict[str, str]) -> Optional[str]:
        """Create marking scheme in Payload and return its ID."""
        try:
            # First create the marking scheme container (without criteria relationships)
            marking_scheme_data = {
                'final_answers_summary': [
                    {
                        'answer_text': answer.answer_text,
                        'value_numeric': answer.value_numeric,
                        'unit': answer.unit
                    }
                    for answer in candidate_question.solution_and_marking_scheme.final_answers_summary
                ],
                'total_marks_for_part': candidate_question.solution_and_marking_scheme.total_marks_for_part
                # Note: mark_allocation_criteria will be added later
            }

            response = requests.post(
                f"{self.api_url}/solutionMarkingSchemes",
                json=marking_scheme_data,
                headers=headers,
                timeout=90
            )

            if response.status_code != 201:
                logger.error(f"Failed to create marking scheme: {response.status_code} - {response.text}")
                return None

            scheme = response.json()
            scheme_id = scheme.get('doc', {}).get('id')
            logger.info(f"Created marking scheme with ID: {scheme_id}")

            # Then create individual mark criteria with parent relationship
            criteria_ids = []
            for criterion in candidate_question.solution_and_marking_scheme.mark_allocation_criteria:
                criterion_data = {
                    'criterion_id': criterion.criterion_id,
                    'criterion_text': criterion.criterion_text,
                    'mark_code_display': criterion.mark_code_display,
                    'marks_value': criterion.marks_value,
                    'mark_type_primary': criterion.mark_type_primary,
                    'qualifiers_and_notes': getattr(criterion, 'qualifiers_and_notes', None),
                    'solutionMarkingScheme': scheme_id  # Link to parent immediately
                }

                response = requests.post(
                    f"{self.api_url}/markCriteria",
                    json=criterion_data,
                    headers=headers,
                    timeout=90
                )

                if response.status_code == 201:
                    criteria_ids.append(response.json().get('doc', {}).get('id'))
                else:
                    logger.error(f"Failed to create mark criterion: {response.status_code} - {response.text}")
                    return None

            # Finally update the marking scheme with child relationships
            if criteria_ids:
                update_data = {'mark_allocation_criteria': criteria_ids}
                requests.patch(
                    f"{self.api_url}/solutionMarkingSchemes/{scheme_id}",
                    json=update_data,
                    headers=headers,
                    timeout=30
                )
                logger.info(f"Linked {len(criteria_ids)} criteria to scheme {scheme_id}")

            return scheme_id

        except Exception as e:
            logger.error(f"Error creating marking scheme: {str(e)}")
            return None

    async def _create_solver_algorithm(self, candidate_question: CandidateQuestion, headers: Dict[str, str]) -> Optional[str]:
        """Create solver algorithm in Payload and return its ID."""
        try:
            # First create the solver algorithm container (without steps relationships)
            solver_data = {}  # Algorithm container has no direct fields besides steps relationship

            response = requests.post(
                f"{self.api_url}/solverAlgorithms",
                json=solver_data,
                headers=headers,
                timeout=90
            )

            if response.status_code != 201:
                logger.error(f"Failed to create solver algorithm: {response.status_code} - {response.text}")
                return None

            algorithm = response.json()
            algorithm_id = algorithm.get('doc', {}).get('id')
            logger.info(f"Created solver algorithm with ID: {algorithm_id}")

            # Then create individual solver steps with parent relationship
            step_ids = []
            for step in candidate_question.solver_algorithm.steps:
                step_data = {
                    'step_number': step.step_number,
                    'description_text': step.description_text,
                    'mathematical_expression_latex': step.mathematical_expression_latex,
                    'justification_or_reasoning': getattr(step, 'justification_or_reasoning', None),
                    'solverAlgorithm': algorithm_id  # Link to parent immediately
                }

                response = requests.post(
                    f"{self.api_url}/solverSteps",
                    json=step_data,
                    headers=headers,
                    timeout=90
                )

                if response.status_code == 201:
                    step_ids.append(response.json().get('doc', {}).get('id'))
                else:
                    logger.error(f"Failed to create solver step: {response.status_code} - {response.text}")
                    return None

            # Finally update the solver algorithm with child relationships
            if step_ids:
                update_data = {'steps': step_ids}
                requests.patch(
                    f"{self.api_url}/solverAlgorithms/{algorithm_id}",
                    json=update_data,
                    headers=headers,
                    timeout=30
                )
                logger.info(f"Linked {len(step_ids)} steps to algorithm {algorithm_id}")

            return algorithm_id

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
            'question_id_local': candidate_question.question_id_local,
            'question_id_global': candidate_question.question_id_global,
            'question_number_display': candidate_question.question_number_display,
            'marks': candidate_question.marks,
            'command_word': candidate_question.command_word.value,
            'raw_text_content': candidate_question.raw_text_content,
            'formatted_text_latex': candidate_question.formatted_text_latex,
            'origin': 'llm_generated',  # Required field from schema
            'taxonomy': {
                'topic_path': candidate_question.taxonomy.topic_path,
                'subject_content_references': candidate_question.taxonomy.subject_content_references,
                'skill_tags': candidate_question.taxonomy.skill_tags,
                'cognitive_level': candidate_question.taxonomy.cognitive_level,
                'difficulty_estimate_0_to_1': candidate_question.taxonomy.difficulty_estimate_0_to_1
            },
            'solution_and_marking_scheme': marking_scheme_id,  # Relationship ID
            'solver_algorithm': solver_algorithm_id,  # Relationship ID
            'assets': []  # Empty array for now, can be populated later
        }

    async def verify_question_exists(self, question_id_global: str) -> bool:
        """Verify that a question exists in Payload by its global ID."""
        if not self.is_enabled():
            return False

        try:
            headers = await self._get_headers()
            response = requests.get(
                f"{self.api_url}/questions",
                params={'where[question_id_global][equals]': question_id_global},
                headers=headers,
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
            headers = await self._get_headers()
            response = requests.delete(
                f"{self.api_url}/questions/{payload_question_id}",
                headers=headers,
                timeout=90
            )

            if response.status_code in [200, 204]:
                logger.info(f"Successfully deleted question {payload_question_id} from Payload")
                return True
            else:
                logger.error(f"Failed to delete question: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error deleting question: {str(e)}")
            return False

    async def find_questions_by_global_id(self, question_id_global: str) -> List[Dict[str, Any]]:
        """Find questions by global ID."""
        if not self.is_enabled():
            return []

        try:
            headers = await self._get_headers()
            response = requests.get(
                f"{self.api_url}/questions",
                params={'where[question_id_global][equals]': question_id_global},
                headers=headers,
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

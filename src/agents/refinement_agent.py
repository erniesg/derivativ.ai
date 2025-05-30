"""
RefinementAgent for improving questions based on review feedback.

This agent takes an original question and review feedback to generate
an improved version that addresses the identified issues.
"""

import json
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import uuid
import random

from src.models.question_models import (
    CandidateQuestion, GenerationConfig, CommandWord, QuestionTaxonomy,
    SolutionAndMarkingScheme, SolverAlgorithm, AnswerSummary,
    MarkAllocationCriterion, SolverStep
)
from src.services.prompt_loader import PromptLoader
from src.services.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class RefinementAgent:
    """Agent responsible for refining questions based on review feedback."""

    def __init__(self, model, config_manager: ConfigManager):
        """
        Initialize the RefinementAgent.

        Args:
            model: The LLM model for generating refinements
            config_manager: Configuration manager for accessing settings
        """
        self.model = model
        self.config_manager = config_manager
        self.prompt_loader = PromptLoader()

    def refine_question(
        self,
        original_question: CandidateQuestion,
        review_feedback: Dict[str, Any],
        interaction_id: str
    ) -> Tuple[Optional[CandidateQuestion], Dict[str, Any]]:
        """
        Refine a question based on review feedback.

        Args:
            original_question: The original question to refine
            review_feedback: Feedback from the review agent
            interaction_id: ID for tracking this interaction

        Returns:
            Tuple of (refined_question, interaction_data) or (None, interaction_data) if failed
        """
        start_time = datetime.utcnow()

        try:
            # Format the refinement prompt
            prompt = self.prompt_loader.format_refinement_prompt(
                original_question=original_question.model_dump(),
                review_feedback=review_feedback
            )

            # Generate refinement with primary prompt
            response = self._call_model(prompt, interaction_id, attempt=1)

            if response:
                # Try to parse the response into complete CandidateQuestion
                refined_question = self._parse_complete_response_to_question(
                    response, original_question, interaction_id
                )

                if refined_question:
                    interaction_data = self._create_interaction_data(
                        prompt, response, start_time, True, interaction_id, 1
                    )
                    return refined_question, interaction_data

            # Fallback: Try simpler prompt if complete parsing failed
            logger.warning(f"Primary refinement failed for {interaction_id}, trying fallback")
            fallback_response = self._try_fallback_refinement(
                original_question, review_feedback, interaction_id
            )

            if fallback_response:
                refined_question = self._parse_simple_response_to_question(
                    fallback_response, original_question, interaction_id
                )

                if refined_question:
                    interaction_data = self._create_interaction_data(
                        "fallback_prompt", fallback_response, start_time, True, interaction_id, 2
                    )
                    return refined_question, interaction_data

            # Both attempts failed
            interaction_data = self._create_interaction_data(
                prompt, response or "No response", start_time, False, interaction_id, 1
            )
            return None, interaction_data

        except Exception as e:
            logger.error(f"Error in refine_question: {str(e)}")
            interaction_data = self._create_interaction_data(
                "error", str(e), start_time, False, interaction_id, 1
            )
            return None, interaction_data

    def _call_model(self, prompt: str, interaction_id: str, attempt: int) -> Optional[str]:
        """Call the LLM model with the given prompt."""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.model.chat(messages)

            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                logger.error(f"Unexpected response type: {type(response)}")
                return None

        except Exception as e:
            logger.error(f"Model call failed for {interaction_id}, attempt {attempt}: {str(e)}")
            return None

    def _try_fallback_refinement(
        self,
        original_question: CandidateQuestion,
        review_feedback: Dict[str, Any],
        interaction_id: str
    ) -> Optional[str]:
        """Try a simpler fallback approach for refinement."""

        # Extract main issues from feedback
        issues = []
        if 'clarity_score' in review_feedback and review_feedback['clarity_score'] < 0.7:
            issues.append("clarity")
        if 'difficulty_score' in review_feedback and review_feedback['difficulty_score'] < 0.7:
            issues.append("difficulty level")
        if 'curriculum_alignment_score' in review_feedback and review_feedback['curriculum_alignment_score'] < 0.7:
            issues.append("curriculum alignment")

        fallback_prompt = f"""
Please improve this Cambridge IGCSE Mathematics question by addressing these issues: {', '.join(issues)}.

Original Question:
{original_question.raw_text_content}

Please provide an improved version in this exact JSON format:
{{
    "question_text": "improved question here",
    "answer": "improved answer here",
    "working": "step by step solution",
    "marks": {original_question.marks},
    "topic": "topic name",
    "difficulty": "difficulty level"
}}
"""

        return self._call_model(fallback_prompt, interaction_id, attempt=2)

    def _parse_complete_response_to_question(
        self,
        response: str,
        original_question: CandidateQuestion,
        interaction_id: str
    ) -> Optional[CandidateQuestion]:
        """Parse LLM response into a complete CandidateQuestion object using the full schema."""
        try:
            # Extract JSON from response
            json_str = self._extract_json_from_response(response)
            if not json_str:
                logger.error(f"No JSON found in complete response for {interaction_id}")
                return None

            # Parse JSON
            question_data = json.loads(json_str)

            # Validate required top-level fields
            required_fields = [
                'question_id_local', 'question_id_global', 'question_number_display',
                'marks', 'command_word', 'raw_text_content', 'taxonomy',
                'solution_and_marking_scheme', 'solver_algorithm'
            ]

            for field in required_fields:
                if field not in question_data:
                    logger.error(f"Missing required field '{field}' in complete refined question for {interaction_id}")
                    return None

            # Parse nested structures
            taxonomy_data = question_data.get("taxonomy", {})
            solution_data = question_data.get("solution_and_marking_scheme", {})
            solver_data = question_data.get("solver_algorithm", {})

            # Create taxonomy
            taxonomy = QuestionTaxonomy(
                topic_path=taxonomy_data.get("topic_path", original_question.taxonomy.topic_path),
                subject_content_references=taxonomy_data.get("subject_content_references", original_question.taxonomy.subject_content_references),
                skill_tags=taxonomy_data.get("skill_tags", original_question.taxonomy.skill_tags),
                cognitive_level=taxonomy_data.get("cognitive_level", original_question.taxonomy.cognitive_level),
                difficulty_estimate_0_to_1=taxonomy_data.get("difficulty_estimate_0_to_1", original_question.taxonomy.difficulty_estimate_0_to_1)
            )

            # Create solution and marking scheme
            answers = []
            for answer_data in solution_data.get("final_answers_summary", []):
                answers.append(AnswerSummary(
                    answer_text=answer_data.get("answer_text", ""),
                    value_numeric=answer_data.get("value_numeric"),
                    unit=answer_data.get("unit")
                ))

            criteria = []
            for criterion_data in solution_data.get("mark_allocation_criteria", []):
                criteria.append(MarkAllocationCriterion(
                    criterion_id=criterion_data.get("criterion_id", f"ref_crit_{random.randint(1, 999)}"),
                    criterion_text=criterion_data.get("criterion_text", ""),
                    mark_code_display=criterion_data.get("mark_code_display", "M1"),
                    marks_value=criterion_data.get("marks_value", 1.0),
                    mark_type_primary=criterion_data.get("mark_type_primary"),
                    qualifiers_and_notes=criterion_data.get("qualifiers_and_notes")
                ))

            solution = SolutionAndMarkingScheme(
                final_answers_summary=answers,
                mark_allocation_criteria=criteria,
                total_marks_for_part=solution_data.get("total_marks_for_part", question_data.get("marks", original_question.marks))
            )

            # Create solver algorithm
            steps = []
            for step_data in solver_data.get("steps", []):
                steps.append(SolverStep(
                    step_number=step_data.get("step_number", 1),
                    description_text=step_data.get("description_text", ""),
                    mathematical_expression_latex=step_data.get("mathematical_expression_latex"),
                    skill_applied_tag=step_data.get("skill_applied_tag"),
                    justification_or_reasoning=step_data.get("justification_or_reasoning")
                ))

            solver_algorithm = SolverAlgorithm(steps=steps)

            # Parse command word
            command_word_str = question_data.get("command_word", original_question.command_word.value)
            try:
                command_word = CommandWord(command_word_str)
            except ValueError:
                logger.warning(f"Invalid command word '{command_word_str}', using original")
                command_word = original_question.command_word

            # Create refined question
            refined_question = CandidateQuestion(
                question_id_local=question_data.get("question_id_local", f"Ref_Q{random.randint(1000, 9999)}"),
                question_id_global=question_data.get("question_id_global", f"ref_{original_question.question_id_local}_{random.randint(100, 999)}"),
                question_number_display=question_data.get("question_number_display", "Refined Question"),
                marks=question_data.get("marks", original_question.marks),
                command_word=command_word,
                raw_text_content=question_data.get("raw_text_content", ""),
                formatted_text_latex=question_data.get("formatted_text_latex"),
                taxonomy=taxonomy,
                solution_and_marking_scheme=solution,
                solver_algorithm=solver_algorithm,
                generation_id=original_question.generation_id,
                seed_question_id=original_question.seed_question_id,
                target_grade_input=original_question.target_grade_input,
                llm_model_used_generation=str(self.model),
                llm_model_used_marking_scheme=original_question.llm_model_used_marking_scheme,
                llm_model_used_review=original_question.llm_model_used_review,
                prompt_template_version_generation=original_question.prompt_template_version_generation,
                prompt_template_version_marking_scheme=original_question.prompt_template_version_marking_scheme,
                prompt_template_version_review=original_question.prompt_template_version_review,
                generation_timestamp=datetime.utcnow(),
                status=original_question.status,
                reviewer_notes=f"Refined from {original_question.question_id_local}",
                confidence_score=None,
                validation_errors=[]
            )

            logger.info(f"Successfully created complete refined question for {interaction_id}")
            return refined_question

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in complete parsing for {interaction_id}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error parsing complete response for {interaction_id}: {str(e)}")
            return None

    def _parse_simple_response_to_question(
        self,
        response: str,
        original_question: CandidateQuestion,
        interaction_id: str
    ) -> Optional[CandidateQuestion]:
        """Parse simple fallback LLM response into a CandidateQuestion object."""
        try:
            # Extract JSON from response
            json_str = self._extract_json_from_response(response)
            if not json_str:
                logger.error(f"No JSON found in simple response for {interaction_id}")
                return None

            # Parse JSON
            question_data = json.loads(json_str)

            # Validate required fields for simple response
            required_fields = ['question_text']
            for field in required_fields:
                if field not in question_data:
                    logger.error(f"Missing required field '{field}' in simple refined question for {interaction_id}")
                    return None

            # Create refined question based on original but with updated content from simple response
            refined_question = CandidateQuestion(
                question_id_local=f"Ref_Q{random.randint(1000, 9999)}",
                question_id_global=f"ref_{original_question.question_id_local}_{random.randint(100, 999)}",
                question_number_display="Refined Question",
                marks=question_data.get('marks', original_question.marks),
                command_word=original_question.command_word,
                raw_text_content=question_data['question_text'],
                formatted_text_latex=None,
                taxonomy=original_question.taxonomy,
                solution_and_marking_scheme=original_question.solution_and_marking_scheme,
                solver_algorithm=original_question.solver_algorithm,
                generation_id=original_question.generation_id,
                seed_question_id=original_question.seed_question_id,
                target_grade_input=original_question.target_grade_input,
                llm_model_used_generation=str(self.model),
                llm_model_used_marking_scheme=original_question.llm_model_used_marking_scheme,
                llm_model_used_review=original_question.llm_model_used_review,
                prompt_template_version_generation=original_question.prompt_template_version_generation,
                prompt_template_version_marking_scheme=original_question.prompt_template_version_marking_scheme,
                prompt_template_version_review=original_question.prompt_template_version_review,
                generation_timestamp=datetime.utcnow(),
                status=original_question.status,
                reviewer_notes=f"Refined from {original_question.question_id_local} (simple mode)",
                confidence_score=None,
                validation_errors=[]
            )

            logger.info(f"Successfully created simple refined question for {interaction_id}")
            return refined_question

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in simple parsing for {interaction_id}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error parsing simple response for {interaction_id}: {str(e)}")
            return None

    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """Extract JSON object from LLM response."""
        try:
            # Look for JSON between ```json and ``` or just {} brackets
            import re

            # Try to find JSON in code blocks first
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                return json_match.group(1)

            # Try to find JSON directly in curly braces
            json_match = re.search(r'(\{[^}]*\})', response, re.DOTALL)
            if json_match:
                return json_match.group(1)

            # Try to find the largest JSON-like structure
            start_idx = response.find('{')
            if start_idx == -1:
                return None

            brace_count = 0
            end_idx = start_idx

            for i in range(start_idx, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break

            if brace_count == 0:
                return response[start_idx:end_idx + 1]

            return None

        except Exception as e:
            logger.error(f"Error extracting JSON: {str(e)}")
            return None

    def _create_interaction_data(
        self,
        prompt: str,
        response: str,
        start_time: datetime,
        success: bool,
        interaction_id: str,
        attempt: int
    ) -> Dict[str, Any]:
        """Create interaction data for database logging."""
        return {
            'interaction_id': interaction_id,
            'agent_type': 'refinement',
            'prompt_used': prompt[:1000],  # Truncate for storage
            'model_response': response[:2000],  # Truncate for storage
            'success': success,
            'timestamp': start_time,
            'processing_time': (datetime.utcnow() - start_time).total_seconds(),
            'attempt_number': attempt,
            'metadata': {
                'model_name': str(self.model),
                'agent_version': '1.0'
            }
        }

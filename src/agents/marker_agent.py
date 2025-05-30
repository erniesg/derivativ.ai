"""
MarkerAgent - Specialized agent for generating detailed marking schemes.

This agent focuses specifically on creating high-quality marking schemes that follow
Cambridge marking principles and standards. It uses the markscheme.json data
to ensure consistency with official marking conventions.
"""

import json
import os
from typing import Dict, List, Optional, Any
from ..models import (
    SolutionAndMarkingScheme, MarkAllocationCriterion, AnswerSummary,
    GenerationConfig, LLMModel
)
from ..services.prompt_loader import PromptLoader


class MarkerAgent:
    """
    Specialized agent for generating detailed marking schemes.

    Focuses on:
    - Creating detailed mark allocation criteria
    - Following Cambridge marking principles
    - Ensuring mark consistency and accuracy
    - Generating comprehensive final answer summaries
    """

    def __init__(self, model, db_client=None, debug: bool = False):
        self.model = model
        self.db_client = db_client
        self.debug = debug

        # Initialize prompt loader
        self.prompt_loader = PromptLoader()

        # Load marking scheme principles and data
        self.marking_data = self._load_marking_data()
        self.syllabus_data = self._load_syllabus_data()

    def _load_marking_data(self) -> Dict[str, Any]:
        """Load Cambridge marking principles from markscheme.json"""
        try:
            with open("data/markscheme.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            if self.debug:
                print("⚠️ markscheme.json not found, using fallback marking data")
            return self._get_fallback_marking_data()

    def _load_syllabus_data(self) -> Dict[str, Any]:
        """Load syllabus structure from syllabus_command.json"""
        try:
            with open("data/syllabus_command.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            if self.debug:
                print("⚠️ syllabus_command.json not found")
            return {}

    def _get_fallback_marking_data(self) -> Dict[str, Any]:
        """Fallback marking data if file not available"""
        return {
            "generic_marking_principles": [
                {"principle_id": "GENERIC MARKING PRINCIPLE 1", "details": "Marks must be awarded positively"}
            ],
            "mark_scheme_notes": {
                "types_of_mark": [
                    {"mark_type": "M", "description": "Method mark"},
                    {"mark_type": "A", "description": "Accuracy mark"},
                    {"mark_type": "B", "description": "Independent result mark"}
                ]
            }
        }

    async def generate_marking_scheme(
        self,
        question_text: str,
        config: GenerationConfig,
        template_version: str = "v1.0",
        expected_answer: Optional[str] = None,
        solution_steps: Optional[List[str]] = None
    ) -> SolutionAndMarkingScheme:
        """
        Generate a detailed marking scheme for a given question.

        Args:
            question_text: The raw question text
            config: Generation configuration
            template_version: Prompt template version to use
            expected_answer: Expected final answer (if known)
            solution_steps: Solution steps (if available)

        Returns:
            Complete SolutionAndMarkingScheme object
        """

        if self.debug:
            print(f"🎯 MarkerAgent generating marking scheme for {config.desired_marks} marks")

        # Construct specialized marking prompt using template
        marking_prompt = self._construct_marking_prompt(
            question_text, config, template_version, expected_answer, solution_steps
        )

        # Call LLM for marking scheme generation
        if self.debug:
            print(f"[DEBUG] Calling MarkerAgent LLM: {type(self.model).__name__}")

        response = await self._call_model(marking_prompt)

        # Parse and validate marking scheme
        marking_scheme = self._parse_marking_response(response, config)

        if self.debug:
            print(f"✅ MarkerAgent generated marking scheme with {len(marking_scheme.mark_allocation_criteria)} criteria")

        return marking_scheme

    def _construct_marking_prompt(
        self,
        question_text: str,
        config: GenerationConfig,
        template_version: str,
        expected_answer: Optional[str] = None,
        solution_steps: Optional[List[str]] = None
    ) -> str:
        """Construct specialized prompt for marking scheme generation using template"""

        # Get marking principles and mark types for context
        marking_principles = self._get_marking_principles_text()
        mark_types = self._get_mark_types_text()

        # Use prompt loader to format the marking scheme prompt
        formatted_prompt = self.prompt_loader.format_marking_scheme_prompt(
            template_version=template_version,
            question_text=question_text,
            target_grade=config.target_grade,
            desired_marks=config.desired_marks,
            subject_content_references=config.subject_content_references,
            calculator_policy=config.calculator_policy.value,
            marking_principles=marking_principles,
            mark_types=mark_types,
            expected_answer=expected_answer,
            solution_steps=solution_steps
        )

        return formatted_prompt

    def _get_marking_principles_text(self) -> str:
        """Extract key marking principles as text"""
        principles = []
        for principle in self.marking_data.get("generic_marking_principles", []):
            details = principle.get('details', '')
            if isinstance(details, str):
                principles.append(f"- {details}")
            elif isinstance(details, list):
                # Handle structured principles with points
                main_detail = details[0] if details else ''
                principles.append(f"- {main_detail}")
        return '\n'.join(principles[:5])  # First 5 principles for brevity

    def _get_mark_types_text(self) -> str:
        """Extract mark types as text"""
        mark_types = []
        for mark_type in self.marking_data.get("mark_scheme_notes", {}).get("types_of_mark", []):
            mark_types.append(f"- **{mark_type['mark_type']}**: {mark_type['description']}")
        return '\n'.join(mark_types)

    async def _call_model(self, prompt: str) -> str:
        """Call the LLM model with the marking prompt"""
        try:
            # Handle different model types - fix Claude 4 content format issue
            if hasattr(self.model, 'model_id') and 'claude' in str(self.model.model_id).lower():
                # Claude models need content as list format for Bedrock
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            else:
                # OpenAI and other models use string content format
                messages = [{"role": "user", "content": prompt}]

            response = self.model(messages)

            # Extract content from response
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                return str(response)

        except Exception as e:
            if self.debug:
                print(f"❌ MarkerAgent model call error: {e}")
            raise

    def _parse_marking_response(self, response: str, config: GenerationConfig) -> SolutionAndMarkingScheme:
        """Parse LLM response into SolutionAndMarkingScheme object"""
        try:
            # Clean and extract JSON from response
            response_clean = self._extract_json_from_response(response)

            if self.debug:
                print(f"[DEBUG] Cleaned response: {response_clean[:200]}...")

            # Parse JSON
            marking_data = json.loads(response_clean)

            # Create answer summaries
            answers = []
            for answer_data in marking_data.get("final_answers_summary", []):
                # Handle different value_numeric formats
                value_numeric = answer_data.get("value_numeric")
                if isinstance(value_numeric, dict):
                    # If it's a dict, try to extract a single numeric value or set to None
                    value_numeric = None

                answers.append(AnswerSummary(
                    answer_text=answer_data.get("answer_text", ""),
                    value_numeric=value_numeric,
                    unit=answer_data.get("unit")
                ))

            # Create mark allocation criteria
            criteria = []
            for criterion_data in marking_data.get("mark_allocation_criteria", []):
                criteria.append(MarkAllocationCriterion(
                    criterion_id=criterion_data.get("criterion_id", "crit_1"),
                    criterion_text=criterion_data.get("criterion_text", ""),
                    mark_code_display=criterion_data.get("mark_code_display", "B1"),
                    marks_value=float(criterion_data.get("marks_value", 1.0)),
                    mark_type_primary=criterion_data.get("mark_type_primary", "B"),
                    qualifiers_and_notes=criterion_data.get("qualifiers_and_notes", "oe")
                ))

            # Create complete marking scheme
            return SolutionAndMarkingScheme(
                final_answers_summary=answers,
                mark_allocation_criteria=criteria,
                total_marks_for_part=marking_data.get("total_marks_for_part", config.desired_marks)
            )

        except Exception as e:
            if self.debug:
                print(f"❌ Error parsing marking scheme response: {e}")
                print(f"Raw response: {response[:500]}...")

            # Fallback: create basic marking scheme
            return self._create_fallback_marking_scheme(config)

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON content from LLM response, handling various formats"""

        # First, try to find JSON blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()

        # If no explicit JSON block, look for { ... } pattern
        start = response.find("{")
        end = response.rfind("}")

        if start != -1 and end != -1 and end > start:
            return response[start:end+1].strip()

        # Last resort: return the whole response
        return response.strip()

    def _create_fallback_marking_scheme(self, config: GenerationConfig) -> SolutionAndMarkingScheme:
        """Create a basic fallback marking scheme if parsing fails"""
        return SolutionAndMarkingScheme(
            final_answers_summary=[
                AnswerSummary(
                    answer_text="Answer not parsed",
                    value_numeric=None,
                    unit=None
                )
            ],
            mark_allocation_criteria=[
                MarkAllocationCriterion(
                    criterion_id="fallback_1",
                    criterion_text="Correct method and answer",
                    mark_code_display=f"B{config.desired_marks}",
                    marks_value=float(config.desired_marks),
                    mark_type_primary="B",
                    qualifiers_and_notes="oe"
                )
            ],
            total_marks_for_part=config.desired_marks
        )

    async def refine_marking_scheme(
        self,
        existing_scheme: SolutionAndMarkingScheme,
        question_text: str,
        config: GenerationConfig,
        feedback: Optional[str] = None
    ) -> SolutionAndMarkingScheme:
        """
        Refine an existing marking scheme based on feedback or quality checks.

        This method allows for iterative improvement of marking schemes.
        """

        if self.debug:
            print("🔄 MarkerAgent refining existing marking scheme")

        # Create refinement prompt (simpler, direct approach for refinement)
        refinement_prompt = f"""Review and improve this marking scheme for the given question. Respond with ONLY the improved JSON object.

## QUESTION
{question_text}
**Marks:** {config.desired_marks}
**Grade:** {config.target_grade}

## CURRENT MARKING SCHEME
{json.dumps(existing_scheme.model_dump(), indent=2)}

{f"## FEEDBACK TO ADDRESS{chr(10)}{feedback}" if feedback else ""}

## TASK
Improve the marking scheme by:
1. Making criteria more specific and clear
2. Ensuring proper mark distribution (M/A/B types)
3. Adding appropriate qualifiers and notes
4. Verifying total marks match {config.desired_marks}

Respond with the improved marking scheme JSON only:"""

        response = await self._call_model(refinement_prompt)
        return self._parse_marking_response(response, config)

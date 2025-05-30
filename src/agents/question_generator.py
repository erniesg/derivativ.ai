"""
Question Generation Agent using smolagents.
Handles the generation of IGCSE Mathematics candidate questions.
"""

import json
import os
import random
from typing import Dict, Any, List, Optional, Union
from uuid import uuid4
import asyncio

from smolagents import CodeAgent, LiteLLMModel, OpenAIServerModel
from ..models.question_models import (
    GenerationConfig, CandidateQuestion, CommandWord,
    CalculatorPolicy, QuestionTaxonomy, SolutionAndMarkingScheme,
    SolverAlgorithm, AnswerSummary, MarkAllocationCriterion, SolverStep
)
from ..database.neon_client import NeonDBClient


class QuestionGeneratorAgent:
    """Agent responsible for generating candidate questions"""

    def __init__(self, model: Union[LiteLLMModel, OpenAIServerModel], db_client: NeonDBClient):
        self.model = model
        self.db_client = db_client
        self.agent = CodeAgent(
            tools=[],
            model=model,
            max_steps=1  # Single-step generation for MVP
        )

        # Load prompt template
        self.prompt_template = self._load_prompt_template()

        # Load marking principles and other static data
        self.marking_principles = self._load_marking_principles()

    def _load_prompt_template(self) -> str:
        """Load the generation prompt template"""
        try:
            with open("prompts/question_generation_v1.0.txt", "r") as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError("Question generation prompt template not found")

    def _load_marking_principles(self) -> str:
        """Load marking scheme principles from data"""
        try:
            with open("data/markscheme.json", "r") as f:
                markscheme_data = json.load(f)

            principles = []
            for principle in markscheme_data.get("maths_specific_marking_principles", []):
                principles.append(f"- {principle.get('details', '')}")

            return "\n".join(principles)
        except Exception as e:
            print(f"Warning: Could not load marking principles: {e}")
            return "Follow Cambridge IGCSE marking standards"

    async def generate_question(self, config: GenerationConfig) -> Optional[CandidateQuestion]:
        """Generate a single candidate question based on configuration"""
        try:
            # Gather context data
            syllabus_content = await self._get_syllabus_context(config.subject_content_references)
            command_word_definition = await self._get_command_word_definition(config.command_word_override)
            seed_context = await self._get_seed_context(config.seed_question_id) if config.seed_question_id else "No seed question provided - create an original question."

            # Prepare prompt
            prompt = self._prepare_generation_prompt(
                config, syllabus_content, command_word_definition, seed_context
            )

            # Generate using LLM
            response = await self._call_llm(prompt, config)

            # Parse and validate response
            question_data = self._parse_llm_response(response)
            if not question_data:
                return None

            # Create CandidateQuestion object
            candidate_question = self._create_candidate_question(question_data, config)

            # Validate question
            validation_errors = self._validate_question(candidate_question)
            candidate_question.validation_errors = validation_errors

            return candidate_question

        except Exception as e:
            print(f"Error generating question: {e}")
            return None

    def _prepare_generation_prompt(
        self,
        config: GenerationConfig,
        syllabus_content: str,
        command_word_definition: str,
        seed_context: str
    ) -> str:
        """Prepare the final prompt for generation"""

        command_word = config.command_word_override.value if config.command_word_override else "Work out"

        return self.prompt_template.format(
            target_grade=config.target_grade,
            calculator_policy=config.calculator_policy.value,
            desired_marks=config.desired_marks,
            subject_content_references=", ".join(config.subject_content_references),
            command_word=command_word,
            syllabus_content=syllabus_content,
            command_word_definition=command_word_definition,
            seed_question_context=seed_context,
            marking_principles=self.marking_principles,
            generation_id=str(config.generation_id)
        )

    async def _get_syllabus_context(self, content_refs: List[str]) -> str:
        """Get formatted syllabus content for the prompt"""
        syllabus_data = await self.db_client.get_syllabus_content(content_refs)

        if not syllabus_data:
            return "No syllabus content found for provided references."

        context_parts = []
        for item in syllabus_data:
            context_parts.append(f"""
**{item['ref']}: {item['title']}**
- Topic: {item['topic']}
- Details: {'; '.join(item['details'])}
- Examples: {'; '.join(item['notes_and_examples'])}
""")

        return "\n".join(context_parts)

    async def _get_command_word_definition(self, command_word: Optional[CommandWord]) -> str:
        """Get the definition of the command word"""
        if not command_word:
            return "Standard mathematical instruction word - follow typical IGCSE usage."

        definition = await self.db_client.get_command_word_definition(command_word.value)
        return definition or "Follow standard IGCSE usage for this command word."

    async def _get_seed_context(self, seed_question_id: str) -> str:
        """Get context from seed question if provided"""
        if not seed_question_id:
            return ""

        seed_data = await self.db_client.get_past_paper_question(seed_question_id)
        if not seed_data:
            return f"Seed question {seed_question_id} not found."

        return f"""
**Seed Question for Inspiration:**
- Question: {seed_data.get('raw_text_content', 'N/A')}
- Marks: {seed_data.get('marks', 'N/A')}
- Command Word: {seed_data.get('command_word', 'N/A')}
- Topic: {seed_data.get('taxonomy', {}).get('topic_path', 'N/A')}

Use this as inspiration but create a NEW question with different context/numbers/approach.
"""

    async def _call_llm(self, prompt: str, config: GenerationConfig) -> str:
        """Call the LLM with the prepared prompt"""
        try:
            # Prepare messages in the format expected by smolagents
            messages = [
                {"role": "system", "content": "You are a helpful assistant designed to output JSON. You must respond with valid JSON only, no additional text or formatting."},
                {"role": "user", "content": prompt}
            ]

            # Call the model directly with JSON mode
            response = self.model(messages)

            # Debug: print the actual response
            print(f"DEBUG - LLM Response length: {len(response)}")
            print(f"DEBUG - LLM Response (first 200 chars): {response[:200]}")
            print(f"DEBUG - LLM Response (last 200 chars): {response[-200:]}")

            return response
        except Exception as e:
            print(f"Error calling LLM: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse the LLM response to extract JSON question data"""
        try:
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                print("No JSON found in LLM response")
                return None

            json_str = response[start_idx:end_idx]
            question_data = json.loads(json_str)

            return question_data

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            return None
        except Exception as e:
            print(f"Error processing LLM response: {e}")
            return None

    def _create_candidate_question(self, question_data: Dict[str, Any], config: GenerationConfig) -> CandidateQuestion:
        """Create a CandidateQuestion object from parsed data"""

        # Parse taxonomy
        taxonomy_data = question_data.get("taxonomy", {})
        taxonomy = QuestionTaxonomy(
            topic_path=taxonomy_data.get("topic_path", []),
            subject_content_references=taxonomy_data.get("subject_content_references", config.subject_content_references),
            skill_tags=taxonomy_data.get("skill_tags", []),
            cognitive_level=taxonomy_data.get("cognitive_level"),
            difficulty_estimate_0_to_1=taxonomy_data.get("difficulty_estimate_0_to_1")
        )

        # Parse solution and marking scheme
        solution_data = question_data.get("solution_and_marking_scheme", {})
        answers = [
            AnswerSummary(**answer)
            for answer in solution_data.get("final_answers_summary", [])
        ]
        criteria = [
            MarkAllocationCriterion(**criterion)
            for criterion in solution_data.get("mark_allocation_criteria", [])
        ]
        solution = SolutionAndMarkingScheme(
            final_answers_summary=answers,
            mark_allocation_criteria=criteria,
            total_marks_for_part=solution_data.get("total_marks_for_part", config.desired_marks)
        )

        # Parse solver algorithm
        solver_data = question_data.get("solver_algorithm", {})
        steps = [
            SolverStep(**step)
            for step in solver_data.get("steps", [])
        ]
        solver_algorithm = SolverAlgorithm(steps=steps)

        # Determine command word
        command_word_str = question_data.get("command_word", config.command_word_override.value if config.command_word_override else "Work out")
        try:
            command_word = CommandWord(command_word_str)
        except ValueError:
            command_word = CommandWord.WORK_OUT  # Default fallback

        # Create the candidate question
        candidate_question = CandidateQuestion(
            question_id_local=question_data.get("question_id_local", f"Gen_Q{random.randint(1000, 9999)}"),
            question_id_global=question_data.get("question_id_global", f"gen_{config.generation_id}_q{random.randint(100, 999)}"),
            question_number_display=question_data.get("question_number_display", "Generated Question"),
            marks=question_data.get("marks", config.desired_marks),
            command_word=command_word,
            raw_text_content=question_data.get("raw_text_content", ""),
            formatted_text_latex=question_data.get("formatted_text_latex"),
            taxonomy=taxonomy,
            solution_and_marking_scheme=solution,
            solver_algorithm=solver_algorithm,
            generation_id=config.generation_id,
            seed_question_id=config.seed_question_id,
            target_grade_input=config.target_grade,
            llm_model_used_generation=config.llm_model_generation.value,
            llm_model_used_marking_scheme=config.llm_model_marking_scheme.value,
            prompt_template_version_generation=config.prompt_template_version_generation,
            prompt_template_version_marking_scheme=config.prompt_template_version_marking_scheme
        )

        return candidate_question

    def _validate_question(self, question: CandidateQuestion) -> List[str]:
        """Validate the generated question and return list of errors"""
        errors = []

        # Check required fields
        if not question.raw_text_content.strip():
            errors.append("Question text is empty")

        if question.marks <= 0:
            errors.append("Question must have positive marks")

        if not question.taxonomy.subject_content_references:
            errors.append("Question must have subject content references")

        if not question.solution_and_marking_scheme.final_answers_summary:
            errors.append("Question must have at least one answer")

        if not question.solver_algorithm.steps:
            errors.append("Question must have solution steps")

        # Check mark consistency
        total_criteria_marks = sum(
            criterion.marks_value
            for criterion in question.solution_and_marking_scheme.mark_allocation_criteria
        )
        if abs(total_criteria_marks - question.marks) > 0.01:
            errors.append(f"Mark allocation criteria total ({total_criteria_marks}) doesn't match question marks ({question.marks})")

        return errors

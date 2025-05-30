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

from ..models import (
    GenerationConfig, CandidateQuestion, CommandWord,
    CalculatorPolicy, QuestionTaxonomy, SolutionAndMarkingScheme,
    SolverAlgorithm, AnswerSummary, MarkAllocationCriterion, SolverStep
)
from ..database import NeonDBClient


class QuestionGeneratorAgent:
    """Agent responsible for generating candidate questions"""

    def __init__(self, model: Union[LiteLLMModel, OpenAIServerModel], db_client: NeonDBClient, debug: bool = None):
        self.model = model
        self.db_client = db_client

        # Set debug mode from parameter, environment variable, or default to False
        if debug is not None:
            self.debug = debug
        else:
            self.debug = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes", "on")

        self.agent = CodeAgent(
            tools=[],
            model=model,
            max_steps=1  # Single-step generation for MVP
        )

        # Load prompt template
        self.prompt_template = self._load_prompt_template()

        # Load marking principles and other static data
        self.marking_principles = self._load_marking_principles()

    def _debug_print(self, *args, **kwargs):
        """Print debug messages only if debug mode is enabled"""
        if self.debug:
            print(*args, **kwargs)

    def _load_prompt_template(self) -> str:
        """Load the generation prompt template"""
        try:
            # Try v1.1 first, fallback to v1.0
            try:
                with open("prompts/question_generation_v1.1.txt", "r") as f:
                    self._debug_print("[DEBUG] Using prompt template v1.1")
                    return f.read()
            except FileNotFoundError:
                with open("prompts/question_generation_v1.0.txt", "r") as f:
                    self._debug_print("[DEBUG] Using prompt template v1.0")
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
        if not self.db_client:
            # Fallback when no database client is available
            return self._get_fallback_syllabus_context(content_refs)

        try:
            syllabus_data = await self.db_client.get_syllabus_content(content_refs)

            if not syllabus_data:
                return self._get_fallback_syllabus_context(content_refs)

            context_parts = []
            for item in syllabus_data:
                context_parts.append(f"""
**{item['ref']}: {item['title']}**
- Topic: {item['topic']}
- Details: {'; '.join(item['details'])}
- Examples: {'; '.join(item['notes_and_examples'])}
""")

            return "\n".join(context_parts)
        except Exception as e:
            self._debug_print(f"[DEBUG] Error getting syllabus content: {e}")
            return self._get_fallback_syllabus_context(content_refs)

    def _get_fallback_syllabus_context(self, content_refs: List[str]) -> str:
        """Provide fallback syllabus context when database is not available"""
        context_parts = []

        # Basic syllabus mapping for common references
        syllabus_mapping = {
            "A1.1": "Number: Natural numbers, integers, rational and irrational numbers",
            "A1.2": "Number: Order of operations (BODMAS/PEMDAS)",
            "A1.3": "Number: Basic arithmetic operations",
            "C1.1": "Number: Natural numbers, integers, prime numbers",
            "C1.4": "Number: Fractions, decimals, percentages",
            "C1.5": "Number: Ordering and comparing numbers",
            "C1.6": "Number: Four operations with whole numbers and decimals",
            "C1.7": "Number: Indices and powers",
            "C1.8": "Number: Standard form (scientific notation)",
            "C1.11": "Number: Ratio and proportion",
            "C1.13": "Number: Percentage calculations",
            "G1.1": "Geometry: Properties of shapes and angles",
            "G1.2": "Geometry: Perimeter and area calculations",
            "S1.1": "Statistics: Data collection and representation",
            "S1.2": "Statistics: Mean, median, mode and range"
        }

        for ref in content_refs:
            title = syllabus_mapping.get(ref, f"Topic {ref}: Mathematical concepts and problem solving")
            context_parts.append(f"""
**{ref}**: {title}
- Focus on grade-appropriate mathematical understanding
- Apply problem-solving techniques
- Show clear working and reasoning
""")

        if not context_parts:
            context_parts.append("Focus on appropriate IGCSE Mathematics concepts for the target grade level.")

        return "\n".join(context_parts)

    async def _get_command_word_definition(self, command_word: Optional[CommandWord]) -> str:
        """Get the definition of the command word"""
        if not command_word:
            return "Standard mathematical instruction word - follow typical IGCSE usage."

        if not self.db_client:
            return self._get_fallback_command_word_definition(command_word)

        try:
            definition = await self.db_client.get_command_word_definition(command_word.value)
            return definition or self._get_fallback_command_word_definition(command_word)
        except Exception as e:
            self._debug_print(f"[DEBUG] Error getting command word definition: {e}")
            return self._get_fallback_command_word_definition(command_word)

    def _get_fallback_command_word_definition(self, command_word: CommandWord) -> str:
        """Provide fallback command word definitions"""
        definitions = {
            CommandWord.CALCULATE: "Work out from given facts, figures or information",
            CommandWord.CONSTRUCT: "Make an accurate drawing",
            CommandWord.DETERMINE: "Establish with certainty",
            CommandWord.DESCRIBE: "State the points of a topic / give characteristics and main features",
            CommandWord.EXPLAIN: "Set out purposes or reasons / make the relationships between things clear / say why and/or how and support with relevant evidence",
            CommandWord.GIVE: "Produce an answer from a given source or recall/memory",
            CommandWord.PLOT: "Mark point(s) on a graph",
            CommandWord.SHOW: "Provide structured evidence that leads to a given result",
            CommandWord.SKETCH: "Make a simple freehand drawing showing the key features",
            CommandWord.STATE: "Express in clear terms",
            CommandWord.WORK_OUT: "Calculate from given facts, figures or information with or without the use of a calculator",
            CommandWord.WRITE: "Give an answer in a specific form",
            CommandWord.WRITE_DOWN: "Give an answer without significant working"
        }
        return definitions.get(command_word, "Follow standard IGCSE mathematical instruction usage.")

    async def _get_seed_context(self, seed_question_id: str) -> str:
        """Get context from seed question if provided"""
        if not seed_question_id:
            return ""

        if not self.db_client:
            return f"Seed question {seed_question_id} referenced but no database connection available - create an original question."

        try:
            # Use the new intelligent question set retrieval
            self._debug_print(f"[DEBUG] Getting intelligent question set for: {seed_question_id}")

            try:
                # Try the intelligent method first
                question_set = await self.db_client.get_intelligent_question_set(seed_question_id, source="auto")
            except Exception as e:
                self._debug_print(f"[DEBUG] Intelligent retrieval failed: {e}, falling back to basic methods")
                # Fallback to original methods
                try:
                    question_set = await self.db_client.get_full_question_set(seed_question_id)
                except Exception:
                    # Final fallback to local file method
                    question_set = await self.db_client.get_question_set_from_local_file(seed_question_id)

            self._debug_print(f"[DEBUG] Retrieved {len(question_set) if question_set else 0} questions in set")

            if not question_set:
                # Fallback to single question
                try:
                    seed_data = await self.db_client.get_past_paper_question(seed_question_id)
                except Exception:
                    # Fallback to random question from local file
                    local_questions = await self.db_client.get_questions_from_local_file(limit=1)
                    seed_data = local_questions[0] if local_questions else None

                if not seed_data:
                    return f"Seed question {seed_question_id} not found - create an original question."

                return self._format_single_seed_context(seed_data)

            # Determine if this is a multi-part question set or single question
            if len(question_set) > 1:
                self._debug_print(f"[DEBUG] Using multi-part question set context ({len(question_set)} parts)")
                return self._format_question_set_context(question_set)
            else:
                self._debug_print(f"[DEBUG] Using single question context")
                return self._format_single_seed_context(question_set[0])

        except Exception as e:
            self._debug_print(f"[DEBUG] Error getting seed context: {e}")
            return f"Error retrieving seed question {seed_question_id} - create an original question."

    def _format_single_seed_context(self, seed_data: Dict[str, Any]) -> str:
        """Format a single seed question for context"""
        taxonomy = seed_data.get('taxonomy', {})

        return f"""
**Seed Question for Inspiration:**
- Question ID: {seed_data.get('question_id_global', 'N/A')}
- Display: {seed_data.get('question_number_display', 'N/A')}
- Marks: {seed_data.get('marks', 'N/A')}
- Command Word: {seed_data.get('command_word', 'N/A')}
- Question Text: {seed_data.get('raw_text_content', 'N/A')}

**Seed Question Context:**
- Topic Path: {' > '.join(taxonomy.get('topic_path', [])) if isinstance(taxonomy.get('topic_path'), list) else taxonomy.get('topic_path', 'N/A')}
- Content References: {', '.join(taxonomy.get('subject_content_references', [])) if isinstance(taxonomy.get('subject_content_references'), list) else taxonomy.get('subject_content_references', 'N/A')}
- Difficulty Level: {taxonomy.get('difficulty_estimate_0_to_1', 'N/A')}
- Has Assets: {seed_data.get('asset_count', 0) > 0}

**Instructions:** Use this as inspiration but create a COMPLETELY NEW question with different context, numbers, and approach. Maintain similar complexity and topic focus.
"""

    def _format_question_set_context(self, question_set: List[Dict[str, Any]]) -> str:
        """Format a full question set (multiple parts) for context"""
        if not question_set:
            return "No question set found."

        main_question = question_set[0]
        taxonomy = main_question.get('taxonomy', {})

        context = f"""
**Seed Question Set for Inspiration ({len(question_set)} parts):**

**Question Overview:**
- Base Question: {main_question.get('question_id_global', 'N/A').split('_q')[1][0] if '_q' in main_question.get('question_id_global', '') else 'N/A'}
- Topic Path: {' > '.join(taxonomy.get('topic_path', [])) if isinstance(taxonomy.get('topic_path'), list) else taxonomy.get('topic_path', 'N/A')}
- Content References: {', '.join(taxonomy.get('subject_content_references', [])) if isinstance(taxonomy.get('subject_content_references'), list) else taxonomy.get('subject_content_references', 'N/A')}
- Total Marks: {sum(q.get('marks', 0) for q in question_set)}

**Question Parts:**
"""

        for i, part in enumerate(question_set, 1):
            context += f"""
Part {part.get('question_number_display', i)}: {part.get('command_word', 'Work out')} ({part.get('marks', 0)} mark{'s' if part.get('marks', 0) != 1 else ''})
- {part.get('raw_text_content', 'N/A')[:100]}{'...' if len(part.get('raw_text_content', '')) > 100 else ''}
"""

        context += f"""

**Instructions:** Create a COMPLETELY NEW multi-part question inspired by this structure and topic. Use different context, numbers, and scenarios while maintaining similar:
- Question progression and complexity
- Mark allocation pattern
- Command word usage
- Topic focus and learning objectives
"""

        return context

    async def _call_llm(self, prompt: str, config: GenerationConfig) -> str:
        """Call the LLM with the prepared prompt"""
        try:
            self._debug_print(f"[DEBUG] =================== LLM CALL START ===================")
            self._debug_print(f"[DEBUG] Model: {config.llm_model_generation.value}")
            self._debug_print(f"[DEBUG] Temperature: {config.temperature}")
            self._debug_print(f"[DEBUG] Max tokens: {config.max_tokens}")
            self._debug_print(f"[DEBUG] Prompt length: {len(prompt)} characters")

            # Show first and last 300 chars of prompt
            self._debug_print(f"[DEBUG] Prompt start: {prompt[:300]}...")
            self._debug_print(f"[DEBUG] Prompt end: ...{prompt[-300:]}")

            # Prepare messages in the format expected by smolagents
            messages = [
                {"role": "system", "content": "You are a helpful assistant designed to output JSON. You must respond with valid JSON only, no additional text or formatting."},
                {"role": "user", "content": prompt}
            ]

            self._debug_print(f"[DEBUG] Calling model with {len(messages)} messages")

            # Call the model directly with JSON mode
            response = self.model(messages)

            # Extract content if response is a ChatMessage object
            content = getattr(response, "content", response)

            self._debug_print(f"[DEBUG] =================== LLM RESPONSE ===================")
            self._debug_print(f"[DEBUG] Response type: {type(content)}")
            self._debug_print(f"[DEBUG] Response length: {len(str(content))} characters")
            self._debug_print(f"[DEBUG] Full response:")
            self._debug_print(str(content))
            self._debug_print(f"[DEBUG] =================== END RESPONSE ===================")

            return content
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse the LLM response to extract JSON question data"""
        try:
            self._debug_print(f"[DEBUG] =================== JSON PARSING START ===================")
            self._debug_print(f"[DEBUG] Original response length: {len(response)} characters")

            # Step 1: Remove <think>...</think> reasoning tokens
            content = self._strip_thinking_tokens(response)
            self._debug_print(f"[DEBUG] After stripping thinking tokens: {len(content)} characters")

            if len(content) != len(response):
                self._debug_print(f"[DEBUG] Removed {len(response) - len(content)} characters of thinking tokens")

            # Step 2: Try to extract JSON from code blocks first
            self._debug_print(f"[DEBUG] Attempting to extract JSON from code blocks...")
            json_from_code_block = self._extract_json_from_code_block(content)
            if json_from_code_block:
                self._debug_print(f"[DEBUG] ✅ Successfully extracted JSON from code block!")
                self._debug_print(f"[DEBUG] Extracted JSON keys: {list(json_from_code_block.keys())}")
                return json_from_code_block

            self._debug_print(f"[DEBUG] No JSON found in code blocks, trying raw JSON extraction...")

            # Step 3: Try to find raw JSON in the response
            json_from_raw = self._extract_raw_json(content)
            if json_from_raw:
                self._debug_print(f"[DEBUG] ✅ Successfully extracted raw JSON!")
                self._debug_print(f"[DEBUG] Extracted JSON keys: {list(json_from_raw.keys())}")
                return json_from_raw

            self._debug_print(f"[DEBUG] ❌ No valid JSON found in LLM response")
            self._debug_print(f"[DEBUG] Content preview (first 500 chars): {content[:500]}")
            self._debug_print(f"[DEBUG] =================== JSON PARSING END ===================")
            return None

        except Exception as e:
            print(f"[ERROR] Exception in _parse_llm_response: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _strip_thinking_tokens(self, text: str) -> str:
        """Remove <think>...</think> blocks from the response"""
        import re
        # Remove <think>...</think> blocks (case insensitive, multiline)
        pattern = r'<think>.*?</think>'
        cleaned = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()

    def _extract_json_from_code_block(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from ```json ... ``` code blocks"""
        import re
        import json

        # Look for ```json ... ``` blocks
        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        # Also try generic ``` blocks
        generic_pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(generic_pattern, text, re.DOTALL)

        for match in matches:
            try:
                # Skip if it looks like code (contains keywords)
                if any(keyword in match.lower() for keyword in ['def ', 'import ', 'class ', 'function']):
                    continue
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        return None

    def _extract_raw_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract raw JSON object from text"""
        import json

        # Try to find JSON object boundaries
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            return None

        json_str = text[start_idx:end_idx]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self._debug_print(f"[DEBUG] Error parsing raw JSON: {e}")
            self._debug_print(f"[DEBUG] JSON string attempted: {json_str[:200]}...")
            return None

    def _create_candidate_question(self, question_data: Dict[str, Any], config: GenerationConfig) -> CandidateQuestion:
        """Create a CandidateQuestion object from parsed data"""
        import uuid

        self._debug_print(f"[DEBUG] =================== FIELD VALIDATION START ===================")
        self._debug_print(f"[DEBUG] Parsed question_data keys: {list(question_data.keys())}")

        # Check for required top-level fields
        required_fields = [
            'question_id_local', 'question_id_global', 'question_number_display',
            'marks', 'command_word', 'raw_text_content', 'taxonomy',
            'solution_and_marking_scheme', 'solver_algorithm'
        ]

        missing_fields = []
        for field in required_fields:
            if field not in question_data:
                missing_fields.append(field)

        if missing_fields:
            self._debug_print(f"[DEBUG] ❌ Missing required fields: {missing_fields}")
        else:
            self._debug_print(f"[DEBUG] ✅ All top-level fields present")

        # Validate nested structures
        taxonomy_data = question_data.get("taxonomy", {})
        self._debug_print(f"[DEBUG] Taxonomy keys: {list(taxonomy_data.keys())}")

        solution_data = question_data.get("solution_and_marking_scheme", {})
        self._debug_print(f"[DEBUG] Solution keys: {list(solution_data.keys())}")

        solver_data = question_data.get("solver_algorithm", {})
        self._debug_print(f"[DEBUG] Solver keys: {list(solver_data.keys())}")

        # Parse taxonomy
        taxonomy = QuestionTaxonomy(
            topic_path=taxonomy_data.get("topic_path", ["Unknown"]),
            subject_content_references=self._parse_subject_content_refs(
                taxonomy_data.get("subject_content_references", config.subject_content_references)
            ),
            skill_tags=taxonomy_data.get("skill_tags", ["basic"]),
            cognitive_level=taxonomy_data.get("cognitive_level"),
            difficulty_estimate_0_to_1=taxonomy_data.get("difficulty_estimate_0_to_1")
        )

        # Parse solution and marking scheme
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
            self._debug_print(f"[DEBUG] ⚠️ Invalid command word '{command_word_str}', using default")
            command_word = CommandWord.WORK_OUT  # Default fallback

        # Handle question_id_local and question_id_global gracefully
        question_id_local = question_data.get("question_id_local")
        if not question_id_local or "random_number" in str(question_id_local) or "Generated_Q" in str(question_id_local):
            question_id_local = f"Gen_Q{random.randint(1000, 9999)}"
            self._debug_print(f"[DEBUG] Generated question_id_local: {question_id_local}")

        question_id_global = question_data.get("question_id_global")
        if not question_id_global or "random_number" in str(question_id_global) or "gen_{generation_id}" in str(question_id_global):
            question_id_global = f"gen_{config.generation_id}_q{random.randint(100, 999)}"
            self._debug_print(f"[DEBUG] Generated question_id_global: {question_id_global}")

        # Debug key field values
        self._debug_print(f"[DEBUG] Question text length: {len(question_data.get('raw_text_content', ''))}")
        self._debug_print(f"[DEBUG] Marks: {question_data.get('marks', 'MISSING')}")
        self._debug_print(f"[DEBUG] Number of answers: {len(answers)}")
        self._debug_print(f"[DEBUG] Number of criteria: {len(criteria)}")
        self._debug_print(f"[DEBUG] Number of steps: {len(steps)}")
        self._debug_print(f"[DEBUG] =================== FIELD VALIDATION END ===================")

        # Create the candidate question
        candidate_question = CandidateQuestion(
            question_id_local=question_id_local,
            question_id_global=question_id_global,
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

    def _parse_subject_content_refs(self, content_refs) -> List[str]:
        """Parse subject content references, handling both string and list inputs"""
        if isinstance(content_refs, str):
            # Split by comma and clean up
            refs = [ref.strip() for ref in content_refs.split(',')]
            self._debug_print(f"[DEBUG] Converted string subject_content_references to list: {refs}")
            return refs
        elif isinstance(content_refs, list):
            return content_refs
        else:
            self._debug_print(f"[DEBUG] ⚠️ Unexpected subject_content_references type: {type(content_refs)}")
            return ["Unknown"]

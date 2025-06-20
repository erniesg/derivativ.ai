"""
Marker Agent for Cambridge IGCSE Mathematics.

Generates detailed marking schemes following Cambridge marking principles
with async patterns, dependency injection, and comprehensive error handling.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from ..models.enums import LLMModel
from ..models.question_models import (
    FinalAnswer,
    GenerationRequest,
    MarkingCriterion,
    MarkType,
    SolutionAndMarkingScheme,
)
from ..services import JSONParser, LLMService, PromptManager
from ..services.prompt_manager import PromptConfig
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class MarkerAgentError(Exception):
    """Raised when marking scheme generation fails"""

    pass


class MarkerAgent(BaseAgent):
    """
    Agent responsible for generating Cambridge IGCSE Mathematics marking schemes.

    Features:
    - Async marking scheme generation with proper error handling
    - Cambridge marking principles compliance
    - Multiple LLM provider support with retry logic
    - Automatic fallback strategies
    - Comprehensive logging and observability
    """

    def __init__(
        self,
        name: str = "Marker",
        llm_service: Optional[LLMService] = None,
        prompt_manager: Optional[PromptManager] = None,
        json_parser: Optional[JSONParser] = None,
        config: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize Marker Agent.

        Args:
            name: Agent name for logging and identification
            llm_service: LLM service for content generation
            prompt_manager: Prompt management service
            json_parser: JSON extraction service
            config: Agent configuration
        """
        # Initialize services if not provided (dependency injection pattern)
        if llm_service is None:
            from ..services import MockLLMService

            llm_service = MockLLMService()

        if prompt_manager is None:
            prompt_manager = PromptManager()

        if json_parser is None:
            json_parser = JSONParser()

        super().__init__(name, llm_service, config)

        self.llm_service = llm_service
        self.prompt_manager = prompt_manager
        self.json_parser = json_parser

        # Agent configuration with defaults
        self.agent_config = {
            "max_retries": 3,
            "generation_timeout": 45,
            "quality_threshold": 0.7,
            "enable_fallback": True,
            **self.config,
        }

        logger.info(f"Initialized {name} with LLM service: {type(llm_service).__name__}")

    async def _execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Execute marking scheme generation based on input request.

        Args:
            input_data: Dictionary containing question and configuration

        Returns:
            Dictionary with generated marking scheme data

        Raises:
            MarkerAgentError: If marking generation fails after all retries
        """
        try:
            # Parse and validate input
            question, config = self._parse_marking_request(input_data)
            self._observe(
                f"Parsed marking request for question: {question['question_text'][:50]}..."
            )

            # Generate marking scheme with retries
            marking_scheme = await self._generate_marking_scheme_with_retries(question, config)

            if not marking_scheme:
                raise MarkerAgentError("Failed to generate marking scheme after all retry attempts")

            # Validate generated marking scheme
            await self._validate_generated_marking_scheme(marking_scheme, config.marks)
            self._act("Successfully generated and validated marking scheme")

            return {
                "marking_scheme": marking_scheme.model_dump(),
                "question": question,
                "config": config.model_dump(),
                "generation_metadata": {
                    "agent_name": self.name,
                    "timestamp": datetime.now().isoformat(),
                    "model_used": config.llm_model.value,
                },
            }

        except Exception as e:
            self._observe(f"Marking generation failed: {e!s}")
            raise MarkerAgentError(f"Marking scheme generation failed: {e}")

    def _parse_marking_request(
        self, input_data: dict[str, Any]
    ) -> tuple[dict[str, Any], GenerationRequest]:
        """Parse and validate marking request"""
        try:
            self._think("Parsing marking request parameters")

            # Extract question data
            if "question" not in input_data:
                raise ValueError("Question data is required for marking scheme generation")

            question = input_data["question"]
            if not isinstance(question, dict) or "question_text" not in question:
                raise ValueError("Question must contain question_text")

            # Extract config data
            if "config" not in input_data:
                raise ValueError("Configuration is required for marking scheme generation")

            config_data = input_data["config"]
            if isinstance(config_data, dict):
                config = GenerationRequest(**config_data)
            elif isinstance(config_data, GenerationRequest):
                config = config_data
            else:
                raise ValueError("Config must be a dictionary or GenerationRequest")

            self._observe(f"Validated request: marks={config.marks}, topic={config.topic}")
            return question, config

        except ValueError as e:
            raise MarkerAgentError(str(e))
        except Exception as e:
            raise MarkerAgentError(f"Invalid marking request: {e}")

    async def _generate_marking_scheme_with_retries(
        self, question: dict[str, Any], config: GenerationRequest
    ) -> Optional[SolutionAndMarkingScheme]:
        """Generate marking scheme with retry logic and fallbacks"""
        max_retries = self.agent_config["max_retries"]

        for attempt in range(1, max_retries + 1):
            try:
                self._think(f"Generation attempt {attempt}/{max_retries}")

                # Generate using primary method
                marking_scheme = await self._generate_marking_scheme_primary(
                    question, config, attempt
                )

                if marking_scheme:
                    self._act(f"Successfully generated marking scheme on attempt {attempt}")
                    return marking_scheme

            except Exception as e:
                self._observe(f"Attempt {attempt} failed: {e!s}")

                # Try fallback on final attempt
                if attempt == max_retries and self.agent_config.get("enable_fallback"):
                    try:
                        self._think("Trying fallback marking generation method")
                        marking_scheme = await self._generate_marking_scheme_fallback(
                            question, config
                        )
                        if marking_scheme:
                            self._act("Successfully generated marking scheme using fallback method")
                            return marking_scheme
                    except Exception as fallback_error:
                        self._observe(f"Fallback method also failed: {fallback_error!s}")

                # Wait before retry (exponential backoff)
                if attempt < max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)

        return None

    async def _generate_marking_scheme_primary(
        self, question: dict[str, Any], config: GenerationRequest, attempt: int
    ) -> Optional[SolutionAndMarkingScheme]:
        """Primary marking scheme generation method"""
        try:
            # Prepare prompt
            prompt_config = PromptConfig(
                template_name="marking_scheme",
                version="latest",
                variables={
                    "question_text": question["question_text"],
                    "total_marks": config.marks,
                    "command_word": question.get("command_word", "Calculate"),
                    "subject_content_references": question.get("subject_content_refs", []),
                    "calculator_policy": config.calculator_policy.value,
                    "target_grade": config.grade_level or 7,
                },
            )

            # Render prompt
            prompt = await self.prompt_manager.render_prompt(prompt_config, config.llm_model.value)
            self._observe(f"Generated prompt for marking scheme (length: {len(prompt)})")

            # Call LLM service
            llm_response = await self.llm_service.generate(
                prompt=prompt,
                model=config.llm_model,
                temperature=config.temperature,
                timeout=self.agent_config["generation_timeout"],
            )

            self._observe(f"Received LLM response (tokens: {llm_response.tokens_used})")

            # Parse JSON response
            extraction_result = await self.json_parser.extract_json(
                llm_response.content, model_name=config.llm_model.value
            )

            if not extraction_result.success:
                raise MarkerAgentError(f"Failed to extract JSON: {extraction_result.error}")

            # Convert to SolutionAndMarkingScheme object
            marking_scheme = self._convert_to_marking_scheme_object(
                extraction_result.data, question_marks=config.marks
            )

            return marking_scheme

        except Exception as e:
            logger.warning(f"Primary marking generation failed on attempt {attempt}: {e}")
            raise

    async def _generate_marking_scheme_fallback(
        self, question: dict[str, Any], config: GenerationRequest
    ) -> Optional[SolutionAndMarkingScheme]:
        """Fallback marking generation method with simpler prompt"""
        try:
            self._think("Using simplified fallback marking generation")

            # Use simpler prompt for fallback
            simplified_prompt = f"""Generate a Cambridge IGCSE marking scheme for this question:
"{question['question_text']}"

Total marks: {config.marks}

Return JSON with: total_marks, mark_allocation_criteria (with criterion_text, marks_value, mark_type), final_answers"""

            llm_response = await self.llm_service.generate(
                prompt=simplified_prompt,
                model=LLMModel.GPT_4O_MINI,  # Use faster model for fallback
                temperature=0.5,
                timeout=30,
            )

            extraction_result = await self.json_parser.extract_json(
                llm_response.content, model_name=LLMModel.GPT_4O_MINI.value
            )

            if extraction_result.success:
                return self._convert_to_marking_scheme_object(
                    extraction_result.data, question_marks=config.marks, is_fallback=True
                )

            return None

        except Exception as e:
            logger.warning(f"Fallback marking generation failed: {e}")
            return None

    def _convert_to_marking_scheme_object(
        self, json_data: dict[str, Any], question_marks: int, is_fallback: bool = False
    ) -> SolutionAndMarkingScheme:
        """Convert parsed JSON to SolutionAndMarkingScheme object"""
        try:
            self._think("Converting JSON response to SolutionAndMarkingScheme object")

            # Extract basic marking data
            total_marks = json_data.get("total_marks", question_marks)
            criteria_data = json_data.get("mark_allocation_criteria", [])
            answers_data = json_data.get("final_answers", [])

            # Generate unique IDs for criteria
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Convert marking criteria
            mark_criteria = []
            for i, criterion_data in enumerate(criteria_data):
                if isinstance(criterion_data, dict):
                    criterion_text = criterion_data.get("criterion_text", f"Criterion {i+1}")
                    marks_value = criterion_data.get("marks_value", 1)
                    mark_type = criterion_data.get("mark_type", "M")

                    # Validate mark type
                    try:
                        mark_type_enum = MarkType(mark_type)
                    except ValueError:
                        mark_type_enum = MarkType.M  # Default to Method mark

                    mark_criteria.append(
                        MarkingCriterion(
                            criterion_id=f"crit_{timestamp}_{i+1}",
                            criterion_text=criterion_text,
                            mark_code_display=f"{mark_type}{i+1}",
                            marks_value=marks_value,
                            mark_type_primary=mark_type_enum,
                        )
                    )

            # If no criteria provided or fallback mode, create defaults
            if not mark_criteria or is_fallback:
                if total_marks == 1:
                    mark_criteria = [
                        MarkingCriterion(
                            criterion_id=f"crit_{timestamp}_1",
                            criterion_text="Correct method and answer",
                            mark_code_display="M1",
                            marks_value=1,
                            mark_type_primary=MarkType.M,
                        )
                    ]
                else:
                    # Create method and accuracy marks
                    mark_criteria = []
                    for i in range(total_marks - 1):
                        mark_criteria.append(
                            MarkingCriterion(
                                criterion_id=f"crit_{timestamp}_{i+1}",
                                criterion_text=f"Correct method step {i+1}",
                                mark_code_display=f"M{i+1}",
                                marks_value=1,
                                mark_type_primary=MarkType.M,
                            )
                        )
                    # Final mark for accuracy
                    mark_criteria.append(
                        MarkingCriterion(
                            criterion_id=f"crit_{timestamp}_{total_marks}",
                            criterion_text="Correct final answer",
                            mark_code_display=f"A{total_marks}",
                            marks_value=1,
                            mark_type_primary=MarkType.A,
                        )
                    )

            # Convert final answers
            final_answers = []
            for answer_data in answers_data:
                if isinstance(answer_data, dict):
                    final_answers.append(
                        FinalAnswer(
                            answer_text=answer_data.get("answer_text", "Answer"),
                            value_numeric=answer_data.get("value_numeric"),
                            unit=answer_data.get("unit"),
                        )
                    )

            # If no answers provided, create default
            if not final_answers:
                final_answers = [FinalAnswer(answer_text="Correct answer")]

            # Create SolutionAndMarkingScheme object
            marking_scheme = SolutionAndMarkingScheme(
                final_answers_summary=final_answers,
                mark_allocation_criteria=mark_criteria,
                total_marks_for_part=total_marks,
            )

            self._act(
                f"Converted to SolutionAndMarkingScheme: {len(mark_criteria)} criteria, {total_marks} marks"
            )
            return marking_scheme

        except Exception as e:
            raise MarkerAgentError(f"Failed to convert JSON to SolutionAndMarkingScheme: {e}")

    async def _validate_generated_marking_scheme(
        self, marking_scheme: SolutionAndMarkingScheme, expected_marks: int
    ):
        """Validate the generated marking scheme meets requirements"""
        try:
            self._think("Validating generated marking scheme")

            # Basic validation
            if not marking_scheme.mark_allocation_criteria:
                raise ValueError("Marking scheme must have allocation criteria")

            if marking_scheme.total_marks_for_part < 1:
                raise ValueError("Total marks must be at least 1")

            # Check mark allocation consistency
            allocated_marks = sum(
                criterion.marks_value for criterion in marking_scheme.mark_allocation_criteria
            )
            if allocated_marks != marking_scheme.total_marks_for_part:
                logger.warning(
                    f"Mark allocation mismatch: criteria={allocated_marks}, total={marking_scheme.total_marks_for_part}"
                )

            # Validate mark types
            mark_types = [
                criterion.mark_type_primary for criterion in marking_scheme.mark_allocation_criteria
            ]
            if not any(
                mark_type in [MarkType.M, MarkType.A, MarkType.B] for mark_type in mark_types
            ):
                logger.warning("No standard mark types (M/A/B) found in criteria")

            # Check final answers
            if not marking_scheme.final_answers_summary:
                logger.warning("No final answers provided in marking scheme")

            self._observe("Marking scheme validation completed successfully")

        except Exception as e:
            raise MarkerAgentError(f"Marking scheme validation failed: {e}")

    def get_marking_stats(self) -> dict[str, Any]:
        """Get statistics about marking generation performance"""
        # This could be enhanced with actual metrics tracking
        return {
            "total_markings": getattr(self, "_marking_count", 0),
            "success_rate": getattr(self, "_success_rate", 0.0),
            "average_marking_time": getattr(self, "_avg_time", 0.0),
            "cache_stats": self.json_parser.get_extraction_stats()
            if hasattr(self.json_parser, "get_extraction_stats")
            else {},
        }

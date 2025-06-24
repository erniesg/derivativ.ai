"""
Review Agent for Cambridge IGCSE Mathematics Questions.

Provides comprehensive quality assessment with multi-dimensional scoring,
feedback generation, and actionable recommendations for improvement.
"""

import asyncio
import logging
from typing import Any, Optional

from src.agents.base_agent import BaseAgent
from src.models.enums import QualityAction
from src.models.question_models import QualityDecision
from src.services.json_parser import JSONExtractionResult
from src.services.llm_service import LLMService, MockLLMService
from src.services.prompt_manager import PromptConfig, PromptManager

logger = logging.getLogger(__name__)


class ReviewAgent(BaseAgent):
    """
    Cambridge IGCSE Mathematics quality review agent.

    Provides multi-dimensional quality assessment with:
    - Mathematical accuracy scoring (0-1)
    - Cambridge syllabus compliance (0-1)
    - Grade-level appropriateness (0-1)
    - Question clarity assessment (0-1)
    - Marking scheme accuracy (0-1)
    - Actionable feedback and improvement suggestions
    """

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        prompt_manager: Optional[PromptManager] = None,
        json_parser: Optional[Any] = None,
        name: str = "Review",
    ):
        """
        Initialize Review Agent with service dependencies.

        Args:
            llm_service: LLM service for quality assessment reasoning
            prompt_manager: Prompt management for review templates
            json_parser: JSON extraction from LLM responses
            name: Agent name for identification
        """
        # Initialize with mock services if none provided (for testing)
        if llm_service is None:
            llm_service = MockLLMService()

        if prompt_manager is None:
            prompt_manager = PromptManager()

        if json_parser is None:
            from src.services.json_parser import JSONParser

            json_parser = JSONParser()

        # Create agent-compatible LLM interface
        from src.services.agent_llm_interface import AgentLLMInterface

        self.llm_interface = AgentLLMInterface(llm_service)

        super().__init__(name, llm_service)
        self.llm_service = llm_service  # Store for easy access
        self.prompt_manager = prompt_manager
        self.json_parser = json_parser

        # Quality assessment configuration
        self.quality_thresholds = {
            "auto_approve": 0.85,
            "manual_review_upper": 0.84,
            "manual_review_lower": 0.60,
            "refine_upper": 0.59,
            "refine_lower": 0.50,  # Adjusted so 0.45 falls into regenerate/reject
            "reject_threshold": 0.40,
        }

        # Dimension weights for overall score calculation
        self.dimension_weights = {
            "mathematical_accuracy": 0.25,
            "cambridge_compliance": 0.25,
            "grade_appropriateness": 0.20,
            "question_clarity": 0.15,
            "marking_accuracy": 0.15,
        }

    async def _execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Execute quality review assessment.

        Args:
            input_data: Dictionary containing question_data with question and marking_scheme

        Returns:
            Dictionary with quality_decision and assessment_metadata

        Raises:
            ValueError: If input validation fails
            Exception: If quality assessment fails
        """
        try:
            # Step 1: Parse and validate input
            self._observe("Received quality review request", input_data)
            question_data = self._parse_review_request(input_data)

            self._think("Preparing quality assessment with Cambridge IGCSE standards")

            # Step 2: Generate quality assessment using LLM
            (
                quality_assessment_result,
                model_used,
            ) = await self._generate_quality_assessment_with_retries(question_data, max_retries=3)

            # Step 3: Convert to structured quality decision
            quality_decision = self._convert_to_quality_decision_object(quality_assessment_result)

            self._act(
                "Quality assessment completed",
                {
                    "quality_score": quality_decision.quality_score,
                    "action": quality_decision.action,
                    "issues_count": len(quality_decision.suggested_improvements),
                },
            )

            return {
                "quality_decision": quality_decision.model_dump(),
                "assessment_metadata": {
                    "agent_name": self.name,
                    "timestamp": self._get_timestamp(),
                    "model_used": model_used,
                    "assessment_confidence": quality_decision.confidence,
                },
            }

        except ValueError as e:
            error_msg = f"Review input validation failed: {e}"
            self._observe(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Review failed: {e}"
            self._observe(error_msg)
            raise Exception(error_msg)

    def _parse_review_request(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Parse and validate review request input.

        Args:
            input_data: Raw input data

        Returns:
            Validated question data with question and marking scheme

        Raises:
            ValueError: If required fields are missing or invalid
        """
        if "question_data" not in input_data:
            raise ValueError("Missing required field: question_data")

        question_data = input_data["question_data"]

        if "question" not in question_data:
            raise ValueError("Missing required field: question_data.question")

        if "marking_scheme" not in question_data:
            raise ValueError("Missing required field: question_data.marking_scheme")

        # Validate question has required fields
        question = question_data["question"]
        required_question_fields = ["question_text", "marks", "command_word", "grade_level"]
        for field in required_question_fields:
            if field not in question:
                raise ValueError(f"Missing required question field: {field}")

        # Validate marking scheme has required fields
        marking_scheme = question_data["marking_scheme"]
        if (
            "total_marks_for_part" not in marking_scheme
            and "mark_allocation_criteria" not in marking_scheme
        ):
            raise ValueError(
                "Marking scheme must have either total_marks_for_part or mark_allocation_criteria"
            )

        return question_data

    async def _generate_quality_assessment_with_retries(
        self, question_data: dict[str, Any], max_retries: int = 3
    ) -> tuple[dict[str, Any], str]:
        """
        Generate quality assessment with retry logic.

        Args:
            question_data: Question and marking scheme data
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple of (quality assessment data from LLM, model used)

        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None

        for attempt in range(max_retries):
            try:
                self._think(f"Generating quality assessment (attempt {attempt + 1}/{max_retries})")

                # Prepare prompt for quality review
                prompt_config = PromptConfig(
                    template_name="quality_review",
                    version="latest",
                    variables={"question_data": question_data},
                )

                # Generate prompt and get LLM response
                prompt = await self.prompt_manager.render_prompt(prompt_config, "gpt-4o")
                llm_response = await self.llm_interface.generate(
                    prompt=prompt,
                    max_tokens=1000,
                    temperature=0.3,  # Lower temperature for consistent quality assessment
                )

                self._observe(
                    f"Generated quality assessment response: {len(llm_response.content)} chars"
                )

                # Extract JSON from response
                json_result: JSONExtractionResult = await self.json_parser.extract_json(
                    llm_response.content, llm_response.model_used
                )

                if not json_result.success or not json_result.data:
                    raise ValueError(
                        f"Failed to extract JSON from quality assessment: {json_result.error}"
                    )

                # Validate quality scores
                if not self._validate_quality_scores(json_result.data):
                    raise ValueError("Quality scores validation failed")

                self._act("Quality assessment generated successfully")
                return json_result.data, llm_response.model_used

            except Exception as e:
                last_exception = e
                self._observe(f"Quality assessment attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff

        raise Exception(f"Quality assessment failed after {max_retries} attempts: {last_exception}")

    def _convert_to_quality_decision_object(
        self, assessment_data: dict[str, Any]
    ) -> QualityDecision:
        """
        Convert LLM assessment response to QualityDecision object.

        Args:
            assessment_data: Raw assessment data from LLM

        Returns:
            Structured QualityDecision object

        Raises:
            ValueError: If conversion fails or data is invalid
        """
        try:
            # Extract and validate overall quality score
            quality_score = float(assessment_data.get("overall_quality_score", 0))
            if not (0 <= quality_score <= 1):
                raise ValueError(f"Invalid quality score: {quality_score}")

            # Determine action based on quality score and thresholds
            action = self._determine_quality_action(quality_score)

            # Calculate confidence based on score clarity (distance from thresholds)
            confidence = self._calculate_confidence(quality_score)

            # Extract feedback and improvements
            feedback_summary = assessment_data.get("feedback_summary", "")
            specific_issues = assessment_data.get("specific_issues", [])
            suggested_improvements = assessment_data.get("suggested_improvements", [])

            # Combine feedback for reasoning
            reasoning_parts = [feedback_summary]
            if specific_issues:
                reasoning_parts.append(f"Issues: {', '.join(specific_issues)}")
            if suggested_improvements:
                reasoning_parts.append(f"Improvements: {', '.join(suggested_improvements)}")

            reasoning = " | ".join(filter(None, reasoning_parts))

            # Create QualityDecision object
            quality_decision = QualityDecision(
                action=action,
                confidence=confidence,
                reasoning=reasoning,
                quality_score=quality_score,
                suggested_improvements=suggested_improvements,
                mathematical_accuracy=assessment_data.get("mathematical_accuracy"),
                cambridge_compliance=assessment_data.get("cambridge_compliance"),
                grade_appropriateness=assessment_data.get("grade_appropriateness"),
            )

            return quality_decision

        except Exception as e:
            raise ValueError(f"Failed to convert quality assessment to QualityDecision: {e}")

    def _determine_quality_action(self, quality_score: float) -> QualityAction:
        """
        Determine quality action based on score and thresholds.

        Args:
            quality_score: Overall quality score (0-1)

        Returns:
            Recommended quality action
        """
        if quality_score >= self.quality_thresholds["auto_approve"]:
            return QualityAction.APPROVE
        elif quality_score >= self.quality_thresholds["manual_review_lower"]:
            return QualityAction.MANUAL_REVIEW
        elif quality_score >= self.quality_thresholds["refine_lower"]:
            return QualityAction.REFINE
        elif quality_score > self.quality_thresholds["reject_threshold"]:
            return QualityAction.REGENERATE
        else:
            return QualityAction.REJECT

    def _calculate_confidence(self, quality_score: float) -> float:
        """
        Calculate confidence in quality decision based on score clarity.

        Args:
            quality_score: Overall quality score (0-1)

        Returns:
            Confidence level (0-1)
        """
        # Higher confidence when score is clearly in one category
        thresholds = [
            self.quality_thresholds["reject_threshold"],
            self.quality_thresholds["refine_lower"],
            self.quality_thresholds["manual_review_lower"],
            self.quality_thresholds["auto_approve"],
        ]

        # Find distance to nearest threshold
        distances = [abs(quality_score - threshold) for threshold in thresholds]
        min_distance = min(distances)

        # Convert distance to confidence (farther from threshold = higher confidence)
        # Maximum distance in any range is ~0.25, so normalize to 0-1
        confidence = min(0.95, 0.5 + (min_distance * 2))

        return round(confidence, 2)

    def _calculate_overall_quality_score(self, dimension_scores: dict[str, float]) -> float:
        """
        Calculate weighted overall quality score from dimension scores.

        Args:
            dimension_scores: Individual dimension scores

        Returns:
            Weighted overall quality score (0-1)
        """
        total_score = 0.0
        total_weight = 0.0

        for dimension, weight in self.dimension_weights.items():
            if dimension in dimension_scores and dimension_scores[dimension] is not None:
                total_score += dimension_scores[dimension] * weight
                total_weight += weight

        # Normalize by actual weights used
        if total_weight == 0:
            return 0.0

        return round(total_score / total_weight, 2)

    def _validate_quality_scores(self, assessment_data: dict[str, Any]) -> bool:
        """
        Validate that quality scores are within valid ranges.

        Args:
            assessment_data: Assessment data to validate

        Returns:
            True if valid, False otherwise
        """
        score_fields = [
            "overall_quality_score",
            "mathematical_accuracy",
            "cambridge_compliance",
            "grade_appropriateness",
            "question_clarity",
            "marking_accuracy",
        ]

        for field in score_fields:
            if field in assessment_data:
                value = assessment_data[field]
                if value is not None:
                    try:
                        score = float(value)
                        if not (0 <= score <= 1):
                            logger.warning(f"Invalid score range for {field}: {score}")
                            return False
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid score type for {field}: {value}")
                        return False

        # Must have overall quality score
        if "overall_quality_score" not in assessment_data:
            logger.warning("Missing required overall_quality_score")
            return False

        return True

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime

        return datetime.now().isoformat()

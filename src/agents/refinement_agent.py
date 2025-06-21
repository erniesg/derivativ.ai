"""
Refinement Agent for Cambridge IGCSE Mathematics Questions.

Improves questions based on review feedback with configurable quality thresholds,
fallback strategies, and comprehensive improvement tracking.
"""

import logging
from typing import Any, Optional

from ..services.llm_service import LLMService, MockLLMService
from ..services.prompt_manager import PromptManager
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class RefinementAgent(BaseAgent):
    """
    Cambridge IGCSE Mathematics question refinement agent.

    Improves questions based on review feedback with:
    - Targeted vs comprehensive refinement strategies
    - Fallback mechanisms for failed refinements
    - Quality improvement prediction
    - Preservation of essential question structure
    - Configurable improvement approaches
    """

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        prompt_manager: Optional[PromptManager] = None,
        json_parser: Optional[Any] = None,
        name: str = "Refinement",
    ):
        """
        Initialize Refinement Agent with service dependencies.

        Args:
            llm_service: LLM service for refinement generation
            prompt_manager: Prompt management for refinement templates
            json_parser: JSON extraction from LLM responses
            name: Agent name for identification
        """
        # Initialize with mock services if none provided (for testing)
        if llm_service is None:
            llm_service = MockLLMService()

        if prompt_manager is None:
            prompt_manager = PromptManager()

        if json_parser is None:
            from ..services.json_parser import JSONParser

            json_parser = JSONParser()

        # Create agent-compatible LLM interface
        from ..services.agent_llm_interface import AgentLLMInterface
        self.llm_interface = AgentLLMInterface(llm_service)
        
        super().__init__(name, llm_service)
        self.llm_service = llm_service  # Store for easy access
        self.prompt_manager = prompt_manager
        self.json_parser = json_parser

    async def _execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Execute question refinement.

        Args:
            input_data: Dictionary containing original_question and quality_decision

        Returns:
            Dictionary with refined_question and refinement_metadata

        Raises:
            ValueError: If input validation fails
            Exception: If refinement fails
        """
        try:
            # Step 1: Parse and validate input
            self._observe("Received refinement request", input_data)
            refinement_data = self._parse_refinement_request(input_data)

            # Step 2: Determine refinement strategy
            strategy = self._determine_refinement_strategy(refinement_data["quality_decision"])
            self._think(f"Using {strategy['approach']} refinement strategy")

            # Step 3: Generate refinement
            refined_question = await self._generate_refinement_with_fallback(
                refinement_data["original_question"], strategy
            )

            # Step 4: Generate metadata
            metadata = self._generate_refinement_metadata(
                refinement_data["quality_decision"],
                refined_question.get("improvements_made", []),
                refined_question.get("strategy_used", "primary"),
            )

            self._act(
                "Refinement completed successfully",
                {
                    "strategy": strategy["approach"],
                    "improvements_count": len(refined_question.get("improvements_made", [])),
                },
            )

            return {"refined_question": refined_question, "refinement_metadata": metadata}

        except ValueError as e:
            error_msg = f"Refinement input validation failed: {e}"
            self._observe(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Refinement failed: {e}"
            self._observe(error_msg)
            raise Exception(error_msg)

    def _parse_refinement_request(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Parse and validate refinement request input.

        Args:
            input_data: Raw input data

        Returns:
            Validated refinement data

        Raises:
            ValueError: If required fields are missing or invalid
        """
        if "original_question" not in input_data:
            raise ValueError("Missing required field: original_question")

        if "quality_decision" not in input_data:
            raise ValueError("Missing required field: quality_decision")

        original_question = input_data["original_question"]
        quality_decision = input_data["quality_decision"]

        # Validate original question has required fields
        required_question_fields = ["question_text", "marks", "command_word", "grade_level"]
        for field in required_question_fields:
            if field not in original_question:
                raise ValueError(f"Missing required original question field: {field}")

        # Validate quality decision has required fields
        required_decision_fields = ["action", "quality_score", "suggested_improvements"]
        for field in required_decision_fields:
            if field not in quality_decision:
                raise ValueError(f"Missing required quality decision field: {field}")

        return {"original_question": original_question, "quality_decision": quality_decision}

    def _determine_refinement_strategy(self, quality_decision: dict[str, Any]) -> dict[str, Any]:
        """
        Determine refinement strategy based on quality assessment.

        Args:
            quality_decision: Quality decision with scores and improvements

        Returns:
            Strategy configuration for refinement
        """
        quality_score = quality_decision.get("quality_score", 0.0)
        suggested_improvements = quality_decision.get("suggested_improvements", [])

        # Analyze specific quality dimensions
        low_scores = []
        for dimension in ["mathematical_accuracy", "cambridge_compliance", "grade_appropriateness"]:
            if quality_decision.get(dimension, 1.0) < 0.7:  # More sensitive threshold
                low_scores.append(dimension)

        # Determine strategy based on quality score and number of issues
        if quality_score >= 0.7 or len(suggested_improvements) <= 2:
            return {
                "approach": "targeted",
                "focus_areas": low_scores or ["question_clarity"],
                "specific_issues": suggested_improvements[:2],  # Focus on top issues
            }
        else:
            return {
                "approach": "comprehensive",
                "focus_areas": [*low_scores, "question_clarity"],
                "specific_issues": suggested_improvements,
            }

    def _create_refinement_prompt(
        self, original_question: dict[str, Any], strategy: dict[str, Any]
    ) -> str:
        """
        Create refinement prompt based on strategy.

        Args:
            original_question: Original question data
            strategy: Refinement strategy configuration

        Returns:
            Formatted refinement prompt
        """
        if strategy["approach"] == "targeted":
            return f"""
Improve this Cambridge IGCSE Mathematics question by focusing on: {', '.join(strategy['focus_areas'])}.

Original Question: {original_question['question_text']}
Grade Level: {original_question['grade_level']}
Current Marks: {original_question['marks']}

Specific Issues to Address:
{chr(10).join(f"- {issue}" for issue in strategy['specific_issues'])}

Please provide the improved question in JSON format with improvements_made field.
"""
        else:
            return f"""
Comprehensively improve this Cambridge IGCSE Mathematics question addressing all areas: {', '.join(strategy['focus_areas'])}.

Original Question: {original_question['question_text']}
Grade Level: {original_question['grade_level']}
Current Marks: {original_question['marks']}

Issues to Address:
{chr(10).join(f"- {issue}" for issue in strategy['specific_issues'])}

Please provide a completely improved question in JSON format with improvements_made field.
"""

    async def _generate_refinement_with_fallback(
        self, original_question: dict[str, Any], strategy: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Generate refinement with fallback strategy.

        Args:
            original_question: Original question data
            strategy: Refinement strategy

        Returns:
            Refined question data
        """
        primary_error = None
        fallback_error = None

        # Try primary approach
        try:
            prompt = self._create_refinement_prompt(original_question, strategy)

            llm_response = await self.llm_interface.generate(
                prompt=prompt, max_tokens=1500, temperature=0.7
            )

            # Parse response
            from ..services.json_parser import JSONExtractionResult

            json_result: JSONExtractionResult = await self.json_parser.extract_json(
                llm_response.content, llm_response.model_used
            )

            if json_result.success:
                # Extract refined_question from response if present
                response_data = json_result.data
                question_data = response_data.get("refined_question", response_data)

                if self._validate_refinement_response(question_data):
                    refined = self._apply_refinement_improvements(original_question, question_data)
                    refined["strategy_used"] = "primary"
                    return refined

        except Exception as e:
            primary_error = e
            self._observe(f"Primary refinement failed: {e}")

        # Fallback approach
        try:
            self._think("Using fallback refinement strategy")
            fallback_prompt = f"""
Improve this question: {original_question['question_text']}
Make it more suitable for grade {original_question['grade_level']}.

Provide improved question in JSON format:
{{"question_text": "improved question", "marks": {original_question['marks']}, "improvements_made": ["list of improvements"]}}
"""

            llm_response = await self.llm_interface.generate(
                prompt=fallback_prompt, max_tokens=800, temperature=0.7
            )

            from ..services.json_parser import JSONExtractionResult

            json_result: JSONExtractionResult = await self.json_parser.extract_json(
                llm_response.content, llm_response.model_used
            )

            if json_result.success:
                # Extract refined_question from response if present
                response_data = json_result.data
                question_data = response_data.get("refined_question", response_data)

                refined = self._apply_refinement_improvements(original_question, question_data)
                refined["strategy_used"] = "fallback"
                return refined

        except Exception as e:
            fallback_error = e
            self._observe(f"Fallback refinement failed: {e}")

        # If both primary and fallback failed with LLM service errors, we should fail
        if primary_error and fallback_error:
            self._think("Both primary and fallback refinement strategies failed")
            raise Exception(
                f"All refinement strategies failed - primary: {primary_error}, fallback: {fallback_error}"
            )

        # If we get here, provide minimal improvement as last resort
        self._think("Using minimal improvement as last resort")
        return {
            **original_question,
            "question_text": f"{original_question['question_text']} Show your working.",
            "improvements_made": ["Added working instruction"],
            "strategy_used": "minimal",
        }

    def _validate_refinement_response(self, response: dict[str, Any]) -> bool:
        """
        Validate refinement response.

        Args:
            response: Response data to validate

        Returns:
            True if valid, False otherwise
        """
        # Required fields for a complete refinement response
        required_fields = ["question_text", "marks", "command_word", "grade_level"]

        # Check if we have minimal required fields (just question_text for fallback)
        if "question_text" not in response:
            return False

        # For comprehensive validation, we need all fields
        # But for fallback scenarios, question_text alone might be acceptable
        if len(response) == 1 and "question_text" in response:
            # This is likely a fallback response - needs more structure
            return False

        # Check that marks is a valid integer if present
        return not ("marks" in response and not isinstance(response["marks"], int))

    def _apply_refinement_improvements(
        self, original_question: dict[str, Any], improvements: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Apply refinement improvements to original question.

        Args:
            original_question: Original question data
            improvements: Improvement data from LLM

        Returns:
            Refined question with improvements applied
        """
        # Start with original question as base
        refined_question = original_question.copy()

        # Apply improvements while preserving essential structure
        if "question_text" in improvements:
            refined_question["question_text"] = improvements["question_text"]

        if "marks" in improvements:
            refined_question["marks"] = improvements["marks"]

        # Preserve certain fields unless explicitly changed
        preserve_fields = ["command_word", "grade_level", "tier", "subject_content_references"]
        for field in preserve_fields:
            if field in improvements:
                refined_question[field] = improvements[field]
            # Otherwise keep original value

        # Add improvement tracking
        refined_question["improvements_made"] = improvements.get("improvements_made", [])

        return refined_question

    def _generate_refinement_metadata(
        self,
        quality_decision: dict[str, Any],
        improvements_made: list[str],
        strategy_used: str = "primary",
    ) -> dict[str, Any]:
        """
        Generate refinement metadata.

        Args:
            quality_decision: Original quality decision
            improvements_made: List of improvements applied
            strategy_used: Which strategy was used for refinement

        Returns:
            Refinement metadata
        """
        from datetime import datetime

        return {
            "agent_name": self.name,
            "original_quality_score": quality_decision.get("quality_score", 0.0),
            "improvements_made": improvements_made,
            "refinement_timestamp": datetime.now().isoformat(),
            "improvement_count": len(improvements_made),
            "strategy_used": strategy_used,
        }

    def _calculate_expected_quality_improvement(
        self, original_scores: dict[str, float], improvements_made: list[str]
    ) -> dict[str, float]:
        """
        Calculate expected quality improvement.

        Args:
            original_scores: Original quality dimension scores
            improvements_made: List of improvements made

        Returns:
            Expected improved scores
        """
        expected_scores = original_scores.copy()

        # Simple heuristic improvement based on improvements made
        improvement_keywords = {
            "cambridge": "cambridge_compliance",
            "grade": "grade_appropriateness",
            "clarity": "question_clarity",
            "mathematical": "mathematical_accuracy",
        }

        for improvement in improvements_made:
            improvement_lower = improvement.lower()
            for keyword, dimension in improvement_keywords.items():
                if keyword in improvement_lower and dimension in expected_scores:
                    # Boost score by 0.1-0.2 based on current score
                    current = expected_scores[dimension]
                    boost = min(0.2, (1.0 - current) * 0.3)
                    expected_scores[dimension] = min(1.0, current + boost)

        return expected_scores

    def _determine_refinement_fallback_strategy(
        self, quality_decision: dict[str, Any]
    ) -> dict[str, str]:
        """
        Determine fallback strategy when refinement fails.

        Args:
            quality_decision: Quality decision data

        Returns:
            Fallback strategy configuration
        """
        quality_score = quality_decision.get("quality_score", 0.0)
        math_accuracy = quality_decision.get("mathematical_accuracy", 1.0)

        if math_accuracy < 0.5:
            return {
                "strategy": "regenerate",
                "reason": "mathematical_accuracy too low for refinement",
            }
        elif quality_score < 0.4:
            return {
                "strategy": "regenerate",
                "reason": "Overall quality too low for effective refinement",
            }
        else:
            return {
                "strategy": "manual_review",
                "reason": "Refinement attempted but manual review recommended",
            }

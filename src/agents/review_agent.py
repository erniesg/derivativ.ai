"""
ReviewAgent - Specialized agent for quality assurance and review.

This agent focuses on reviewing generated questions and marking schemes to ensure
they meet Cambridge IGCSE standards, are pedagogically sound, and are appropriate
for the target grade level.
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from ..models import (
    CandidateQuestion, SolutionAndMarkingScheme,
    GenerationConfig, LLMModel, CalculatorPolicy
)
from ..services.prompt_loader import PromptLoader


class ReviewOutcome(Enum):
    """Possible outcomes of a review"""
    APPROVE = "approve"
    MINOR_REVISIONS = "minor_revisions"
    MAJOR_REVISIONS = "major_revisions"
    REJECT = "reject"


class ReviewFeedback:
    """Structured feedback from review process"""

    def __init__(
        self,
        outcome: ReviewOutcome,
        overall_score: float,  # 0.0 to 1.0
        feedback_summary: str,
        specific_feedback: Dict[str, Any],
        suggested_improvements: List[str],
        syllabus_compliance: float,  # 0.0 to 1.0
        difficulty_alignment: float,  # 0.0 to 1.0
        marking_quality: float  # 0.0 to 1.0
    ):
        self.outcome = outcome
        self.overall_score = overall_score
        self.feedback_summary = feedback_summary
        self.specific_feedback = specific_feedback
        self.suggested_improvements = suggested_improvements
        self.syllabus_compliance = syllabus_compliance
        self.difficulty_alignment = difficulty_alignment
        self.marking_quality = marking_quality


class ReviewAgent:
    """
    Specialized agent for quality assurance and review of questions and marking schemes.

    Focuses on:
    - Quality assessment and validation
    - Syllabus compliance checking
    - Grade-appropriate difficulty validation
    - Marking scheme accuracy review
    - Pedagogical soundness evaluation
    """

    def __init__(self, model, db_client=None, debug: bool = False):
        self.model = model
        self.db_client = db_client
        self.debug = debug

        # Initialize prompt loader
        self.prompt_loader = PromptLoader()

        # Load syllabus and marking data
        self.syllabus_data = self._load_syllabus_data()
        self.marking_data = self._load_marking_data()

    def _load_syllabus_data(self) -> Dict[str, Any]:
        """Load syllabus structure from syllabus_command.json"""
        try:
            with open("data/syllabus_command.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            if self.debug:
                print("⚠️ syllabus_command.json not found")
            return {}

    def _load_marking_data(self) -> Dict[str, Any]:
        """Load Cambridge marking principles from markscheme.json"""
        try:
            with open("data/markscheme.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            if self.debug:
                print("⚠️ markscheme.json not found")
            return {}

    async def review_question(
        self,
        question: CandidateQuestion,
        config: GenerationConfig,
        template_version: str = "v1.0"
    ) -> ReviewFeedback:
        """
        Comprehensive review of a generated question and its marking scheme.

        Args:
            question: The candidate question to review
            config: Generation configuration used
            template_version: Review template version to use

        Returns:
            Detailed review feedback with scores and suggestions
        """

        if self.debug:
            print(f"🔍 ReviewAgent reviewing question: {question.question_id_local}")

        # Construct review prompt
        review_prompt = self._construct_review_prompt(
            question, config, template_version
        )

        # Call LLM for review
        if self.debug:
            print(f"[DEBUG] Calling ReviewAgent LLM: {type(self.model).__name__}")

        response = await self._call_model(review_prompt)

        # Parse review response
        feedback = self._parse_review_response(response, question, config)

        if self.debug:
            print(f"✅ ReviewAgent completed review - Outcome: {feedback.outcome.value}")
            print(f"   Overall Score: {feedback.overall_score:.2f}")

        return feedback

    def _construct_review_prompt(
        self,
        question: CandidateQuestion,
        config: GenerationConfig,
        template_version: str
    ) -> str:
        """Construct review prompt using template"""

        # Get relevant syllabus content for the subject references
        syllabus_context = self._get_syllabus_context(config.subject_content_references)

        # Get marking principles for context
        marking_principles = self._get_marking_principles_text()

        # Serialize question data for review
        question_json = json.dumps(question.model_dump(mode='json'), indent=2)

        # Use prompt loader to format review prompt
        formatted_prompt = self.prompt_loader.format_review_prompt(
            template_version=template_version,
            question_json=question_json,
            target_grade=config.target_grade,
            desired_marks=config.desired_marks,
            subject_content_references=', '.join(config.subject_content_references),
            calculator_policy=config.calculator_policy.value,
            syllabus_context=syllabus_context,
            marking_principles=marking_principles
        )

        return formatted_prompt

    def _get_syllabus_context(self, subject_refs: List[str]) -> str:
        """Extract relevant syllabus content for the given references"""
        context_parts = []

        # Look through syllabus data for matching references
        core_content = self.syllabus_data.get("core_subject_content", [])

        for topic in core_content:
            for sub_topic in topic.get("sub_topics", []):
                ref = sub_topic.get("subject_content_ref", "")
                if ref in subject_refs:
                    title = sub_topic.get("title", "")
                    details = sub_topic.get("details", [])

                    if isinstance(details, list):
                        details_text = "; ".join(details[:3])  # First 3 details
                    else:
                        details_text = str(details)

                    context_parts.append(f"**{ref}**: {title} - {details_text}")

        return '\n'.join(context_parts) if context_parts else "No specific syllabus context found"

    def _get_marking_principles_text(self) -> str:
        """Extract key marking principles as text"""
        principles = []
        for principle in self.marking_data.get("generic_marking_principles", [])[:3]:
            details = principle.get('details', '')
            if isinstance(details, str):
                principles.append(f"- {details}")
            elif isinstance(details, list) and details:
                principles.append(f"- {details[0]}")
        return '\n'.join(principles)

    async def _call_model(self, prompt: str) -> str:
        """Call the LLM model with the review prompt"""
        try:
            # Handle different model types
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
                print(f"❌ ReviewAgent model call error: {e}")
            raise

    def _parse_review_response(
        self,
        response: str,
        question: CandidateQuestion,
        config: GenerationConfig
    ) -> ReviewFeedback:
        """Parse LLM review response into ReviewFeedback object"""

        try:
            # Extract JSON from response
            response_clean = self._extract_json_from_response(response)

            if self.debug:
                print(f"[DEBUG] Cleaned review response: {response_clean[:300]}...")

            # Parse JSON
            review_data = json.loads(response_clean)

            # Map outcome string to enum
            outcome_str = review_data.get("outcome", "minor_revisions").lower()
            outcome = ReviewOutcome(outcome_str) if outcome_str in [e.value for e in ReviewOutcome] else ReviewOutcome.MINOR_REVISIONS

            # Create ReviewFeedback object
            feedback = ReviewFeedback(
                outcome=outcome,
                overall_score=float(review_data.get("overall_score", 0.7)),
                feedback_summary=review_data.get("feedback_summary", "Review completed"),
                specific_feedback=review_data.get("specific_feedback", {}),
                suggested_improvements=review_data.get("suggested_improvements", []),
                syllabus_compliance=float(review_data.get("syllabus_compliance", 0.8)),
                difficulty_alignment=float(review_data.get("difficulty_alignment", 0.7)),
                marking_quality=float(review_data.get("marking_quality", 0.8))
            )

            return feedback

        except Exception as e:
            if self.debug:
                print(f"❌ Error parsing review response: {e}")
                print(f"Raw response: {response[:500]}...")

            # Fallback: create basic feedback
            return self._create_fallback_feedback()

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON content from LLM response using robust parser"""
        try:
            from ..utils.json_parser import extract_json_robust

            result = extract_json_robust(response)
            if result.success:
                return result.raw_json
            else:
                # Fallback to original method
                return self._extract_json_from_response_fallback(response)

        except Exception as e:
            if self.debug:
                print(f"❌ Robust JSON extraction failed: {e}")
            # Fallback to original method
            return self._extract_json_from_response_fallback(response)

    def _extract_json_from_response_fallback(self, response: str) -> str:
        """Fallback JSON extraction method (original implementation)"""
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

    def _create_fallback_feedback(self) -> ReviewFeedback:
        """Create fallback feedback if parsing fails"""
        return ReviewFeedback(
            outcome=ReviewOutcome.MINOR_REVISIONS,
            overall_score=0.7,
            feedback_summary="Review completed with basic assessment",
            specific_feedback={"note": "Review parsing failed, using fallback"},
            suggested_improvements=["Review and improve question clarity"],
            syllabus_compliance=0.8,
            difficulty_alignment=0.7,
            marking_quality=0.8
        )

    async def batch_review(
        self,
        questions: List[CandidateQuestion],
        configs: List[GenerationConfig]
    ) -> List[ReviewFeedback]:
        """
        Review multiple questions in batch.

        Args:
            questions: List of questions to review
            configs: Corresponding generation configs

        Returns:
            List of review feedback for each question
        """

        if self.debug:
            print(f"🔍 ReviewAgent batch reviewing {len(questions)} questions")

        feedbacks = []
        for i, (question, config) in enumerate(zip(questions, configs)):
            if self.debug:
                print(f"   Reviewing question {i+1}/{len(questions)}")

            feedback = await self.review_question(question, config)
            feedbacks.append(feedback)

        return feedbacks

    def get_review_summary(self, feedbacks: List[ReviewFeedback]) -> Dict[str, Any]:
        """
        Generate summary statistics from multiple review feedbacks.

        Args:
            feedbacks: List of review feedbacks

        Returns:
            Summary statistics and insights
        """

        if not feedbacks:
            return {"total": 0, "message": "No feedbacks to summarize"}

        # Count outcomes
        outcome_counts = {}
        for outcome in ReviewOutcome:
            outcome_counts[outcome.value] = sum(1 for f in feedbacks if f.outcome == outcome)

        # Calculate averages
        avg_overall = sum(f.overall_score for f in feedbacks) / len(feedbacks)
        avg_syllabus = sum(f.syllabus_compliance for f in feedbacks) / len(feedbacks)
        avg_difficulty = sum(f.difficulty_alignment for f in feedbacks) / len(feedbacks)
        avg_marking = sum(f.marking_quality for f in feedbacks) / len(feedbacks)

        # Collect common improvements
        all_improvements = []
        for f in feedbacks:
            all_improvements.extend(f.suggested_improvements)

        # Count improvement frequency
        improvement_counts = {}
        for improvement in all_improvements:
            improvement_counts[improvement] = improvement_counts.get(improvement, 0) + 1

        common_improvements = sorted(improvement_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_questions": len(feedbacks),
            "outcome_distribution": outcome_counts,
            "average_scores": {
                "overall": round(avg_overall, 3),
                "syllabus_compliance": round(avg_syllabus, 3),
                "difficulty_alignment": round(avg_difficulty, 3),
                "marking_quality": round(avg_marking, 3)
            },
            "top_improvement_suggestions": [imp[0] for imp in common_improvements],
            "quality_grade": self._calculate_quality_grade(avg_overall)
        }

    def _calculate_quality_grade(self, overall_score: float) -> str:
        """Calculate quality grade from overall score"""
        if overall_score >= 0.9:
            return "Excellent"
        elif overall_score >= 0.8:
            return "Good"
        elif overall_score >= 0.7:
            return "Satisfactory"
        elif overall_score >= 0.6:
            return "Needs Improvement"
        else:
            return "Poor"

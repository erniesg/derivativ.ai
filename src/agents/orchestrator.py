"""
Multi-Agent Orchestrator for coordinating agent workflows.
Supports both async operation and smolagents integration.
"""

import asyncio
import logging
from typing import Any, Optional, Union

from src.agents.base_agent import BaseAgent
from src.agents.marker_agent import MarkerAgent
from src.agents.question_generator import QuestionGeneratorAgent
from src.agents.refinement_agent import RefinementAgent
from src.agents.review_agent import ReviewAgent
from src.agents.sync_wrapper import SyncAgentWrapper, make_sync_agent
from src.core.config import load_config
from src.models.enums import QualityAction
from src.models.question_models import GenerationRequest
from src.services.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """
    Orchestrates multiple agents for complete question generation workflow.

    Workflow:
    1. QuestionGenerator creates initial question
    2. Marker creates marking scheme
    3. Reviewer assesses quality
    4. Refiner improves if needed (based on quality score)
    """

    def __init__(
        self,
        llm_factory: Optional[LLMFactory] = None,
        quality_thresholds: Optional[dict[str, float]] = None,
        use_sync: bool = False,
    ):
        """
        Initialize orchestrator with agents.

        Args:
            llm_factory: LLM factory for creating services
            quality_thresholds: Quality control thresholds
            use_sync: Whether to use synchronous wrappers (for smolagents)
        """
        self.llm_factory = llm_factory or LLMFactory()
        self.use_sync = use_sync

        # Load quality thresholds from config
        config = load_config()
        self.quality_thresholds = quality_thresholds or {
            "auto_approve": config.quality_control.get("thresholds", {}).get("auto_approve", 0.85),
            "refine": config.quality_control.get("thresholds", {}).get("refine", 0.60),
            "regenerate": config.quality_control.get("thresholds", {}).get("regenerate", 0.40),
            "reject": config.quality_control.get("thresholds", {}).get("reject", 0.20),
        }

        # Initialize agents (created on demand)
        self._agents = {}

    def _get_agent(
        self, agent_type: str, force_async: bool = False
    ) -> Union[BaseAgent, SyncAgentWrapper]:
        """Get or create an agent instance."""
        cache_key = (
            f"{agent_type}_{'async' if force_async else 'sync' if self.use_sync else 'async'}"
        )

        if cache_key not in self._agents:
            # Create LLM service for agent
            llm_service = self._create_llm_service_for_agent(agent_type)

            # Create agent
            if agent_type == "generator":
                agent = QuestionGeneratorAgent(llm_service=llm_service)
            elif agent_type == "marker":
                agent = MarkerAgent(llm_service=llm_service)
            elif agent_type == "reviewer":
                agent = ReviewAgent(llm_service=llm_service)
            elif agent_type == "refiner":
                agent = RefinementAgent(llm_service=llm_service)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

            # Wrap in sync wrapper if needed (but not if force_async)
            if self.use_sync and not force_async:
                agent = make_sync_agent(type(agent), llm_service=llm_service)

            self._agents[cache_key] = agent

        return self._agents[cache_key]

    def _create_llm_service_for_agent(self, agent_type: str):
        """Create appropriate LLM service for agent based on config."""
        try:
            config = load_config()

            # Get agent-specific config
            agent_configs = {
                "generator": config.agents.get("question_generator", {}),
                "marker": config.agents.get("marker", {}),
                "reviewer": config.agents.get("reviewer", {}),
                "refiner": config.agents.get("refinement", {}),
            }

            agent_config = agent_configs.get(agent_type, {})
            model = agent_config.get("model", "gpt-4o-mini")

            # Detect provider from model name
            try:
                provider = self.llm_factory.detect_provider(model)
                return self.llm_factory.get_service(provider)
            except ValueError:
                # Fallback to openai for unknown models
                return self.llm_factory.get_service("openai")

        except Exception as e:
            # Fallback to mock service for testing
            from src.services.mock_llm_service import MockLLMService

            logger.warning(f"Could not create LLM service for {agent_type}, using mock: {e}")
            return MockLLMService()

    async def generate_question_async(  # noqa: PLR0915
        self, request: GenerationRequest, max_refinement_cycles: int = 2
    ) -> dict[str, Any]:
        """
        Generate a question using multi-agent workflow (async).

        Args:
            request: Generation request parameters
            max_refinement_cycles: Maximum refinement attempts

        Returns:
            Complete question with marking scheme and quality assessment
        """
        workflow_result = {
            "agents_used": [],
            "reasoning_steps": {},
            "final_quality_score": 0.0,
            "refinement_cycles": 0,
        }

        try:
            # Step 1: Generate question
            generator = self._get_agent("generator", force_async=True)
            gen_result = await generator.process(request.model_dump())

            if not gen_result.success:
                raise Exception(f"Question generation failed: {gen_result.error}")

            workflow_result["agents_used"].append("generator")
            workflow_result["reasoning_steps"]["generator"] = gen_result.reasoning_steps
            question_data = gen_result.output

            # Step 2: Create marking scheme
            marker = self._get_agent("marker", force_async=True)
            # Extract the question from the question_data if it's nested
            actual_question = question_data.get("question", question_data)

            # Ensure the question has the expected format for marker
            marker_question = (
                actual_question.copy() if isinstance(actual_question, dict) else actual_question
            )
            if isinstance(marker_question, dict) and "raw_text_content" in marker_question:
                marker_question["question_text"] = marker_question["raw_text_content"]

            mark_result = await marker.process(
                {"question": marker_question, "config": request.model_dump()}
            )

            if mark_result.success:
                workflow_result["agents_used"].append("marker")
                workflow_result["reasoning_steps"]["marker"] = mark_result.reasoning_steps
                question_data["marking_scheme"] = mark_result.output

            # Step 3: Quality review
            reviewer = self._get_agent("reviewer", force_async=True)
            # Prepare review data in expected format
            review_question = (
                actual_question.copy() if isinstance(actual_question, dict) else actual_question
            )
            if isinstance(review_question, dict) and "raw_text_content" in review_question:
                review_question["question_text"] = review_question["raw_text_content"]
                review_question["grade_level"] = review_question.get(
                    "grade_level", request.grade_level
                )

            review_result = await reviewer.process(
                {
                    "question_data": {
                        "question": review_question,
                        "marking_scheme": mark_result.output if mark_result.success else {},
                    }
                }
            )

            if not review_result.success:
                logger.warning(f"Review failed: {review_result.error}")
                workflow_result["final_quality_score"] = 0.5  # Default medium score
            else:
                workflow_result["agents_used"].append("reviewer")
                workflow_result["reasoning_steps"]["reviewer"] = review_result.reasoning_steps
                quality_data = review_result.output
                workflow_result["final_quality_score"] = quality_data.get("quality_score", 0.5)

                # Step 4: Refinement if needed
                if quality_data.get("quality_score", 0) < self.quality_thresholds["auto_approve"]:
                    for cycle in range(max_refinement_cycles):
                        if (
                            quality_data.get("quality_score", 0)
                            >= self.quality_thresholds["auto_approve"]
                        ):
                            break

                        refiner = self._get_agent("refiner", force_async=True)
                        refine_result = await refiner.process(
                            {"original_question": question_data, "review_feedback": quality_data}
                        )

                        if refine_result.success:
                            workflow_result["agents_used"].append(f"refiner_cycle_{cycle + 1}")
                            workflow_result["reasoning_steps"][
                                f"refiner_cycle_{cycle + 1}"
                            ] = refine_result.reasoning_steps
                            workflow_result["refinement_cycles"] += 1

                            # Update question with refinements
                            question_data.update(refine_result.output)

                            # Re-review refined question
                            updated_review_question = review_question.copy()
                            updated_review_question.update(refine_result.output.get("question", {}))

                            review_result = await reviewer.process(
                                {
                                    "question_data": {
                                        "question": updated_review_question,
                                        "marking_scheme": question_data.get("marking_scheme", {}),
                                    }
                                }
                            )

                            if review_result.success:
                                quality_data = review_result.output
                                workflow_result["final_quality_score"] = quality_data.get(
                                    "quality_score", 0.5
                                )

            # Compile final result
            workflow_result["question"] = question_data
            workflow_result["quality_decision"] = self._determine_quality_action(
                workflow_result["final_quality_score"]
            )

            return workflow_result

        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            workflow_result["error"] = str(e)
            return workflow_result

    def generate_question_sync(
        self, request: Union[dict[str, Any], GenerationRequest], max_refinement_cycles: int = 2
    ) -> dict[str, Any]:
        """
        Generate a question using multi-agent workflow (sync).
        For smolagents integration - handles both event loop and non-event loop contexts.
        """
        # Convert dict to GenerationRequest if needed
        if isinstance(request, dict):
            try:
                request = GenerationRequest(**request)
            except Exception as e:
                return {"error": f"Invalid request format: {e}"}
        elif not isinstance(request, GenerationRequest):
            return {"error": f"Invalid request type: {type(request)}"}
        # Ensure we're using sync wrappers
        original_use_sync = self.use_sync
        self.use_sync = True

        try:
            # Check if we're already in an event loop
            try:
                current_loop = asyncio.get_running_loop()
                if current_loop.is_running():
                    # We're in a running event loop, use ThreadPoolExecutor
                    import concurrent.futures

                    def run_in_new_loop():
                        # Create new event loop in thread
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(
                                self.generate_question_async(request, max_refinement_cycles)
                            )
                        finally:
                            new_loop.close()

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_new_loop)
                        return future.result()

            except RuntimeError:
                # No event loop running, we can use asyncio.run()
                return asyncio.run(self.generate_question_async(request, max_refinement_cycles))

        finally:
            self.use_sync = original_use_sync

    def _determine_quality_action(self, quality_score: float) -> str:
        """Determine action based on quality score."""
        if quality_score >= self.quality_thresholds["auto_approve"]:
            return QualityAction.APPROVE.value
        elif quality_score >= self.quality_thresholds["refine"]:
            return QualityAction.REFINE.value
        elif quality_score >= self.quality_thresholds["regenerate"]:
            return QualityAction.REGENERATE.value
        else:
            return QualityAction.REJECT.value

    def get_workflow_summary(self, workflow_result: dict[str, Any]) -> str:
        """Get human-readable summary of workflow execution."""
        summary = "=== Multi-Agent Workflow Summary ===\n\n"

        summary += f"Agents Used: {', '.join(workflow_result.get('agents_used', []))}\n"
        summary += f"Final Quality Score: {workflow_result.get('final_quality_score', 0):.2f}\n"
        summary += f"Quality Decision: {workflow_result.get('quality_decision', 'unknown')}\n"
        summary += f"Refinement Cycles: {workflow_result.get('refinement_cycles', 0)}\n\n"

        # Add reasoning highlights
        for agent, steps in workflow_result.get("reasoning_steps", {}).items():
            summary += f"\n{agent.upper()} Reasoning:\n"
            for step in steps[:3]:  # First 3 steps only
                summary += f"  - {step}\n"
            if len(steps) > 3:
                summary += f"  ... and {len(steps) - 3} more steps\n"

        return summary


# For smolagents integration
class SmolagentsOrchestrator(MultiAgentOrchestrator):
    """
    Synchronous orchestrator specifically for smolagents integration.
    Always uses sync wrappers.
    """

    def __init__(self, *args, **kwargs):
        kwargs["use_sync"] = True
        super().__init__(*args, **kwargs)

    def generate_question(
        self, request: Union[dict[str, Any], GenerationRequest]
    ) -> dict[str, Any]:
        """Synchronous method for smolagents tool interface."""
        from ..models.question_models import GenerationRequest

        # Convert dict to GenerationRequest if needed
        if isinstance(request, dict):
            try:
                request = GenerationRequest(**request)
            except Exception as e:
                return {"error": f"Invalid request format: {e}"}
        elif not isinstance(request, GenerationRequest):
            return {"error": f"Invalid request type: {type(request)}"}

        return self.generate_question_sync(request)

"""
Multi-Agent Orchestrator for coordinating agent workflows.
Supports both async operation and smolagents integration.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from ..core.config import load_config
from ..models.enums import QualityAction
from ..models.question_models import AgentResult, GenerationRequest
from ..services.llm_factory import LLMFactory
from .base_agent import BaseAgent
from .marker_agent import MarkerAgent
from .question_generator import QuestionGeneratorAgent
from .refinement_agent import RefinementAgent
from .review_agent import ReviewAgent
from .sync_wrapper import SyncAgentWrapper, make_sync_agent

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
        quality_thresholds: Optional[Dict[str, float]] = None,
        use_sync: bool = False
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
    
    def _get_agent(self, agent_type: str) -> Union[BaseAgent, SyncAgentWrapper]:
        """Get or create an agent instance."""
        if agent_type not in self._agents:
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
            
            # Wrap in sync wrapper if needed
            if self.use_sync:
                agent = make_sync_agent(type(agent), llm_service=llm_service)
            
            self._agents[agent_type] = agent
        
        return self._agents[agent_type]
    
    def _create_llm_service_for_agent(self, agent_type: str):
        """Create appropriate LLM service for agent based on config."""
        config = load_config()
        
        # Get agent-specific config
        agent_configs = {
            "generator": config.agents.get("question_generator", {}),
            "marker": config.agents.get("marker", {}),
            "reviewer": config.agents.get("reviewer", {}),
            "refiner": config.agents.get("refinement", {})
        }
        
        agent_config = agent_configs.get(agent_type, {})
        model = agent_config.get("model", "gpt-4o-mini")
        
        return self.llm_factory.create_from_config(config, model_override=model)
    
    async def generate_question_async(
        self, 
        request: GenerationRequest,
        max_refinement_cycles: int = 2
    ) -> Dict[str, Any]:
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
            "refinement_cycles": 0
        }
        
        try:
            # Step 1: Generate question
            generator = self._get_agent("generator")
            gen_result = await generator.process(request.model_dump())
            
            if not gen_result.success:
                raise Exception(f"Question generation failed: {gen_result.error}")
            
            workflow_result["agents_used"].append("generator")
            workflow_result["reasoning_steps"]["generator"] = gen_result.reasoning_steps
            question_data = gen_result.output
            
            # Step 2: Create marking scheme
            marker = self._get_agent("marker")
            mark_result = await marker.process({
                "question_text": question_data.get("question_text"),
                "marks": question_data.get("marks", request.marks),
                "solution_steps": question_data.get("solution_steps", [])
            })
            
            if mark_result.success:
                workflow_result["agents_used"].append("marker")
                workflow_result["reasoning_steps"]["marker"] = mark_result.reasoning_steps
                question_data["marking_scheme"] = mark_result.output
            
            # Step 3: Quality review
            reviewer = self._get_agent("reviewer")
            review_result = await reviewer.process({
                "question": question_data,
                "marking_scheme": question_data.get("marking_scheme", {})
            })
            
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
                        if quality_data.get("quality_score", 0) >= self.quality_thresholds["auto_approve"]:
                            break
                        
                        refiner = self._get_agent("refiner")
                        refine_result = await refiner.process({
                            "original_question": question_data,
                            "review_feedback": quality_data
                        })
                        
                        if refine_result.success:
                            workflow_result["agents_used"].append(f"refiner_cycle_{cycle + 1}")
                            workflow_result["reasoning_steps"][f"refiner_cycle_{cycle + 1}"] = refine_result.reasoning_steps
                            workflow_result["refinement_cycles"] += 1
                            
                            # Update question with refinements
                            question_data.update(refine_result.output)
                            
                            # Re-review refined question
                            review_result = await reviewer.process({
                                "question": question_data,
                                "marking_scheme": question_data.get("marking_scheme", {})
                            })
                            
                            if review_result.success:
                                quality_data = review_result.output
                                workflow_result["final_quality_score"] = quality_data.get("quality_score", 0.5)
            
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
        self,
        request: GenerationRequest,
        max_refinement_cycles: int = 2
    ) -> Dict[str, Any]:
        """
        Generate a question using multi-agent workflow (sync).
        For smolagents integration.
        """
        # Ensure we're using sync wrappers
        original_use_sync = self.use_sync
        self.use_sync = True
        
        try:
            # Run async method in sync context
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
    
    def get_workflow_summary(self, workflow_result: Dict[str, Any]) -> str:
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
    
    def generate_question(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous method for smolagents tool interface."""
        # Convert dict to GenerationRequest
        if isinstance(request, dict):
            from ..models.question_models import GenerationRequest
            request = GenerationRequest(**request)
        
        return self.generate_question_sync(request)
"""
Smolagents integration for Derivativ AI.
Provides both our custom agents as smolagents tools AND creates smolagents-native agents.
"""

import json
import os
from typing import Any, Dict, List, Optional

from smolagents import CodeAgent, InferenceClientModel, tool

from ..models.enums import CalculatorPolicy, CommandWord, Tier
from ..models.question_models import GenerationRequest
from ..services.llm_factory import LLMFactory
from ..services.mock_llm_service import MockLLMService


# Create our custom agents as smolagents tools
@tool
def generate_math_question(
    topic: str,
    grade_level: int = 8,
    marks: int = 3,
    calculator_policy: str = "not_allowed",
    command_word: str = "Calculate",
) -> str:
    """
    Generate a Cambridge IGCSE Mathematics question using our custom agent.
    
    Args:
        topic: The mathematics topic for the question
        grade_level: Target grade level (1-10)
        marks: Number of marks for the question
        calculator_policy: Whether calculator is allowed ('allowed' or 'not_allowed')
        command_word: Cambridge command word to use
        
    Returns:
        JSON string containing the generated question with marking scheme
    """
    try:
        from ..agents.question_generator import QuestionGeneratorAgent
        
        # Create agent with mock LLM (or real LLM if API keys available)
        llm_service = _get_llm_service()
        agent = QuestionGeneratorAgent(llm_service=llm_service)
        
        # Prepare request
        request_data = {
            "topic": topic,
            "grade_level": grade_level,
            "marks": marks,
            "calculator_policy": calculator_policy,
            "command_word": command_word,
            "tier": "Core",  # Default tier
        }
        
        # Process synchronously (smolagents expects sync)
        import asyncio
        result = asyncio.run(agent.process(request_data))
        
        if result.success:
            return json.dumps(result.output, indent=2)
        else:
            return f"Error generating question: {result.error}"
            
    except Exception as e:
        return f"Error in question generation: {e}"


@tool
def review_question_quality(question_data: str) -> str:
    """
    Review the quality of a mathematics question using our custom review agent.
    
    Args:
        question_data: JSON string containing the question to review
        
    Returns:
        JSON string containing quality assessment and feedback
    """
    try:
        from ..agents.review_agent import ReviewAgent
        
        # Parse input
        question = json.loads(question_data)
        
        # Create agent
        llm_service = _get_llm_service()
        agent = ReviewAgent(llm_service=llm_service)
        
        # Process synchronously
        import asyncio
        result = asyncio.run(agent.process({"question": question}))
        
        if result.success:
            return json.dumps(result.output, indent=2)
        else:
            return f"Error reviewing question: {result.error}"
            
    except Exception as e:
        return f"Error in question review: {e}"


@tool
def refine_question(original_question: str, feedback: str) -> str:
    """
    Refine a mathematics question based on review feedback.
    
    Args:
        original_question: JSON string of the original question
        feedback: JSON string of review feedback
        
    Returns:
        JSON string containing the refined question
    """
    try:
        from ..agents.refinement_agent import RefinementAgent
        
        # Parse inputs
        question = json.loads(original_question)
        review_feedback = json.loads(feedback)
        
        # Create agent
        llm_service = _get_llm_service()
        agent = RefinementAgent(llm_service=llm_service)
        
        # Process synchronously
        import asyncio
        result = asyncio.run(agent.process({
            "original_question": question,
            "review_feedback": review_feedback
        }))
        
        if result.success:
            return json.dumps(result.output, indent=2)
        else:
            return f"Error refining question: {result.error}"
            
    except Exception as e:
        return f"Error in question refinement: {e}"


def _get_llm_service():
    """Get appropriate LLM service based on available API keys."""
    # Check if any API keys are available
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    }
    
    # If we have API keys, use real LLM service
    for key_name, key_value in api_keys.items():
        if key_value:
            try:
                factory = LLMFactory()
                config = {"llm_providers": {"default_provider": key_name.lower().replace("_api_key", "")}}
                return factory.create_from_config(config)
            except Exception:
                pass
    
    # Fall back to mock service
    return MockLLMService()


class DerivativSmolagents:
    """
    Main interface for using Derivativ AI with smolagents.
    Provides both native smolagents agents and our custom tools.
    """
    
    def __init__(self, model_id: Optional[str] = None):
        """
        Initialize Derivativ smolagents integration.
        
        Args:
            model_id: Model to use (defaults to best available)
        """
        self.model_id = model_id or self._get_best_model()
        self.model = self._create_model()
        
    def _get_best_model(self) -> str:
        """Determine best available model based on API keys."""
        if os.getenv("OPENAI_API_KEY"):
            return "gpt-4o-mini"
        elif os.getenv("ANTHROPIC_API_KEY"):
            return "claude-3-5-haiku-20241022"
        elif os.getenv("GOOGLE_API_KEY"):
            return "gemini-2.0-flash-exp"
        else:
            return "gpt-4o-mini"  # Will use mock if no API key
    
    def _create_model(self):
        """Create appropriate model instance."""
        try:
            return InferenceClientModel(model_id=self.model_id)
        except Exception:
            # If InferenceClientModel fails, we'll handle it in the tools
            return None
    
    def create_question_generator_agent(self) -> CodeAgent:
        """
        Create a smolagents CodeAgent specialized for question generation.
        """
        return CodeAgent(
            tools=[generate_math_question],
            model=self.model,
            name="question_generator",
            description="Generates Cambridge IGCSE Mathematics questions with proper marking schemes"
        )
    
    def create_quality_control_agent(self) -> CodeAgent:
        """
        Create a smolagents CodeAgent for quality control workflows.
        """
        return CodeAgent(
            tools=[generate_math_question, review_question_quality, refine_question],
            model=self.model,
            name="quality_controller",
            description="Generates, reviews, and refines mathematics questions for optimal quality"
        )
    
    def create_multi_agent_system(self) -> CodeAgent:
        """
        Create a comprehensive multi-agent system using smolagents.
        """
        # Add base tools for web search, etc.
        agent = CodeAgent(
            tools=[generate_math_question, review_question_quality, refine_question],
            model=self.model,
            add_base_tools=True,  # Adds web search and other utilities
            name="derivativ_ai",
            description="Complete AI-powered mathematics education platform"
        )
        
        return agent


# Convenience function for quick setup
def create_derivativ_agent(
    model_id: Optional[str] = None,
    agent_type: str = "multi_agent"
) -> CodeAgent:
    """
    Quick setup function for creating Derivativ AI smolagents.
    
    Args:
        model_id: Model to use (auto-detected if None)
        agent_type: Type of agent ("question_generator", "quality_control", or "multi_agent")
        
    Returns:
        Configured CodeAgent instance
    """
    derivativ = DerivativSmolagents(model_id=model_id)
    
    if agent_type == "question_generator":
        return derivativ.create_question_generator_agent()
    elif agent_type == "quality_control":
        return derivativ.create_quality_control_agent()
    elif agent_type == "multi_agent":
        return derivativ.create_multi_agent_system()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# Example usage functions
def demo_smolagents_question_generation():
    """Demo: Generate a question using smolagents."""
    print("\n🤖 SMOLAGENTS DEMO: Question Generation")
    print("=" * 50)
    
    agent = create_derivativ_agent("question_generator")
    
    result = agent.run(
        "Generate a mathematics question about algebra for grade 8 students, "
        "worth 3 marks, without calculator allowed. Use the command word 'Calculate'."
    )
    
    print("Result:", result)
    return result


def demo_smolagents_quality_workflow():
    """Demo: Complete quality workflow using smolagents."""
    print("\n🎯 SMOLAGENTS DEMO: Quality Control Workflow")
    print("=" * 50)
    
    agent = create_derivativ_agent("quality_control")
    
    result = agent.run(
        "Generate a geometry question for grade 9, review its quality, "
        "and refine it if the quality score is below 0.8. "
        "The question should be worth 5 marks and allow calculator use."
    )
    
    print("Result:", result)
    return result


if __name__ == "__main__":
    # Run demos
    try:
        demo_smolagents_question_generation()
        demo_smolagents_quality_workflow()
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()
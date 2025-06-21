#!/usr/bin/env python3
"""
Demo script showing multi-agent workflow in action.
Demonstrates both async and sync (smolagents-compatible) usage.
"""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.orchestrator import MultiAgentOrchestrator, SmolagentsOrchestrator
from src.agents.sync_wrapper import create_sync_question_generator
from src.services.mock_llm_service import MockLLMService
from src.services.llm_factory import LLMFactory
from src.models.enums import CalculatorPolicy, CommandWord, Tier
from src.models.question_models import GenerationRequest


def demo_standalone_sync_agent():
    """Demo: Using a single agent synchronously (smolagents-compatible)."""
    print("\n" + "="*60)
    print("DEMO 1: Standalone Synchronous Agent")
    print("="*60)
    
    # Create sync wrapper for question generator with mock LLM
    mock_llm = MockLLMService()
    generator = create_sync_question_generator(llm_service=mock_llm)
    
    # Prepare request
    request = {
        "topic": "algebra",
        "tier": "Core",
        "grade_level": 8,
        "marks": 3,
        "calculator_policy": "not_allowed"
    }
    
    print(f"\nGenerating question about: {request['topic']}")
    print("Using synchronous wrapper (smolagents-compatible)...")
    
    # Process synchronously
    result = generator.process(request)
    
    if result.success:
        print("\n✅ Question generated successfully!")
        print(f"Question: {result.output.get('question_text', 'N/A')}")
        print(f"Marks: {result.output.get('marks', 'N/A')}")
        print("\nAgent reasoning steps:")
        for step in result.reasoning_steps[:5]:
            print(f"  - {step}")
    else:
        print(f"\n❌ Generation failed: {result.error}")


async def demo_standalone_async_agent():
    """Demo: Using a single agent asynchronously."""
    print("\n" + "="*60)
    print("DEMO 2: Standalone Asynchronous Agent")
    print("="*60)
    
    from src.agents.question_generator import QuestionGeneratorAgent
    
    # Create async agent with mock LLM
    mock_llm = MockLLMService()
    generator = QuestionGeneratorAgent(llm_service=mock_llm)
    
    # Prepare request
    request = {
        "topic": "geometry",
        "tier": "Extended",
        "grade_level": 9,
        "marks": 5,
        "calculator_policy": "allowed"
    }
    
    print(f"\nGenerating question about: {request['topic']}")
    print("Using native async interface...")
    
    # Process asynchronously
    result = await generator.process(request)
    
    if result.success:
        print("\n✅ Question generated successfully!")
        print(f"Question: {result.output.get('question_text', 'N/A')}")
        print(f"Marks: {result.output.get('marks', 'N/A')}")
    else:
        print(f"\n❌ Generation failed: {result.error}")


async def demo_multi_agent_workflow():
    """Demo: Full multi-agent orchestration."""
    print("\n" + "="*60)
    print("DEMO 3: Multi-Agent Orchestration (Async)")
    print("="*60)
    
    # Create orchestrator with mock LLM factory
    from src.services.llm_factory import LLMFactory
    mock_factory = LLMFactory()
    # Override factory to return mock services
    mock_factory._create_service = lambda provider, config: MockLLMService(provider)
    orchestrator = MultiAgentOrchestrator(llm_factory=mock_factory)
    
    # Prepare request
    request = GenerationRequest(
        topic="quadratic equations",
        tier=Tier.EXTENDED,
        grade_level=9,
        marks=6,
        calculator_policy=CalculatorPolicy.ALLOWED,
        command_word=CommandWord.SOLVE
    )
    
    print(f"\nOrchestrating multi-agent workflow for: {request.topic}")
    print("Agents will coordinate: Generator → Marker → Reviewer → Refiner (if needed)")
    
    # Run orchestration
    result = await orchestrator.generate_question_async(request)
    
    # Display results
    print("\n" + orchestrator.get_workflow_summary(result))
    
    if "question" in result:
        print("\nFinal Question:")
        print(f"Text: {result['question'].get('question_text', 'N/A')}")
        print(f"Quality Score: {result.get('final_quality_score', 0):.2f}")
        print(f"Decision: {result.get('quality_decision', 'unknown')}")


def demo_smolagents_orchestration():
    """Demo: Synchronous orchestration for smolagents."""
    print("\n" + "="*60)
    print("DEMO 4: Smolagents-Compatible Orchestration (Sync)")
    print("="*60)
    
    # Create smolagents orchestrator with mock LLM
    mock_factory = LLMFactory()
    mock_factory._create_service = lambda provider, config: MockLLMService(provider)
    orchestrator = SmolagentsOrchestrator(llm_factory=mock_factory)
    
    # Prepare request (as dict, like smolagents would)
    request = {
        "topic": "probability",
        "tier": "Core",
        "grade_level": 7,
        "marks": 4,
        "calculator_policy": "not_allowed"
    }
    
    print(f"\nRunning synchronous orchestration for: {request['topic']}")
    print("This demonstrates smolagents compatibility...")
    
    # Run synchronously
    result = orchestrator.generate_question(request)
    
    if "error" not in result:
        print("\n✅ Workflow completed successfully!")
        print(f"Agents used: {', '.join(result.get('agents_used', []))}")
        print(f"Final quality: {result.get('final_quality_score', 0):.2f}")
    else:
        print(f"\n❌ Workflow failed: {result.get('error', 'Unknown error')}")


def demo_smolagents_native():
    """Demo: Using smolagents natively with our tools."""
    print("\n" + "="*60)
    print("DEMO 5: Native Smolagents Integration")
    print("="*60)
    
    try:
        from src.agents.smolagents_integration import create_derivativ_agent
        
        print("\nCreating smolagents CodeAgent with our custom tools...")
        
        # Create a smolagents agent with our tools
        agent = create_derivativ_agent("question_generator")
        
        print("Running smolagents agent to generate a math question...")
        
        # Let smolagents handle the entire workflow
        result = agent.run(
            "Generate a mathematics question about quadratic equations for grade 10 students, "
            "worth 4 marks, calculator allowed. Make sure it follows Cambridge IGCSE standards."
        )
        
        print("\n✅ Smolagents completed the task!")
        print("Result:")
        print(result)
        
    except Exception as e:
        print(f"\n❌ Smolagents demo failed: {e}")
        import traceback
        traceback.print_exc()


def demo_smolagents_quality_workflow():
    """Demo: Complete quality workflow using smolagents."""
    print("\n" + "="*60)
    print("DEMO 6: Smolagents Quality Control Workflow")
    print("="*60)
    
    try:
        from src.agents.smolagents_integration import create_derivativ_agent
        
        print("\nCreating smolagents agent with quality control tools...")
        
        # Create agent with all our tools
        agent = create_derivativ_agent("quality_control")
        
        print("Running complete generate → review → refine workflow...")
        
        # Complex workflow that smolagents will orchestrate
        result = agent.run(
            "Generate a probability question for grade 8, then review its quality. "
            "If the quality score is below 0.8, refine the question to improve it. "
            "The question should be worth 3 marks with no calculator allowed."
        )
        
        print("\n✅ Smolagents quality workflow completed!")
        print("Result:")
        print(result)
        
    except Exception as e:
        print(f"\n❌ Smolagents quality workflow failed: {e}")
        import traceback
        traceback.print_exc()


def check_api_keys():
    """Check if API keys are configured."""
    providers = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "Google": "GOOGLE_API_KEY"
    }
    
    print("\n🔑 API Key Status:")
    configured = False
    for provider, env_var in providers.items():
        if os.getenv(env_var):
            print(f"  ✅ {provider}: Configured")
            configured = True
        else:
            print(f"  ❌ {provider}: Not configured")
    
    if not configured:
        print("\n⚠️  No API keys configured! Agents will use mock LLM service.")
        print("   Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY to use real LLMs.")
    
    return configured


async def main():
    """Run all demos."""
    print("\n🚀 DERIVATIV AI - MULTI-AGENT SYSTEM DEMO")
    print("==========================================")
    
    # Check API keys
    has_keys = check_api_keys()
    
    if not has_keys:
        print("\n⚠️  Running with mock LLM service...")
        print("   Agents will return example responses.")
    
    # Run demos
    try:
        # Demo 1: Sync single agent
        demo_standalone_sync_agent()
        
        # Demo 2: Async single agent
        await demo_standalone_async_agent()
        
        # Demo 3: Multi-agent async workflow
        await demo_multi_agent_workflow()
        
        # Demo 4: Smolagents-compatible sync workflow
        demo_smolagents_orchestration()
        
        # Demo 5: Native smolagents integration
        demo_smolagents_native()
        
        # Demo 6: Smolagents quality workflow
        demo_smolagents_quality_workflow()
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✨ Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Demo script showing multi-agent workflow in action.
Demonstrates both async and sync (smolagents-compatible) usage.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

from dotenv import load_dotenv

from src.agents.orchestrator import MultiAgentOrchestrator, SmolagentsOrchestrator
from src.agents.sync_wrapper import create_sync_question_generator
from src.models.document_models import DetailLevel, DocumentGenerationRequest, DocumentType
from src.models.enums import CalculatorPolicy, CommandWord, Tier
from src.models.question_models import GenerationRequest
from src.services.document_generation_service import DocumentGenerationService
from src.services.llm_factory import LLMFactory
from src.services.mock_llm_service import MockLLMService
from src.services.prompt_manager import PromptManager

# Load environment variables
load_dotenv()


def demo_standalone_sync_agent():
    """Demo: Using a single agent synchronously (smolagents-compatible)."""
    print("\n" + "=" * 60)
    print("DEMO 1: Standalone Synchronous Agent")
    print("=" * 60)

    # Create sync wrapper for question generator with mock LLM
    mock_llm = MockLLMService()
    generator = create_sync_question_generator(llm_service=mock_llm)

    # Prepare request
    request = {
        "topic": "algebra",
        "tier": "Core",
        "grade_level": 8,
        "marks": 3,
        "calculator_policy": "not_allowed",
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
    print("\n" + "=" * 60)
    print("DEMO 2: Standalone Asynchronous Agent")
    print("=" * 60)

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
        "calculator_policy": "allowed",
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
    print("\n" + "=" * 60)
    print("DEMO 3: Multi-Agent Orchestration (Async)")
    print("=" * 60)

    # Create orchestrator with mock LLM factory
    from src.services.llm_factory import LLMFactory

    mock_factory = LLMFactory()
    # Override factory to return mock services
    mock_factory._create_service = lambda provider: MockLLMService()
    orchestrator = MultiAgentOrchestrator(llm_factory=mock_factory)

    # Prepare request
    request = GenerationRequest(
        topic="quadratic equations",
        tier=Tier.EXTENDED,
        grade_level=9,
        marks=6,
        calculator_policy=CalculatorPolicy.ALLOWED,
        command_word=CommandWord.SOLVE,
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
    print("\n" + "=" * 60)
    print("DEMO 4: Smolagents-Compatible Orchestration (Sync)")
    print("=" * 60)

    # Create smolagents orchestrator with mock LLM
    mock_factory = LLMFactory()
    mock_factory._create_service = lambda provider: MockLLMService()
    orchestrator = SmolagentsOrchestrator(llm_factory=mock_factory)

    # Prepare request (as dict, like smolagents would)
    request = {
        "topic": "probability",
        "tier": "Core",
        "grade_level": 7,
        "marks": 4,
        "calculator_policy": "not_allowed",
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
    print("\n" + "=" * 60)
    print("DEMO 5: Native Smolagents Integration")
    print("=" * 60)

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
    print("\n" + "=" * 60)
    print("DEMO 6: Smolagents Quality Control Workflow")
    print("=" * 60)

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
        "Google": "GOOGLE_API_KEY",
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


async def demo_document_generation():  # noqa: PLR0915
    """Demo: Document generation workflow with mock data."""
    print("\n" + "=" * 60)
    print("DEMO 7: Document Generation Service")
    print("=" * 60)

    try:
        # Create mock question repository
        mock_repo = MagicMock()
        mock_repo.get_question = AsyncMock(return_value=None)
        mock_repo.list_questions = AsyncMock(return_value=[])

        # Create real LLM factory and prompt manager
        llm_factory = LLMFactory()
        prompt_manager = PromptManager()

        # Create document generation service
        doc_service = DocumentGenerationService(
            question_repository=mock_repo, llm_factory=llm_factory, prompt_manager=prompt_manager
        )

        print("\n📋 Creating document generation request...")

        # Create document generation request
        request = DocumentGenerationRequest(
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            title="Linear Equations Practice",
            topic="linear_equations",
            tier=Tier.CORE,
            grade_level=7,
            auto_include_questions=False,
            max_questions=3,
            custom_instructions="Focus on basic substitution and simple solving techniques",
        )

        print(f"📝 Document type: {request.document_type.value}")
        print(f"🎯 Detail level: {request.detail_level.value}")
        print(f"📚 Topic: {request.topic}")
        print(f"✏️  Custom instructions: {request.custom_instructions}")

        # Test template retrieval
        print("\n🔧 Testing template retrieval...")
        templates = await doc_service.get_document_templates()
        print(f"✅ Found {len(templates)} document templates:")
        for template_name, template_data in templates.items():
            print(f"  • {template_name}: {template_data.supported_detail_levels}")

        # Test structure patterns
        print("\n🏗️  Testing structure patterns...")
        patterns = await doc_service.get_structure_patterns()
        worksheet_patterns = patterns.get("worksheet", {})
        print(f"✅ Worksheet structure patterns: {list(worksheet_patterns.keys())}")

        print("\n🔄 Generating document...")
        if os.getenv("OPENAI_API_KEY"):
            print("Using real OpenAI API...")
            result = await doc_service.generate_document(request)
        else:
            print("⚠️  No OpenAI API key - skipping actual generation")
            return

        if result.success:
            print("✅ Document generation successful!")
            print(f"⏱️  Processing time: {result.processing_time:.2f}s")
            print(f"📊 Questions processed: {result.questions_processed}")
            print(f"📑 Sections generated: {result.sections_generated}")
            print(f"🎨 Customizations applied: {result.customizations_applied}")

            document = result.document
            if document:
                print("\n📄 Generated Document:")
                print(f"  Title: {document.title}")
                print(f"  Type: {document.document_type.value}")
                print(f"  Detail level: {document.detail_level.value}")
                print(f"  Estimated duration: {document.estimated_duration} minutes")
                print(f"  Total questions: {document.total_questions}")

                print(f"\n📋 Sections ({len(document.sections)}):")
                for i, section in enumerate(document.sections, 1):
                    print(f"  {i}. {section.title} ({section.content_type})")
        else:
            print(f"❌ Document generation failed: {result.error_message}")

    except Exception as e:
        print(f"❌ Document generation demo failed: {e}")
        import traceback

        traceback.print_exc()


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

        # Demo 7: Document generation
        await demo_document_generation()

    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback

        traceback.print_exc()

    print("\n✨ Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())

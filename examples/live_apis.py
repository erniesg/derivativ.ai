#!/usr/bin/env python3
"""
Live LLM Demo - Using Real API Calls
Shows agents making actual calls to OpenAI, Anthropic, and Google
"""

import asyncio
import os

from dotenv import load_dotenv

from src.agents.question_generator import QuestionGeneratorAgent
from src.models.enums import CalculatorPolicy, CommandWord, LLMModel, Tier
from src.services.llm_factory import LLMFactory

# Load environment variables
load_dotenv()


async def demo_live_openai():
    """Demo with live OpenAI API calls."""
    print("\n" + "=" * 60)
    print("LIVE OpenAI Demo")
    print("=" * 60)

    try:
        # Create real OpenAI LLM service
        factory = LLMFactory()
        openai_service = factory.get_service("openai")
        print(f"✅ Created OpenAI service: {openai_service}")

        # Create agent with live LLM service
        agent = QuestionGeneratorAgent(name="LiveQuestionGenerator", llm_service=openai_service)

        # Test generation request
        request_data = {
            "topic": "quadratic equations",
            "tier": Tier.CORE,
            "grade_level": 8,
            "marks": 4,
            "calculator_policy": CalculatorPolicy.ALLOWED,
            "command_word": CommandWord.SOLVE,
            "llm_model": LLMModel.GPT_4O_MINI,  # Use cheap model for testing
        }

        print(f"📝 Generating question about: {request_data['topic']}")
        print(f"🤖 Using model: {request_data['llm_model'].value}")

        # Make live API call
        result = await agent.process(request_data)

        if result.success:
            print("✅ Live generation successful!")
            question_data = result.output.get("question", {})
            print(f"📋 Question: {question_data.get('raw_text_content', 'N/A')[:100]}...")
            print(f"🎯 Marks: {question_data.get('marks', 'N/A')}")
            print(
                f"⚡ Model used: {result.output.get('generation_metadata', {}).get('model_used', 'N/A')}"
            )

            # Show agent reasoning
            print(f"\n🧠 Agent Reasoning ({len(result.reasoning_steps)} steps):")
            for i, step in enumerate(result.reasoning_steps[-5:], 1):  # Show last 5 steps
                if isinstance(step, dict):
                    print(
                        f"  {i}. {step.get('type', 'unknown')}: {step.get('content', '')[:80]}..."
                    )
                else:
                    print(f"  {i}. {str(step)[:80]}...")
        else:
            print(f"❌ Generation failed: {result.error}")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


async def demo_live_anthropic():
    """Demo with live Anthropic API calls."""
    print("\n" + "=" * 60)
    print("LIVE Anthropic Demo")
    print("=" * 60)

    try:
        # Create real Anthropic LLM service
        factory = LLMFactory()
        anthropic_service = factory.get_service("anthropic")
        print(f"✅ Created Anthropic service: {anthropic_service}")

        # Create agent with live LLM service
        agent = QuestionGeneratorAgent(name="LiveAnthropicGenerator", llm_service=anthropic_service)

        # Test generation request
        request_data = {
            "topic": "trigonometry",
            "tier": Tier.EXTENDED,
            "grade_level": 9,
            "marks": 6,
            "calculator_policy": CalculatorPolicy.ALLOWED,
            "command_word": CommandWord.CALCULATE,
            "llm_model": LLMModel.CLAUDE_3_5_HAIKU,  # Use cheaper Haiku model
        }

        print(f"📝 Generating question about: {request_data['topic']}")
        print(f"🤖 Using model: {request_data['llm_model'].value}")

        # Make live API call
        result = await agent.process(request_data)

        if result.success:
            print("✅ Live generation successful!")
            question_data = result.output.get("question", {})
            print(f"📋 Question: {question_data.get('raw_text_content', 'N/A')[:100]}...")
            print(f"🎯 Marks: {question_data.get('marks', 'N/A')}")
            print(
                f"⚡ Model used: {result.output.get('generation_metadata', {}).get('model_used', 'N/A')}"
            )
        else:
            print(f"❌ Generation failed: {result.error}")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


async def demo_live_gemini():
    """Demo with live Google Gemini API calls."""
    print("\n" + "=" * 60)
    print("LIVE Google Gemini Demo")
    print("=" * 60)

    try:
        # Create real Gemini LLM service
        factory = LLMFactory()
        gemini_service = factory.get_service("google")
        print(f"✅ Created Gemini service: {gemini_service}")

        # Create agent with live LLM service
        agent = QuestionGeneratorAgent(name="LiveGeminiGenerator", llm_service=gemini_service)

        # Test generation request
        request_data = {
            "topic": "probability",
            "tier": Tier.CORE,
            "grade_level": 7,
            "marks": 3,
            "calculator_policy": CalculatorPolicy.NOT_ALLOWED,
            "command_word": CommandWord.FIND,
            "llm_model": LLMModel.GEMINI_1_5_FLASH,  # Use Flash model
        }

        print(f"📝 Generating question about: {request_data['topic']}")
        print(f"🤖 Using model: {request_data['llm_model'].value}")

        # Make live API call
        result = await agent.process(request_data)

        if result.success:
            print("✅ Live generation successful!")
            question_data = result.output.get("question", {})
            print(f"📋 Question: {question_data.get('raw_text_content', 'N/A')[:100]}...")
            print(f"🎯 Marks: {question_data.get('marks', 'N/A')}")
            print(
                f"⚡ Model used: {result.output.get('generation_metadata', {}).get('model_used', 'N/A')}"
            )
        else:
            print(f"❌ Generation failed: {result.error}")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


def check_api_keys():
    """Check which API keys are available."""
    print("🔑 API Key Status:")

    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    print(f"  ✅ OpenAI: {'Configured' if openai_key else 'Missing'}")
    print(f"  ✅ Anthropic: {'Configured' if anthropic_key else 'Missing'}")
    print(f"  ✅ Google: {'Configured' if google_key else 'Missing'}")

    return {
        "openai": bool(openai_key),
        "anthropic": bool(anthropic_key),
        "google": bool(google_key),
    }


async def main():
    """Run live LLM demos."""
    print("🚀 DERIVATIV AI - LIVE LLM CALLS DEMO")
    print("=====================================")

    # Check API keys
    api_status = check_api_keys()

    # Only run demos for configured APIs
    if api_status["openai"]:
        await demo_live_openai()
    else:
        print("\n⚠️  Skipping OpenAI demo - no API key configured")

    if api_status["anthropic"]:
        await demo_live_anthropic()
    else:
        print("\n⚠️  Skipping Anthropic demo - no API key configured")

    if api_status["google"]:
        await demo_live_gemini()
    else:
        print("\n⚠️  Skipping Google demo - no API key configured")

    if not any(api_status.values()):
        print("\n❌ No API keys configured! Please set environment variables:")
        print("   - OPENAI_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        print("   - GOOGLE_API_KEY")

    print("\n✨ Live demo complete!")


if __name__ == "__main__":
    asyncio.run(main())

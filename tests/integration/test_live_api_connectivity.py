"""
Integration tests for live API connectivity with all LLM providers.
Tests the cheapest models with simple requests to validate API connectivity.
"""

import asyncio
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.core.config import load_config, validate_api_keys
from src.models.llm_models import LLMRequest

# Load environment variables from .env file
load_dotenv()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openai_api_connectivity():
    """Test OpenAI API with gpt-4o-mini (cheapest available)."""
    print("🧪 Testing OpenAI API...")

    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY must be set for integration tests"

    try:
        import openai

        client = openai.AsyncOpenAI(api_key=api_key)

        response = await client.chat.completions.create(
            model="gpt-4o-mini",  # Cheapest model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello from OpenAI!' in exactly 3 words."},
            ],
            max_tokens=10,
            temperature=0.1,
        )

        content = response.choices[0].message.content.strip()
        cost_estimate = response.usage.total_tokens * 0.00000015  # Rough estimate for gpt-4o-mini

        print("✅ OpenAI API working!")
        print(f"   Model: {response.model}")
        print(f"   Response: {content}")
        print(f"   Tokens: {response.usage.total_tokens}")
        print(f"   Cost: ~${cost_estimate:.6f}")
        return True

    except Exception as e:
        print(f"❌ OpenAI API failed: {e}")
        return False


async def test_anthropic_api():
    """Test Anthropic API with claude-3-5-haiku (cheapest)."""
    print("\n🧪 Testing Anthropic API...")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set")
        return False

    try:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=api_key)

        response = await client.messages.create(
            model="claude-3-5-haiku-20241022",  # Cheapest model
            max_tokens=10,
            temperature=0.1,
            system="You are a helpful assistant.",
            messages=[
                {"role": "user", "content": "Say 'Hello from Anthropic!' in exactly 3 words."}
            ],
        )

        content = response.content[0].text.strip()
        cost_estimate = (
            response.usage.input_tokens * 0.0000008 + response.usage.output_tokens * 0.000004
        )  # Haiku pricing

        print("✅ Anthropic API working!")
        print(f"   Model: {response.model}")
        print(f"   Response: {content}")
        print(f"   Tokens: {response.usage.input_tokens + response.usage.output_tokens}")
        print(f"   Cost: ~${cost_estimate:.6f}")
        return True

    except Exception as e:
        print(f"❌ Anthropic API failed: {e}")
        return False


async def test_google_api():
    """Test Google Gemini API with flash-lite (free tier)."""
    print("\n🧪 Testing Google Gemini API...")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not set")
        return False

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-1.5-flash")  # Free tier model

        response = await model.generate_content_async(
            "Say 'Hello from Google!' in exactly 3 words.",
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=10,
                temperature=0.1,
            ),
        )

        content = response.text.strip()

        print("✅ Google Gemini API working!")
        print("   Model: gemini-1.5-flash")
        print(f"   Response: {content}")
        print("   Cost: Free tier")
        return True

    except Exception as e:
        print(f"❌ Google Gemini API failed: {e}")
        return False


async def test_config_system():
    """Test our configuration system."""
    print("\n🧪 Testing Configuration System...")

    try:
        config = load_config()

        print("✅ Configuration loaded successfully!")
        print(f"   Default provider: {config.llm_providers.default_provider}")
        print(f"   OpenAI model: {config.llm_providers.openai.default_model}")
        print(f"   Anthropic model: {config.llm_providers.anthropic.default_model}")
        print(f"   Google model: {config.llm_providers.google.default_model}")
        print(f"   Default temp: {config.llm_defaults.temperature}")

        # Test API key validation
        api_status = validate_api_keys()
        print("\n   API Key Status:")
        for provider, available in api_status.items():
            status = "✅" if available else "❌"
            print(f"     {provider}: {status}")

        return True

    except Exception as e:
        print(f"❌ Configuration system failed: {e}")
        return False


async def test_llm_request_creation():
    """Test LLM request creation with config."""
    print("\n🧪 Testing LLM Request Creation...")

    try:
        config = load_config()

        # Test creating request for question generator
        request = config.create_llm_request_for_agent(
            agent_name="question_generator",
            prompt="Generate a simple math question about {{topic}}",
            topic="fractions",  # This would normally be handled by PromptManager
        )

        print("✅ LLM Request created successfully!")
        print(f"   Model: {request.model}")
        print(f"   Temperature: {request.temperature}")
        print(f"   Max tokens: {request.max_tokens}")
        print(f"   Prompt: {request.prompt}")

        # Test runtime override
        overridden_request = LLMRequest(
            **{**request.model_dump(), "temperature": 0.2, "model": "claude-3-5-haiku-20241022"}
        )

        print("\n   Runtime Override Test:")
        print(f"   New model: {overridden_request.model}")
        print(f"   New temperature: {overridden_request.temperature}")

        return True

    except Exception as e:
        print(f"❌ LLM Request creation failed: {e}")
        return False


async def main():
    """Run all API tests."""
    print("🚀 Starting Live API Tests for Derivativ LLM System")
    print("=" * 60)

    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("ℹ️  No .env file found. Please create one with your API keys.")
        print("   You can copy .env.example and add your keys.")
        print("\n   Required environment variables:")
        print("   - OPENAI_API_KEY=sk-...")
        print("   - ANTHROPIC_API_KEY=sk-ant-...")
        print("   - GOOGLE_API_KEY=...")
        return

    results = []

    # Test configuration system first
    results.append(await test_config_system())
    results.append(await test_llm_request_creation())

    # Test APIs (only if keys are available)
    results.append(await test_openai_api_connectivity())
    results.append(await test_anthropic_api())
    results.append(await test_google_api())

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")

    test_names = [
        "Configuration System",
        "LLM Request Creation",
        "OpenAI API",
        "Anthropic API",
        "Google Gemini API",
    ]

    passed = sum(results)
    total = len(results)

    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All systems ready for live development!")
    else:
        print("⚠️  Some issues found. Check the output above.")


if __name__ == "__main__":
    # Install required packages if not available
    required_packages = ["openai", "anthropic", "google-generativeai"]

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            print(f"📦 Installing {package}...")
            os.system(f"pip install {package}")

    asyncio.run(main())

#!/usr/bin/env python3
"""Debug script to test LLM service directly."""

import asyncio
import json
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.models.llm_models import LLMRequest  # noqa: E402
from src.services.llm_factory import LLMFactory  # noqa: E402

# Set environment for testing
os.environ["DEMO_MODE"] = "false"

print(f"🔑 OpenAI API Key loaded: {bool(os.getenv('OPENAI_API_KEY'))}")


async def test_llm_direct():
    """Test LLM service directly to find the issue."""
    print("🔧 Testing LLM service directly...")

    try:
        # Create LLM factory
        llm_factory = LLMFactory()

        # Get OpenAI service
        llm_service = llm_factory.get_service("openai")
        print(f"✅ LLM service created: {type(llm_service)}")

        # Simple test request
        request = LLMRequest(
            model="gpt-4o-mini",
            prompt='Say \'Hello, World!\' in JSON format: {"message": "Hello, World!"}',
            temperature=0.3,
            max_tokens=100,
            response_format={"type": "json_object"},
        )

        print(f"📤 Sending request: {request.model} - {request.prompt[:50]}...")

        # Make the call
        response = await llm_service.generate_non_stream(request)

        print("✅ Response received!")
        print(f"   Content: {response.content}")
        print(f"   Tokens: {getattr(response, 'total_tokens', 'Unknown')}")

        # Try to parse as JSON
        try:
            parsed = json.loads(response.content)
            print(f"   ✅ JSON parsing successful: {parsed}")
        except json.JSONDecodeError as e:
            print(f"   ❌ JSON parsing failed: {e}")

        return True

    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_template_rendering():
    """Test template rendering separately."""
    print("\n🎨 Testing template rendering...")

    try:
        from src.services.prompt_manager import PromptConfig, PromptManager

        # Create prompt manager
        prompt_manager = PromptManager()

        # Test context (simplified) - include ALL required template variables
        context = {
            "document_type": "worksheet",
            "title": "Test Worksheet",
            "topic": "Number",
            "detail_level": 3,
            "target_duration_minutes": 30,
            "grade_level": 8,
            "tier": "Core",
            "custom_instructions": "Focus on practical applications",  # Add missing variable
            "subtopics": ["Basic arithmetic"],  # Add missing variable
            "syllabus_refs": ["N1.1"],  # Add missing variable
            "detailed_syllabus_content": {},  # Add missing variable
            "personalization_context": {},  # Add missing variable
            "selected_blocks": [
                {
                    "block_type": "practice_questions",
                    "content_guidelines": {"difficulty": "medium"},
                    "estimated_content_volume": {"num_questions": 5},
                    "schema": {"type": "object", "properties": {"questions": {"type": "array"}}},
                }
            ],
            "output_schema": {"type": "object", "required": ["blocks"]},
        }

        prompt_config = PromptConfig(template_name="document_content_generation", variables=context)

        rendered_prompt = await prompt_manager.render_prompt(
            prompt_config, model_name="gpt-4o-mini"
        )

        print("✅ Template rendered successfully!")
        print(f"   Length: {len(rendered_prompt)} characters")
        print(f"   First 200 chars: {rendered_prompt[:200]}...")

        return True

    except Exception as e:
        print(f"❌ Template rendering failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all debug tests."""
    print("🧪 LLM Debug Test Suite")
    print("=" * 40)

    # Test 1: Direct LLM service
    llm_success = await test_llm_direct()

    # Test 2: Template rendering
    template_success = await test_template_rendering()

    print("\n📊 Debug Results:")
    print(f"   LLM Service: {'✅ Working' if llm_success else '❌ Failed'}")
    print(f"   Template Rendering: {'✅ Working' if template_success else '❌ Failed'}")

    if llm_success and template_success:
        print("\n🎯 Both components working - issue may be in integration")
    else:
        print("\n🔍 Found the problem! Check the failed component above.")


if __name__ == "__main__":
    asyncio.run(main())

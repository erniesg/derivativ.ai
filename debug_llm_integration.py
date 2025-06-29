#!/usr/bin/env python3
"""Debug LLM integration in document generation service."""

import asyncio
import json
import logging

from dotenv import load_dotenv

# Set up logging BEFORE imports
logging.basicConfig(level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s")

# Load environment
load_dotenv()

from src.models.document_generation_v2 import DocumentGenerationRequestV2  # noqa: E402
from src.models.enums import Tier, TopicName  # noqa: E402
from src.services.document_generation_service_v2 import DocumentGenerationServiceV2  # noqa: E402
from src.services.llm_factory import LLMFactory  # noqa: E402
from src.services.prompt_manager import PromptManager  # noqa: E402

# Skip repositories for now - focus on LLM issue

logger = logging.getLogger(__name__)


async def test_document_generation_with_logging():
    """Test document generation with detailed logging."""
    print("\n🔍 Testing Document Generation V2 with Debug Logging\n")

    # Create services
    llm_factory = LLMFactory()
    prompt_manager = PromptManager()

    # Use None for repositories to focus on LLM issue
    question_repo = None
    document_repo = None
    print("⚠️ Using mock repositories (focusing on LLM debugging)")

    # Create document generation service
    service = DocumentGenerationServiceV2(
        llm_factory=llm_factory,
        prompt_manager=prompt_manager,
        question_repository=question_repo,
        document_storage_repository=document_repo,
    )

    # Create simple request
    request = DocumentGenerationRequestV2(
        title="Debug Algebra Worksheet",
        document_type="worksheet",
        topic=TopicName.ALGEBRA_AND_GRAPHS,
        tier=Tier.CORE,
        grade_level=8,
        detail_level=3,
        target_duration_minutes=20,
        custom_instructions="Create simple linear equation problems",
    )

    print(f"📋 Request: {request.title}")
    print(f"   Type: {request.document_type}")
    print(f"   Topic: {request.topic.value}")
    print(f"   Detail Level: {request.detail_level}")

    try:
        # Generate document
        print("\n🚀 Starting generation...")
        result = await service.generate_document(request)

        if result.success:
            print("\n✅ Generation Successful!")
            print(f"   Document ID: {result.document.document_id}")
            print(f"   Processing Time: {result.processing_time:.2f}s")
            print(f"   Blocks Generated: {len(result.document.content_structure.blocks)}")

            # Show block details
            for i, block in enumerate(result.document.content_structure.blocks):
                print(f"\n   Block {i+1}: {block.block_type}")
                print(f"     - Reasoning: {block.reasoning}")
                print(f"     - Estimated Minutes: {block.estimated_minutes}")

                # Show content preview
                content_str = (
                    json.dumps(block.content)[:100]
                    if isinstance(block.content, dict)
                    else str(block.content)[:100]
                )
                print(f"     - Content Preview: {content_str}...")

            # Check generation insights
            insights = result.generation_insights
            print("\n📊 Generation Insights:")
            print(f"   - LLM Reasoning: {insights.get('llm_reasoning', 'No reasoning provided')}")

        else:
            print(f"\n❌ Generation Failed: {result.error_message}")

    except Exception as e:
        print(f"\n💥 Exception occurred: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


async def test_llm_generation_directly():
    """Test the exact LLM generation that's failing."""
    print("\n🔬 Testing Direct LLM Generation\n")

    from src.models.document_generation_v2 import BLOCK_CONTENT_SCHEMAS, DOCUMENT_CONTENT_SCHEMA
    from src.models.llm_models import LLMRequest

    llm_factory = LLMFactory()
    prompt_manager = PromptManager()

    # Simulate the exact context that would be sent
    context = {
        "document_type": "worksheet",
        "title": "Test Worksheet",
        "topic": "Algebra and graphs",
        "subtopics": None,
        "detail_level": 3,
        "target_duration_minutes": 20,
        "grade_level": 8,
        "difficulty": None,
        "tier": "Core",
        "custom_instructions": "Simple problems",
        "personalization_context": None,
        "syllabus_refs": [],
        "detailed_syllabus_content": {},
        "selected_blocks": [
            {
                "block_type": "practice_questions",
                "content_guidelines": {"difficulty": "medium", "num_questions": 5},
                "estimated_content_volume": {"num_questions": 5},
                "schema": BLOCK_CONTENT_SCHEMAS.get("practice_questions", {}),
            }
        ],
        "output_schema": DOCUMENT_CONTENT_SCHEMA,
    }

    try:
        # Render the template
        from src.services.prompt_manager import PromptConfig

        prompt_config = PromptConfig(template_name="document_content_generation", variables=context)
        rendered_prompt = await prompt_manager.render_prompt(
            prompt_config, model_name="gpt-4o-mini"
        )

        print(f"✅ Template rendered: {len(rendered_prompt)} chars")
        print(f"First 500 chars:\n{rendered_prompt[:500]}...\n")

        # Create LLM request
        llm_service = llm_factory.get_service("openai")
        llm_request = LLMRequest(
            model="gpt-4o-mini",
            prompt=rendered_prompt,
            temperature=0.3,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )

        print("📤 Sending to LLM...")
        response = await llm_service.generate_non_stream(llm_request)

        print(f"✅ LLM Response received: {len(response.content)} chars")

        # Try to parse
        try:
            parsed = json.loads(response.content)
            print("✅ JSON parsed successfully!")
            print(f"   Keys: {list(parsed.keys())}")
            print(f"   Blocks: {len(parsed.get('blocks', []))}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error: {e}")
            print(f"Response content:\n{response.content}")

    except Exception as e:
        print(f"💥 Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Run all debug tests."""
    print("🧪 Document Generation Debug Suite")
    print("=" * 50)

    # Test 1: Full document generation
    await test_document_generation_with_logging()

    # Test 2: Direct LLM generation
    await test_llm_generation_directly()


if __name__ == "__main__":
    asyncio.run(main())

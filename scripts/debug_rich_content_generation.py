#!/usr/bin/env python3
"""
Debug script for rich content generation.
Tests the complete document generation pipeline to ensure real rich content is generated.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.api.dependencies import get_dependencies
from src.core.config import get_settings
from src.models.document_models import (
    DetailLevel,
    DocumentGenerationRequest,
    DocumentType,
)
from src.models.enums import Tier

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("debug_rich_content.log")],
)
logger = logging.getLogger(__name__)


async def test_rich_content_generation():  # noqa: PLR0915
    """Test document generation to ensure rich content is generated, not fallback mock data."""

    print("🧪 Testing Rich Content Generation")
    print("=" * 50)

    try:
        # Get dependencies
        dependencies = await get_dependencies()
        doc_service = dependencies["document_generation_service"]

        # Create test request for worksheet generation
        request = DocumentGenerationRequest(
            title="Algebra Practice: Linear Equations",
            topic="Solving Linear Equations",
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            tier=Tier.CORE,
            grade_level=8,
            auto_include_questions=True,
            max_questions=5,
            custom_instructions="Include step-by-step solutions with detailed explanations",
        )

        print("📋 Request Details:")
        print(f"   Title: {request.title}")
        print(f"   Type: {request.document_type.value}")
        print(f"   Detail Level: {request.detail_level.value}")
        print(f"   Auto Include Questions: {request.auto_include_questions}")
        print(f"   Max Questions: {request.max_questions}")
        print()

        # Generate document
        print("🚀 Generating document...")
        start_time = datetime.now()

        result = await doc_service.generate_document(request)

        processing_time = (datetime.now() - start_time).total_seconds()
        print(f"⏱️  Processing completed in {processing_time:.2f}s")
        print()

        # Analyze results
        if result.success:
            print("✅ Document generation SUCCEEDED")
            print()

            doc = result.document
            print("📄 Generated Document Analysis:")
            print(f"   Document ID: {doc.document_id}")
            print(f"   Title: {doc.title}")
            print(f"   Template Used: {doc.template_used}")
            print(f"   Total Sections: {len(doc.sections)}")
            print(f"   Total Questions: {doc.total_questions}")
            print(f"   Estimated Duration: {doc.estimated_duration} minutes")
            print(f"   Questions Used: {len(doc.questions_used)}")
            print()

            # Analyze content richness
            print("🔍 Content Richness Analysis:")
            print("=" * 30)

            total_content_length = 0
            sections_with_rich_content = 0

            for i, section in enumerate(doc.sections):
                print(f"\n{i+1}. Section: {section.title}")
                print(f"   Type: {section.content_type}")
                print(f"   Order: {section.order_index}")

                # Analyze content data
                if section.content_data:
                    content_str = json.dumps(section.content_data, indent=2)
                    content_length = len(content_str)
                    total_content_length += content_length

                    print(f"   Content Size: {content_length} chars")

                    # Check for rich content indicators
                    rich_indicators = []
                    if "questions" in section.content_data:
                        questions = section.content_data["questions"]
                        if isinstance(questions, list) and len(questions) > 0:
                            rich_indicators.append(f"{len(questions)} questions")
                            sections_with_rich_content += 1

                    if "examples" in section.content_data:
                        examples = section.content_data["examples"]
                        if isinstance(examples, list) and len(examples) > 0:
                            rich_indicators.append(f"{len(examples)} examples")
                            sections_with_rich_content += 1

                    if "solution_steps" in section.content_data:
                        steps = section.content_data["solution_steps"]
                        if isinstance(steps, list) and len(steps) > 0:
                            rich_indicators.append(f"{len(steps)} solution steps")

                    if "text" in section.content_data:
                        text = section.content_data["text"]
                        if isinstance(text, str) and len(text) > 50:
                            rich_indicators.append(f"rich text ({len(text)} chars)")

                    if rich_indicators:
                        print(f"   Rich Content: {', '.join(rich_indicators)}")
                    else:
                        print("   ⚠️  Basic/Mock Content Detected")
                        print(f"   Content Preview: {content_str[:200]}...")
                else:
                    print("   ❌ No content data")

            print("\n📊 Summary:")
            print(f"   Total Content Size: {total_content_length} characters")
            print(
                f"   Sections with Rich Content: {sections_with_rich_content}/{len(doc.sections)}"
            )
            print("   Processing Stats:")
            print(f"     - Questions Processed: {result.questions_processed}")
            print(f"     - Sections Generated: {result.sections_generated}")
            print(f"     - Customizations Applied: {result.customizations_applied}")
            print(f"     - Personalization Success: {result.personalization_success}")

            # Check if this is fallback content
            is_fallback_content = (
                total_content_length < 500  # Very short content
                or sections_with_rich_content == 0  # No rich content
                or all(
                    "Generated content for" in str(section.content_data.get("text", ""))
                    for section in doc.sections
                    if section.content_data and "text" in section.content_data
                )  # Fallback placeholder text
            )

            if is_fallback_content:
                print("\n⚠️  FALLBACK CONTENT DETECTED")
                print(
                    "   This appears to be mock/fallback content, not real LLM-generated rich content"
                )
                print("   Possible causes:")
                print("     - LLM JSON parsing failed")
                print("     - API connectivity issues")
                print("     - Template rendering problems")
                return False
            else:
                print("\n✅ RICH CONTENT CONFIRMED")
                print("   Document contains real LLM-generated content with proper structure")
                return True

        else:
            print("❌ Document generation FAILED")
            print(f"   Error: {result.error_message}")
            print(f"   Processing Time: {result.processing_time:.2f}s")
            return False

    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        print(f"\n💥 Test Exception: {e}")
        return False


async def test_llm_connectivity():
    """Test LLM service connectivity and response quality."""

    print("\n🔌 Testing LLM Connectivity")
    print("=" * 30)

    try:
        # Get LLM service
        dependencies = await get_dependencies()
        llm_factory = dependencies["llm_factory"]
        llm_service = llm_factory.get_service("openai")

        from src.models.llm_models import LLMRequest

        # Simple test request
        test_request = LLMRequest(
            model="gpt-4o-mini",
            prompt="Generate a simple JSON object with fields 'status' and 'message'. The status should be 'success' and message should be 'LLM connectivity test passed'.",
            temperature=0.1,
            max_tokens=100,
        )

        print("🧪 Testing basic LLM request...")
        response = await llm_service.generate_non_stream(test_request)

        print("✅ LLM Response received:")
        print(f"   Length: {len(response.content)} chars")
        print(f"   Content: {response.content}")

        # Test JSON parsing
        try:
            parsed = json.loads(response.content)
            print(f"✅ JSON parsing successful: {parsed}")
            return True
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"   Raw content: {response.content}")
            return False

    except Exception as e:
        logger.error(f"LLM connectivity test failed: {e}", exc_info=True)
        print(f"💥 LLM Test Exception: {e}")
        return False


async def main():
    """Run all debugging tests."""

    print("🚀 Rich Content Generation Debug Suite")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Check environment
    settings = get_settings()
    print("\nEnvironment:")
    print(f"   Demo Mode: {os.getenv('DEMO_MODE', 'False')}")
    print(f"   Database URL: {'***' if settings.database_url else 'Not set'}")
    print(f"   OpenAI Key: {'***' if settings.openai_api_key else 'Not set'}")

    # Run tests
    tests_passed = 0
    total_tests = 2

    # Test 1: LLM Connectivity
    if await test_llm_connectivity():
        tests_passed += 1

    # Test 2: Rich Content Generation
    if await test_rich_content_generation():
        tests_passed += 1

    # Final report
    print("\n🏁 Debug Results")
    print("=" * 20)
    print(f"Tests Passed: {tests_passed}/{total_tests}")

    if tests_passed == total_tests:
        print("✅ All tests PASSED - Rich content generation is working correctly")
        return True
    else:
        print("❌ Some tests FAILED - Rich content generation needs debugging")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

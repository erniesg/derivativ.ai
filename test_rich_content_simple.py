#!/usr/bin/env python3
"""
Simple test for rich content generation.
Tests the document generation pipeline with direct dependency injection.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.api.dependencies import (  # noqa: E402
    get_document_generation_service,
)
from src.models.document_models import (  # noqa: E402
    DetailLevel,
    DocumentGenerationRequest,
    DocumentType,
)
from src.models.enums import Tier  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_rich_content():  # noqa: PLR0915
    """Test document generation for rich content."""

    print("🧪 Testing Rich Content Generation")
    print("=" * 50)

    # Get document generation service (handles dependency injection internally)
    doc_service = get_document_generation_service()

    # Create test request
    request = DocumentGenerationRequest(
        title="Linear Equations Practice",
        topic="Solving Linear Equations",
        document_type=DocumentType.WORKSHEET,
        detail_level=DetailLevel.MEDIUM,
        tier=Tier.CORE,
        grade_level=8,
        auto_include_questions=True,
        max_questions=3,
        custom_instructions="Include detailed step-by-step solutions",
    )

    print("📋 Test Request:")
    print(f"   Title: {request.title}")
    print(f"   Topic: {request.topic}")
    print(f"   Type: {request.document_type.value}")
    print(f"   Detail Level: {request.detail_level.value}")
    print()

    # Generate document
    print("🚀 Generating document...")
    start_time = datetime.now()

    try:
        result = await doc_service.generate_document(request)
        processing_time = (datetime.now() - start_time).total_seconds()

        print(f"⏱️  Processing completed in {processing_time:.2f}s")
        print()

        if result.success:
            print("✅ Document generation SUCCEEDED")
            doc = result.document

            print("📄 Document Details:")
            print(f"   ID: {doc.document_id}")
            print(f"   Title: {doc.title}")
            print(f"   Template: {doc.template_used}")
            print(f"   Sections: {len(doc.sections)}")
            print(f"   Duration: {doc.estimated_duration} min")
            print()

            # Analyze content
            print("🔍 Content Analysis:")
            total_content = 0
            rich_sections = 0

            for i, section in enumerate(doc.sections):
                print(f"\n{i+1}. {section.title} ({section.content_type})")

                if section.content_data:
                    content_str = json.dumps(section.content_data, indent=2)
                    content_size = len(content_str)
                    total_content += content_size

                    print(f"   Content size: {content_size} chars")

                    # Check for rich content
                    if any(
                        key in section.content_data
                        for key in ["questions", "examples", "solution_steps"]
                    ):
                        rich_sections += 1
                        print("   ✅ Rich content detected")
                    elif content_size > 100:
                        print("   📝 Text content")
                    else:
                        print("   ⚠️  Basic content")
                        print(f"   Preview: {content_str[:100]}...")
                else:
                    print("   ❌ No content")

            print("\n📊 Summary:")
            print(f"   Total content: {total_content} chars")
            print(f"   Rich sections: {rich_sections}/{len(doc.sections)}")

            # Determine if this is real rich content
            is_rich = total_content > 1000 and rich_sections > 0

            if is_rich:
                print("✅ RICH CONTENT CONFIRMED - Real LLM-generated content")
                return True
            else:
                print("⚠️  BASIC CONTENT - Likely fallback/mock data")
                return False

        else:
            print("❌ Document generation FAILED")
            print(f"   Error: {result.error_message}")
            return False

    except Exception as e:
        print(f"💥 Exception: {e}")
        logger.error("Test failed", exc_info=True)
        return False


async def main():
    """Run the test."""
    print("🚀 Rich Content Test")
    print(f"Demo mode: {os.getenv('DEMO_MODE', 'False')}")
    print()

    success = await test_rich_content()

    if success:
        print("\n🎉 Test PASSED - Rich content generation working!")
    else:
        print("\n❌ Test FAILED - Needs investigation")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

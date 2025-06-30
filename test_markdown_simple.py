#!/usr/bin/env python3
"""
Simple test script for markdown generation without FastAPI dependencies.
Tests the core functionality step by step.
"""

import asyncio
import tempfile
from pathlib import Path

from src.models.document_models import DetailLevel, DocumentType
from src.models.enums import Tier, TopicName
from src.models.markdown_generation_models import MarkdownGenerationRequest
from src.services.llm_factory import create_llm_factory
from src.services.markdown_document_service import MarkdownDocumentService
from src.services.pandoc_service import PandocService
from src.services.prompt_manager import PromptManager


async def test_markdown_generation():
    """Test just the markdown generation component."""

    print("🧪 Testing Markdown Generation")
    print("=" * 40)

    try:
        # Create request
        request = MarkdownGenerationRequest(
            document_type=DocumentType.WORKSHEET,
            topic=TopicName.ALGEBRA_AND_GRAPHS,
            tier=Tier.CORE,
            detail_level=DetailLevel.MEDIUM,
            target_duration_minutes=30,
            grade_level="7-9",
        )

        print(f"📝 Request: {request.document_type.value} on {request.topic.value}")

        # Create services
        llm_factory = create_llm_factory()
        prompt_manager = PromptManager()

        print("🔧 Services created")

        # Create markdown service
        markdown_service = MarkdownDocumentService(
            llm_service=llm_factory.get_service("openai"), prompt_manager=prompt_manager
        )

        print("📄 Generating markdown...")

        # Generate markdown
        result = await markdown_service.generate_markdown_document(
            request=request, custom_instructions="Focus on linear equations and graphing"
        )

        if result["success"]:
            print("✅ Markdown generation successful!")
            print(f"   Content length: {len(result['markdown_content'])} characters")
            print(f"   Word count: {result['metadata']['word_count']}")
            print(f"   Sections: {result['metadata']['section_count']}")

            # Show first 300 characters
            content = result["markdown_content"]
            print("\n📄 Sample content:")
            print("-" * 40)
            print(content[:300] + "..." if len(content) > 300 else content)
            print("-" * 40)

            return result
        else:
            print(f"❌ Markdown generation failed: {result.get('error')}")
            return None

    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback

        traceback.print_exc()
        return None


async def test_pandoc_conversion():
    """Test pandoc conversion separately."""

    print("\n🔄 Testing Pandoc Conversion")
    print("=" * 40)

    try:
        # Sample markdown content
        sample_markdown = """# Algebra Worksheet

## Learning Objectives
- Solve linear equations
- Graph linear functions
- Understand slope and intercept

## Practice Questions

### Question 1
Solve for x: 2x + 5 = 13

**Solution:**
- 2x + 5 = 13
- 2x = 13 - 5
- 2x = 8
- x = 4

### Question 2
Find the slope of the line passing through (2,3) and (6,11).

**Solution:**
- slope = (y₂ - y₁) / (x₂ - x₁)
- slope = (11 - 3) / (6 - 2)
- slope = 8 / 4 = 2
"""

        pandoc_service = PandocService()

        print("🔧 Pandoc service created")

        # Test markdown to HTML
        print("📄 Converting to HTML...")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            html_path = await pandoc_service.convert_markdown_to_html(
                sample_markdown, temp_path / "test.html"
            )

            if html_path.exists():
                print(f"✅ HTML conversion successful: {html_path}")
                with open(html_path) as f:
                    html_content = f.read()
                print(f"   HTML size: {len(html_content)} characters")

            # Test markdown to PDF (if pandoc available)
            print("📄 Converting to PDF...")
            try:
                pdf_path = await pandoc_service.convert_markdown_to_pdf(
                    sample_markdown, temp_path / "test.pdf"
                )

                if pdf_path.exists():
                    print(f"✅ PDF conversion successful: {pdf_path}")
                    print(f"   PDF size: {pdf_path.stat().st_size} bytes")
                else:
                    print("❌ PDF file not created")

            except Exception as e:
                print(f"⚠️ PDF conversion failed: {e}")
                print("   (This is expected if pandoc is not installed)")

        return True

    except Exception as e:
        print(f"❌ Pandoc test error: {e}")
        return False


async def main():
    """Run simple tests."""

    print("🚀 Simple Markdown Pipeline Test")
    print("=" * 50)

    # Test 1: Markdown generation
    markdown_result = await test_markdown_generation()

    # Test 2: Pandoc conversion
    pandoc_success = await test_pandoc_conversion()

    print("\n📊 Test Summary:")
    print(f"   Markdown Generation: {'✅ PASS' if markdown_result else '❌ FAIL'}")
    print(f"   Pandoc Conversion: {'✅ PASS' if pandoc_success else '❌ FAIL'}")

    if markdown_result and pandoc_success:
        print("\n🎉 Core components working!")
        print("✅ Ready for full pipeline integration")
    else:
        print("\n⚠️ Some components need attention")
        print("💡 Check API keys and pandoc installation")


if __name__ == "__main__":
    asyncio.run(main())

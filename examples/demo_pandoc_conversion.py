#!/usr/bin/env python3
"""
Demo script for pandoc-based document conversion.

Shows how the DocumentFormatterAgent can convert documents to PDF, DOCX, and PowerPoint
using pandoc as the conversion engine.

Requirements:
- pandoc installed (brew install pandoc on macOS)
- pdflatex for PDF generation (brew install --cask mactex)
"""

import asyncio
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.document_formatter_agent import DocumentFormatterAgent  # noqa: E402
from src.models.document_models import (  # noqa: E402
    ContentSection,
    DetailLevel,
    DocumentGenerationRequest,
    DocumentType,
    ExportFormat,
    GeneratedDocument,
)
from src.models.enums import Tier  # noqa: E402
from src.services.llm_service import LLMService  # noqa: E402


def check_pandoc_installation():
    """Check if pandoc is installed and available."""
    try:
        result = subprocess.run(
            ["pandoc", "--version"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            version = result.stdout.split("\n")[0]
            print(f"✅ Found {version}")
            return True
        else:
            print("❌ Pandoc not found")
            return False
    except FileNotFoundError:
        print("❌ Pandoc not installed")
        print("   Install with: brew install pandoc")
        return False


def create_sample_document():
    """Create a sample document for conversion testing."""
    sections = [
        ContentSection(
            title="Learning Objectives",
            content_type="learning_objectives",
            content_data={
                "objectives_text": "• Understand quadratic equations\n• Apply factoring methods\n• Solve real-world problems\n• Graph parabolic functions"
            },
            order_index=0,
        ),
        ContentSection(
            title="Key Formulas",
            content_type="key_formulas",
            content_data={
                "text": "**Standard Form:** ax² + bx + c = 0\n\n**Quadratic Formula:** x = (-b ± √(b² - 4ac)) / 2a\n\n**Vertex Form:** y = a(x - h)² + k"
            },
            order_index=1,
        ),
        ContentSection(
            title="Practice Questions",
            content_type="practice_questions",
            content_data={
                "questions": [
                    {
                        "question_text": "Solve: x² + 5x + 6 = 0",
                        "marks": 3,
                        "command_word": "Calculate",
                    },
                    {
                        "question_text": "Factor completely: x² - 4x + 4",
                        "marks": 2,
                        "command_word": "Factor",
                    },
                    {
                        "question_text": "Find the vertex of: y = x² - 6x + 8",
                        "marks": 4,
                        "command_word": "Find",
                    },
                ],
                "total_marks": 9,
            },
            order_index=2,
        ),
        ContentSection(
            title="Worked Example",
            content_type="worked_examples",
            content_data={
                "examples": [
                    {
                        "question_text": "Solve: x² - 7x + 12 = 0",
                        "marks": 3,
                        "solution_steps": [
                            {
                                "step_number": 1,
                                "description_text": "Look for two numbers that multiply to 12 and add to -7",
                            },
                            {"step_number": 2, "description_text": "The numbers are -3 and -4"},
                            {"step_number": 3, "description_text": "Factor: (x - 3)(x - 4) = 0"},
                            {"step_number": 4, "description_text": "Therefore: x = 3 or x = 4"},
                        ],
                    }
                ]
            },
            order_index=3,
        ),
    ]

    request = DocumentGenerationRequest(
        document_type=DocumentType.WORKSHEET,
        detail_level=DetailLevel.COMPREHENSIVE,
        title="Quadratic Equations Practice",
        topic="quadratic_equations",
        tier=Tier.EXTENDED,
        custom_instructions="Include step-by-step solutions and visual learning aids",
        personalization_context={"learning_style": "visual", "difficulty_preference": "gradual"},
    )

    return GeneratedDocument(
        title="Quadratic Equations Practice Worksheet",
        document_type=DocumentType.WORKSHEET,
        detail_level=DetailLevel.COMPREHENSIVE,
        generated_at="2025-06-21T12:00:00Z",
        template_used="worksheet_default",
        generation_request=request,
        sections=sections,
        total_questions=3,
        estimated_duration=25,
        applied_customizations={
            "custom_instructions": "Include step-by-step solutions and visual learning aids",
            "personalization_context": {
                "learning_style": "visual",
                "difficulty_preference": "gradual",
            },
        },
    )


async def demo_document_conversion():
    """Demonstrate document conversion to multiple formats."""
    print("🔄 Pandoc Document Conversion Demo")
    print("=" * 50)

    # Check if pandoc is available
    if not check_pandoc_installation():
        print("\n❌ Cannot proceed without pandoc")
        return

    # Create formatter agent
    mock_llm = MagicMock(spec=LLMService)
    formatter = DocumentFormatterAgent(mock_llm)

    # Create sample document
    document = create_sample_document()
    print(f"\n📄 Created sample document: {document.title}")
    print(f"   Type: {document.document_type.value}")
    print(f"   Detail Level: {document.detail_level.value}")
    print(f"   Sections: {len(document.sections)}")
    print(f"   Questions: {document.total_questions}")

    # Test formats to convert
    formats_to_test = [
        (ExportFormat.HTML, "📄 HTML", "Open in browser"),
        (ExportFormat.PDF, "📄 PDF", "Ready for printing"),
        (ExportFormat.DOCX, "📄 Word Document", "Editable in Microsoft Word"),
        (ExportFormat.SLIDES_PPTX, "📄 PowerPoint", "Ready for presentation"),
        (ExportFormat.LATEX, "📄 LaTeX", "Academic formatting"),
    ]

    results = {}

    for format_type, description, usage in formats_to_test:
        print(f"\n{description}:")
        try:
            # Add personalization for demonstration
            personalization = document.applied_customizations.get("personalization_context", {})

            output_file = await formatter._format_with_pandoc(
                document, format_type, personalization
            )

            if Path(output_file).exists():
                file_size = Path(output_file).stat().st_size
                print(f"   ✅ Success: {output_file}")
                print(f"   📏 Size: {file_size:,} bytes")
                print(f"   💡 {usage}")
                results[format_type] = output_file
            else:
                print(f"   ❌ File not created: {output_file}")

        except Exception as e:
            print(f"   ❌ Failed: {e}")

    # Display summary
    print("\n📊 Conversion Summary")
    print("=" * 30)
    print(f"✅ Successful: {len(results)}/{len(formats_to_test)} formats")

    if results:
        print("\n📁 Generated files:")
        for format_type, file_path in results.items():
            print(f"   {format_type.value.upper()}: {file_path}")

        print("\n🧹 Cleanup:")
        cleanup_choice = input("Delete generated files? (y/N): ").lower().strip()
        if cleanup_choice == "y":
            for file_path in results.values():
                try:
                    Path(file_path).unlink()
                    print(f"   🗑️  Deleted: {file_path}")
                except Exception as e:
                    print(f"   ❌ Failed to delete {file_path}: {e}")
        else:
            print("   📁 Files preserved for your review")

    # Show sample content
    print("\n📝 Sample Markdown Content (first 500 chars):")
    print("-" * 50)
    markdown_content = formatter._format_to_markdown(document, personalization)
    print(markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content)


async def demo_slides_conversion():
    """Demonstrate slide-specific conversion."""
    print("\n🎯 Slides-Specific Demo")
    print("=" * 30)

    mock_llm = MagicMock(spec=LLMService)
    formatter = DocumentFormatterAgent(mock_llm)

    # Create document optimized for slides
    document = create_sample_document()
    document.document_type = DocumentType.SLIDES
    document.title = "Quadratic Equations Presentation"

    personalization = {"learning_style": "visual"}

    slides_markdown = formatter._format_to_markdown_for_slides(document, personalization)

    print("📝 Slides Markdown Structure:")
    lines = slides_markdown.split("\n")
    slide_count = 0
    for i, line in enumerate(lines):
        if line.strip() == "---":
            slide_count += 1
            print(f"   📊 Slide {slide_count}")
        elif line.startswith("## "):
            print(f"      🏷️  {line}")

    print(f"\n📊 Total slides: {slide_count}")
    print(
        f"💡 Visual learner optimizations: {'Applied' if 'diagram' in slides_markdown else 'None'}"
    )


if __name__ == "__main__":
    print("🚀 Starting Pandoc Document Conversion Demo")

    try:
        asyncio.run(demo_document_conversion())
        asyncio.run(demo_slides_conversion())
        print("\n✅ Demo completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()

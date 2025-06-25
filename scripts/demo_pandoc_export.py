#!/usr/bin/env python3
"""
Demo script showing complete pandoc export functionality.
Generates a sample worksheet and exports it to PDF, DOCX, and HTML.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.document_models import (
    DetailLevel,
    DocumentSection,
    DocumentStructure,
    DocumentType,
)
from services.document_export_service import DocumentExportService


async def demo_pandoc_export():  # noqa: PLR0915
    """Demonstrate complete document export functionality."""
    print("🚀 Derivativ Pandoc Export Demo")
    print("=" * 50)

    # Create sample document structure
    document = DocumentStructure(
        title="Linear Equations Practice Worksheet",
        document_type=DocumentType.WORKSHEET,
        detail_level=DetailLevel.MEDIUM,
        estimated_duration=30,
        total_questions=3,
        sections=[
            DocumentSection(
                title="Learning Objectives",
                content_type="learning_objectives",
                content_data={
                    "objectives_text": "• Solve linear equations in one variable\n• Apply the balance method\n• Check solutions by substitution"
                },
                order_index=0,
            ),
            DocumentSection(
                title="Key Formula",
                content_type="theory_content",
                content_data={
                    "theory_text": "For linear equations of the form ax + b = c:\n\n**Steps to solve:**\n1. Isolate the variable term\n2. Divide by the coefficient\n3. Check your answer"
                },
                order_index=1,
            ),
            DocumentSection(
                title="Worked Example",
                content_type="worked_examples",
                content_data={
                    "examples": [
                        {
                            "title": "Example 1",
                            "problem": "Solve: 3x + 7 = 22",
                            "solution": "3x + 7 = 22\n3x = 22 - 7\n3x = 15\nx = 15 ÷ 3\nx = 5\n\n**Check:** 3(5) + 7 = 15 + 7 = 22 ✓",
                        }
                    ]
                },
                order_index=2,
            ),
            DocumentSection(
                title="Practice Questions",
                content_type="practice_questions",
                content_data={
                    "questions": [
                        {
                            "question_id": "q1",
                            "question_text": "Solve: 2x + 5 = 13",
                            "marks": 2,
                            "command_word": "Solve",
                        },
                        {
                            "question_id": "q2",
                            "question_text": "Find the value of y: 4y - 3 = 17",
                            "marks": 3,
                            "command_word": "Find",
                        },
                        {
                            "question_id": "q3",
                            "question_text": "Calculate x when: 7x + 2 = 3x + 18",
                            "marks": 4,
                            "command_word": "Calculate",
                        },
                    ],
                    "total_marks": 9,
                    "estimated_time": 15,
                },
                order_index=3,
            ),
            DocumentSection(
                title="Answers",
                content_type="answers",
                content_data={
                    "answers": [{"answer": "x = 4"}, {"answer": "y = 5"}, {"answer": "x = 4"}]
                },
                order_index=4,
            ),
        ],
    )

    print(f"📄 Created document: '{document.title}'")
    print(f"   Type: {document.document_type.value}")
    print(f"   Detail Level: {document.detail_level.value}")
    print(f"   Sections: {len(document.sections)}")
    print(f"   Questions: {document.total_questions}")
    print()

    # Initialize export service
    export_service = DocumentExportService()

    # Export to different formats
    formats = ["pdf", "docx", "html"]
    export_paths = {}

    for format_type in formats:
        print(f"📤 Exporting to {format_type.upper()}...")

        try:
            start_time = asyncio.get_event_loop().time()

            if format_type == "pdf":
                path = await export_service.export_to_pdf(document)
            elif format_type == "docx":
                path = await export_service.export_to_docx(document)
            elif format_type == "html":
                path = await export_service.export_to_html(document)

            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time

            export_paths[format_type] = path
            file_size = path.stat().st_size

            print(f"   ✅ Success! Created {path.name}")
            print(f"   📊 Size: {file_size:,} bytes")
            print(f"   ⏱️  Time: {duration:.2f} seconds")
            print()

        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()

    # Display results summary
    print("📋 Export Summary")
    print("-" * 30)

    for format_type, path in export_paths.items():
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"{format_type.upper():4} | {path.name:40} | {size_kb:6.1f} KB")

    print()
    print("🔍 To view the files:")
    for format_type, path in export_paths.items():
        if path.exists():
            print(f"   {format_type.upper()}: open '{path}'")

    print()
    print("🧹 Cleanup files:")
    print("   rm " + " ".join(f"'{path}'" for path in export_paths.values() if path.exists()))

    # Optional: Clean up files automatically after demo
    cleanup_input = input("\n🗑️  Delete generated files? (y/N): ").strip().lower()
    if cleanup_input == "y":
        for path in export_paths.values():
            if path.exists():
                path.unlink()
                print(f"   Deleted: {path.name}")

    print("\n✨ Demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_pandoc_export())

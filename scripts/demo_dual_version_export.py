#!/usr/bin/env python3
"""
Demo script for dual version (student/teacher) worksheet export.
Shows the complete workflow from worksheet creation to PDF/DOCX/HTML export.
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


async def demo_dual_version_export():  # noqa: PLR0915
    """Demonstrate student and teacher version export functionality."""
    print("🎓 Derivativ Dual Version Export Demo")
    print("=" * 50)

    # Create comprehensive worksheet
    worksheet = DocumentStructure(
        title="Algebraic Expressions Mastery Worksheet",
        document_type=DocumentType.WORKSHEET,
        detail_level=DetailLevel.MEDIUM,
        estimated_duration=40,
        total_questions=5,
        sections=[
            DocumentSection(
                title="Learning Objectives",
                content_type="learning_objectives",
                content_data={
                    "objectives_text": "• Simplify algebraic expressions\n• Expand brackets using distributive property\n• Factorize simple expressions\n• Substitute values into expressions"
                },
                order_index=0,
            ),
            DocumentSection(
                title="Key Concepts",
                content_type="theory_content",
                content_data={
                    "theory_text": "**Algebraic Expressions** contain variables (letters) and constants (numbers).\n\n**Key Rules:**\n- Like terms can be combined: 3x + 5x = 8x\n- Distributive property: a(b + c) = ab + ac\n- Order of operations: BIDMAS/PEMDAS"
                },
                order_index=1,
            ),
            DocumentSection(
                title="Worked Examples",
                content_type="worked_examples",
                content_data={
                    "examples": [
                        {
                            "title": "Simplifying Expressions",
                            "problem": "Simplify: 3x + 2y + 5x - y",
                            "solution": "Collect like terms:\n3x + 5x + 2y - y = 8x + y",
                        },
                        {
                            "title": "Expanding Brackets",
                            "problem": "Expand: 3(2x + 4)",
                            "solution": "Using distributive property:\n3(2x + 4) = 3 × 2x + 3 × 4 = 6x + 12",
                        },
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
                            "question_text": "Simplify: 5a + 3b + 2a - b",
                            "marks": 2,
                            "command_word": "Simplify",
                        },
                        {
                            "question_id": "q2",
                            "question_text": "Expand: 4(x + 3)",
                            "marks": 2,
                            "command_word": "Expand",
                        },
                        {
                            "question_id": "q3",
                            "question_text": "Factorize: 6x + 9",
                            "marks": 3,
                            "command_word": "Factorize",
                        },
                        {
                            "question_id": "q4",
                            "question_text": "Find the value of 2x² + 3x when x = 4",
                            "marks": 3,
                            "command_word": "Find",
                        },
                        {
                            "question_id": "q5",
                            "question_text": "Simplify: 3(2x + 1) + 2(x - 3)",
                            "marks": 4,
                            "command_word": "Simplify",
                        },
                    ],
                    "total_marks": 14,
                    "estimated_time": 25,
                },
                order_index=3,
            ),
            DocumentSection(
                title="Answers",
                content_type="answers",
                content_data={
                    "answers": [
                        {"answer": "7a + 2b"},
                        {"answer": "4x + 12"},
                        {"answer": "3(2x + 3)"},
                        {"answer": "44"},
                        {"answer": "8x - 3"},
                    ]
                },
                order_index=4,
            ),
            DocumentSection(
                title="Detailed Solutions",
                content_type="detailed_solutions",
                content_data={
                    "solutions": [
                        {"solution": "5a + 3b + 2a - b\n= 5a + 2a + 3b - b\n= 7a + 2b"},
                        {"solution": "4(x + 3)\n= 4 × x + 4 × 3\n= 4x + 12"},
                        {"solution": "6x + 9\n= 3 × 2x + 3 × 3\n= 3(2x + 3)"},
                        {
                            "solution": "2x² + 3x when x = 4\n= 2(4)² + 3(4)\n= 2(16) + 12\n= 32 + 12 = 44"
                        },
                        {"solution": "3(2x + 1) + 2(x - 3)\n= 6x + 3 + 2x - 6\n= 8x - 3"},
                    ]
                },
                order_index=5,
            ),
        ],
    )

    print(f"📄 Created worksheet: '{worksheet.title}'")
    print(f"   Duration: {worksheet.estimated_duration} minutes")
    print(f"   Questions: {worksheet.total_questions}")
    print(f"   Sections: {len(worksheet.sections)}")
    print()

    # Initialize export service
    export_service = DocumentExportService()

    # Show version differences first
    print("🔍 Analyzing version differences...")
    student_version = export_service.version_service.create_student_version(worksheet)
    teacher_version = export_service.version_service.create_teacher_version(worksheet)

    differences = export_service.version_service.get_version_differences(
        student_version, teacher_version
    )

    print(f"   Student sections: {differences['student_sections']}")
    print(f"   Teacher sections: {differences['teacher_sections']}")
    print(f"   Removed from student: {differences['sections_removed_for_student']}")
    print(f"   Teacher-only sections: {', '.join(differences['teacher_only_sections'])}")
    print()

    # Export both versions in all formats
    print("📤 Exporting worksheet in both versions (all formats)...")

    start_time = asyncio.get_event_loop().time()

    results = await export_service.export_worksheet_versions_all_formats(worksheet)

    end_time = asyncio.get_event_loop().time()
    total_time = end_time - start_time

    print(f"⏱️  Total export time: {total_time:.2f} seconds")
    print()

    # Display results by format
    print("📋 Export Results")
    print("=" * 50)

    total_files = 0
    total_size = 0

    for format_type, format_results in results.items():
        print(f"\n🗂️  {format_type.upper()} Format:")

        if "error" in format_results:
            print(f"   ❌ Error: {format_results['error']}")
            continue

        for version, file_path in format_results.items():
            if file_path.exists():
                size_kb = file_path.stat().st_size / 1024
                print(f"   {version.title():8} | {file_path.name:50} | {size_kb:6.1f} KB")
                total_files += 1
                total_size += file_path.stat().st_size

    print(f"\n📊 Summary: {total_files} files created, {total_size/1024:.1f} KB total")

    # Show content preview
    print("\n📝 Content Preview Comparison")
    print("-" * 50)

    # Convert to markdown for preview
    student_md = export_service._convert_document_to_markdown(student_version)
    teacher_md = export_service._convert_document_to_markdown(teacher_version)

    print(f"Student version: {len(student_md):,} characters")
    print(f"Teacher version: {len(teacher_md):,} characters")
    print(f"Teacher has {len(teacher_md) - len(student_md):,} more characters")

    # Show unique teacher content
    print("\n🍎 Teacher-Only Content:")
    teacher_lines = teacher_md.split("\n")
    unique_teacher_sections = []

    for line in teacher_lines:
        if any(
            keyword in line
            for keyword in ["## Answers", "## Detailed Solutions", "## Teaching Notes"]
        ):
            unique_teacher_sections.append(line.strip())

    for section in unique_teacher_sections[:3]:  # Show first 3
        print(f"   • {section}")

    # Show file access information
    print("\n🔍 Generated Files Location:")
    for format_type, format_results in results.items():
        if "error" not in format_results:
            student_path = format_results["student"]
            teacher_path = format_results["teacher"]
            if student_path.exists() and teacher_path.exists():
                print(f"\n{format_type.upper()}:")
                print(f"   Student: open '{student_path}'")
                print(f"   Teacher: open '{teacher_path}'")

    # Cleanup option
    print("\n🧹 Cleanup command:")
    all_files = []
    for format_results in results.values():
        if "error" not in format_results:
            all_files.extend([str(p) for p in format_results.values() if p.exists()])

    if all_files:
        quoted_files = [f"'{f}'" for f in all_files]
        print(f"   rm {' '.join(quoted_files)}")

    cleanup_input = input("\n🗑️  Delete generated files? (y/N): ").strip().lower()
    if cleanup_input == "y":
        deleted_count = 0
        for format_results in results.values():
            if "error" not in format_results:
                for file_path in format_results.values():
                    if file_path.exists():
                        file_path.unlink()
                        deleted_count += 1
        print(f"   Deleted {deleted_count} files")

    print("\n✨ Dual version export demo complete!")
    print("\n💡 Key Benefits:")
    print("   • Students get clean worksheets without answers")
    print("   • Teachers get complete materials with solutions")
    print("   • Automatic teaching notes and guidance")
    print("   • Multiple export formats (PDF, DOCX, HTML)")
    print("   • Fast batch export (~3 seconds for all formats)")


if __name__ == "__main__":
    try:
        asyncio.run(demo_dual_version_export())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except EOFError:
        print("\n✅ Demo completed (non-interactive mode)")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)

"""
Integration tests for dual version (student/teacher) document export.
Tests complete workflow from document creation to file export.
"""

import tempfile
from pathlib import Path

import pytest

from src.models.document_models import (
    DetailLevel,
    DocumentSection,
    DocumentStructure,
    DocumentType,
)
from src.services.document_export_service import DocumentExportService


class TestDualVersionExport:
    """Integration tests for student/teacher dual export functionality."""

    @pytest.fixture
    def export_service(self):
        """Create DocumentExportService instance."""
        return DocumentExportService()

    @pytest.fixture
    def sample_worksheet_base(self):
        """Create comprehensive worksheet for dual export testing."""
        return DocumentStructure(
            title="Quadratic Equations Practice",
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            estimated_duration=45,
            total_questions=4,
            sections=[
                DocumentSection(
                    title="Learning Objectives",
                    content_type="learning_objectives",
                    content_data={
                        "objectives_text": "• Solve quadratic equations by factoring\n• Use the quadratic formula\n• Verify solutions by substitution"
                    },
                    order_index=0,
                ),
                DocumentSection(
                    title="Key Formula",
                    content_type="theory_content",
                    content_data={
                        "theory_text": "**Quadratic Formula:** x = (-b ± √(b²-4ac)) / 2a\n\nFor equation ax² + bx + c = 0"
                    },
                    order_index=1,
                ),
                DocumentSection(
                    title="Worked Example",
                    content_type="worked_examples",
                    content_data={
                        "examples": [
                            {
                                "title": "Factoring Example",
                                "problem": "Solve: x² + 5x + 6 = 0",
                                "solution": "Factor: (x + 2)(x + 3) = 0\nSolutions: x = -2 or x = -3",
                            },
                            {
                                "title": "Quadratic Formula Example",
                                "problem": "Solve: 2x² - 3x - 1 = 0",
                                "solution": "x = (3 ± √(9 + 8)) / 4 = (3 ± √17) / 4",
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
                                "question_text": "Solve by factoring: x² - 4x + 3 = 0",
                                "marks": 3,
                                "command_word": "Solve",
                            },
                            {
                                "question_id": "q2",
                                "question_text": "Use the quadratic formula to solve: x² + 2x - 8 = 0",
                                "marks": 4,
                                "command_word": "Solve",
                            },
                            {
                                "question_id": "q3",
                                "question_text": "Find the solutions: 2x² - 5x + 2 = 0",
                                "marks": 4,
                                "command_word": "Find",
                            },
                            {
                                "question_id": "q4",
                                "question_text": "Verify that x = 2 is a solution to x² - 3x + 2 = 0",
                                "marks": 2,
                                "command_word": "Verify",
                            },
                        ],
                        "total_marks": 13,
                        "estimated_time": 30,
                    },
                    order_index=3,
                ),
                DocumentSection(
                    title="Answers",
                    content_type="answers",
                    content_data={
                        "answers": [
                            {"answer": "x = 1, x = 3"},
                            {"answer": "x = 2, x = -4"},
                            {"answer": "x = 2, x = 0.5"},
                            {"answer": "Verified: 4 - 6 + 2 = 0 ✓"},
                        ]
                    },
                    order_index=4,
                ),
                DocumentSection(
                    title="Detailed Solutions",
                    content_type="detailed_solutions",
                    content_data={
                        "solutions": [
                            {"solution": "x² - 4x + 3 = 0\n(x - 1)(x - 3) = 0\nx = 1 or x = 3"},
                            {
                                "solution": "x² + 2x - 8 = 0\na=1, b=2, c=-8\nx = (-2 ± √(4 + 32))/2 = (-2 ± 6)/2\nx = 2 or x = -4"
                            },
                            {
                                "solution": "2x² - 5x + 2 = 0\na=2, b=-5, c=2\nx = (5 ± √(25 - 16))/4 = (5 ± 3)/4\nx = 2 or x = 0.5"
                            },
                            {"solution": "Substitute x = 2:\n(2)² - 3(2) + 2 = 4 - 6 + 2 = 0 ✓"},
                        ]
                    },
                    order_index=5,
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_export_both_versions_pdf(self, export_service, sample_worksheet_base):
        """Test exporting both student and teacher versions as PDF."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            results = await export_service.export_worksheet_both_versions(
                sample_worksheet_base, format_type="pdf", output_dir=output_dir
            )

            # Verify both files were created
            assert "student" in results
            assert "teacher" in results

            student_pdf = results["student"]
            teacher_pdf = results["teacher"]

            # Check files exist and have content
            assert student_pdf.exists()
            assert teacher_pdf.exists()
            assert student_pdf.stat().st_size > 1000  # Should be substantial
            assert teacher_pdf.stat().st_size > 1000

            # Teacher version should be larger (more content)
            assert teacher_pdf.stat().st_size > student_pdf.stat().st_size

            # Check file names
            assert "Student" in student_pdf.name
            assert "Teacher" in teacher_pdf.name
            assert student_pdf.suffix == ".pdf"
            assert teacher_pdf.suffix == ".pdf"

    @pytest.mark.asyncio
    async def test_export_both_versions_docx(self, export_service, sample_worksheet_base):
        """Test exporting both versions as DOCX."""
        results = await export_service.export_worksheet_both_versions(
            sample_worksheet_base, format_type="docx"
        )

        try:
            student_docx = results["student"]
            teacher_docx = results["teacher"]

            # Verify DOCX files
            assert student_docx.exists()
            assert teacher_docx.exists()
            assert student_docx.suffix == ".docx"
            assert teacher_docx.suffix == ".docx"

            # Basic DOCX file verification (ZIP format)
            with open(student_docx, "rb") as f:
                assert f.read(4) == b"PK\x03\x04"

            with open(teacher_docx, "rb") as f:
                assert f.read(4) == b"PK\x03\x04"

        finally:
            # Cleanup
            for result_path in results.values():
                if result_path.exists():
                    result_path.unlink()

    @pytest.mark.asyncio
    async def test_export_both_versions_html(self, export_service, sample_worksheet_base):
        """Test exporting both versions as HTML."""
        results = await export_service.export_worksheet_both_versions(
            sample_worksheet_base, format_type="html"
        )

        try:
            student_html = results["student"]
            teacher_html = results["teacher"]

            # Verify HTML files
            assert student_html.exists()
            assert teacher_html.exists()
            assert student_html.suffix == ".html"
            assert teacher_html.suffix == ".html"

            # Check HTML content differences
            student_content = student_html.read_text()
            teacher_content = teacher_html.read_text()

            # Both should have basic structure
            assert "<!DOCTYPE html>" in student_content or "<html" in student_content
            assert "<!DOCTYPE html>" in teacher_content or "<html" in teacher_content

            # Both should have learning objectives and questions
            assert "Learning Objectives" in student_content
            assert "Learning Objectives" in teacher_content
            assert "Practice Questions" in student_content
            assert "Practice Questions" in teacher_content

            # Only teacher should have answers and solutions
            assert "Detailed Solutions" not in student_content
            assert "Detailed Solutions" in teacher_content
            assert "Teaching Notes" not in student_content
            assert "Teaching Notes" in teacher_content

            # Teacher version should be longer
            assert len(teacher_content) > len(student_content)

        finally:
            # Cleanup
            for result_path in results.values():
                if result_path.exists():
                    result_path.unlink()

    @pytest.mark.asyncio
    async def test_export_all_formats_both_versions(self, export_service, sample_worksheet_base):
        """Test exporting worksheet in all formats and both versions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            results = await export_service.export_worksheet_versions_all_formats(
                sample_worksheet_base, output_dir=output_dir
            )

            # Should have results for all formats
            expected_formats = ["pdf", "docx", "html"]
            for format_type in expected_formats:
                assert format_type in results

                if "error" not in results[format_type]:
                    format_results = results[format_type]

                    # Should have both versions
                    assert "student" in format_results
                    assert "teacher" in format_results

                    # Files should exist
                    assert format_results["student"].exists()
                    assert format_results["teacher"].exists()

                    # Should have correct extensions
                    assert format_results["student"].suffix == f".{format_type}"
                    assert format_results["teacher"].suffix == f".{format_type}"

    @pytest.mark.asyncio
    async def test_version_content_differences(self, export_service, sample_worksheet_base):
        """Test that content differences between versions are correct."""
        # Create both versions manually to inspect differences
        student_version = export_service.version_service.create_student_version(
            sample_worksheet_base
        )
        teacher_version = export_service.version_service.create_teacher_version(
            sample_worksheet_base
        )

        # Convert to markdown to inspect content
        student_markdown = export_service._convert_document_to_markdown(student_version)
        teacher_markdown = export_service._convert_document_to_markdown(teacher_version)

        # Student version checks
        assert "Student Copy" in student_markdown
        assert "Practice Questions" in student_markdown
        assert "Learning Objectives" in student_markdown
        assert "Worked Example" in student_markdown

        # Student should NOT have answers or solutions
        assert "## Answers" not in student_markdown
        assert "## Detailed Solutions" not in student_markdown
        assert "## Teaching Notes" not in student_markdown

        # Teacher version checks
        assert "Teacher Copy" in teacher_markdown
        assert "Practice Questions" in teacher_markdown
        assert "Learning Objectives" in teacher_markdown

        # Teacher SHOULD have answers and solutions
        assert "## Answers" in teacher_markdown
        assert "## Detailed Solutions" in teacher_markdown
        assert "## Teaching Notes" in teacher_markdown

        # Teacher should have more content
        assert len(teacher_markdown) > len(student_markdown)

    @pytest.mark.asyncio
    async def test_error_handling_non_worksheet(self, export_service):
        """Test error handling when trying to export dual versions for non-worksheets."""
        notes_document = DocumentStructure(
            title="Study Notes",
            document_type=DocumentType.NOTES,
            detail_level=DetailLevel.MEDIUM,
            sections=[
                DocumentSection(
                    title="Theory",
                    content_type="theory_content",
                    content_data={"theory_text": "Mathematical concepts..."},
                    order_index=0,
                )
            ],
        )

        # Should raise error for non-worksheet
        with pytest.raises(Exception) as exc_info:
            await export_service.export_worksheet_both_versions(notes_document)

        assert "worksheet" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_filename_generation(self, export_service, sample_worksheet_base):
        """Test that filenames are generated correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            results = await export_service.export_worksheet_both_versions(
                sample_worksheet_base, format_type="pdf", output_dir=output_dir
            )

            student_path = results["student"]
            teacher_path = results["teacher"]

            # Check filename patterns
            expected_base = "Quadratic_Equations_Practice"
            assert expected_base in student_path.name
            assert expected_base in teacher_path.name
            assert "Student" in student_path.name
            assert "Teacher" in teacher_path.name

    @pytest.mark.asyncio
    async def test_performance_dual_export(self, export_service, sample_worksheet_base):
        """Test performance of dual version export."""
        import time

        start_time = time.time()

        results = await export_service.export_worksheet_both_versions(
            sample_worksheet_base, format_type="pdf"
        )

        end_time = time.time()
        export_time = end_time - start_time

        try:
            # Should complete reasonably quickly (both versions in under 10 seconds)
            assert export_time < 10.0

            # Both files should be created
            assert len(results) == 2
            assert all(path.exists() for path in results.values())

            print(f"Dual export completed in {export_time:.2f} seconds")

        finally:
            # Cleanup
            for result_path in results.values():
                if result_path.exists():
                    result_path.unlink()


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

"""
Integration tests for DocumentExportService.
Tests complete document structure to file export pipeline.
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
from src.services.pandoc_service import PandocService


class TestDocumentExportServiceIntegration:
    """Integration tests for DocumentExportService with real file generation."""

    @pytest.fixture
    def pandoc_service(self):
        """Create PandocService instance."""
        return PandocService()

    @pytest.fixture
    def export_service(self, pandoc_service):
        """Create DocumentExportService instance."""
        return DocumentExportService(pandoc_service)

    @pytest.fixture
    def sample_document_structure(self):
        """Create sample DocumentStructure for testing."""
        return DocumentStructure(
            title="Algebra Practice Worksheet",
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            estimated_duration=30,
            total_questions=5,
            sections=[
                DocumentSection(
                    title="Learning Objectives",
                    content_type="learning_objectives",
                    content_data={
                        "objectives_text": "• Solve linear equations\n• Apply algebraic methods\n• Check solutions"
                    },
                    order_index=0,
                ),
                DocumentSection(
                    title="Practice Questions",
                    content_type="practice_questions",
                    content_data={
                        "questions": [
                            {
                                "question_id": "q1",
                                "question_text": "Solve for x: 2x + 3 = 11",
                                "marks": 2,
                                "command_word": "Solve",
                            },
                            {
                                "question_id": "q2",
                                "question_text": "Find the value of y when 3y - 5 = 7",
                                "marks": 3,
                                "command_word": "Find",
                            },
                        ],
                        "total_marks": 5,
                        "estimated_time": 15,
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
                                "problem": "Solve: x + 4 = 9",
                                "solution": "x + 4 = 9\nx = 9 - 4\nx = 5",
                            }
                        ]
                    },
                    order_index=2,
                ),
                DocumentSection(
                    title="Answers",
                    content_type="answers",
                    content_data={"answers": [{"answer": "x = 4"}, {"answer": "y = 4"}]},
                    order_index=3,
                ),
            ],
        )

    def test_convert_document_to_markdown(self, export_service, sample_document_structure):
        """Test conversion of DocumentStructure to markdown format."""
        markdown = export_service._convert_document_to_markdown(sample_document_structure)

        # Verify basic structure
        assert "# Algebra Practice Worksheet" in markdown
        assert "**Estimated Duration:** 30 minutes" in markdown
        assert "**Total Questions:** 5" in markdown

        # Verify sections are included
        assert "## Learning Objectives" in markdown
        assert "## Practice Questions" in markdown
        assert "## Worked Example" in markdown
        assert "## Answers" in markdown

        # Verify section content
        assert "• Solve linear equations" in markdown
        assert "**Question 1** (2 marks)" in markdown
        assert "Solve for x: 2x + 3 = 11" in markdown
        assert "**Question 2** (3 marks)" in markdown
        assert "Find the value of y when 3y - 5 = 7" in markdown

        # Verify worked example
        assert "### Example 1" in markdown
        assert "**Problem:** Solve: x + 4 = 9" in markdown
        assert "**Solution:** x + 4 = 9" in markdown

        # Verify answers
        assert "### Answers" in markdown
        assert "1. x = 4" in markdown
        assert "2. y = 4" in markdown

    @pytest.mark.asyncio
    async def test_export_to_pdf(self, export_service, sample_document_structure):
        """Test exporting DocumentStructure to PDF."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            output_path = Path(temp_pdf.name)

        try:
            result_path = await export_service.export_to_pdf(sample_document_structure, output_path)

            # Verify file was created
            assert result_path == output_path
            assert output_path.exists()
            assert output_path.stat().st_size > 0

            # Verify it's a PDF file
            with open(output_path, "rb") as f:
                header = f.read(4)
                assert header == b"%PDF"

        finally:
            if output_path.exists():
                output_path.unlink()

    @pytest.mark.asyncio
    async def test_export_to_docx(self, export_service, sample_document_structure):
        """Test exporting DocumentStructure to DOCX."""
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp_docx:
            output_path = Path(temp_docx.name)

        try:
            result_path = await export_service.export_to_docx(
                sample_document_structure, output_path
            )

            # Verify file was created
            assert result_path == output_path
            assert output_path.exists()
            assert output_path.stat().st_size > 0

            # Verify it's a DOCX file (ZIP format)
            with open(output_path, "rb") as f:
                header = f.read(4)
                assert header == b"PK\x03\x04"

        finally:
            if output_path.exists():
                output_path.unlink()

    @pytest.mark.asyncio
    async def test_export_to_html(self, export_service, sample_document_structure):
        """Test exporting DocumentStructure to HTML."""
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as temp_html:
            output_path = Path(temp_html.name)

        try:
            result_path = await export_service.export_to_html(
                sample_document_structure, output_path
            )

            # Verify file was created
            assert result_path == output_path
            assert output_path.exists()
            assert output_path.stat().st_size > 0

            # Verify HTML content
            content = output_path.read_text()
            assert "<!DOCTYPE html>" in content or "<html" in content
            assert "<h1" in content and "Algebra Practice Worksheet" in content
            assert "<h2" in content and "Learning Objectives" in content
            assert "<h2" in content and "Practice Questions" in content
            assert "Solve for x: 2x + 3 = 11" in content

        finally:
            if output_path.exists():
                output_path.unlink()

    @pytest.mark.asyncio
    async def test_export_with_temp_files(self, export_service, sample_document_structure):
        """Test export using temporary files (no output path specified)."""
        # Test PDF export
        pdf_path = await export_service.export_to_pdf(sample_document_structure)

        try:
            assert pdf_path.exists()
            assert pdf_path.suffix == ".pdf"
            assert pdf_path.stat().st_size > 0
        finally:
            pdf_path.unlink(missing_ok=True)

        # Test DOCX export
        docx_path = await export_service.export_to_docx(sample_document_structure)

        try:
            assert docx_path.exists()
            assert docx_path.suffix == ".docx"
            assert docx_path.stat().st_size > 0
        finally:
            docx_path.unlink(missing_ok=True)

        # Test HTML export
        html_path = await export_service.export_to_html(sample_document_structure)

        try:
            assert html_path.exists()
            assert html_path.suffix == ".html"
            assert html_path.stat().st_size > 0
        finally:
            html_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_export_different_document_types(self, export_service):
        """Test export with different document types and structures."""
        # Test Notes document
        notes_document = DocumentStructure(
            title="Quadratic Equations Study Notes",
            document_type=DocumentType.NOTES,
            detail_level=DetailLevel.COMPREHENSIVE,
            sections=[
                DocumentSection(
                    title="Theory Content",
                    content_type="theory_content",
                    content_data={
                        "theory_text": "Quadratic equations are polynomial equations of degree 2.\n\nGeneral form: ax² + bx + c = 0"
                    },
                    order_index=0,
                ),
                DocumentSection(
                    title="Key Formulas",
                    content_type="theory_content",
                    content_data={"theory_text": "Quadratic formula: x = (-b ± √(b²-4ac)) / 2a"},
                    order_index=1,
                ),
            ],
        )

        pdf_path = await export_service.export_to_pdf(notes_document)

        try:
            assert pdf_path.exists()
            assert pdf_path.stat().st_size > 0
        finally:
            pdf_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_export_error_handling(self, sample_document_structure):
        """Test error handling during export operations."""
        # Create export service with invalid pandoc path
        from src.services.pandoc_service import PandocError, PandocService

        # The error should occur during PandocService initialization
        with pytest.raises(PandocError):
            invalid_pandoc = PandocService(pandoc_path="/invalid/pandoc/path")

    def test_get_supported_formats(self, export_service):
        """Test getting supported export formats."""
        formats = export_service.get_supported_formats()

        assert isinstance(formats, dict)
        assert "pdf" in formats
        assert "docx" in formats
        assert "html" in formats

        # Verify descriptions
        assert formats["pdf"] == "Portable Document Format"
        assert formats["docx"] == "Microsoft Word Document"
        assert formats["html"] == "HyperText Markup Language"

    @pytest.mark.asyncio
    async def test_export_complex_document_with_all_sections(self, export_service):
        """Test export with a complex document containing all section types."""
        complex_document = DocumentStructure(
            title="Comprehensive Mathematics Worksheet",
            document_type=DocumentType.TEXTBOOK,
            detail_level=DetailLevel.COMPREHENSIVE,
            estimated_duration=90,
            total_questions=10,
            sections=[
                DocumentSection(
                    title="Learning Objectives",
                    content_type="learning_objectives",
                    content_data={
                        "objectives_text": "• Master quadratic equations\n• Apply factoring techniques\n• Solve real-world problems"
                    },
                    order_index=0,
                ),
                DocumentSection(
                    title="Theory Introduction",
                    content_type="theory_content",
                    content_data={
                        "theory_text": "Quadratic equations are fundamental in algebra. They appear in many real-world applications including physics, engineering, and finance."
                    },
                    order_index=1,
                ),
                DocumentSection(
                    title="Worked Examples",
                    content_type="worked_examples",
                    content_data={
                        "examples": [
                            {
                                "title": "Factoring Example",
                                "problem": "Factor: x² + 5x + 6",
                                "solution": "x² + 5x + 6 = (x + 2)(x + 3)",
                            },
                            {
                                "title": "Quadratic Formula Example",
                                "problem": "Solve: 2x² - 3x - 1 = 0",
                                "solution": "Using quadratic formula:\nx = (3 ± √(9 + 8)) / 4 = (3 ± √17) / 4",
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
                            {"question_text": "Factor: x² - 4x + 3", "marks": 2},
                            {"question_text": "Solve: x² - 6x + 9 = 0", "marks": 3},
                            {"question_text": "Find the vertex of y = x² - 4x + 1", "marks": 4},
                        ]
                    },
                    order_index=3,
                ),
                DocumentSection(
                    title="Detailed Solutions",
                    content_type="detailed_solutions",
                    content_data={
                        "solutions": [
                            {"solution": "x² - 4x + 3 = (x - 1)(x - 3)"},
                            {"solution": "x² - 6x + 9 = (x - 3)² = 0, so x = 3"},
                            {
                                "solution": "Complete the square: y = (x - 2)² - 3, vertex at (2, -3)"
                            },
                        ]
                    },
                    order_index=4,
                ),
            ],
        )

        # Test PDF export of complex document
        pdf_path = await export_service.export_to_pdf(complex_document)

        try:
            assert pdf_path.exists()
            assert pdf_path.stat().st_size > 5000  # Should be substantial
        finally:
            pdf_path.unlink(missing_ok=True)


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

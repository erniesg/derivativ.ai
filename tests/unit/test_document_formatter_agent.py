"""
Unit tests for DocumentFormatterAgent with pandoc integration.
Tests document formatting and conversion to various formats.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.document_formatter_agent import DocumentFormatterAgent
from src.models.document_models import (
    ContentSection,
    DetailLevel,
    DocumentType,
    ExportFormat,
    GeneratedDocument,
)
from src.models.enums import Tier
from src.services.llm_service import LLMService


class TestDocumentFormatterAgent:
    """Test DocumentFormatterAgent functionality."""


    @pytest.fixture
    def formatter_agent(self, mock_llm_service):
        """Create DocumentFormatterAgent instance."""
        return DocumentFormatterAgent(mock_llm_service)

    @pytest.fixture
    def sample_document(self):
        """Create sample document for testing."""
        from src.models.document_models import DocumentGenerationRequest

        sections = [
            ContentSection(
                title="Learning Objectives",
                content_type="learning_objectives",
                content_data={
                    "objectives_text": "• Understand quadratic equations\n• Apply factoring methods\n• Solve real-world problems"
                },
                order_index=0,
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
                            "question_text": "Factor: x² - 4x + 4",
                            "marks": 2,
                            "command_word": "Factor",
                        },
                    ],
                    "total_marks": 5,
                },
                order_index=1,
            ),
        ]

        request = DocumentGenerationRequest(
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            title="Algebra Practice",
            topic="quadratic_equations",
            tier=Tier.CORE,
        )

        return GeneratedDocument(
            title="Algebra Practice Worksheet",
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            generated_at="2025-06-21T12:00:00Z",
            template_used="worksheet_default",
            generation_request=request,
            sections=sections,
            total_questions=2,
            estimated_duration=15,
        )

    def test_html_formatting(self, formatter_agent, sample_document):
        """Test HTML formatting functionality."""
        html_output = formatter_agent._format_to_html(sample_document)

        assert "<html" in html_output  # Match both <html> and <html lang="en">
        assert "<title>Algebra Practice Worksheet</title>" in html_output
        assert "Practice Questions" in html_output
        assert "x² + 5x + 6 = 0" in html_output
        assert "[3 marks]" in html_output

    def test_markdown_formatting(self, formatter_agent, sample_document):
        """Test Markdown formatting functionality."""
        markdown_output = formatter_agent._format_to_markdown(sample_document)

        assert "# Algebra Practice Worksheet" in markdown_output
        assert "## Practice Questions" in markdown_output
        assert "**Question 1** (3 marks)" in markdown_output
        assert "x² + 5x + 6 = 0" in markdown_output

    def test_markdown_for_slides_formatting(self, formatter_agent, sample_document):
        """Test slide-optimized Markdown formatting."""
        slides_markdown = formatter_agent._format_to_markdown_for_slides(sample_document)

        assert "# Algebra Practice Worksheet" in slides_markdown
        assert "---" in slides_markdown  # Slide separators
        assert "## Document Information" in slides_markdown
        assert "## Learning Objectives" in slides_markdown
        assert "## Practice Questions" in slides_markdown
        assert "## Thank You" in slides_markdown

    def test_markdown_for_slides_with_personalization(self, formatter_agent, sample_document):
        """Test slide formatting with visual learner personalization."""
        personalization = {"learning_style": "visual"}
        slides_markdown = formatter_agent._format_to_markdown_for_slides(
            sample_document, personalization
        )

        assert "💡 *Consider drawing a diagram" in slides_markdown
        # Visual learners get fewer questions per slide
        assert slides_markdown.count("---") >= 4  # More slide breaks

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_pandoc_pdf_conversion_success(
        self, mock_subprocess, formatter_agent, sample_document
    ):
        """Test successful PDF conversion with pandoc."""
        # Mock successful pandoc execution
        mock_subprocess.return_value = MagicMock(returncode=0)

        with patch("tempfile.NamedTemporaryFile") as mock_tempfile, patch(
            "pathlib.Path.unlink"
        ) as mock_unlink:
            # Mock temporary file
            mock_file = MagicMock()
            mock_file.name = "/tmp/test_doc.md"
            mock_tempfile.return_value.__enter__.return_value = mock_file

            # Execute conversion
            result = await formatter_agent._format_with_pandoc(sample_document, ExportFormat.PDF)

            # Verify pandoc was called with correct arguments
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args[0][0]

            assert "pandoc" in call_args
            assert "--pdf-engine=pdflatex" in call_args
            assert "--variable" in call_args
            assert "geometry:margin=2.5cm" in call_args
            assert "--mathjax" in call_args

            # Should return output file path
            assert result.endswith(".pdf")

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_pandoc_docx_conversion_success(
        self, mock_subprocess, formatter_agent, sample_document
    ):
        """Test successful DOCX conversion with pandoc."""
        mock_subprocess.return_value = MagicMock(returncode=0)

        with patch("tempfile.NamedTemporaryFile") as mock_tempfile, patch(
            "pathlib.Path.unlink"
        ) as mock_unlink:
            mock_file = MagicMock()
            mock_file.name = "/tmp/test_doc.md"
            mock_tempfile.return_value.__enter__.return_value = mock_file

            result = await formatter_agent._format_with_pandoc(sample_document, ExportFormat.DOCX)

            # Verify DOCX-specific arguments
            call_args = mock_subprocess.call_args[0][0]
            assert "pandoc" in call_args
            assert result.endswith(".docx")

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_pandoc_slides_conversion_success(
        self, mock_subprocess, formatter_agent, sample_document
    ):
        """Test successful PowerPoint slides conversion with pandoc."""
        mock_subprocess.return_value = MagicMock(returncode=0)

        with patch("tempfile.NamedTemporaryFile") as mock_tempfile, patch(
            "pathlib.Path.unlink"
        ) as mock_unlink:
            mock_file = MagicMock()
            mock_file.name = "/tmp/test_doc.md"
            mock_tempfile.return_value.__enter__.return_value = mock_file

            result = await formatter_agent._format_with_pandoc(
                sample_document, ExportFormat.SLIDES_PPTX
            )

            # Verify slides-specific arguments
            call_args = mock_subprocess.call_args[0][0]
            assert "pandoc" in call_args
            assert "-t" in call_args
            assert "pptx" in call_args
            assert "--slide-level=2" in call_args
            assert result.endswith(".pptx")

    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_pandoc_conversion_failure(
        self, mock_subprocess, formatter_agent, sample_document
    ):
        """Test pandoc conversion failure handling."""
        # Mock pandoc failure
        mock_subprocess.side_effect = subprocess.CalledProcessError(
            1, "pandoc", stderr="pandoc: pdflatex not found"
        )

        with patch("tempfile.NamedTemporaryFile") as mock_tempfile, patch(
            "pathlib.Path.unlink"
        ) as mock_unlink:
            mock_file = MagicMock()
            mock_file.name = "/tmp/test_doc.md"
            mock_tempfile.return_value.__enter__.return_value = mock_file

            # Should raise ValueError with pandoc error
            with pytest.raises(ValueError, match="Document conversion failed"):
                await formatter_agent._format_with_pandoc(sample_document, ExportFormat.PDF)

            # Should clean up files on failure
            mock_unlink.assert_called()

    @pytest.mark.asyncio
    async def test_format_document_execution(self, formatter_agent, sample_document):
        """Test document formatting execution flow."""
        format_request = {"document": sample_document, "format": ExportFormat.HTML, "options": {}}

        result = await formatter_agent._execute(format_request)

        assert result["success"] is True
        assert "formatted_content" in result
        assert result["format"] == ExportFormat.HTML
        assert "<html" in result["formatted_content"]  # Match both <html> and <html lang="en">

    def test_personalization_extraction(self, formatter_agent, sample_document):
        """Test personalization settings extraction."""
        # Document with applied customizations
        sample_document.applied_customizations = {
            "personalization_context": {"learning_style": "visual", "font_size": "large"}
        }

        export_options = {"export_personalization": {"high_contrast": True}}

        personalization = formatter_agent._extract_personalization(sample_document, export_options)

        assert personalization["learning_style"] == "visual"
        assert personalization["font_size"] == "large"
        assert personalization["high_contrast"] is True

    def test_custom_css_generation(self, formatter_agent):
        """Test custom CSS generation for personalization."""
        personalization = {"learning_style": "visual", "font_size": "large", "high_contrast": True}

        css = formatter_agent._generate_custom_css(personalization)

        assert ".visual-enhanced" in css
        assert "font-size: 18px" in css
        assert "background-color: #000" in css  # High contrast

    @pytest.mark.asyncio
    async def test_unsupported_format_error(self, formatter_agent, sample_document):
        """Test error handling for unsupported formats."""
        format_request = {
            "document": sample_document,
            "format": "unsupported_format",
            "options": {},
        }

        result = await formatter_agent._execute(format_request)

        assert result["success"] is False
        assert "error" in result
        assert "Unsupported format" in result["error"]

    def test_parse_format_request_validation(self, formatter_agent, sample_document):
        """Test format request parsing and validation."""
        # Valid request
        valid_request = {
            "document": sample_document,
            "format": ExportFormat.PDF,
            "options": {"custom": "value"},
        }

        parsed = formatter_agent._parse_format_request(valid_request)
        assert parsed["document"] == sample_document
        assert parsed["format"] == ExportFormat.PDF
        assert parsed["options"]["custom"] == "value"

        # Missing document
        with pytest.raises(ValueError, match="Missing required field: document"):
            formatter_agent._parse_format_request({"format": ExportFormat.PDF})

        # Invalid format
        with pytest.raises(ValueError, match="Unsupported format"):
            formatter_agent._parse_format_request(
                {"document": sample_document, "format": "invalid_format"}
            )


# Integration test requiring actual pandoc installation
@pytest.mark.integration
@pytest.mark.skipif(
    not Path("/usr/bin/pandoc").exists() and not Path("/usr/local/bin/pandoc").exists(),
    reason="pandoc not installed",
)
class TestDocumentFormatterIntegration:
    """Integration tests with real pandoc (requires pandoc installation)."""

    @pytest.fixture
    def formatter_agent(self):
        """Create real DocumentFormatterAgent."""
        llm_service = MagicMock(spec=LLMService)
        return DocumentFormatterAgent(llm_service)

    @pytest.fixture
    def sample_document(self):
        """Create sample document for integration testing."""
        # Simpler document for integration testing
        from src.models.document_models import DocumentGenerationRequest

        sections = [
            ContentSection(
                title="Introduction",
                content_type="introduction",
                content_data={"text": "This is a test document for pandoc integration."},
                order_index=0,
            )
        ]

        request = DocumentGenerationRequest(
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MINIMAL,
            title="Integration Test",
            topic="testing",
        )

        return GeneratedDocument(
            title="Integration Test Document",
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MINIMAL,
            generated_at="2025-06-21T12:00:00Z",
            template_used="worksheet_default",
            generation_request=request,
            sections=sections,
            total_questions=0,
            estimated_duration=5,
        )

    @pytest.mark.asyncio
    async def test_real_pandoc_html_conversion(self, formatter_agent, sample_document):
        """Test real pandoc HTML conversion."""
        try:
            output_file = await formatter_agent._format_with_pandoc(
                sample_document, ExportFormat.HTML
            )

            assert Path(output_file).exists()
            assert output_file.endswith(".html")

            # Read and verify content
            with open(output_file) as f:
                content = f.read()
                assert "Integration Test Document" in content
                assert "<html>" in content

            # Clean up
            Path(output_file).unlink()

        except FileNotFoundError:
            pytest.skip("pandoc not found in PATH")
        except Exception as e:
            pytest.fail(f"Real pandoc conversion failed: {e}")

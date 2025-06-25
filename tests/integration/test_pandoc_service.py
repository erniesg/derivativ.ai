"""
Integration tests for PandocService.
Tests actual pandoc binary integration and file conversion.
"""

import tempfile
from pathlib import Path

import pytest

from src.services.pandoc_service import PandocError, PandocService


class TestPandocServiceIntegration:
    """Integration tests for PandocService with real pandoc binary."""

    @pytest.fixture
    def pandoc_service(self):
        """Create PandocService instance."""
        return PandocService()

    @pytest.fixture
    def sample_markdown(self):
        """Sample markdown content for testing."""
        return """# Test Document

This is a test document for pandoc conversion.

## Section 1

Some content with **bold** and *italic* text.

### Subsection

- Bullet point 1
- Bullet point 2
- Bullet point 3

## Section 2

A table:

| Name | Age | City |
|------|-----|------|
| John | 25  | NYC  |
| Jane | 30  | LA   |

Some math: $E = mc^2$

"""

    @pytest.mark.asyncio
    async def test_pandoc_installation_verification(self, pandoc_service):
        """Test that pandoc is properly installed and accessible."""
        # PandocService constructor should verify installation
        assert pandoc_service.pandoc_path == "pandoc"

        # Should not raise an exception
        result = await pandoc_service._run_pandoc_command(["--version"])
        assert "pandoc" in result.lower()

    @pytest.mark.asyncio
    async def test_convert_markdown_to_pdf(self, pandoc_service, sample_markdown):
        """Test converting markdown to PDF."""
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            output_path = Path(temp_pdf.name)

        try:
            # Convert to PDF
            result_path = await pandoc_service.convert_markdown_to_pdf(sample_markdown, output_path)

            # Verify file was created
            assert result_path == output_path
            assert output_path.exists()
            assert output_path.stat().st_size > 0

            # Verify it's a PDF file (basic check)
            with open(output_path, "rb") as f:
                header = f.read(4)
                assert header == b"%PDF"  # PDF file signature

        finally:
            # Cleanup
            if output_path.exists():
                output_path.unlink()

    @pytest.mark.asyncio
    async def test_convert_markdown_to_pdf_with_template_options(
        self, pandoc_service, sample_markdown
    ):
        """Test PDF conversion with template options."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            output_path = Path(temp_pdf.name)

        try:
            template_options = {
                "title": "Test Document Title",
                "author": "Test Author",
                "fontsize": "14pt",
            }

            result_path = await pandoc_service.convert_markdown_to_pdf(
                sample_markdown, output_path, template_options
            )

            assert result_path.exists()
            assert result_path.stat().st_size > 0

        finally:
            if output_path.exists():
                output_path.unlink()

    @pytest.mark.asyncio
    async def test_convert_markdown_to_docx(self, pandoc_service, sample_markdown):
        """Test converting markdown to DOCX."""
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp_docx:
            output_path = Path(temp_docx.name)

        try:
            result_path = await pandoc_service.convert_markdown_to_docx(
                sample_markdown, output_path
            )

            assert result_path == output_path
            assert output_path.exists()
            assert output_path.stat().st_size > 0

            # Basic DOCX file verification (ZIP format)
            with open(output_path, "rb") as f:
                header = f.read(4)
                assert header == b"PK\x03\x04"  # ZIP file signature

        finally:
            if output_path.exists():
                output_path.unlink()

    @pytest.mark.asyncio
    async def test_convert_markdown_to_html(self, pandoc_service, sample_markdown):
        """Test converting markdown to HTML."""
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as temp_html:
            output_path = Path(temp_html.name)

        try:
            result_path = await pandoc_service.convert_markdown_to_html(
                sample_markdown, output_path
            )

            assert result_path == output_path
            assert output_path.exists()
            assert output_path.stat().st_size > 0

            # Verify HTML content
            content = output_path.read_text()
            assert "<!DOCTYPE html>" in content or "<html" in content
            assert "<h1" in content and "Test Document" in content
            assert "<h2" in content and "Section 1" in content
            assert "<table>" in content  # Table should be converted

        finally:
            if output_path.exists():
                output_path.unlink()

    @pytest.mark.asyncio
    async def test_convert_with_temp_files(self, pandoc_service, sample_markdown):
        """Test conversion using temporary files (no output path specified)."""
        # PDF conversion
        pdf_path = await pandoc_service.convert_markdown_to_pdf(sample_markdown)

        try:
            assert pdf_path.exists()
            assert pdf_path.suffix == ".pdf"
            assert pdf_path.stat().st_size > 0
        finally:
            pdf_path.unlink(missing_ok=True)

        # DOCX conversion
        docx_path = await pandoc_service.convert_markdown_to_docx(sample_markdown)

        try:
            assert docx_path.exists()
            assert docx_path.suffix == ".docx"
            assert docx_path.stat().st_size > 0
        finally:
            docx_path.unlink(missing_ok=True)

        # HTML conversion
        html_path = await pandoc_service.convert_markdown_to_html(sample_markdown)

        try:
            assert html_path.exists()
            assert html_path.suffix == ".html"
            assert html_path.stat().st_size > 0
        finally:
            html_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_pandoc_error_handling(self, pandoc_service):
        """Test error handling for invalid pandoc operations."""
        # Test with invalid output path
        with pytest.raises(PandocError):
            await pandoc_service.convert_markdown_to_pdf(
                "# Test", Path("/invalid/path/that/does/not/exist.pdf")
            )

    @pytest.mark.asyncio
    async def test_cleanup_temp_files(self, pandoc_service):
        """Test temporary file cleanup functionality."""
        # Create some temporary files
        temp_files = []
        for i in range(3):
            temp_file = Path(tempfile.mktemp(suffix=f".test{i}"))
            temp_file.write_text(f"Test content {i}")
            temp_files.append(temp_file)

        # Verify files exist
        for temp_file in temp_files:
            assert temp_file.exists()

        # Cleanup
        await pandoc_service.cleanup_temp_files(temp_files)

        # Verify files are gone
        for temp_file in temp_files:
            assert not temp_file.exists()

    def test_get_supported_formats(self, pandoc_service):
        """Test getting list of supported formats."""
        formats = pandoc_service.get_supported_formats()

        assert isinstance(formats, list)
        assert "pdf" in formats
        assert "docx" in formats
        assert "html" in formats

    @pytest.mark.asyncio
    async def test_pandoc_with_math_content(self, pandoc_service):
        """Test pandoc conversion with mathematical content."""
        math_markdown = """# Mathematics Test

## Inline Math
The equation $E = mc^2$ is famous.

## Block Math
$$\\int_{0}^{\\infty} e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}$$

## More Examples
- Quadratic formula: $x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$
- Pythagorean theorem: $a^2 + b^2 = c^2$
"""

        # Test PDF conversion with math
        pdf_path = await pandoc_service.convert_markdown_to_pdf(math_markdown)

        try:
            assert pdf_path.exists()
            assert pdf_path.stat().st_size > 0
        finally:
            pdf_path.unlink(missing_ok=True)


# Test configuration
@pytest.mark.integration
class TestPandocPerformance:
    """Performance tests for pandoc conversions."""

    @pytest.fixture
    def large_markdown(self):
        """Generate large markdown content for performance testing."""
        content = ["# Large Document Test\n"]

        for i in range(100):
            content.append(f"\n## Section {i+1}\n")
            content.append(f"This is section {i+1} with some content. " * 20)
            content.append("\n")

            # Add a table every 10 sections
            if (i + 1) % 10 == 0:
                content.append("\n| Col 1 | Col 2 | Col 3 |\n")
                content.append("|-------|-------|-------|\n")
                for row in range(5):
                    content.append(f"| Data {row} | Value {row} | Result {row} |\n")

        return "".join(content)

    @pytest.mark.asyncio
    async def test_large_document_pdf_conversion(self, large_markdown):
        """Test PDF conversion performance with large document."""
        import time

        pandoc_service = PandocService()

        start_time = time.time()
        pdf_path = await pandoc_service.convert_markdown_to_pdf(large_markdown)
        end_time = time.time()

        try:
            # Verify conversion completed
            assert pdf_path.exists()
            assert pdf_path.stat().st_size > 10000  # Should be a substantial file

            # Performance assertion (should complete within reasonable time)
            conversion_time = end_time - start_time
            assert conversion_time < 30.0  # Should complete within 30 seconds

            print(f"Large document PDF conversion took: {conversion_time:.2f} seconds")

        finally:
            pdf_path.unlink(missing_ok=True)


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

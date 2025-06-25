"""
Pandoc service for converting documents to various formats.
Handles markdown to PDF/DOCX/HTML conversion using pandoc binary.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PandocError(Exception):
    """Raised when pandoc conversion fails"""

    pass


class PandocService:
    """Service for document conversion using pandoc binary."""

    def __init__(self, pandoc_path: str = "pandoc"):
        """
        Initialize pandoc service.

        Args:
            pandoc_path: Path to pandoc binary
        """
        self.pandoc_path = pandoc_path
        self._verify_pandoc_installation()

    def _verify_pandoc_installation(self) -> None:
        """Verify pandoc is installed and accessible."""
        try:
            # Use subprocess directly for sync verification
            import subprocess

            result = subprocess.run(
                [self.pandoc_path, "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                raise PandocError(f"Pandoc command failed: {result.stderr}")
            logger.info(f"Pandoc available: {result.stdout[:50]}...")
        except subprocess.TimeoutExpired:
            raise PandocError("Pandoc command timed out")
        except FileNotFoundError:
            raise PandocError(f"Pandoc binary not found at: {self.pandoc_path}")
        except Exception as e:
            raise PandocError(f"Pandoc not available: {e}")

    async def _run_pandoc_command(self, args: list[str]) -> str:
        """
        Run pandoc command asynchronously.

        Args:
            args: Command line arguments for pandoc

        Returns:
            Command output

        Raises:
            PandocError: If command fails
        """
        cmd = [self.pandoc_path] + args

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown pandoc error"
                raise PandocError(f"Pandoc command failed: {error_msg}")

            return stdout.decode()

        except FileNotFoundError:
            raise PandocError(f"Pandoc binary not found at: {self.pandoc_path}")
        except Exception as e:
            raise PandocError(f"Failed to run pandoc: {e}")

    async def convert_markdown_to_pdf(
        self,
        markdown_content: str,
        output_path: Optional[Path] = None,
        template_options: Optional[dict[str, str]] = None,
    ) -> Path:
        """
        Convert markdown content to PDF.

        Args:
            markdown_content: Markdown text to convert
            output_path: Output file path (temp file if None)
            template_options: PDF template options

        Returns:
            Path to generated PDF file
        """
        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=".pdf"))

        # Create temporary markdown file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as temp_md:
            temp_md.write(markdown_content)
            temp_md_path = Path(temp_md.name)

        try:
            # Build pandoc command with Unicode support
            args = [
                str(temp_md_path),
                "-o",
                str(output_path),
                "--pdf-engine=xelatex",  # Better Unicode support than pdflatex
                "-V",
                "geometry:margin=1in",
            ]

            # Add template options
            if template_options:
                for key, value in template_options.items():
                    args.extend(["-V", f"{key}={value}"])

            # Run conversion
            await self._run_pandoc_command(args)

            if not output_path.exists():
                raise PandocError("PDF file was not created")

            logger.info(f"Successfully converted markdown to PDF: {output_path}")
            return output_path

        finally:
            # Cleanup temporary markdown file
            temp_md_path.unlink(missing_ok=True)

    async def convert_markdown_to_docx(
        self, markdown_content: str, output_path: Optional[Path] = None
    ) -> Path:
        """
        Convert markdown content to DOCX.

        Args:
            markdown_content: Markdown text to convert
            output_path: Output file path (temp file if None)

        Returns:
            Path to generated DOCX file
        """
        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=".docx"))

        # Create temporary markdown file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as temp_md:
            temp_md.write(markdown_content)
            temp_md_path = Path(temp_md.name)

        try:
            # Build pandoc command
            args = [str(temp_md_path), "-o", str(output_path)]

            # Run conversion
            await self._run_pandoc_command(args)

            if not output_path.exists():
                raise PandocError("DOCX file was not created")

            logger.info(f"Successfully converted markdown to DOCX: {output_path}")
            return output_path

        finally:
            # Cleanup temporary markdown file
            temp_md_path.unlink(missing_ok=True)

    async def convert_markdown_to_html(
        self,
        markdown_content: str,
        output_path: Optional[Path] = None,
        css_file: Optional[Path] = None,
    ) -> Path:
        """
        Convert markdown content to HTML.

        Args:
            markdown_content: Markdown text to convert
            output_path: Output file path (temp file if None)
            css_file: Optional CSS file for styling

        Returns:
            Path to generated HTML file
        """
        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=".html"))

        # Create temporary markdown file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as temp_md:
            temp_md.write(markdown_content)
            temp_md_path = Path(temp_md.name)

        try:
            # Build pandoc command
            args = [str(temp_md_path), "-o", str(output_path), "--standalone"]

            # Add CSS if provided
            if css_file and css_file.exists():
                args.extend(["--css", str(css_file)])

            # Run conversion
            await self._run_pandoc_command(args)

            if not output_path.exists():
                raise PandocError("HTML file was not created")

            logger.info(f"Successfully converted markdown to HTML: {output_path}")
            return output_path

        finally:
            # Cleanup temporary markdown file
            temp_md_path.unlink(missing_ok=True)

    def get_supported_formats(self) -> list[str]:
        """Get list of supported output formats."""
        return ["pdf", "docx", "html"]

    async def cleanup_temp_files(self, file_paths: list[Path]) -> None:
        """
        Clean up temporary files.

        Args:
            file_paths: List of file paths to remove
        """
        for path in file_paths:
            try:
                if path.exists():
                    path.unlink()
                    logger.debug(f"Cleaned up temp file: {path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {path}: {e}")

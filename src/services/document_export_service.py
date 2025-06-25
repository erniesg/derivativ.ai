"""
Document export service that converts DocumentStructure to various formats.
Integrates with PandocService for file generation.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from src.models.document_models import DocumentStructure, DocumentType
from src.services.document_version_service import DocumentVersionService
from src.services.pandoc_service import PandocError, PandocService

logger = logging.getLogger(__name__)


class DocumentExportError(Exception):
    """Raised when document export fails"""

    pass


class DocumentExportService:
    """Service for exporting DocumentStructure to various file formats."""

    def __init__(self, pandoc_service: Optional[PandocService] = None):
        """
        Initialize document export service.

        Args:
            pandoc_service: PandocService instance (creates new if None)
        """
        self.pandoc_service = pandoc_service or PandocService()
        self.version_service = DocumentVersionService()

    def _convert_document_to_markdown(self, document: DocumentStructure) -> str:
        """
        Convert DocumentStructure to markdown format.

        Args:
            document: Document structure to convert

        Returns:
            Markdown content string
        """
        markdown_parts = []

        # Document title
        markdown_parts.append(f"# {document.title}\n")

        # Document metadata
        if document.estimated_duration:
            markdown_parts.append(
                f"**Estimated Duration:** {document.estimated_duration} minutes\n"
            )

        if document.total_questions and document.total_questions > 0:
            markdown_parts.append(f"**Total Questions:** {document.total_questions}\n")

        markdown_parts.append("\n---\n")

        # Process sections in order
        sorted_sections = sorted(document.sections, key=lambda s: s.order_index)

        for section in sorted_sections:
            # Section title
            markdown_parts.append(f"\n## {section.title}\n")

            # Section content based on type
            content_data = section.content_data

            if section.content_type == "learning_objectives":
                objectives = content_data.get("objectives_text", "")
                markdown_parts.append(f"{objectives}\n")

            elif section.content_type == "practice_questions":
                questions = content_data.get("questions", [])
                if questions:
                    for i, question in enumerate(questions, 1):
                        question_text = question.get("question_text", "")
                        marks = question.get("marks", "")
                        markdown_parts.append(f"\n**Question {i}** ({marks} marks)\n")
                        markdown_parts.append(f"{question_text}\n")
                else:
                    markdown_parts.append("*Questions to be added*\n")

            elif section.content_type == "worked_examples":
                examples = content_data.get("examples", [])
                for example in examples:
                    title = example.get("title", "Example")
                    problem = example.get("problem", "")
                    solution = example.get("solution", "")
                    markdown_parts.append(f"\n### {title}\n")
                    markdown_parts.append(f"**Problem:** {problem}\n")
                    markdown_parts.append(f"**Solution:** {solution}\n")

            elif section.content_type == "theory_content":
                theory_text = content_data.get("theory_text", "")
                markdown_parts.append(f"{theory_text}\n")

            elif section.content_type == "answers":
                answers = content_data.get("answers", [])
                if answers:
                    markdown_parts.append("\n### Answers\n")
                    for i, answer in enumerate(answers, 1):
                        answer_text = answer.get("answer", "")
                        markdown_parts.append(f"{i}. {answer_text}\n")

            elif section.content_type == "detailed_solutions":
                solutions = content_data.get("solutions", [])
                if solutions:
                    markdown_parts.append("\n### Detailed Solutions\n")
                    for i, solution in enumerate(solutions, 1):
                        solution_text = solution.get("solution", "")
                        markdown_parts.append(f"\n**Solution {i}:**\n{solution_text}\n")

            else:
                # Generic content handling
                text_content = content_data.get("text", content_data.get("content", ""))
                if text_content:
                    markdown_parts.append(f"{text_content}\n")

        return "\n".join(markdown_parts)

    async def export_to_pdf(
        self, document: DocumentStructure, output_path: Optional[Path] = None
    ) -> Path:
        """
        Export document to PDF format.

        Args:
            document: Document structure to export
            output_path: Output file path (temp file if None)

        Returns:
            Path to generated PDF file

        Raises:
            DocumentExportError: If export fails
        """
        try:
            # Convert to markdown
            markdown_content = self._convert_document_to_markdown(document)

            # Set up PDF template options
            template_options = {
                "documentclass": "article",
                "fontsize": "12pt",
                "title": document.title,
            }

            # Convert to PDF using pandoc
            pdf_path = await self.pandoc_service.convert_markdown_to_pdf(
                markdown_content, output_path, template_options
            )

            logger.info(f"Successfully exported document '{document.title}' to PDF: {pdf_path}")
            return pdf_path

        except PandocError as e:
            raise DocumentExportError(f"PDF export failed: {e}")
        except Exception as e:
            raise DocumentExportError(f"Unexpected error during PDF export: {e}")

    async def export_to_docx(
        self, document: DocumentStructure, output_path: Optional[Path] = None
    ) -> Path:
        """
        Export document to DOCX format.

        Args:
            document: Document structure to export
            output_path: Output file path (temp file if None)

        Returns:
            Path to generated DOCX file

        Raises:
            DocumentExportError: If export fails
        """
        try:
            # Convert to markdown
            markdown_content = self._convert_document_to_markdown(document)

            # Convert to DOCX using pandoc
            docx_path = await self.pandoc_service.convert_markdown_to_docx(
                markdown_content, output_path
            )

            logger.info(f"Successfully exported document '{document.title}' to DOCX: {docx_path}")
            return docx_path

        except PandocError as e:
            raise DocumentExportError(f"DOCX export failed: {e}")
        except Exception as e:
            raise DocumentExportError(f"Unexpected error during DOCX export: {e}")

    async def export_to_html(
        self,
        document: DocumentStructure,
        output_path: Optional[Path] = None,
        css_file: Optional[Path] = None,
    ) -> Path:
        """
        Export document to HTML format.

        Args:
            document: Document structure to export
            output_path: Output file path (temp file if None)
            css_file: Optional CSS file for styling

        Returns:
            Path to generated HTML file

        Raises:
            DocumentExportError: If export fails
        """
        try:
            # Convert to markdown
            markdown_content = self._convert_document_to_markdown(document)

            # Convert to HTML using pandoc
            html_path = await self.pandoc_service.convert_markdown_to_html(
                markdown_content, output_path, css_file
            )

            logger.info(f"Successfully exported document '{document.title}' to HTML: {html_path}")
            return html_path

        except PandocError as e:
            raise DocumentExportError(f"HTML export failed: {e}")
        except Exception as e:
            raise DocumentExportError(f"Unexpected error during HTML export: {e}")

    def get_supported_formats(self) -> dict[str, str]:
        """
        Get supported export formats.

        Returns:
            Dictionary mapping format names to descriptions
        """
        return {
            "pdf": "Portable Document Format",
            "docx": "Microsoft Word Document",
            "html": "HyperText Markup Language",
        }

    async def export_worksheet_both_versions(
        self,
        base_document: DocumentStructure,
        format_type: str = "pdf",
        output_dir: Optional[Path] = None,
    ) -> dict[str, Path]:
        """
        Export worksheet in both student and teacher versions.

        Args:
            base_document: Base document structure
            format_type: Export format (pdf, docx, html)
            output_dir: Output directory (temp if None)

        Returns:
            Dictionary with student and teacher file paths

        Raises:
            DocumentExportError: If export fails
        """
        if base_document.document_type != DocumentType.WORKSHEET:
            raise DocumentExportError("Both versions export only supported for worksheets")

        logger.info(f"Exporting worksheet '{base_document.title}' in both versions ({format_type})")

        try:
            # Create both versions
            student_version, teacher_version = self.version_service.create_both_versions(
                base_document
            )

            # Set up output paths
            if output_dir is None:
                output_dir = Path(tempfile.mkdtemp())

            base_filename = base_document.title.replace(" ", "_").replace("-", "_")
            student_filename = f"{base_filename}_Student.{format_type}"
            teacher_filename = f"{base_filename}_Teacher.{format_type}"

            student_path = output_dir / student_filename
            teacher_path = output_dir / teacher_filename

            # Export both versions
            results = {}

            # Export student version
            logger.info("Exporting student version...")
            if format_type == "pdf":
                results["student"] = await self.export_to_pdf(student_version, student_path)
            elif format_type == "docx":
                results["student"] = await self.export_to_docx(student_version, student_path)
            elif format_type == "html":
                results["student"] = await self.export_to_html(student_version, student_path)
            else:
                raise DocumentExportError(f"Unsupported format: {format_type}")

            # Export teacher version
            logger.info("Exporting teacher version...")
            if format_type == "pdf":
                results["teacher"] = await self.export_to_pdf(teacher_version, teacher_path)
            elif format_type == "docx":
                results["teacher"] = await self.export_to_docx(teacher_version, teacher_path)
            elif format_type == "html":
                results["teacher"] = await self.export_to_html(teacher_version, teacher_path)

            # Log success
            student_size = results["student"].stat().st_size
            teacher_size = results["teacher"].stat().st_size

            logger.info("Successfully exported both versions:")
            logger.info(f"  Student: {results['student'].name} ({student_size:,} bytes)")
            logger.info(f"  Teacher: {results['teacher'].name} ({teacher_size:,} bytes)")

            return results

        except Exception as e:
            raise DocumentExportError(f"Failed to export both versions: {e}")

    async def export_worksheet_versions_all_formats(
        self, base_document: DocumentStructure, output_dir: Optional[Path] = None
    ) -> dict[str, dict[str, Path]]:
        """
        Export worksheet in both versions across all formats.

        Args:
            base_document: Base document structure
            output_dir: Output directory (temp if None)

        Returns:
            Nested dictionary: {format: {version: path}}
        """
        if base_document.document_type != DocumentType.WORKSHEET:
            raise DocumentExportError("Multi-format export only supported for worksheets")

        logger.info(f"Exporting '{base_document.title}' in all formats and versions")

        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp())

        results = {}
        formats = ["pdf", "docx", "html"]

        for format_type in formats:
            try:
                format_results = await self.export_worksheet_both_versions(
                    base_document, format_type, output_dir
                )
                results[format_type] = format_results
                logger.info(f"✅ {format_type.upper()} exports completed")

            except Exception as e:
                logger.error(f"❌ {format_type.upper()} export failed: {e}")
                results[format_type] = {"error": str(e)}

        return results

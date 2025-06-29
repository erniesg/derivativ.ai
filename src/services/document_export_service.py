"""
Document Export Service

Handles exporting generated documents to various formats (PDF, DOCX) with
student and teacher versions.
"""

import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

logger = logging.getLogger(__name__)


class DocumentExportService:
    """Service for exporting documents to various formats."""

    def __init__(self, output_directory: Optional[str] = None):
        """Initialize the export service.

        Args:
            output_directory: Directory to save exported files. Defaults to temp directory.
        """
        self.output_directory = Path(output_directory or tempfile.gettempdir())
        self.output_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ReportLab styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom styles for PDF generation."""
        # Title style
        self.styles.add(ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.darkblue,
            alignment=1  # Center alignment
        ))

        # Section header style
        self.styles.add(ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=10,
            textColor=colors.darkgreen,
            borderWidth=1,
            borderColor=colors.lightgrey,
            borderPadding=5,
            backColor=colors.lightgrey
        ))

        # Question style
        self.styles.add(ParagraphStyle(
            'Question',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceBefore=10,
            spaceAfter=5,
            leftIndent=20
        ))

        # Answer style
        self.styles.add(ParagraphStyle(
            'Answer',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceBefore=5,
            spaceAfter=10,
            leftIndent=40,
            textColor=colors.blue,
            fontName='Helvetica-Oblique'
        ))

    async def export_document(
        self,
        document: dict[str, Any],
        formats: list[str],
        create_versions: bool = True,
        output_prefix: str = None
    ) -> dict[str, list[str]]:
        """Export a document to specified formats.

        Args:
            document: Generated document data
            formats: List of formats to export to ('pdf', 'docx')
            create_versions: Whether to create separate student/teacher versions
            output_prefix: Prefix for output files

        Returns:
            Dictionary mapping format to list of generated file paths
        """
        if not output_prefix:
            # Generate prefix from document title and timestamp
            title = document.get('title', 'document').replace(' ', '_').lower()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_prefix = f"{title}_{timestamp}"

        exported_files = {}

        for format_type in formats:
            if format_type.lower() == 'pdf':
                exported_files['pdf'] = await self._export_pdf(
                    document, output_prefix, create_versions
                )
            elif format_type.lower() == 'docx':
                exported_files['docx'] = await self._export_docx(
                    document, output_prefix, create_versions
                )
            else:
                logger.warning(f"Unsupported export format: {format_type}")

        return exported_files

    async def _export_pdf(
        self,
        document: dict[str, Any],
        output_prefix: str,
        create_versions: bool
    ) -> list[str]:
        """Export document to PDF format."""
        exported_files = []

        if create_versions:
            # Create student version (no answers)
            student_file = self.output_directory / f"{output_prefix}_student.pdf"
            await self._create_pdf(document, student_file, version='student')
            exported_files.append(str(student_file))

            # Create teacher version (with answers)
            teacher_file = self.output_directory / f"{output_prefix}_teacher.pdf"
            await self._create_pdf(document, teacher_file, version='teacher')
            exported_files.append(str(teacher_file))
        else:
            # Create combined version
            combined_file = self.output_directory / f"{output_prefix}_combined.pdf"
            await self._create_pdf(document, combined_file, version='combined')
            exported_files.append(str(combined_file))

        return exported_files

    async def _create_pdf(
        self,
        document: dict[str, Any],
        output_path: Path,
        version: str
    ):
        """Create a PDF file for the document."""
        # Create PDF document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=1*inch
        )

        # Build content
        story = []

        # Add title
        title = document.get('title', 'Generated Document')
        if version == 'teacher':
            title += " (Teacher Version)"
        elif version == 'student':
            title += " (Student Version)"

        story.append(Paragraph(title, self.styles['CustomTitle']))
        story.append(Spacer(1, 20))

        # Add document metadata
        metadata = self._extract_document_metadata(document)
        if metadata:
            story.append(Paragraph("Document Information", self.styles['SectionHeader']))
            for key, value in metadata.items():
                story.append(Paragraph(f"<b>{key}:</b> {value}", self.styles['Normal']))
            story.append(Spacer(1, 15))

        # Add content blocks
        content_structure = document.get('content_structure', {})
        blocks = content_structure.get('blocks', [])

        for block in blocks:
            await self._add_block_to_pdf(story, block, version)

        # Build PDF
        doc.build(story)
        logger.info(f"Created PDF: {output_path}")

    async def _add_block_to_pdf(
        self,
        story: list,
        block: dict[str, Any],
        version: str
    ):
        """Add a content block to the PDF story."""
        block_type = block.get('block_type', '')
        content = block.get('content', {})

        # Add block header
        block_title = self._format_block_title(block_type)
        story.append(Paragraph(block_title, self.styles['SectionHeader']))

        if block_type == 'learning_objectives':
            objectives = content.get('objectives', [])
            for i, objective in enumerate(objectives, 1):
                story.append(Paragraph(f"{i}. {objective}", self.styles['Normal']))
            story.append(Spacer(1, 10))

        elif block_type == 'concept_explanation':
            concepts = content.get('concepts', [])
            for concept in concepts:
                name = concept.get('name', '')
                explanation = concept.get('explanation', '')
                story.append(Paragraph(f"<b>{name}</b>", self.styles['Normal']))
                story.append(Paragraph(explanation, self.styles['Normal']))
                story.append(Spacer(1, 10))

        elif block_type == 'practice_questions':
            questions = content.get('questions', [])
            await self._add_questions_to_pdf(story, questions, version)

        elif block_type == 'summary':
            key_points = content.get('key_points', [])
            for point in key_points:
                story.append(Paragraph(f"• {point}", self.styles['Normal']))
            story.append(Spacer(1, 10))

        else:
            # Generic content handling
            story.append(Paragraph(str(content), self.styles['Normal']))
            story.append(Spacer(1, 10))

    async def _add_questions_to_pdf(
        self,
        story: list,
        questions: list[dict[str, Any]],
        version: str
    ):
        """Add questions to PDF with appropriate formatting based on version."""
        for i, question in enumerate(questions, 1):
            # Question text
            question_text = question.get('text', '')
            marks = question.get('marks', 0)

            # Format question with marks
            formatted_question = f"Q{i}. {question_text}"
            if marks:
                formatted_question += f" [{marks} marks]"

            story.append(Paragraph(formatted_question, self.styles['Question']))

            # Add answer space for student version
            if version == 'student':
                story.append(Spacer(1, 30))  # Space for writing
                story.append(Paragraph("_" * 50, self.styles['Normal']))
                story.append(Spacer(1, 20))

            # Add answers for teacher/combined version
            elif version in ['teacher', 'combined']:
                answer = question.get('answer', '')
                if answer:
                    story.append(Paragraph(f"<b>Answer:</b> {answer}", self.styles['Answer']))

                hint = question.get('hint', '')
                if hint:
                    story.append(Paragraph(f"<b>Hint:</b> {hint}", self.styles['Answer']))

                story.append(Spacer(1, 15))

    async def _export_docx(
        self,
        document: dict[str, Any],
        output_prefix: str,
        create_versions: bool
    ) -> list[str]:
        """Export document to DOCX format."""
        # For now, return placeholder - DOCX export can be implemented later
        logger.info("DOCX export not yet implemented")
        return []

    def _extract_document_metadata(self, document: dict[str, Any]) -> dict[str, str]:
        """Extract metadata from document for display."""
        metadata = {}

        # Get generation request details
        generation_request = document.get('generation_request', {})
        if generation_request:
            topic = getattr(generation_request, 'topic', None)
            if hasattr(topic, 'value'):
                metadata['Topic'] = topic.value
            elif topic:
                metadata['Topic'] = str(topic)

            grade_level = getattr(generation_request, 'grade_level', None)
            if grade_level:
                metadata['Grade Level'] = str(grade_level)

            difficulty = getattr(generation_request, 'difficulty', None)
            if difficulty:
                metadata['Difficulty'] = str(difficulty)

        # Get document details
        estimated_minutes = document.get('total_estimated_minutes')
        if estimated_minutes:
            metadata['Estimated Duration'] = f"{estimated_minutes} minutes"

        detail_level = document.get('actual_detail_level')
        if detail_level:
            metadata['Detail Level'] = str(detail_level)

        return metadata

    def _format_block_title(self, block_type: str) -> str:
        """Format block type into a readable title."""
        title_map = {
            'learning_objectives': 'Learning Objectives',
            'concept_explanation': 'Concept Explanation',
            'worked_example': 'Worked Examples',
            'practice_questions': 'Practice Questions',
            'summary': 'Summary',
            'extension_activities': 'Extension Activities',
            'assessment_rubric': 'Assessment Rubric'
        }

        return title_map.get(block_type, block_type.replace('_', ' ').title())

    def get_export_directory(self) -> str:
        """Get the export directory path."""
        return str(self.output_directory)

    def cleanup_old_exports(self, max_age_hours: int = 24):
        """Clean up old exported files."""
        import time

        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)

        for file_path in self.output_directory.glob("*.pdf"):
            if file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    logger.info(f"Cleaned up old export: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {file_path}: {e}")

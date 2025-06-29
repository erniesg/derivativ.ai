"""
Document version service for creating student and teacher versions.
Handles content filtering and section customization based on target audience.
"""

import logging

from src.models.document_models import (
    DetailLevel,
    DocumentSection,
    DocumentStructure,
    DocumentType,
    DocumentVersion,
)

logger = logging.getLogger(__name__)


class DocumentVersionService:
    """Service for creating different versions of documents for students and teachers."""

    def _get_detail_level_display(self, detail_level: DetailLevel) -> str:
        """Convert DetailLevel integer to display string."""
        level_names = {
            DetailLevel.MINIMAL: "minimal",
            DetailLevel.BASIC: "basic",
            DetailLevel.MEDIUM: "medium",
            DetailLevel.DETAILED: "detailed",
            DetailLevel.COMPREHENSIVE: "comprehensive",
            DetailLevel.GUIDED: "guided"
        }
        return level_names.get(detail_level, f"level {detail_level.value}")

    def __init__(self):
        """Initialize document version service."""
        pass

    def create_student_version(self, base_document: DocumentStructure) -> DocumentStructure:
        """
        Create student version by filtering out answers and solutions.

        Args:
            base_document: Base document structure

        Returns:
            Student version with questions only
        """
        logger.info(f"Creating student version of '{base_document.title}'")

        # Filter sections for student version
        student_sections = []

        for section in base_document.sections:
            # Skip answer and solution sections for students
            if section.content_type in ["answers", "detailed_solutions", "marking_scheme"]:
                logger.debug(f"Skipping section '{section.title}' for student version")
                continue

            # Keep learning objectives, theory, examples, and questions
            if section.content_type in [
                "learning_objectives",
                "theory_content",
                "worked_examples",
                "practice_questions",
                "instructions",
            ]:
                # For practice questions, remove any embedded answers
                if section.content_type == "practice_questions":
                    student_section = self._clean_questions_for_student(section)
                else:
                    student_section = section.model_copy()

                student_sections.append(student_section)

        # Create student document
        student_document = base_document.model_copy()
        student_document.version = DocumentVersion.STUDENT
        student_document.title = f"{base_document.title} - Student Copy"
        student_document.sections = student_sections

        logger.info(f"Student version created with {len(student_sections)} sections")
        return student_document

    def create_teacher_version(self, base_document: DocumentStructure) -> DocumentStructure:
        """
        Create teacher version with full content including answers and marking schemes.

        Args:
            base_document: Base document structure

        Returns:
            Teacher version with complete content
        """
        logger.info(f"Creating teacher version of '{base_document.title}'")

        # Teacher gets all sections, but we might want to reorganize them
        teacher_sections = []

        # Add all existing sections
        for section in base_document.sections:
            teacher_sections.append(section.model_copy())

        # Add teaching notes section if it doesn't exist
        has_teaching_notes = any(s.content_type == "teaching_notes" for s in teacher_sections)
        if not has_teaching_notes and base_document.document_type == DocumentType.WORKSHEET:
            teaching_notes = DocumentSection(
                title="Teaching Notes",
                content_type="teaching_notes",
                content_data={"notes": self._generate_teaching_notes(base_document)},
                order_index=len(teacher_sections),
            )
            teacher_sections.append(teaching_notes)

        # Create teacher document
        teacher_document = base_document.model_copy()
        teacher_document.version = DocumentVersion.TEACHER
        teacher_document.title = f"{base_document.title} - Teacher Copy"
        teacher_document.sections = teacher_sections

        logger.info(f"Teacher version created with {len(teacher_sections)} sections")
        return teacher_document

    def _clean_questions_for_student(self, questions_section: DocumentSection) -> DocumentSection:
        """
        Remove answers and solutions from practice questions for student version.

        Args:
            questions_section: Section containing practice questions

        Returns:
            Cleaned section without answers
        """
        cleaned_section = questions_section.model_copy()

        if "questions" in cleaned_section.content_data:
            cleaned_questions = []

            for question in cleaned_section.content_data["questions"]:
                # Create clean question without answers
                clean_question = {
                    "question_id": question.get("question_id", ""),
                    "question_text": question.get("question_text", ""),
                    "marks": question.get("marks", 1),
                    "command_word": question.get("command_word", ""),
                    # Remove answer-related fields
                }
                cleaned_questions.append(clean_question)

            cleaned_section.content_data["questions"] = cleaned_questions

            # Keep timing info but remove answer references
            if "total_marks" in cleaned_section.content_data:
                del cleaned_section.content_data["total_marks"]

        return cleaned_section

    def _generate_teaching_notes(self, document: DocumentStructure) -> str:
        """
        Generate teaching notes for the teacher version.

        Args:
            document: Document structure

        Returns:
            Teaching notes text
        """
        notes = []

        # Basic teaching guidance
        notes.append("**Teaching Guidance:**")
        notes.append(
            f"• Estimated completion time: {document.estimated_duration or 'Not specified'} minutes"
        )
        notes.append(f"• Total questions: {document.total_questions}")
        notes.append(f"• Detail level: {self._get_detail_level_display(document.detail_level)}")

        # Question-specific notes
        practice_sections = [s for s in document.sections if s.content_type == "practice_questions"]
        if practice_sections:
            notes.append("\n**Question Notes:**")

            for section in practice_sections:
                questions = section.content_data.get("questions", [])
                for i, question in enumerate(questions, 1):
                    marks = question.get("marks", 1)
                    command_word = question.get("command_word", "")
                    notes.append(f"• Q{i}: {marks} marks, {command_word} - Focus on method marks")

        # Common misconceptions
        notes.append("\n**Common Student Errors:**")
        notes.append("• Check arithmetic carefully")
        notes.append("• Ensure all working is shown")
        notes.append("• Verify answers make sense in context")

        return "\n".join(notes)

    def create_both_versions(
        self, base_document: DocumentStructure
    ) -> tuple[DocumentStructure, DocumentStructure]:
        """
        Create both student and teacher versions.

        Args:
            base_document: Base document structure

        Returns:
            Tuple of (student_version, teacher_version)
        """
        student_version = self.create_student_version(base_document)
        teacher_version = self.create_teacher_version(base_document)

        return student_version, teacher_version

    def get_version_differences(
        self, student_doc: DocumentStructure, teacher_doc: DocumentStructure
    ) -> dict:
        """
        Analyze differences between student and teacher versions.

        Args:
            student_doc: Student version
            teacher_doc: Teacher version

        Returns:
            Dictionary with difference analysis
        """
        return {
            "student_sections": len(student_doc.sections),
            "teacher_sections": len(teacher_doc.sections),
            "sections_removed_for_student": len(teacher_doc.sections) - len(student_doc.sections),
            "student_section_types": [s.content_type for s in student_doc.sections],
            "teacher_section_types": [s.content_type for s in teacher_doc.sections],
            "teacher_only_sections": [
                s.content_type
                for s in teacher_doc.sections
                if s.content_type not in [ss.content_type for ss in student_doc.sections]
            ],
        }

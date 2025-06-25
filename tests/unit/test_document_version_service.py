"""
Unit tests for DocumentVersionService.
Tests student/teacher version creation logic.
"""

import pytest

from src.models.document_models import (
    DetailLevel,
    DocumentSection,
    DocumentStructure,
    DocumentType,
    DocumentVersion,
)
from src.services.document_version_service import DocumentVersionService


class TestDocumentVersionService:
    """Test DocumentVersionService functionality."""

    @pytest.fixture
    def version_service(self):
        """Create DocumentVersionService instance."""
        return DocumentVersionService()

    @pytest.fixture
    def sample_worksheet(self):
        """Create sample worksheet with all section types."""
        return DocumentStructure(
            title="Linear Equations Worksheet",
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            estimated_duration=30,
            total_questions=3,
            sections=[
                DocumentSection(
                    title="Learning Objectives",
                    content_type="learning_objectives",
                    content_data={
                        "objectives_text": "• Solve linear equations\n• Apply algebraic methods"
                    },
                    order_index=0,
                ),
                DocumentSection(
                    title="Worked Example",
                    content_type="worked_examples",
                    content_data={
                        "examples": [
                            {
                                "title": "Example 1",
                                "problem": "Solve: 2x + 3 = 11",
                                "solution": "x = 4",
                            }
                        ]
                    },
                    order_index=1,
                ),
                DocumentSection(
                    title="Practice Questions",
                    content_type="practice_questions",
                    content_data={
                        "questions": [
                            {
                                "question_id": "q1",
                                "question_text": "Solve: 3x + 5 = 14",
                                "marks": 2,
                                "command_word": "Solve",
                                "answer": "x = 3",  # This should be removed for students
                            },
                            {
                                "question_id": "q2",
                                "question_text": "Find: 2y - 1 = 7",
                                "marks": 3,
                                "command_word": "Find",
                                "solution": "y = 4",  # This should be removed for students
                            },
                        ],
                        "total_marks": 5,
                    },
                    order_index=2,
                ),
                DocumentSection(
                    title="Answers",
                    content_type="answers",
                    content_data={"answers": [{"answer": "x = 3"}, {"answer": "y = 4"}]},
                    order_index=3,
                ),
                DocumentSection(
                    title="Detailed Solutions",
                    content_type="detailed_solutions",
                    content_data={
                        "solutions": [
                            {"solution": "3x + 5 = 14\n3x = 9\nx = 3"},
                            {"solution": "2y - 1 = 7\n2y = 8\ny = 4"},
                        ]
                    },
                    order_index=4,
                ),
            ],
        )

    def test_create_student_version(self, version_service, sample_worksheet):
        """Test creating student version removes answers and solutions."""
        student_version = version_service.create_student_version(sample_worksheet)

        # Check basic properties
        assert student_version.version == DocumentVersion.STUDENT
        assert "Student Copy" in student_version.title
        assert student_version.document_type == DocumentType.WORKSHEET
        assert student_version.detail_level == sample_worksheet.detail_level

        # Check section filtering
        section_types = [s.content_type for s in student_version.sections]

        # Should keep these sections
        assert "learning_objectives" in section_types
        assert "worked_examples" in section_types
        assert "practice_questions" in section_types

        # Should remove these sections
        assert "answers" not in section_types
        assert "detailed_solutions" not in section_types
        assert "marking_scheme" not in section_types

        # Should have fewer sections than original
        assert len(student_version.sections) < len(sample_worksheet.sections)

    def test_create_teacher_version(self, version_service, sample_worksheet):
        """Test creating teacher version keeps all content and adds teaching notes."""
        teacher_version = version_service.create_teacher_version(sample_worksheet)

        # Check basic properties
        assert teacher_version.version == DocumentVersion.TEACHER
        assert "Teacher Copy" in teacher_version.title
        assert teacher_version.document_type == DocumentType.WORKSHEET

        # Should have all original sections plus teaching notes
        assert len(teacher_version.sections) >= len(sample_worksheet.sections)

        section_types = [s.content_type for s in teacher_version.sections]

        # Should keep all original sections
        assert "learning_objectives" in section_types
        assert "worked_examples" in section_types
        assert "practice_questions" in section_types
        assert "answers" in section_types
        assert "detailed_solutions" in section_types

        # Should add teaching notes for worksheets
        assert "teaching_notes" in section_types

    def test_clean_questions_for_student(self, version_service, sample_worksheet):
        """Test that student questions are cleaned of answers."""
        # Get the practice questions section
        practice_section = next(
            s for s in sample_worksheet.sections if s.content_type == "practice_questions"
        )

        cleaned_section = version_service._clean_questions_for_student(practice_section)

        # Check that questions are cleaned
        questions = cleaned_section.content_data["questions"]

        for question in questions:
            # Should keep essential fields
            assert "question_id" in question
            assert "question_text" in question
            assert "marks" in question
            assert "command_word" in question

            # Should remove answer fields
            assert "answer" not in question
            assert "solution" not in question

        # Should remove total_marks reference
        assert "total_marks" not in cleaned_section.content_data

    def test_create_both_versions(self, version_service, sample_worksheet):
        """Test creating both versions simultaneously."""
        student_version, teacher_version = version_service.create_both_versions(sample_worksheet)

        # Verify both versions are correct
        assert student_version.version == DocumentVersion.STUDENT
        assert teacher_version.version == DocumentVersion.TEACHER

        # Student should have fewer sections
        assert len(student_version.sections) < len(teacher_version.sections)

        # Teacher should have teaching notes
        teacher_section_types = [s.content_type for s in teacher_version.sections]
        assert "teaching_notes" in teacher_section_types

    def test_generate_teaching_notes(self, version_service, sample_worksheet):
        """Test teaching notes generation."""
        notes = version_service._generate_teaching_notes(sample_worksheet)

        # Should be a string with useful content
        assert isinstance(notes, str)
        assert len(notes) > 50  # Should be substantial

        # Should include key information
        assert "30 minutes" in notes or "Not specified" in notes
        assert "3" in notes  # total questions
        assert "medium" in notes  # detail level
        assert "marks" in notes.lower()

    def test_get_version_differences(self, version_service, sample_worksheet):
        """Test version difference analysis."""
        student_version, teacher_version = version_service.create_both_versions(sample_worksheet)

        differences = version_service.get_version_differences(student_version, teacher_version)

        # Check difference structure
        assert "student_sections" in differences
        assert "teacher_sections" in differences
        assert "sections_removed_for_student" in differences
        assert "student_section_types" in differences
        assert "teacher_section_types" in differences
        assert "teacher_only_sections" in differences

        # Verify numbers make sense
        assert differences["student_sections"] < differences["teacher_sections"]
        assert differences["sections_removed_for_student"] > 0

        # Teacher should have answer-related sections student doesn't
        teacher_only = differences["teacher_only_sections"]
        expected_teacher_only = ["answers", "detailed_solutions", "teaching_notes"]
        for section_type in expected_teacher_only:
            assert section_type in teacher_only

    def test_version_service_with_non_worksheet(self, version_service):
        """Test version service with non-worksheet document types."""
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

        # Should still work but not add teaching notes
        student_version = version_service.create_student_version(notes_document)
        teacher_version = version_service.create_teacher_version(notes_document)

        assert student_version.version == DocumentVersion.STUDENT
        assert teacher_version.version == DocumentVersion.TEACHER

        # Notes documents shouldn't get teaching notes added
        teacher_section_types = [s.content_type for s in teacher_version.sections]
        assert "teaching_notes" not in teacher_section_types

    def test_section_order_preservation(self, version_service, sample_worksheet):
        """Test that section order is preserved in both versions."""
        student_version, teacher_version = version_service.create_both_versions(sample_worksheet)

        # Student sections should maintain relative order
        student_indices = [s.order_index for s in student_version.sections]
        assert student_indices == sorted(student_indices)

        # Teacher sections should maintain order (teaching notes added at end)
        teacher_base_sections = [
            s for s in teacher_version.sections if s.content_type != "teaching_notes"
        ]
        teacher_base_indices = [s.order_index for s in teacher_base_sections]
        assert teacher_base_indices == sorted(teacher_base_indices)

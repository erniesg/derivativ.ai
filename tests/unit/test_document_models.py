"""
Unit tests for document generation models and schemas.
Tests validation, serialization, and model behavior.
"""

import pytest
from pydantic import ValidationError

from src.models.document_models import (
    ContentSection,
    DetailLevel,
    DocumentGenerationRequest,
    DocumentGenerationResult,
    DocumentTemplate,
    DocumentType,
    ExportFormat,
    ExportRequest,
    ExportResult,
    GeneratedDocument,
    QuestionReference,
)
from src.models.enums import SubjectContentReference, Tier


class TestDocumentModels:
    """Test document generation models."""

    def test_document_type_enum_values(self):
        """Test DocumentType enum has correct values."""
        assert DocumentType.WORKSHEET == "worksheet"
        assert DocumentType.NOTES == "notes"
        assert DocumentType.TEXTBOOK == "textbook"
        assert DocumentType.SLIDES == "slides"

    def test_detail_level_enum_values(self):
        """Test DetailLevel enum has correct values."""
        assert DetailLevel.MINIMAL == "minimal"
        assert DetailLevel.MEDIUM == "medium"
        assert DetailLevel.COMPREHENSIVE == "comprehensive"
        assert DetailLevel.GUIDED == "guided"

    def test_export_format_enum_values(self):
        """Test ExportFormat enum has correct values."""
        assert ExportFormat.HTML == "html"
        assert ExportFormat.PDF == "pdf"
        assert ExportFormat.DOCX == "docx"
        assert ExportFormat.MARKDOWN == "markdown"
        assert ExportFormat.LATEX == "latex"
        assert ExportFormat.SLIDES_PPTX == "pptx"


class TestContentSection:
    """Test ContentSection model."""

    def test_content_section_valid_creation(self):
        """Test creating a valid ContentSection."""
        section = ContentSection(
            title="Introduction",
            content_type="text",
            content_data={"text": "This is an introduction"},
            order_index=0,
        )

        assert section.title == "Introduction"
        assert section.content_type == "text"
        assert section.content_data["text"] == "This is an introduction"
        assert section.order_index == 0
        assert section.subsections == []
        assert section.section_id is not None

    def test_content_section_with_subsections(self):
        """Test ContentSection with nested subsections."""
        subsection = ContentSection(
            title="Sub-topic",
            content_type="example",
            content_data={"example": "Sample"},
            order_index=1,
        )

        main_section = ContentSection(
            title="Main Topic",
            content_type="section",
            content_data={},
            order_index=0,
            subsections=[subsection],
        )

        assert len(main_section.subsections) == 1
        assert main_section.subsections[0].title == "Sub-topic"

    def test_content_section_empty_title_validation(self):
        """Test that empty title raises validation error."""
        with pytest.raises(ValidationError, match="title cannot be empty"):
            ContentSection(title="", content_type="text", content_data={}, order_index=0)

    def test_content_section_whitespace_title_validation(self):
        """Test that whitespace-only title raises validation error."""
        with pytest.raises(ValidationError, match="title cannot be empty"):
            ContentSection(title="   ", content_type="text", content_data={}, order_index=0)


class TestQuestionReference:
    """Test QuestionReference model."""

    def test_question_reference_valid_creation(self):
        """Test creating a valid QuestionReference."""
        ref = QuestionReference(
            question_id="0580_SP_25_P1_q1a",
            include_solution=True,
            include_marking=True,
            context_note="Example of area calculation",
            order_index=1,
        )

        assert ref.question_id == "0580_SP_25_P1_q1a"
        assert ref.include_solution is True
        assert ref.include_marking is True
        assert ref.context_note == "Example of area calculation"
        assert ref.order_index == 1

    def test_question_reference_defaults(self):
        """Test QuestionReference with default values."""
        ref = QuestionReference(question_id="test_id", order_index=0)

        assert ref.include_solution is True  # Default
        assert ref.include_marking is True  # Default
        assert ref.context_note is None  # Default


class TestDocumentTemplate:
    """Test DocumentTemplate model."""

    def test_document_template_valid_creation(self):
        """Test creating a valid DocumentTemplate."""
        template = DocumentTemplate(
            name="Standard Worksheet",
            document_type=DocumentType.WORKSHEET,
            supported_detail_levels=[DetailLevel.MEDIUM, DetailLevel.COMPREHENSIVE],
            structure_patterns={
                DetailLevel.MEDIUM: ["introduction", "questions", "solutions"],
                DetailLevel.COMPREHENSIVE: [
                    "introduction",
                    "theory",
                    "examples",
                    "questions",
                    "solutions",
                ],
            },
            content_rules={
                DetailLevel.MEDIUM: {"include_solutions": True},
                DetailLevel.COMPREHENSIVE: {"include_solutions": True, "include_theory": True},
            },
        )

        assert template.name == "Standard Worksheet"
        assert template.document_type == DocumentType.WORKSHEET
        assert DetailLevel.MEDIUM in template.supported_detail_levels
        assert template.structure_patterns[DetailLevel.MEDIUM] == [
            "introduction",
            "questions",
            "solutions",
        ]
        assert template.content_rules[DetailLevel.MEDIUM]["include_solutions"] is True
        assert template.template_id is not None

    def test_document_template_empty_structure_validation(self):
        """Test that empty structure pattern raises validation error."""
        with pytest.raises(ValidationError, match="structure_pattern for .* cannot be empty"):
            DocumentTemplate(
                name="Invalid Template",
                document_type=DocumentType.WORKSHEET,
                supported_detail_levels=[DetailLevel.MEDIUM],
                structure_patterns={
                    DetailLevel.MEDIUM: []  # Empty structure
                },
                content_rules={},
            )


class TestDocumentGenerationRequest:
    """Test DocumentGenerationRequest model."""

    def test_document_generation_request_minimal(self):
        """Test creating minimal DocumentGenerationRequest."""
        request = DocumentGenerationRequest(
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            title="Test Worksheet",
            topic="algebra",
        )

        assert request.document_type == DocumentType.WORKSHEET
        assert request.detail_level == DetailLevel.MEDIUM
        assert request.title == "Test Worksheet"
        assert request.topic == "algebra"
        assert request.tier == Tier.CORE  # Default
        assert request.grade_level is None
        assert request.auto_include_questions is True  # Default
        assert request.max_questions == 10  # Default

    def test_document_generation_request_full(self):
        """Test creating full DocumentGenerationRequest."""
        question_refs = [
            QuestionReference(question_id="test_1", order_index=1),
            QuestionReference(question_id="test_2", order_index=2),
        ]

        request = DocumentGenerationRequest(
            document_type=DocumentType.TEXTBOOK,
            detail_level=DetailLevel.COMPREHENSIVE,
            title="Complete Algebra Guide",
            topic="algebraic expressions",
            tier=Tier.EXTENDED,
            grade_level=9,
            subject_content_refs=[SubjectContentReference.C2_1, SubjectContentReference.C2_2],
            question_references=question_refs,
            auto_include_questions=False,
            max_questions=20,
            template_id="custom_template",
            custom_sections=["extensions", "historical_notes"],
            exclude_content_types=["basic_examples"],
            custom_instructions="Use visual diagrams and step-by-step explanations for complex concepts",
            personalization_context={
                "learning_style": "visual",
                "difficulty_preference": "gradual",
            },
            include_answers=True,
            include_working=True,
            include_mark_schemes=True,
        )

        assert request.document_type == DocumentType.TEXTBOOK
        assert request.detail_level == DetailLevel.COMPREHENSIVE
        assert request.tier == Tier.EXTENDED
        assert request.grade_level == 9
        assert len(request.subject_content_refs) == 2
        assert len(request.question_references) == 2
        assert request.auto_include_questions is False
        assert request.max_questions == 20
        assert request.custom_instructions is not None
        assert "visual" in request.custom_instructions
        assert request.personalization_context["learning_style"] == "visual"

    def test_document_generation_request_title_validation(self):
        """Test title validation in DocumentGenerationRequest."""
        with pytest.raises(ValidationError, match="title cannot be empty"):
            DocumentGenerationRequest(
                document_type=DocumentType.WORKSHEET,
                detail_level=DetailLevel.MEDIUM,
                title="",  # Empty title
                topic="algebra",
            )

    def test_document_generation_request_grade_level_validation(self):
        """Test grade level validation."""
        # Valid grade level
        request = DocumentGenerationRequest(
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            title="Test",
            topic="algebra",
            grade_level=5,
        )
        assert request.grade_level == 5

        # Invalid grade level (too low)
        with pytest.raises(ValidationError):
            DocumentGenerationRequest(
                document_type=DocumentType.WORKSHEET,
                detail_level=DetailLevel.MEDIUM,
                title="Test",
                topic="algebra",
                grade_level=0,
            )

        # Invalid grade level (too high)
        with pytest.raises(ValidationError):
            DocumentGenerationRequest(
                document_type=DocumentType.WORKSHEET,
                detail_level=DetailLevel.MEDIUM,
                title="Test",
                topic="algebra",
                grade_level=15,
            )

    def test_document_generation_request_max_questions_validation(self):
        """Test max questions validation."""
        # Valid max questions
        request = DocumentGenerationRequest(
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            title="Test",
            topic="algebra",
            max_questions=25,
        )
        assert request.max_questions == 25

        # Invalid max questions (too low)
        with pytest.raises(ValidationError):
            DocumentGenerationRequest(
                document_type=DocumentType.WORKSHEET,
                detail_level=DetailLevel.MEDIUM,
                title="Test",
                topic="algebra",
                max_questions=0,
            )

        # Invalid max questions (too high)
        with pytest.raises(ValidationError):
            DocumentGenerationRequest(
                document_type=DocumentType.WORKSHEET,
                detail_level=DetailLevel.MEDIUM,
                title="Test",
                topic="algebra",
                max_questions=100,
            )


class TestGeneratedDocument:
    """Test GeneratedDocument model."""

    def test_generated_document_creation(self):
        """Test creating a GeneratedDocument."""
        sections = [
            ContentSection(
                title="Introduction",
                content_type="text",
                content_data={"text": "Introduction text"},
                order_index=0,
            ),
            ContentSection(
                title="Questions",
                content_type="questions",
                content_data={"questions": []},
                order_index=1,
            ),
        ]

        request = DocumentGenerationRequest(
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            title="Test Document",
            topic="algebra",
            custom_instructions="Keep explanations simple",
        )

        document = GeneratedDocument(
            title="Test Document",
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            generated_at="2025-06-21T12:00:00Z",
            template_used="worksheet_default",
            generation_request=request,
            sections=sections,
            total_questions=5,
            estimated_duration=15,
            applied_customizations={"custom_instructions": "Keep explanations simple"},
        )

        assert document.title == "Test Document"
        assert document.document_type == DocumentType.WORKSHEET
        assert len(document.sections) == 2
        assert document.total_questions == 5
        assert document.estimated_duration == 15
        assert document.document_id is not None
        assert "custom_instructions" in document.applied_customizations

    def test_generated_document_empty_sections_validation(self):
        """Test that empty sections raises validation error."""
        request = DocumentGenerationRequest(
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            title="Test Document",
            topic="algebra",
        )

        with pytest.raises(ValidationError, match="Document must have at least one section"):
            GeneratedDocument(
                title="Test Document",
                document_type=DocumentType.WORKSHEET,
                detail_level=DetailLevel.MEDIUM,
                generated_at="2025-06-21T12:00:00Z",
                template_used="worksheet_default",
                generation_request=request,
                sections=[],  # Empty sections
            )


class TestDocumentGenerationResult:
    """Test DocumentGenerationResult model."""

    def test_successful_generation_result(self):
        """Test successful generation result."""
        sections = [
            ContentSection(
                title="Test Section",
                content_type="text",
                content_data={"text": "Test"},
                order_index=0,
            )
        ]

        request = DocumentGenerationRequest(
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            title="Test",
            topic="algebra",
            custom_instructions="Simple language",
        )

        document = GeneratedDocument(
            title="Test",
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            generated_at="2025-06-21T12:00:00Z",
            template_used="test_template",
            generation_request=request,
            sections=sections,
            applied_customizations={"custom_instructions": "Simple language"},
        )

        result = DocumentGenerationResult(
            success=True,
            document=document,
            processing_time=2.5,
            questions_processed=5,
            sections_generated=3,
            customizations_applied=1,
            personalization_success=True,
        )

        assert result.success is True
        assert result.document is not None
        assert result.error_message is None
        assert result.processing_time == 2.5
        assert result.questions_processed == 5
        assert result.sections_generated == 3
        assert result.customizations_applied == 1
        assert result.personalization_success is True

    def test_failed_generation_result(self):
        """Test failed generation result."""
        result = DocumentGenerationResult(
            success=False,
            error_message="Generation failed due to invalid request",
            processing_time=1.0,
        )

        assert result.success is False
        assert result.document is None
        assert result.error_message == "Generation failed due to invalid request"
        assert result.processing_time == 1.0
        assert result.questions_processed == 0  # Default
        assert result.sections_generated == 0  # Default


class TestExportRequest:
    """Test ExportRequest model."""

    def test_export_request_creation(self):
        """Test creating an ExportRequest."""
        request = ExportRequest(
            document_id="test_doc_123",
            format=ExportFormat.PDF,
            include_metadata=True,
            custom_styling={"font_size": "12pt", "margins": "2cm"},
            export_personalization={"language_level": "simplified"},
        )

        assert request.document_id == "test_doc_123"
        assert request.format == ExportFormat.PDF
        assert request.include_metadata is True
        assert request.custom_styling["font_size"] == "12pt"
        assert request.export_personalization["language_level"] == "simplified"

    def test_export_request_defaults(self):
        """Test ExportRequest with default values."""
        request = ExportRequest(document_id="test_doc", format=ExportFormat.HTML)

        assert request.include_metadata is True  # Default
        assert request.custom_styling is None  # Default


class TestExportResult:
    """Test ExportResult model."""

    def test_successful_export_result(self):
        """Test successful export result."""
        result = ExportResult(
            success=True, file_path="/tmp/document.pdf", file_size=1024000, export_time=3.2
        )

        assert result.success is True
        assert result.file_path == "/tmp/document.pdf"
        assert result.file_size == 1024000
        assert result.error_message is None
        assert result.export_time == 3.2

    def test_failed_export_result(self):
        """Test failed export result."""
        result = ExportResult(
            success=False, error_message="Export failed: invalid format", export_time=0.5
        )

        assert result.success is False
        assert result.file_path is None
        assert result.file_size is None
        assert result.error_message == "Export failed: invalid format"
        assert result.export_time == 0.5

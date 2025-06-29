"""
Document generation models and schemas for educational materials.

Supports multiple document types (worksheets, notes, textbooks, slides)
with different detail levels and content structures.
"""

from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from src.models.enums import SubjectContentReference, Tier


class DocumentType(str, Enum):
    """Types of educational documents we can generate."""

    WORKSHEET = "worksheet"
    NOTES = "notes"
    TEXTBOOK = "textbook"
    SLIDES = "slides"


class DetailLevel(int, Enum):
    """Detail levels for document content (1-10 scale)."""

    MINIMAL = 1  # Key points only
    BASIC = 3  # Basic concepts with simple examples
    MEDIUM = 5  # Moderate detail with examples
    DETAILED = 7  # Detailed explanations with multiple examples
    COMPREHENSIVE = 9  # Full detail with solutions and extensions
    GUIDED = 10  # Step-by-step guidance with scaffolding

    @classmethod
    def from_legacy_string(cls, value: str) -> "DetailLevel":
        """Convert legacy string values to new integer enum."""
        mapping = {
            "minimal": cls.MINIMAL,
            "medium": cls.MEDIUM,
            "comprehensive": cls.COMPREHENSIVE,
            "guided": cls.GUIDED,
        }
        return mapping.get(value.lower(), cls.MEDIUM)


class DocumentVersion(str, Enum):
    """Document versions for different audiences."""

    STUDENT = "student"  # Questions only, no answers/solutions
    TEACHER = "teacher"  # Questions + answers + marking schemes + solutions


class ContentSection(BaseModel):
    """A section within a document with specific content."""

    section_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(..., description="Section title")
    content_type: str = Field(..., description="Type of content (text, question, example, etc.)")
    content_data: dict[str, Any] = Field(default_factory=dict, description="Section content")
    order_index: int = Field(..., description="Position in document")
    subsections: list["ContentSection"] = Field(default_factory=list, description="Nested sections")

    @field_validator("title")
    @classmethod
    def validate_title(cls, v):
        """Ensure title is not empty."""
        if not v or not v.strip():
            raise ValueError("title cannot be empty")
        return v.strip()


class QuestionReference(BaseModel):
    """Reference to a question in our database for inclusion in documents."""

    question_id: str = Field(..., description="Global question ID")
    include_solution: bool = Field(default=True, description="Include step-by-step solution")
    include_marking: bool = Field(default=True, description="Include marking scheme")
    context_note: Optional[str] = Field(None, description="Why this question is included")
    order_index: int = Field(..., description="Position within section")


class DocumentTemplate(BaseModel):
    """Template defining document structure and content patterns."""

    template_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Template name")
    document_type: DocumentType = Field(..., description="Type of document this template creates")
    supported_detail_levels: list[DetailLevel] = Field(
        default_factory=lambda: list(DetailLevel),
        description="Detail levels this template supports",
    )
    structure_patterns: dict[DetailLevel, list[str]] = Field(
        ..., description="Structure patterns for each supported detail level"
    )
    content_rules: dict[DetailLevel, dict[str, Any]] = Field(
        default_factory=dict, description="Content rules for each detail level"
    )

    @field_validator("structure_patterns")
    @classmethod
    def validate_structure_patterns(cls, v):
        """Ensure each detail level has a non-empty structure pattern."""
        for detail_level, pattern in v.items():
            if not pattern:
                raise ValueError(f"structure_pattern for {detail_level} cannot be empty")
        return v


class DocumentGenerationRequest(BaseModel):
    """Request for generating an educational document."""

    # Document specification
    document_type: DocumentType = Field(..., description="Type of document to generate")
    detail_level: DetailLevel = Field(..., description="Level of detail to include")
    title: str = Field(..., description="Document title")

    # Content targeting
    topic: str = Field(..., description="Main topic/subject")
    tier: Tier = Field(default=Tier.CORE, description="Core or Extended tier")
    grade_level: Optional[int] = Field(None, ge=1, le=9, description="Target grade level")
    subject_content_refs: list[SubjectContentReference] = Field(
        default_factory=list, description="Specific syllabus references to cover"
    )

    # Question integration
    question_references: list[QuestionReference] = Field(
        default_factory=list, description="Specific questions to include"
    )
    auto_include_questions: bool = Field(
        default=True, description="Automatically include relevant questions from database"
    )
    max_questions: int = Field(default=10, ge=1, le=50, description="Maximum questions to include")

    # Customization
    template_id: Optional[str] = Field(None, description="Template to use (default if None)")
    custom_sections: list[str] = Field(
        default_factory=list, description="Additional sections to include"
    )
    exclude_content_types: list[str] = Field(
        default_factory=list, description="Content types to exclude"
    )

    # Personalization and custom instructions
    custom_instructions: Optional[str] = Field(
        None, description="Custom instructions for personalizing content generation"
    )
    personalization_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for content personalization (e.g., learning style, difficulty preferences)",
    )

    # Output preferences
    include_answers: bool = Field(default=True, description="Include answer sections")
    include_working: bool = Field(default=True, description="Include step-by-step working")
    include_mark_schemes: bool = Field(default=False, description="Include marking schemes")
    
    # Version generation
    generate_versions: list["DocumentVersion"] = Field(
        default_factory=lambda: [DocumentVersion.STUDENT, DocumentVersion.TEACHER],
        description="Which document versions to generate"
    )
    export_formats: list["ExportFormat"] = Field(
        default_factory=lambda: [ExportFormat.PDF, ExportFormat.DOCX],
        description="Which export formats to generate"
    )

    @field_validator("title")
    @classmethod
    def validate_title(cls, v):
        """Ensure title is not empty."""
        if not v or not v.strip():
            raise ValueError("title cannot be empty")
        return v.strip()


class DocumentSection(BaseModel):
    """A section within a document with specific content."""

    title: str = Field(..., description="Section title")
    content_type: str = Field(..., description="Type of content (text, question, example, etc.)")
    content_data: dict[str, Any] = Field(default_factory=dict, description="Section content")
    order_index: int = Field(..., description="Position in document")


class DocumentStructure(BaseModel):
    """Structure of a document with sections and metadata."""

    title: str = Field(..., description="Document title")
    document_type: DocumentType = Field(..., description="Type of document")
    detail_level: DetailLevel = Field(..., description="Detail level used")
    version: DocumentVersion = Field(
        default=DocumentVersion.STUDENT, description="Document version"
    )
    sections: list[DocumentSection] = Field(..., description="Document sections in order")
    estimated_duration: Optional[int] = Field(
        None, description="Estimated completion time in minutes"
    )
    total_questions: int = Field(default=0, description="Total questions included")


class GeneratedDocument(BaseModel):
    """A complete generated educational document."""

    # Metadata
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(..., description="Document title")
    document_type: DocumentType = Field(..., description="Type of document")
    detail_level: DetailLevel = Field(..., description="Detail level used")

    # Generation info
    generated_at: str = Field(..., description="Generation timestamp")
    template_used: str = Field(..., description="Template ID used")
    generation_request: DocumentGenerationRequest = Field(..., description="Original request")

    # Content structure
    sections: list[ContentSection] = Field(..., description="Document sections in order")
    total_questions: int = Field(default=0, description="Total questions included")
    estimated_duration: Optional[int] = Field(
        None, description="Estimated completion time in minutes"
    )

    # References and metadata
    questions_used: list[str] = Field(default_factory=list, description="Question IDs referenced")
    syllabus_coverage: list[SubjectContentReference] = Field(
        default_factory=list, description="Syllabus points covered"
    )

    # Personalization tracking
    applied_customizations: dict[str, Any] = Field(
        default_factory=dict,
        description="Custom instructions and personalizations that were applied",
    )

    # Output formats available
    content_html: Optional[str] = Field(None, description="HTML formatted content")
    content_latex: Optional[str] = Field(None, description="LaTeX formatted content")
    content_markdown: Optional[str] = Field(None, description="Markdown formatted content")

    @field_validator("sections")
    @classmethod
    def validate_sections(cls, v):
        """Ensure document has content."""
        if not v:
            raise ValueError("Document must have at least one section")
        return v


class DocumentGenerationResult(BaseModel):
    """Result of document generation process."""

    success: bool = Field(..., description="Whether generation succeeded")
    document: Optional[GeneratedDocument] = Field(
        None, description="Generated document if successful"
    )
    error_message: Optional[str] = Field(None, description="Error message if failed")

    # Generation metrics
    processing_time: float = Field(..., description="Time taken in seconds")
    questions_processed: int = Field(default=0, description="Questions analyzed")
    sections_generated: int = Field(default=0, description="Sections created")

    # Personalization metrics
    customizations_applied: int = Field(
        default=0, description="Number of custom instructions applied"
    )
    personalization_success: bool = Field(
        default=True, description="Whether personalization instructions were successfully applied"
    )

    # Agent information
    agent_results: list[dict[str, Any]] = Field(
        default_factory=list, description="Results from document generation agents"
    )
    reasoning_steps: list[dict[str, Any]] = Field(
        default_factory=list, description="Reasoning steps taken"
    )


class ExportFormat(str, Enum):
    """Supported export formats for documents."""

    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    LATEX = "latex"
    SLIDES_PPTX = "pptx"  # For slides document type


class ExportRequest(BaseModel):
    """Request for exporting a document to specific format."""

    document_id: str = Field(..., description="Document to export")
    format: ExportFormat = Field(..., description="Export format")
    version: str = Field(default="student", description="Document version (student/teacher)")
    include_metadata: bool = Field(default=True, description="Include generation metadata")
    custom_styling: Optional[dict[str, Any]] = Field(None, description="Custom styling options")
    export_personalization: Optional[dict[str, Any]] = Field(
        None,
        description="Additional personalization for export format (e.g., font preferences, layout adjustments)",
    )


class ExportResult(BaseModel):
    """Result of document export."""

    success: bool = Field(..., description="Whether export succeeded")
    file_path: Optional[str] = Field(None, description="Path to exported file")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    error_message: Optional[str] = Field(None, description="Error if failed")
    export_time: float = Field(..., description="Export time in seconds")
    applied_personalizations: list[str] = Field(
        default_factory=list, description="List of personalizations applied during export"
    )


# Update ContentSection to resolve forward reference
ContentSection.model_rebuild()

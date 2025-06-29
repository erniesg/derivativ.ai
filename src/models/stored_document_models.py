"""
Pydantic models for stored document data structures.
Defines models for document storage, search, and metadata management.
"""

from datetime import datetime
from typing import Any, ClassVar, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field, field_validator

from src.models.document_models import DocumentType


class StoredDocumentMetadata(BaseModel):
    """Metadata for a stored document."""

    id: UUID = Field(default_factory=uuid4, description="Unique document identifier")
    session_id: Optional[UUID] = Field(None, description="Generation session identifier")

    # Document content metadata
    title: str = Field(..., min_length=1, description="Document title")
    document_type: DocumentType = Field(..., description="Document type enum")
    detail_level: Optional[int] = Field(None, ge=1, le=10, description="Detail level (1-10 scale)")
    topic: Optional[str] = Field(None, description="Main topic/subject")
    grade_level: Optional[int] = Field(None, ge=1, le=12, description="Target grade level")

    # Generation metadata
    estimated_duration: Optional[int] = Field(
        None, description="Estimated completion time in minutes"
    )
    total_questions: Optional[int] = Field(None, ge=0, description="Total number of questions")

    # Storage metadata
    status: str = Field(default="pending", description="Document status")
    file_count: int = Field(default=0, ge=0, description="Number of associated files")
    total_file_size: int = Field(default=0, ge=0, description="Total size of all files in bytes")

    # Search and categorization
    tags: list[str] = Field(default_factory=list, description="Document tags for categorization")
    search_content: str = Field(default="", description="Searchable text content")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    deleted_at: Optional[datetime] = Field(None, description="Soft deletion timestamp")

    @field_validator("document_type")
    @classmethod
    def validate_document_type(cls, v):
        """Validate document type."""
        valid_types = ["worksheet", "notes", "textbook", "slides"]
        if v not in valid_types:
            raise ValueError(f"Invalid document type. Must be one of: {valid_types}")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        """Validate document status."""
        valid_statuses = [
            "pending",
            "generating",
            "generated",
            "exporting",
            "exported",
            "failed",
            "deleted",
            "archived",
        ]
        if v not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
        return v

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, v):
        """Normalize tags to lowercase."""
        if isinstance(v, list):
            return [tag.lower().strip() for tag in v if tag and tag.strip()]
        return []

    def generate_search_content(self) -> str:
        """Generate searchable content from document metadata."""
        content_parts = [self.title, self.document_type, self.topic or "", " ".join(self.tags)]
        return " ".join(filter(None, content_parts)).lower()

    class Config:
        """Pydantic configuration."""

        json_encoders: ClassVar = {datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}


class DocumentFile(BaseModel):
    """Information about a document file stored in R2."""

    id: UUID = Field(default_factory=uuid4, description="Unique file identifier")
    document_id: UUID = Field(..., description="Parent document identifier")

    # File location and format
    file_key: str = Field(..., description="R2 storage key/path")
    file_format: str = Field(..., description="File format extension")
    version: str = Field(..., description="Document version (student, teacher, combined)")

    # File metadata
    file_size: int = Field(default=0, ge=0, description="File size in bytes")
    content_type: str = Field(default="", description="MIME content type")

    # R2 storage metadata
    r2_metadata: dict[str, Any] = Field(default_factory=dict, description="R2-specific metadata")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

    @field_validator("file_format")
    @classmethod
    def validate_file_format(cls, v):
        """Validate file format."""
        valid_formats = ["pdf", "docx", "html", "txt", "json", "png", "jpg", "svg"]
        if v not in valid_formats:
            raise ValueError(f"Invalid file format. Must be one of: {valid_formats}")
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v):
        """Validate document version."""
        valid_versions = ["student", "teacher", "combined"]
        if v not in valid_versions:
            raise ValueError(f"Invalid version. Must be one of: {valid_versions}")
        return v

    @field_validator("file_key")
    @classmethod
    def validate_file_key(cls, v):
        """Validate file key format."""
        if not v or not isinstance(v, str):
            raise ValueError("File key cannot be empty")

        # Check for security issues
        if "../" in v or ".." in v:
            raise ValueError("File key contains path traversal attempt")

        if "//" in v:
            raise ValueError("File key contains double slashes")

        if " " in v:
            raise ValueError("File key cannot contain spaces")

        if len(v) > 1024:
            raise ValueError("File key too long (max 1024 characters)")

        return v

    class Config:
        """Pydantic configuration."""

        json_encoders: ClassVar = {datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}


class StoredDocument(BaseModel):
    """Complete stored document with metadata and files."""

    metadata: StoredDocumentMetadata = Field(..., description="Document metadata")
    files: list[DocumentFile] = Field(default_factory=list, description="Associated files")
    session_data: dict[str, Any] = Field(
        default_factory=dict, description="Generation session data"
    )

    class Config:
        """Pydantic configuration."""

        json_encoders: ClassVar = {datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}


class DocumentSearchFilters(BaseModel):
    """Filters for document search operations."""

    # Content filters
    document_type: Optional[str] = Field(None, description="Filter by document type")
    topic: Optional[str] = Field(None, description="Filter by topic")
    grade_level: Optional[int] = Field(None, ge=1, le=12, description="Filter by grade level")
    status: Optional[str] = Field(None, description="Filter by status")

    # Text search
    search_text: str = Field(default="", description="Full-text search query")
    tags: list[str] = Field(default_factory=list, description="Filter by tags")

    # Date filters
    created_after: Optional[datetime] = Field(None, description="Filter by creation date (after)")
    created_before: Optional[datetime] = Field(None, description="Filter by creation date (before)")

    # Pagination
    limit: int = Field(default=50, ge=1, le=100, description="Maximum results to return")
    offset: int = Field(default=0, ge=0, description="Number of results to skip")

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v):
        """Validate pagination limit."""
        if v < 1 or v > 100:
            raise ValueError("Limit must be between 1 and 100")
        return v

    class Config:
        """Pydantic configuration."""

        json_encoders: ClassVar = {datetime: lambda v: v.isoformat()}


class DocumentSearchResults(BaseModel):
    """Results from document search operations."""

    documents: list[StoredDocumentMetadata] = Field(..., description="Found documents")
    total_count: int = Field(..., description="Total number of matching documents")
    offset: int = Field(..., description="Current offset")
    limit: int = Field(..., description="Current limit")

    @computed_field
    @property
    def has_more(self) -> bool:
        """Calculate if more results are available."""
        return (self.offset + self.limit) < self.total_count

    class Config:
        """Pydantic configuration."""

        json_encoders: ClassVar = {datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}


class DocumentStatistics(BaseModel):
    """Statistics about stored documents."""

    total_documents: int = Field(default=0, description="Total number of documents")
    total_file_size: int = Field(default=0, description="Total file size in bytes")

    # Breakdown by categories
    documents_by_type: dict[str, int] = Field(
        default_factory=dict, description="Count by document type"
    )
    documents_by_status: dict[str, int] = Field(default_factory=dict, description="Count by status")

    # Performance metrics
    average_file_size: int = Field(default=0, description="Average file size in bytes")
    largest_document_size: int = Field(default=0, description="Size of largest document in bytes")

    # Popular content
    most_popular_topics: list[str] = Field(default_factory=list, description="Most popular topics")

    # Quality metrics
    generation_success_rate: float = Field(
        default=0.0, description="Success rate of document generation"
    )
    storage_usage_percentage: float = Field(default=0.0, description="Storage usage percentage")

    class Config:
        """Pydantic configuration."""

        json_encoders: ClassVar = {datetime: lambda v: v.isoformat()}


class DocumentExportRequest(BaseModel):
    """Request for document export operations."""

    document_id: UUID = Field(..., description="Document to export")
    formats: list[str] = Field(..., description="Export formats (pdf, docx, html)")
    create_dual_versions: bool = Field(default=False, description="Create student/teacher versions")

    # Export options
    export_options: dict[str, Any] = Field(
        default_factory=dict, description="Format-specific options"
    )

    # Output preferences
    output_directory: Optional[str] = Field(None, description="Custom output directory")
    filename_template: Optional[str] = Field(None, description="Custom filename template")

    @field_validator("formats")
    @classmethod
    def validate_formats(cls, v):
        """Validate export formats."""
        valid_formats = ["pdf", "docx", "html"]
        invalid_formats = [f for f in v if f not in valid_formats]

        if invalid_formats:
            raise ValueError(f"Invalid formats: {invalid_formats}. Valid formats: {valid_formats}")

        if not v:
            raise ValueError("At least one format must be specified")

        return v

    class Config:
        """Pydantic configuration."""

        json_encoders: ClassVar = {UUID: lambda v: str(v)}


class DocumentExportResult(BaseModel):
    """Result of document export operations."""

    export_id: UUID = Field(default_factory=uuid4, description="Unique export identifier")
    document_id: UUID = Field(..., description="Source document identifier")
    status: str = Field(..., description="Export status")

    # Export results
    files: list[DocumentFile] = Field(default_factory=list, description="Generated files")
    errors: list[str] = Field(default_factory=list, description="Export errors")

    # Timing information
    started_at: datetime = Field(default_factory=datetime.now, description="Export start time")
    completed_at: Optional[datetime] = Field(None, description="Export completion time")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        """Validate export status."""
        valid_statuses = ["queued", "processing", "completed", "failed", "cancelled"]
        if v not in valid_statuses:
            raise ValueError(f"Invalid export status. Must be one of: {valid_statuses}")
        return v

    class Config:
        """Pydantic configuration."""

        json_encoders: ClassVar = {datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}


class BulkOperationRequest(BaseModel):
    """Request for bulk operations on documents."""

    document_ids: list[UUID] = Field(..., description="List of document IDs")
    operation: str = Field(..., description="Operation to perform")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Operation parameters")

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(cls, v):
        """Validate document IDs list."""
        if not v:
            raise ValueError("At least one document ID must be provided")

        if len(v) > 100:
            raise ValueError("Maximum 100 documents per bulk operation")

        return v

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, v):
        """Validate bulk operation type."""
        valid_operations = ["update_status", "delete", "export", "archive", "tag"]
        if v not in valid_operations:
            raise ValueError(f"Invalid operation. Must be one of: {valid_operations}")
        return v

    class Config:
        """Pydantic configuration."""

        json_encoders: ClassVar = {UUID: lambda v: str(v)}


class BulkOperationResult(BaseModel):
    """Result of bulk operations."""

    operation_id: UUID = Field(default_factory=uuid4, description="Unique operation identifier")
    operation: str = Field(..., description="Operation performed")

    # Results
    total_requested: int = Field(..., description="Total documents requested")
    successful_count: int = Field(..., description="Number of successful operations")
    failed_count: int = Field(..., description="Number of failed operations")

    # Details
    successful_ids: list[UUID] = Field(
        default_factory=list, description="Successfully processed IDs"
    )
    failed_operations: list[dict[str, Any]] = Field(
        default_factory=list, description="Failed operations with errors"
    )

    # Timing
    started_at: datetime = Field(default_factory=datetime.now, description="Operation start time")
    completed_at: Optional[datetime] = Field(None, description="Operation completion time")

    class Config:
        """Pydantic configuration."""

        json_encoders: ClassVar = {datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}

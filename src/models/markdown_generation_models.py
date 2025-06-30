"""
Models for markdown-first document generation.

These models bridge the gap between simple document requests and
the rich context needed for educational content generation.
"""

from typing import Optional

from pydantic import BaseModel, Field

from src.models.document_models import DetailLevel, DocumentType
from src.models.enums import Tier, TopicName


class MarkdownGenerationRequest(BaseModel):
    """Request for generating educational documents via markdown pipeline."""

    # Core document specification
    document_type: DocumentType = Field(..., description="Type of document to generate")
    topic: TopicName = Field(..., description="Mathematical topic to cover")
    tier: Tier = Field(default=Tier.CORE, description="Core or Extended tier")
    detail_level: DetailLevel = Field(..., description="Level of detail (1-10)")

    # Document parameters
    title: Optional[str] = Field(None, description="Document title (auto-generated if not provided)")
    target_duration_minutes: int = Field(default=30, ge=5, le=120, description="Target completion time")
    grade_level: str = Field(default="7-9", description="Target grade level")

    # Optional enhancements
    custom_instructions: Optional[str] = Field(None, description="Additional generation instructions")

    def get_title(self) -> str:
        """Get document title, generating one if not provided."""
        if self.title:
            return self.title

        # Auto-generate title
        return f"{self.document_type.value.title()}: {self.topic.value}"

    def to_template_context(self) -> dict:
        """Convert to template context variables."""
        return {
            "document_type": self.document_type.value,
            "topic": self.topic.value,
            "tier": self.tier.value,
            "detail_level": self.detail_level.value,
            "target_duration_minutes": self.target_duration_minutes,
            "grade_level": self.grade_level,
            "title": self.get_title(),
            "custom_instructions": self.custom_instructions
        }


class MarkdownGenerationResult(BaseModel):
    """Result of markdown document generation."""

    success: bool = Field(..., description="Whether generation succeeded")
    document_id: str = Field(..., description="Unique document identifier")
    markdown_content: str = Field(..., description="Generated markdown content")

    # Format availability
    formats: dict[str, dict] = Field(default_factory=dict, description="Available format downloads")

    # Metadata
    metadata: dict = Field(default_factory=dict, description="Document metadata")
    generation_info: dict = Field(default_factory=dict, description="Generation details")

    # Error info (if applicable)
    error: Optional[str] = Field(None, description="Error message if generation failed")

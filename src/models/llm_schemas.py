"""
Centralized Pydantic schemas for LLM structured outputs.
Used across OpenAI parse, Gemini response_schema, and Anthropic prompt instructions.
"""

from typing import Any

from pydantic import BaseModel, Field

from src.models.content_blocks import ContentBlock


class DocumentContentBlock(ContentBlock):
    """A single content block in a generated document - extends base ContentBlock."""

    content: dict[str, Any] = Field(..., description="Block-specific content data")
    reasoning: str = Field(..., description="Explanation of why this content was generated")

    async def render(self, content: dict[str, Any], format=None) -> str:
        """Render implementation for LLM-generated blocks."""
        # For LLM blocks, content is already provided in the content field
        return str(self.content)  # Simple implementation for now


class DocumentGenerationResponse(BaseModel):
    """Complete LLM response for document generation."""

    enhanced_title: str = Field(..., description="Enhanced version of the document title")
    introduction: str = Field(default="", description="Brief introduction to the topic")
    blocks: list[DocumentContentBlock] = Field(
        ..., min_items=1, description="Array of content blocks that make up the document"
    )
    total_estimated_minutes: int = Field(
        ..., description="Total estimated time for the entire document"
    )
    actual_detail_level: int = Field(
        ..., ge=1, le=10, description="The detail level actually achieved (1-10)"
    )
    generation_reasoning: str = Field(
        ..., description="Overall reasoning for the document structure and content"
    )
    coverage_notes: str = Field(default="", description="Notes about topic coverage and scope")
    personalization_applied: list[str] = Field(
        default_factory=list, description="List of personalization features that were applied"
    )


# Example usage for different providers:
# OpenAI: completion = client.chat.completions.parse(response_format=DocumentGenerationResponse, ...)
# Gemini: response = client.models.generate_content(config={'response_schema': DocumentGenerationResponse})
# Anthropic: Use the schema in prompt instructions

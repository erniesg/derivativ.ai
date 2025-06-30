"""
Centralized Pydantic schemas for LLM structured outputs.
Used across OpenAI parse, Gemini response_schema, and Anthropic prompt instructions.
"""

from typing import Any

from pydantic import BaseModel, Field


class DocumentGenerationResponse(BaseModel):
    """
    Complete LLM response for document generation.

    IMPORTANT: All fields are required for OpenAI structured output compatibility.
    OpenAI requires every field in 'properties' to be listed in 'required'.

    Flattened structure to avoid nested schema issues with OpenAI.
    """

    enhanced_title: str = Field(..., description="Enhanced version of the document title")
    introduction: str = Field(..., description="Brief introduction to the topic")

    # Flattened block structure - OpenAI structured output compatible (no dict types)
    block_types: list[str] = Field(
        ...,
        min_items=1,
        description="Types of content blocks (e.g., 'learning_objectives', 'practice_questions')",
    )
    block_contents: list[str] = Field(
        ..., min_items=1, description="Content data for each block as JSON strings"
    )
    block_minutes: list[int] = Field(
        ..., min_items=1, description="Estimated minutes for each block"
    )
    block_reasoning: list[str] = Field(..., min_items=1, description="Reasoning for each block")

    total_estimated_minutes: int = Field(
        ..., description="Total estimated time for the entire document"
    )
    actual_detail_level: int = Field(
        ..., ge=1, le=10, description="The detail level actually achieved (1-10)"
    )
    generation_reasoning: str = Field(
        ..., description="Overall reasoning for the document structure and content"
    )
    coverage_notes: str = Field(..., description="Notes about topic coverage and scope")
    personalization_applied: list[str] = Field(
        ..., description="List of personalization features that were applied"
    )

    def to_content_blocks(self) -> list[dict[str, Any]]:
        """Convert the flattened response back to structured content blocks."""
        import json

        blocks = []

        for i in range(len(self.block_types)):
            # Parse JSON content string back to dictionary
            try:
                content_data = (
                    json.loads(self.block_contents[i]) if i < len(self.block_contents) else {}
                )
            except (json.JSONDecodeError, IndexError):
                # If it's not JSON, treat as plain text
                content_data = self.block_contents[i] if i < len(self.block_contents) else ""

            blocks.append(
                {
                    "block_type": self.block_types[i],
                    "block_content": content_data,
                    "estimated_minutes": self.block_minutes[i]
                    if i < len(self.block_minutes)
                    else 5,
                    "reasoning": self.block_reasoning[i] if i < len(self.block_reasoning) else "",
                    "order_index": i,
                }
            )

        return blocks


# Helper class for internal use (not sent to OpenAI)
class DocumentContentBlock(BaseModel):
    """A single content block - for internal processing after OpenAI response."""

    block_type: str = Field(..., description="Type identifier for this block")
    block_content: dict[str, Any] = Field(..., description="Block-specific content data")
    estimated_minutes: int = Field(..., description="Estimated time to complete this block")
    reasoning: str = Field(..., description="Explanation of why this content was generated")


# Example usage for different providers:
# OpenAI: completion = client.chat.completions.parse(response_format=DocumentGenerationResponse, ...)
# Gemini: response = client.models.generate_content(config={'response_schema': DocumentGenerationResponse})
# Anthropic: Use the schema in prompt instructions

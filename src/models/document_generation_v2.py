"""
Document generation v2 models with blocks-based architecture.

Provides models for the new flexible document generation system that uses
content blocks and LLM-driven content creation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from src.models.document_blueprints import BlockConfig
from src.models.document_models import DocumentType, ExportFormat
from src.models.enums import SubjectContentReference, Tier, TopicName, refs_by_topic


class GenerationApproach(str, Enum):
    """Approach for content generation."""

    LLM_DRIVEN = "llm_driven"  # LLM decides block selection
    RULE_BASED = "rule_based"  # System selects blocks by rules
    HYBRID = "hybrid"  # Rules select, LLM can override


class DocumentGenerationRequestV2(BaseModel):
    """Enhanced request for document generation with flexible constraints."""

    # Document specification
    document_type: DocumentType = Field(..., description="Type of document to generate")
    title: str = Field(..., description="Document title")

    # Flexible constraints (user provides what they know)
    target_duration_minutes: Optional[int] = Field(
        None, ge=5, le=120, description="Target time to complete/read the document"
    )
    detail_level: Optional[int] = Field(
        None, ge=1, le=10, description="Level of detail (1=minimal, 10=comprehensive)"
    )

    # If neither provided, we use defaults
    generation_approach: GenerationApproach = Field(
        default=GenerationApproach.LLM_DRIVEN, description="How to handle block selection"
    )

    # Content targeting
    topic: TopicName = Field(..., description="Main topic from Cambridge IGCSE syllabus")
    subtopics: list[str] = Field(default_factory=list, description="Specific subtopics to cover")
    tier: Tier = Field(default=Tier.CORE, description="Core or Extended tier")
    grade_level: Optional[int] = Field(None, ge=1, le=9, description="Target grade level")
    difficulty: Optional[str] = Field(
        None, description="Target difficulty: easy, medium, hard, mixed"
    )

    # Syllabus alignment
    subject_content_refs: list[SubjectContentReference] = Field(
        default_factory=list, description="Specific syllabus references to cover"
    )

    # Question integration
    include_questions: bool = Field(
        default=True, description="Whether to include practice questions"
    )
    num_questions: Optional[int] = Field(
        None, ge=1, le=50, description="Number of practice questions to include"
    )
    question_sources: list[str] = Field(
        default_factory=list, description="Specific question IDs to include"
    )

    # Customization
    custom_instructions: Optional[str] = Field(
        None, description="Custom instructions for content generation"
    )
    personalization_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Context for personalization (learning style, preferences)",
    )
    style_preferences: dict[str, Any] = Field(
        default_factory=dict, description="Style preferences (formal/casual, visual/textual)"
    )

    # Block overrides (advanced)
    force_include_blocks: list[str] = Field(
        default_factory=list, description="Block types to force include"
    )
    exclude_blocks: list[str] = Field(default_factory=list, description="Block types to exclude")

    # Output preferences
    include_answers: bool = Field(default=True, description="Include answers/solutions")
    teacher_version: bool = Field(default=False, description="Generate teacher version with extras")

    @field_validator("target_duration_minutes", "detail_level")
    @classmethod
    def validate_constraints(cls, v, info):
        """Validate constraints without setting defaults."""
        # Note: We don't set defaults here, they're handled in get_effective_detail_level()
        return v

    def get_effective_detail_level(self) -> int:
        """Get effective detail level, computing from time if needed."""
        if self.detail_level:
            return self.detail_level

        # Estimate from time constraint
        if self.target_duration_minutes:
            if self.target_duration_minutes <= 15:
                return 3
            elif self.target_duration_minutes <= 30:
                return 5
            elif self.target_duration_minutes <= 45:
                return 7
            else:
                return 9

        return 5  # Default medium

    def get_syllabus_refs(self) -> list[str]:
        """Get syllabus references for the topic and tier, if not explicitly provided."""
        if self.subject_content_refs:
            return [ref.value for ref in self.subject_content_refs]

        # Auto-generate from topic and tier
        return refs_by_topic(self.topic, self.tier)


class BlockGenerationResult(BaseModel):
    """Result of generating content for a single block."""

    block_type: str = Field(..., description="Type of content block")
    content: dict[str, Any] = Field(..., description="Generated content data")
    estimated_minutes: int = Field(..., description="Estimated time for this block")
    reasoning: Optional[str] = Field(None, description="LLM reasoning for content decisions")
    quality_score: Optional[float] = Field(
        None, ge=0, le=1, description="Self-assessed quality score"
    )


class DocumentContentStructure(BaseModel):
    """
    Structured output from LLM for document content.

    This is what the LLM generates in a single shot with all content.
    """

    # Document metadata
    enhanced_title: Optional[str] = Field(
        None, description="Enhanced title if different from request"
    )
    introduction: Optional[str] = Field(None, description="Document introduction paragraph")

    # Selected blocks with content
    blocks: list[BlockGenerationResult] = Field(..., description="Generated content blocks")

    # Metadata
    total_estimated_minutes: int = Field(..., description="Total estimated completion time")
    actual_detail_level: int = Field(..., ge=1, le=10, description="Actual detail level achieved")

    # Generation insights
    generation_reasoning: str = Field(..., description="Overall reasoning for content decisions")
    coverage_notes: Optional[str] = Field(None, description="Notes on topic coverage and gaps")
    personalization_applied: list[str] = Field(
        default_factory=list, description="List of personalizations applied"
    )


class SelectedBlock(BaseModel):
    """A block selected for inclusion in a document."""

    block_config: BlockConfig = Field(..., description="Block configuration")
    content_guidelines: dict[str, Any] = Field(
        default_factory=dict, description="Guidelines for content generation"
    )
    estimated_content_volume: dict[str, int] = Field(
        default_factory=dict, description="Expected content amounts"
    )


class BlockSelectionResult(BaseModel):
    """Result of block selection process."""

    selected_blocks: list[SelectedBlock] = Field(..., description="Blocks selected for inclusion")
    total_estimated_minutes: int = Field(..., description="Total estimated time")
    selection_reasoning: str = Field(..., description="Reasoning for selection decisions")
    excluded_blocks: list[str] = Field(default_factory=list, description="Blocks excluded and why")


class GeneratedDocumentV2(BaseModel):
    """A document generated using the blocks-based approach."""

    # Metadata
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(..., description="Document title")
    document_type: DocumentType = Field(..., description="Type of document")

    # Generation info
    generated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(), description="Generation timestamp"
    )
    generation_request: DocumentGenerationRequestV2 = Field(..., description="Original request")
    blueprint_used: str = Field(..., description="Blueprint name used")

    # Content
    content_structure: DocumentContentStructure = Field(
        ..., description="Generated content structure"
    )
    rendered_blocks: dict[str, str] = Field(
        default_factory=dict, description="Rendered block content by type"
    )

    # Metrics
    total_estimated_minutes: int = Field(..., description="Total estimated completion time")
    actual_detail_level: int = Field(..., description="Detail level achieved")
    word_count: int = Field(default=0, description="Total word count")

    # Quality indicators
    coverage_completeness: float = Field(
        default=1.0, ge=0, le=1, description="How completely topics were covered"
    )
    personalization_score: float = Field(
        default=0.0, ge=0, le=1, description="How well personalized"
    )

    # Export readiness
    available_formats: list[ExportFormat] = Field(
        default_factory=list, description="Formats this can be exported to"
    )

    def get_markdown_content(self) -> str:
        """Get full document content as markdown."""
        sections = []

        # Title
        sections.append(f"# {self.title}\n")

        # Introduction if present
        if self.content_structure.introduction:
            sections.append(f"{self.content_structure.introduction}\n")

        # Rendered blocks
        for block in self.content_structure.blocks:
            if block.block_type in self.rendered_blocks:
                sections.append(self.rendered_blocks[block.block_type])
                sections.append("")  # Empty line between blocks

        return "\n".join(sections)


class DocumentGenerationResultV2(BaseModel):
    """Result of document generation process."""

    success: bool = Field(..., description="Whether generation succeeded")
    document: Optional[GeneratedDocumentV2] = Field(
        None, description="Generated document if successful"
    )
    error_message: Optional[str] = Field(None, description="Error message if failed")

    # Performance metrics
    processing_time: float = Field(..., description="Total time in seconds")
    llm_calls: int = Field(default=1, description="Number of LLM calls made")
    tokens_used: Optional[int] = Field(None, description="Total tokens consumed")

    # Insights
    generation_insights: dict[str, Any] = Field(
        default_factory=dict, description="Insights from generation process"
    )


# JSON Schema for structured output from LLM
DOCUMENT_CONTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "enhanced_title": {"type": "string"},
        "introduction": {"type": "string"},
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "block_type": {"type": "string"},
                    "content": {"type": "object"},
                    "estimated_minutes": {"type": "integer"},
                    "reasoning": {"type": "string"},
                },
                "required": ["block_type", "content", "estimated_minutes"],
            },
        },
        "total_estimated_minutes": {"type": "integer"},
        "actual_detail_level": {"type": "integer", "minimum": 1, "maximum": 10},
        "generation_reasoning": {"type": "string"},
        "coverage_notes": {"type": "string"},
        "personalization_applied": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "blocks",
        "total_estimated_minutes",
        "actual_detail_level",
        "generation_reasoning",
    ],
}


# Block-specific content schemas
BLOCK_CONTENT_SCHEMAS = {
    "learning_objectives": {
        "type": "object",
        "properties": {
            "objectives": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 8,
            }
        },
        "required": ["objectives"],
    },
    "concept_explanation": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "introduction": {"type": "string"},
            "concepts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "explanation": {"type": "string"},
                        "example": {"type": "string"},
                    },
                    "required": ["name", "explanation"],
                },
            },
        },
        "required": ["concepts"],
    },
    "worked_example": {
        "type": "object",
        "properties": {
            "examples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "problem": {"type": "string"},
                        "steps": {"type": "array", "items": {"type": "string"}},
                        "answer": {"type": "string"},
                        "explanation": {"type": "string"},
                    },
                    "required": ["problem", "steps", "answer"],
                },
            }
        },
        "required": ["examples"],
    },
    "practice_questions": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "marks": {"type": "integer", "minimum": 1},
                        "difficulty": {"type": "string"},
                        "hint": {"type": "string"},
                        "answer": {"type": "string"},
                    },
                    "required": ["text", "marks"],
                },
            },
            "include_answers": {"type": "boolean"},
        },
        "required": ["questions"],
    },
    "quick_reference": {
        "type": "object",
        "properties": {
            "formulas": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "expression": {"type": "string"}},
                    "required": ["name", "expression"],
                },
            },
            "definitions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"term": {"type": "string"}, "definition": {"type": "string"}},
                    "required": ["term", "definition"],
                },
            },
            "key_facts": {"type": "array", "items": {"type": "string"}},
        },
    },
    "summary": {
        "type": "object",
        "properties": {
            "key_points": {"type": "array", "items": {"type": "string"}},
            "insights": {"type": "array", "items": {"type": "string"}},
            "next_steps": {"type": "array", "items": {"type": "string"}},
        },
    },
}

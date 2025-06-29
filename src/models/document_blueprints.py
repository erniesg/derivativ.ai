"""
Document blueprint models for flexible document generation.

Defines how different document types are composed from content blocks,
with dynamic selection based on time and detail level constraints.
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from src.models.document_models import DocumentType, ExportFormat


class BlockPriority(str, Enum):
    """Priority levels for content blocks in documents."""
    REQUIRED = "required"  # Must be included
    HIGH = "high"  # Include unless very constrained
    MEDIUM = "medium"  # Include if time/detail allows
    LOW = "low"  # Include only with high detail/time
    OPTIONAL = "optional"  # User preference


class BlockConfig(BaseModel):
    """Configuration for a content block within a document."""

    block_type: str = Field(..., description="Type of content block")
    priority: BlockPriority = Field(
        default=BlockPriority.MEDIUM,
        description="Priority for inclusion"
    )
    min_detail_level: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Minimum detail level to include this block"
    )
    max_detail_level: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Maximum detail level to include this block"
    )
    time_weight: float = Field(
        default=1.0,
        gt=0,
        description="Multiplier for base time estimate"
    )
    customization_hints: dict[str, Any] = Field(
        default_factory=dict,
        description="Hints for content generation"
    )

    @field_validator("max_detail_level")
    @classmethod
    def validate_max_detail(cls, v, info):
        """Ensure max_detail_level >= min_detail_level if set."""
        if v is not None and info.data and "min_detail_level" in info.data:
            if v < info.data["min_detail_level"]:
                raise ValueError("max_detail_level must be >= min_detail_level")
        return v

    def is_applicable(self, detail_level: int) -> bool:
        """Check if this block should be included at given detail level."""
        if detail_level < self.min_detail_level:
            return False
        if self.max_detail_level and detail_level > self.max_detail_level:
            return False
        return True


class DocumentBlueprint(BaseModel):
    """
    Blueprint defining how a document type is constructed from blocks.

    Instead of rigid structures, blueprints define available blocks
    and rules for their selection based on constraints.
    """

    name: str = Field(..., description="Blueprint name")
    document_type: DocumentType = Field(..., description="Type of document")
    description: str = Field(..., description="Blueprint description")

    # Block configurations
    blocks: list[BlockConfig] = Field(..., description="Available blocks")

    # Output format support
    supported_formats: list[ExportFormat] = Field(
        default_factory=lambda: [
            ExportFormat.MARKDOWN,
            ExportFormat.HTML,
            ExportFormat.PDF
        ],
        description="Supported export formats"
    )

    # Time estimation factors
    base_overhead_minutes: int = Field(
        default=2,
        description="Fixed time overhead (intro, navigation)"
    )
    time_efficiency_factor: float = Field(
        default=1.0,
        gt=0,
        description="Efficiency factor for this document type"
    )

    # Detail level guidance
    detail_level_descriptions: dict[int, str] = Field(
        default_factory=lambda: {
            1: "Essential concepts only",
            3: "Key points with basic examples",
            5: "Standard coverage with practice",
            7: "Extended with applications",
            10: "Comprehensive with enrichment"
        },
        description="Descriptions for detail levels"
    )

    def get_applicable_blocks(self, detail_level: int) -> list[BlockConfig]:
        """Get blocks applicable at given detail level."""
        return [
            block for block in self.blocks
            if block.is_applicable(detail_level)
        ]

    def get_required_blocks(self) -> list[BlockConfig]:
        """Get blocks that must be included."""
        return [
            block for block in self.blocks
            if block.priority == BlockPriority.REQUIRED
        ]

    def estimate_time(self, blocks: list[BlockConfig], content_volume: dict[str, int]) -> int:
        """
        Estimate time for given blocks and content volume.

        Args:
            blocks: Selected block configurations
            content_volume: Expected content amounts (e.g., num_questions)

        Returns:
            Estimated time in minutes
        """
        from src.models.content_blocks import get_block_class

        total_minutes = self.base_overhead_minutes

        for block_config in blocks:
            block_class = get_block_class(block_config.block_type)
            block_instance = block_class()
            base_time = block_instance.estimated_minutes

            # Apply time weight from configuration
            adjusted_time = base_time * block_config.time_weight

            # Adjust for content volume if applicable
            if block_config.block_type == "practice_questions":
                num_items = content_volume.get("num_questions", 5)
                adjusted_time *= num_items
            elif block_config.block_type == "worked_example":
                num_items = content_volume.get("num_examples", 2)
                adjusted_time *= num_items

            total_minutes += adjusted_time

        # Apply document type efficiency factor
        return int(total_minutes * self.time_efficiency_factor)


# Pre-defined blueprints for each document type
def create_worksheet_blueprint() -> DocumentBlueprint:
    """Create blueprint for worksheet documents."""
    return DocumentBlueprint(
        name="Standard Worksheet",
        document_type=DocumentType.WORKSHEET,
        description="Practice-focused document with exercises and solutions",
        blocks=[
            # Always included
            BlockConfig(
                block_type="practice_questions",
                priority=BlockPriority.REQUIRED,
                min_detail_level=1
            ),
            # Include at medium detail
            BlockConfig(
                block_type="learning_objectives",
                priority=BlockPriority.HIGH,
                min_detail_level=3
            ),
            BlockConfig(
                block_type="worked_example",
                priority=BlockPriority.HIGH,
                min_detail_level=4,
                customization_hints={"num_examples": "1-3"}
            ),
            # Include at higher detail
            BlockConfig(
                block_type="quick_reference",
                priority=BlockPriority.MEDIUM,
                min_detail_level=5
            ),
            BlockConfig(
                block_type="concept_explanation",
                priority=BlockPriority.MEDIUM,
                min_detail_level=6,
                max_detail_level=9,  # Skip at highest detail
                customization_hints={"depth": "brief"}
            ),
            # Optional enrichment
            BlockConfig(
                block_type="summary",
                priority=BlockPriority.LOW,
                min_detail_level=7
            ),
        ],
        supported_formats=[
            ExportFormat.PDF,
            ExportFormat.DOCX,
            ExportFormat.HTML
        ],
        time_efficiency_factor=0.9  # Worksheets are quick to complete
    )


def create_notes_blueprint() -> DocumentBlueprint:
    """Create blueprint for study notes documents."""
    return DocumentBlueprint(
        name="Study Notes",
        document_type=DocumentType.NOTES,
        description="Explanatory document with theory and examples",
        blocks=[
            # Core content
            BlockConfig(
                block_type="concept_explanation",
                priority=BlockPriority.REQUIRED,
                min_detail_level=1
            ),
            BlockConfig(
                block_type="summary",
                priority=BlockPriority.REQUIRED,
                min_detail_level=1
            ),
            # Enhanced content
            BlockConfig(
                block_type="learning_objectives",
                priority=BlockPriority.HIGH,
                min_detail_level=2
            ),
            BlockConfig(
                block_type="worked_example",
                priority=BlockPriority.HIGH,
                min_detail_level=3,
                customization_hints={"num_examples": "2-4"}
            ),
            BlockConfig(
                block_type="quick_reference",
                priority=BlockPriority.MEDIUM,
                min_detail_level=4
            ),
            # Practice elements
            BlockConfig(
                block_type="practice_questions",
                priority=BlockPriority.MEDIUM,
                min_detail_level=5,
                customization_hints={"num_questions": "3-5"}
            ),
        ],
        supported_formats=[
            ExportFormat.PDF,
            ExportFormat.MARKDOWN,
            ExportFormat.HTML
        ],
        time_efficiency_factor=1.0
    )


def create_textbook_blueprint() -> DocumentBlueprint:
    """Create blueprint for mini-textbook documents."""
    return DocumentBlueprint(
        name="Mini Textbook",
        document_type=DocumentType.TEXTBOOK,
        description="Comprehensive learning material with full coverage",
        blocks=[
            # Foundation
            BlockConfig(
                block_type="learning_objectives",
                priority=BlockPriority.REQUIRED,
                min_detail_level=1
            ),
            BlockConfig(
                block_type="concept_explanation",
                priority=BlockPriority.REQUIRED,
                min_detail_level=1,
                customization_hints={"depth": "comprehensive"}
            ),
            # Examples and practice
            BlockConfig(
                block_type="worked_example",
                priority=BlockPriority.REQUIRED,
                min_detail_level=2,
                customization_hints={"num_examples": "3-6"}
            ),
            BlockConfig(
                block_type="practice_questions",
                priority=BlockPriority.HIGH,
                min_detail_level=3,
                customization_hints={"difficulty": "graded"}
            ),
            # Reference materials
            BlockConfig(
                block_type="quick_reference",
                priority=BlockPriority.HIGH,
                min_detail_level=4
            ),
            BlockConfig(
                block_type="summary",
                priority=BlockPriority.HIGH,
                min_detail_level=3
            ),
        ],
        supported_formats=[
            ExportFormat.PDF,
            ExportFormat.HTML,
            ExportFormat.LATEX
        ],
        base_overhead_minutes=5,  # More navigation/structure
        time_efficiency_factor=1.2  # Textbooks take longer to digest
    )


def create_slides_blueprint() -> DocumentBlueprint:
    """Create blueprint for presentation slides."""
    return DocumentBlueprint(
        name="Presentation Slides",
        document_type=DocumentType.SLIDES,
        description="Visual presentation material for teaching",
        blocks=[
            # Essential slides
            BlockConfig(
                block_type="learning_objectives",
                priority=BlockPriority.REQUIRED,
                min_detail_level=1,
                customization_hints={"format": "bullet_points"}
            ),
            BlockConfig(
                block_type="concept_explanation",
                priority=BlockPriority.REQUIRED,
                min_detail_level=1,
                customization_hints={"format": "visual", "depth": "concise"}
            ),
            BlockConfig(
                block_type="summary",
                priority=BlockPriority.REQUIRED,
                min_detail_level=1
            ),
            # Enhanced content
            BlockConfig(
                block_type="worked_example",
                priority=BlockPriority.HIGH,
                min_detail_level=3,
                customization_hints={"format": "step_by_step", "num_examples": "1-2"}
            ),
            BlockConfig(
                block_type="quick_reference",
                priority=BlockPriority.MEDIUM,
                min_detail_level=4,
                customization_hints={"format": "visual_chart"}
            ),
            # Interactive elements
            BlockConfig(
                block_type="practice_questions",
                priority=BlockPriority.MEDIUM,
                min_detail_level=5,
                customization_hints={"format": "interactive", "num_questions": "2-3"}
            ),
        ],
        supported_formats=[
            ExportFormat.SLIDES_PPTX,
            ExportFormat.PDF,
            ExportFormat.HTML
        ],
        time_efficiency_factor=0.8  # Slides are quicker to review
    )


# Blueprint registry
BLUEPRINT_REGISTRY = {
    DocumentType.WORKSHEET: create_worksheet_blueprint,
    DocumentType.NOTES: create_notes_blueprint,
    DocumentType.TEXTBOOK: create_textbook_blueprint,
    DocumentType.SLIDES: create_slides_blueprint,
}


def get_blueprint(document_type: DocumentType) -> DocumentBlueprint:
    """Get blueprint for a document type."""
    if document_type not in BLUEPRINT_REGISTRY:
        raise ValueError(f"No blueprint defined for document type: {document_type}")
    return BLUEPRINT_REGISTRY[document_type]()

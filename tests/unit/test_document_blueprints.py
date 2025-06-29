"""Unit tests for document blueprint models."""

import pytest

from src.models.document_blueprints import (
    BlockConfig,
    BlockPriority,
    DocumentBlueprint,
    create_notes_blueprint,
    create_slides_blueprint,
    create_textbook_blueprint,
    create_worksheet_blueprint,
    get_blueprint,
)
from src.models.document_models import DocumentType, ExportFormat


class TestBlockConfig:
    """Test BlockConfig model."""

    def test_block_config_creation(self):
        """Test basic block configuration creation."""
        config = BlockConfig(
            block_type="learning_objectives",
            priority=BlockPriority.REQUIRED,
            min_detail_level=1,
            max_detail_level=5
        )

        assert config.block_type == "learning_objectives"
        assert config.priority == BlockPriority.REQUIRED
        assert config.min_detail_level == 1
        assert config.max_detail_level == 5
        assert config.time_weight == 1.0  # Default

    def test_block_config_is_applicable(self):
        """Test block applicability checking."""
        config = BlockConfig(
            block_type="test_block",
            min_detail_level=3,
            max_detail_level=7
        )

        assert not config.is_applicable(2)  # Below minimum
        assert config.is_applicable(3)     # At minimum
        assert config.is_applicable(5)     # In range
        assert config.is_applicable(7)     # At maximum
        assert not config.is_applicable(8)  # Above maximum

    def test_block_config_no_max_detail(self):
        """Test block config without max detail level."""
        config = BlockConfig(
            block_type="test_block",
            min_detail_level=3
        )

        assert not config.is_applicable(2)
        assert config.is_applicable(3)
        assert config.is_applicable(10)  # No upper limit

    def test_block_config_validation_error(self):
        """Test validation error when max < min detail."""
        with pytest.raises(ValueError, match="max_detail_level must be >= min_detail_level"):
            BlockConfig(
                block_type="test_block",
                min_detail_level=5,
                max_detail_level=3  # Invalid: less than min
            )


class TestDocumentBlueprint:
    """Test DocumentBlueprint model."""

    def test_blueprint_creation(self):
        """Test basic blueprint creation."""
        blocks = [
            BlockConfig(block_type="learning_objectives", priority=BlockPriority.REQUIRED),
            BlockConfig(block_type="summary", priority=BlockPriority.HIGH)
        ]

        blueprint = DocumentBlueprint(
            name="Test Blueprint",
            document_type=DocumentType.WORKSHEET,
            description="Test description",
            blocks=blocks
        )

        assert blueprint.name == "Test Blueprint"
        assert blueprint.document_type == DocumentType.WORKSHEET
        assert len(blueprint.blocks) == 2
        assert blueprint.base_overhead_minutes == 2  # Default

    def test_get_applicable_blocks(self):
        """Test getting blocks applicable at detail level."""
        blocks = [
            BlockConfig(block_type="block1", min_detail_level=1, max_detail_level=5),
            BlockConfig(block_type="block2", min_detail_level=3, max_detail_level=8),
            BlockConfig(block_type="block3", min_detail_level=6)
        ]

        blueprint = DocumentBlueprint(
            name="Test",
            document_type=DocumentType.NOTES,
            description="Test",
            blocks=blocks
        )

        # Detail level 2: only block1
        applicable = blueprint.get_applicable_blocks(2)
        assert len(applicable) == 1
        assert applicable[0].block_type == "block1"

        # Detail level 4: block1 and block2
        applicable = blueprint.get_applicable_blocks(4)
        assert len(applicable) == 2
        assert {b.block_type for b in applicable} == {"block1", "block2"}

        # Detail level 7: block2 and block3
        applicable = blueprint.get_applicable_blocks(7)
        assert len(applicable) == 2
        assert {b.block_type for b in applicable} == {"block2", "block3"}

    def test_get_required_blocks(self):
        """Test getting only required blocks."""
        blocks = [
            BlockConfig(block_type="required1", priority=BlockPriority.REQUIRED),
            BlockConfig(block_type="high1", priority=BlockPriority.HIGH),
            BlockConfig(block_type="required2", priority=BlockPriority.REQUIRED),
            BlockConfig(block_type="medium1", priority=BlockPriority.MEDIUM)
        ]

        blueprint = DocumentBlueprint(
            name="Test",
            document_type=DocumentType.NOTES,
            description="Test",
            blocks=blocks
        )

        required = blueprint.get_required_blocks()
        assert len(required) == 2
        assert {b.block_type for b in required} == {"required1", "required2"}

    def test_estimate_time_basic(self):
        """Test basic time estimation."""
        blocks = [
            BlockConfig(block_type="learning_objectives"),  # 2 min base
            BlockConfig(block_type="summary")  # 3 min base
        ]

        blueprint = DocumentBlueprint(
            name="Test",
            document_type=DocumentType.NOTES,
            description="Test",
            blocks=blocks,
            base_overhead_minutes=5
        )

        selected_blocks = blocks
        content_volume = {}

        estimated = blueprint.estimate_time(selected_blocks, content_volume)
        # 5 (overhead) + 2 (learning_objectives) + 3 (summary) = 10
        assert estimated == 10

    def test_estimate_time_with_content_volume(self):
        """Test time estimation with content scaling."""
        blocks = [
            BlockConfig(block_type="practice_questions"),  # 3 min base
            BlockConfig(block_type="worked_example", time_weight=1.5)  # 5 min base * 1.5
        ]

        blueprint = DocumentBlueprint(
            name="Test",
            document_type=DocumentType.WORKSHEET,
            description="Test",
            blocks=blocks,
            base_overhead_minutes=2,
            time_efficiency_factor=0.9
        )

        content_volume = {
            "num_questions": 4,  # 4 questions * 3 min = 12 min
            "num_examples": 2    # 2 examples * 5 * 1.5 = 15 min
        }

        estimated = blueprint.estimate_time(blocks, content_volume)
        # (2 + 12 + 15) * 0.9 = 26.1 → 26
        assert estimated == 26


class TestWorksheetBlueprint:
    """Test worksheet blueprint."""

    def test_worksheet_blueprint_structure(self):
        """Test worksheet blueprint has correct structure."""
        blueprint = create_worksheet_blueprint()

        assert blueprint.name == "Standard Worksheet"
        assert blueprint.document_type == DocumentType.WORKSHEET
        assert ExportFormat.PDF in blueprint.supported_formats
        assert blueprint.time_efficiency_factor == 0.9

        # Should have practice questions as required
        required = blueprint.get_required_blocks()
        assert len(required) == 1
        assert required[0].block_type == "practice_questions"

        # Check blocks by detail level
        low_detail = blueprint.get_applicable_blocks(2)
        block_types = {b.block_type for b in low_detail}
        assert "practice_questions" in block_types

        high_detail = blueprint.get_applicable_blocks(8)
        high_block_types = {b.block_type for b in high_detail}
        assert len(high_block_types) > len(block_types)
        assert "quick_reference" in high_block_types


class TestNotesBlueprint:
    """Test notes blueprint."""

    def test_notes_blueprint_structure(self):
        """Test notes blueprint has correct structure."""
        blueprint = create_notes_blueprint()

        assert blueprint.name == "Study Notes"
        assert blueprint.document_type == DocumentType.NOTES
        assert ExportFormat.MARKDOWN in blueprint.supported_formats
        assert blueprint.time_efficiency_factor == 1.0

        # Should have concept explanation and summary as required
        required = blueprint.get_required_blocks()
        required_types = {b.block_type for b in required}
        assert "concept_explanation" in required_types
        assert "summary" in required_types


class TestTextbookBlueprint:
    """Test textbook blueprint."""

    def test_textbook_blueprint_structure(self):
        """Test textbook blueprint has correct structure."""
        blueprint = create_textbook_blueprint()

        assert blueprint.name == "Mini Textbook"
        assert blueprint.document_type == DocumentType.TEXTBOOK
        assert ExportFormat.LATEX in blueprint.supported_formats
        assert blueprint.time_efficiency_factor == 1.2
        assert blueprint.base_overhead_minutes == 5

        # Should have comprehensive required blocks
        required = blueprint.get_required_blocks()
        required_types = {b.block_type for b in required}
        assert "learning_objectives" in required_types
        assert "concept_explanation" in required_types
        assert "worked_example" in required_types


class TestSlidesBlueprint:
    """Test slides blueprint."""

    def test_slides_blueprint_structure(self):
        """Test slides blueprint has correct structure."""
        blueprint = create_slides_blueprint()

        assert blueprint.name == "Presentation Slides"
        assert blueprint.document_type == DocumentType.SLIDES
        assert ExportFormat.SLIDES_PPTX in blueprint.supported_formats
        assert blueprint.time_efficiency_factor == 0.8

        # Check customization hints for visual format
        concept_blocks = [
            b for b in blueprint.blocks
            if b.block_type == "concept_explanation"
        ]
        assert len(concept_blocks) == 1
        hints = concept_blocks[0].customization_hints
        assert hints.get("format") == "visual"
        assert hints.get("depth") == "concise"


class TestBlueprintRegistry:
    """Test blueprint registry functions."""

    def test_get_blueprint_valid_types(self):
        """Test getting blueprints for all document types."""
        worksheet = get_blueprint(DocumentType.WORKSHEET)
        assert worksheet.document_type == DocumentType.WORKSHEET

        notes = get_blueprint(DocumentType.NOTES)
        assert notes.document_type == DocumentType.NOTES

        textbook = get_blueprint(DocumentType.TEXTBOOK)
        assert textbook.document_type == DocumentType.TEXTBOOK

        slides = get_blueprint(DocumentType.SLIDES)
        assert slides.document_type == DocumentType.SLIDES

    def test_blueprint_registry_completeness(self):
        """Test that all document types have blueprints."""
        for doc_type in DocumentType:
            blueprint = get_blueprint(doc_type)
            assert blueprint.document_type == doc_type
            assert len(blueprint.blocks) > 0

    def test_all_blueprints_have_required_blocks(self):
        """Test that all blueprints have at least one required block."""
        for doc_type in DocumentType:
            blueprint = get_blueprint(doc_type)
            required = blueprint.get_required_blocks()
            assert len(required) > 0, f"{doc_type} blueprint has no required blocks"


class TestBlueprintDifferences:
    """Test that blueprints are meaningfully different."""

    def test_different_block_compositions(self):
        """Test that different document types have different block emphasis."""
        worksheet = get_blueprint(DocumentType.WORKSHEET)
        notes = get_blueprint(DocumentType.NOTES)
        textbook = get_blueprint(DocumentType.TEXTBOOK)
        slides = get_blueprint(DocumentType.SLIDES)

        # Worksheet focuses on practice
        worksheet_required = {b.block_type for b in worksheet.get_required_blocks()}
        assert "practice_questions" in worksheet_required

        # Notes focuses on explanation
        notes_required = {b.block_type for b in notes.get_required_blocks()}
        assert "concept_explanation" in notes_required

        # Textbook is comprehensive
        textbook_required = {b.block_type for b in textbook.get_required_blocks()}
        assert len(textbook_required) >= 3  # Multiple required blocks

        # Slides has visual customizations
        slides_concept = next(
            b for b in slides.blocks
            if b.block_type == "concept_explanation"
        )
        assert "visual" in slides_concept.customization_hints.get("format", "")

    def test_different_time_factors(self):
        """Test that document types have different time characteristics."""
        worksheet = get_blueprint(DocumentType.WORKSHEET)
        notes = get_blueprint(DocumentType.NOTES)
        textbook = get_blueprint(DocumentType.TEXTBOOK)
        slides = get_blueprint(DocumentType.SLIDES)

        # Different efficiency factors
        factors = [
            worksheet.time_efficiency_factor,
            notes.time_efficiency_factor,
            textbook.time_efficiency_factor,
            slides.time_efficiency_factor
        ]

        # Should have different values
        assert len(set(factors)) > 1

        # Slides should be fastest, textbook slowest
        assert slides.time_efficiency_factor < textbook.time_efficiency_factor

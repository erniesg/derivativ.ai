"""Unit tests for document generation service v2."""

from unittest.mock import AsyncMock, Mock

import pytest

from src.models.document_generation_v2 import (
    DocumentGenerationRequestV2,
    GenerationApproach,
)
from src.models.document_models import DocumentType
from src.models.enums import Tier
from src.services.document_generation_service_v2 import (
    BlockSelector,
    DocumentGenerationServiceV2,
)


class TestDocumentGenerationRequestV2:
    """Test the enhanced request model."""

    def test_request_with_time_constraint(self):
        """Test request with only time constraint."""
        request = DocumentGenerationRequestV2(
            document_type=DocumentType.WORKSHEET,
            title="Quadratic Equations Practice",
            topic="Quadratic Equations",
            target_duration_minutes=25
        )

        assert request.document_type == DocumentType.WORKSHEET
        assert request.target_duration_minutes == 25
        assert request.detail_level is None  # Will be computed
        assert request.get_effective_detail_level() == 5  # 25 min → level 5

    def test_request_with_detail_constraint(self):
        """Test request with only detail level constraint."""
        request = DocumentGenerationRequestV2(
            document_type=DocumentType.NOTES,
            title="Algebra Basics",
            topic="Algebra",
            detail_level=7
        )

        assert request.detail_level == 7
        assert request.target_duration_minutes is None  # Will be computed
        assert request.get_effective_detail_level() == 7

    def test_request_with_both_constraints(self):
        """Test request with both time and detail constraints."""
        request = DocumentGenerationRequestV2(
            document_type=DocumentType.TEXTBOOK,
            title="Comprehensive Geometry",
            topic="Geometry",
            target_duration_minutes=45,
            detail_level=8
        )

        assert request.target_duration_minutes == 45
        assert request.detail_level == 8
        assert request.get_effective_detail_level() == 8  # Uses explicit detail level

    def test_request_default_values(self):
        """Test request uses appropriate defaults."""
        request = DocumentGenerationRequestV2(
            document_type=DocumentType.SLIDES,
            title="Quick Trigonometry",
            topic="Trigonometry"
        )

        assert request.tier == Tier.CORE
        assert request.generation_approach == GenerationApproach.LLM_DRIVEN
        assert request.include_questions is True
        assert request.include_answers is True
        assert request.teacher_version is False

    def test_get_effective_detail_level_from_time(self):
        """Test detail level computation from time."""
        test_cases = [
            (10, 3),   # Short time → low detail
            (20, 5),   # Medium time → medium detail
            (40, 7),   # Longer time → high detail
            (60, 9),   # Very long → very high detail
        ]

        for minutes, expected_level in test_cases:
            request = DocumentGenerationRequestV2(
                document_type=DocumentType.WORKSHEET,
                title="Test",
                topic="Test",
                target_duration_minutes=minutes
            )
            assert request.get_effective_detail_level() == expected_level


class TestBlockSelector:
    """Test block selection logic."""

    @pytest.fixture
    def mock_llm_factory(self):
        """Mock LLM factory."""
        factory = Mock()
        service = Mock()
        service.generate_non_stream = AsyncMock()
        factory.get_service.return_value = service
        return factory

    @pytest.fixture
    def mock_prompt_manager(self):
        """Mock prompt manager."""
        manager = Mock()
        manager.render_prompt = AsyncMock(return_value="Test prompt")
        return manager

    @pytest.fixture
    def block_selector(self, mock_llm_factory, mock_prompt_manager):
        """Create block selector instance."""
        return BlockSelector(mock_llm_factory, mock_prompt_manager)

    @pytest.mark.asyncio
    async def test_rule_based_selection_worksheet(self, block_selector):
        """Test rule-based selection for worksheet."""
        request = DocumentGenerationRequestV2(
            document_type=DocumentType.WORKSHEET,
            title="Algebra Practice",
            topic="Algebra",
            target_duration_minutes=20,
            generation_approach=GenerationApproach.RULE_BASED
        )

        result = await block_selector.select_blocks(request)

        assert len(result.selected_blocks) > 0
        assert result.total_estimated_minutes > 0

        # Should include practice questions for worksheet
        block_types = {block.block_config.block_type for block in result.selected_blocks}
        assert "practice_questions" in block_types

    @pytest.mark.asyncio
    async def test_rule_based_selection_notes(self, block_selector):
        """Test rule-based selection for notes."""
        request = DocumentGenerationRequestV2(
            document_type=DocumentType.NOTES,
            title="Geometry Concepts",
            topic="Geometry",
            detail_level=6,
            generation_approach=GenerationApproach.RULE_BASED
        )

        result = await block_selector.select_blocks(request)

        assert len(result.selected_blocks) > 0

        # Should include concept explanation for notes
        block_types = {block.block_config.block_type for block in result.selected_blocks}
        assert "concept_explanation" in block_types
        assert "summary" in block_types

    @pytest.mark.asyncio
    async def test_time_constraint_respected(self, block_selector):
        """Test that time constraints are respected in selection."""
        short_request = DocumentGenerationRequestV2(
            document_type=DocumentType.WORKSHEET,
            title="Quick Practice",
            topic="Fractions",
            target_duration_minutes=10,
            generation_approach=GenerationApproach.RULE_BASED
        )

        long_request = DocumentGenerationRequestV2(
            document_type=DocumentType.WORKSHEET,
            title="Extended Practice",
            topic="Fractions",
            target_duration_minutes=45,
            generation_approach=GenerationApproach.RULE_BASED
        )

        short_result = await block_selector.select_blocks(short_request)
        long_result = await block_selector.select_blocks(long_request)

        # Longer time should allow more blocks
        assert len(long_result.selected_blocks) >= len(short_result.selected_blocks)
        assert long_result.total_estimated_minutes >= short_result.total_estimated_minutes

    @pytest.mark.asyncio
    async def test_user_overrides_applied(self, block_selector):
        """Test that user block overrides are applied."""
        request = DocumentGenerationRequestV2(
            document_type=DocumentType.NOTES,
            title="Custom Notes",
            topic="Statistics",
            force_include_blocks=["quick_reference"],
            exclude_blocks=["practice_questions"],
            generation_approach=GenerationApproach.RULE_BASED
        )

        result = await block_selector.select_blocks(request)

        block_types = {block.block_config.block_type for block in result.selected_blocks}

        # Should include forced block
        assert "quick_reference" in block_types

        # Should exclude specified block
        assert "practice_questions" not in block_types


class TestDocumentGenerationServiceV2:
    """Test the main document generation service."""

    @pytest.fixture
    def mock_llm_factory(self):
        """Mock LLM factory."""
        factory = Mock()
        service = Mock()

        # Mock successful JSON response
        mock_response = Mock()
        mock_response.content = """{
            "blocks": [
                {
                    "block_type": "learning_objectives",
                    "content": {"objectives": ["Understand algebra", "Solve equations"]},
                    "estimated_minutes": 2
                },
                {
                    "block_type": "practice_questions",
                    "content": {"questions": [{"text": "Solve x + 2 = 5", "marks": 2}]},
                    "estimated_minutes": 6
                }
            ],
            "total_estimated_minutes": 15,
            "actual_detail_level": 5,
            "generation_reasoning": "Selected basic blocks for worksheet"
        }"""

        service.generate_non_stream = AsyncMock(return_value=mock_response)
        factory.get_service.return_value = service
        return factory

    @pytest.fixture
    def mock_prompt_manager(self):
        """Mock prompt manager."""
        manager = Mock()
        manager.render_prompt = AsyncMock(return_value="Test prompt")
        return manager

    @pytest.fixture
    def service(self, mock_llm_factory, mock_prompt_manager):
        """Create service instance."""
        return DocumentGenerationServiceV2(mock_llm_factory, mock_prompt_manager)

    @pytest.mark.asyncio
    async def test_successful_generation(self, service):
        """Test successful document generation."""
        request = DocumentGenerationRequestV2(
            document_type=DocumentType.WORKSHEET,
            title="Algebra Basics",
            topic="Linear Equations",
            target_duration_minutes=20
        )

        result = await service.generate_document(request)

        assert result.success is True
        assert result.document is not None
        assert result.processing_time > 0

        # Check document structure
        document = result.document
        assert document.title == "Algebra Basics"
        assert document.document_type == DocumentType.WORKSHEET
        assert len(document.content_structure.blocks) > 0
        assert document.total_estimated_minutes > 0

    @pytest.mark.asyncio
    async def test_word_count_calculation(self, service):
        """Test that word count is calculated correctly."""
        request = DocumentGenerationRequestV2(
            document_type=DocumentType.NOTES,
            title="Test Notes",
            topic="Test Topic",
            detail_level=5
        )

        result = await service.generate_document(request)

        assert result.success is True
        assert result.document.word_count >= 0

    @pytest.mark.asyncio
    async def test_available_formats_set(self, service):
        """Test that available export formats are set correctly."""
        request = DocumentGenerationRequestV2(
            document_type=DocumentType.SLIDES,
            title="Test Slides",
            topic="Test Topic",
            detail_level=4
        )

        result = await service.generate_document(request)

        assert result.success is True
        assert len(result.document.available_formats) > 0

        # Slides should support PPTX format
        from src.models.document_models import ExportFormat
        assert ExportFormat.SLIDES_PPTX in result.document.available_formats

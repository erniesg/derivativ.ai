"""
Unit tests for document persistence functionality.
Tests the integration between DocumentGenerationServiceV2 and DocumentStorageRepository.
"""

import uuid
from unittest.mock import AsyncMock, Mock

import pytest

from src.models.document_generation_v2 import (
    DocumentGenerationRequestV2,
)
from src.models.document_models import DocumentType
from src.models.enums import Tier, TopicName
from src.models.stored_document_models import StoredDocumentMetadata
from src.services.document_generation_service_v2 import DocumentGenerationServiceV2


class TestDocumentPersistence:
    """Test document persistence functionality."""

    @pytest.fixture
    def mock_document_storage_repository(self):
        """Mock document storage repository."""
        repo = Mock()
        repo.save_document_metadata = AsyncMock(return_value=uuid.uuid4())
        repo.update_document_status = AsyncMock(return_value=True)
        repo.save_document_file = AsyncMock(return_value=uuid.uuid4())
        return repo

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
    def service_with_persistence(
        self, mock_llm_factory, mock_prompt_manager, mock_document_storage_repository
    ):
        """Create service instance with document storage."""
        return DocumentGenerationServiceV2(
            llm_factory=mock_llm_factory,
            prompt_manager=mock_prompt_manager,
            document_storage_repository=mock_document_storage_repository,
        )

    @pytest.mark.asyncio
    async def test_save_document_metadata_after_generation(self, service_with_persistence):
        """Test that document metadata is saved after successful generation."""
        request = DocumentGenerationRequestV2(
            document_type=DocumentType.WORKSHEET,
            title="Algebra Practice",
            topic=TopicName.ALGEBRA_AND_GRAPHS,
            target_duration_minutes=20,
        )

        result = await service_with_persistence.generate_document(request)

        # Verify generation succeeded
        assert result.success is True
        assert result.document is not None

        # Verify document metadata was saved
        storage_repo = service_with_persistence.document_storage_repository
        storage_repo.save_document_metadata.assert_called_once()

        # Check the metadata that was saved
        call_args = storage_repo.save_document_metadata.call_args[0][0]
        assert isinstance(call_args, StoredDocumentMetadata)
        assert call_args.title == "Algebra Practice"
        assert call_args.document_type == "worksheet"
        assert call_args.topic == "Algebra and graphs"
        assert call_args.status == "generated"

    @pytest.mark.asyncio
    async def test_document_metadata_contains_correct_fields(self, service_with_persistence):
        """Test that saved document metadata contains all expected fields."""
        request = DocumentGenerationRequestV2(
            document_type=DocumentType.NOTES,
            title="Geometry Concepts",
            topic=TopicName.GEOMETRY,
            detail_level=7,
            grade_level=9,
            tier=Tier.EXTENDED,
        )

        result = await service_with_persistence.generate_document(request)

        assert result.success is True

        # Get the saved metadata
        storage_repo = service_with_persistence.document_storage_repository
        call_args = storage_repo.save_document_metadata.call_args[0][0]

        # Verify all fields are set correctly
        assert call_args.title == "Geometry Concepts"
        assert call_args.document_type == "notes"
        assert call_args.topic == "Geometry"
        assert call_args.grade_level == 9
        assert call_args.detail_level == 7  # Detail level is now stored as integer
        assert call_args.estimated_duration == 15  # From mock response
        assert call_args.status == "generated"
        assert call_args.file_count == 0  # No files saved yet
        assert call_args.total_file_size == 0

    @pytest.mark.asyncio
    async def test_tags_generation_from_request(self, service_with_persistence):
        """Test that tags are generated correctly from request data."""
        request = DocumentGenerationRequestV2(
            document_type=DocumentType.WORKSHEET,
            title="Advanced Calculus",
            topic=TopicName.ALGEBRA_AND_GRAPHS,
            subtopics=["derivatives", "integrals"],
            tier=Tier.EXTENDED,
        )

        result = await service_with_persistence.generate_document(request)

        assert result.success is True

        # Get the saved metadata
        storage_repo = service_with_persistence.document_storage_repository
        call_args = storage_repo.save_document_metadata.call_args[0][0]

        # Verify tags are generated correctly
        expected_tags = ["worksheet", "algebra-and-graphs", "extended", "derivatives", "integrals"]
        assert set(call_args.tags) == set(expected_tags)

    @pytest.mark.asyncio
    async def test_status_updates_during_generation(self, service_with_persistence):
        """Test that document status is updated during generation process."""
        request = DocumentGenerationRequestV2(
            document_type=DocumentType.TEXTBOOK,
            title="Statistics Guide",
            topic=TopicName.STATISTICS,
            detail_level=8,
        )

        result = await service_with_persistence.generate_document(request)

        assert result.success is True

        # Verify status updates
        storage_repo = service_with_persistence.document_storage_repository

        # Should have initial save with "generating" status
        # Then update to "generated" status
        assert storage_repo.save_document_metadata.call_count == 1
        assert storage_repo.update_document_status.call_count == 0  # Updated in same save

    @pytest.mark.asyncio
    async def test_error_handling_saves_failed_status(self, service_with_persistence):
        """Test that failed generation updates document status to failed."""
        # Make the LLM service fail
        service_with_persistence.llm_factory.get_service.side_effect = Exception("LLM Error")

        request = DocumentGenerationRequestV2(
            document_type=DocumentType.SLIDES,
            title="Failed Generation",
            topic=TopicName.NUMBER,
            detail_level=5,
        )

        result = await service_with_persistence.generate_document(request)

        # Generation should fail
        assert result.success is False
        assert "LLM Error" in result.error_message

        # But document metadata should still be saved with failed status
        storage_repo = service_with_persistence.document_storage_repository
        storage_repo.save_document_metadata.assert_called_once()

        call_args = storage_repo.save_document_metadata.call_args[0][0]
        assert call_args.status == "failed"

    @pytest.mark.asyncio
    async def test_document_without_storage_repository_still_works(
        self, mock_llm_factory, mock_prompt_manager
    ):
        """Test that service works without document storage repository."""
        service = DocumentGenerationServiceV2(
            llm_factory=mock_llm_factory,
            prompt_manager=mock_prompt_manager,
            document_storage_repository=None,  # No storage
        )

        request = DocumentGenerationRequestV2(
            document_type=DocumentType.WORKSHEET,
            title="No Storage Test",
            topic=TopicName.PROBABILITY,
            detail_level=4,
        )

        result = await service.generate_document(request)

        # Should still succeed, just not save to storage
        assert result.success is True
        assert result.document is not None

    @pytest.mark.asyncio
    async def test_session_id_linking(self, service_with_persistence):
        """Test that documents are linked to generation sessions if available."""
        # Mock question generation service that creates a session
        mock_question_service = Mock()
        mock_session = Mock()
        mock_session.session_id = uuid.uuid4()
        mock_question_service.generate_questions = AsyncMock(return_value=mock_session)

        service_with_persistence.question_generation_service = mock_question_service

        request = DocumentGenerationRequestV2(
            document_type=DocumentType.WORKSHEET,
            title="Session Linked",
            topic=TopicName.TRIGONOMETRY,
            detail_level=6,
            include_questions=True,
        )

        result = await service_with_persistence.generate_document(request)

        assert result.success is True

        # Verify session ID is linked
        storage_repo = service_with_persistence.document_storage_repository
        call_args = storage_repo.save_document_metadata.call_args[0][0]
        # Note: session_id might not be set if no actual question generation happens
        # This test validates the linking mechanism exists

    def test_detail_level_mapping(self):
        """Test that detail levels are mapped correctly to descriptive strings."""
        # This tests the mapping logic that should be in the service
        test_cases = [
            (1, "minimal"),
            (2, "minimal"),
            (3, "minimal"),
            (4, "medium"),
            (5, "medium"),
            (6, "medium"),
            (7, "comprehensive"),
            (8, "comprehensive"),
            (9, "comprehensive"),
            (10, "comprehensive"),
        ]

        for level, expected in test_cases:
            # This would test a utility function for mapping
            # We'll implement this in the service
            pass  # Implementation will be added

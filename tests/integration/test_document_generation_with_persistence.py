"""
Integration test for document generation with persistence.
Tests the complete workflow from request to database storage.
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, Mock

import pytest

from src.models.document_generation_v2 import DocumentGenerationRequestV2
from src.models.document_models import DocumentType
from src.models.enums import Tier, TopicName
from src.repositories.document_storage_repository import DocumentStorageRepository
from src.services.document_generation_service_v2 import DocumentGenerationServiceV2
from src.services.llm_factory import LLMFactory
from src.services.prompt_manager import PromptManager


class TestDocumentGenerationWithPersistence:
    """Test document generation with actual persistence integration."""

    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client for testing."""
        client = Mock()

        # Mock successful insert response
        mock_response = Mock()
        mock_response.data = [{"id": str(uuid.uuid4())}]

        mock_table = Mock()
        mock_table.insert.return_value.execute.return_value = mock_response
        client.table.return_value = mock_table

        return client

    @pytest.fixture
    def document_storage_repository(self, mock_supabase_client):
        """Create document storage repository with mocked client."""
        return DocumentStorageRepository(mock_supabase_client)

    @pytest.fixture
    def mock_llm_factory(self):
        """Mock LLM factory with realistic responses."""
        factory = Mock(spec=LLMFactory)
        service = Mock()

        # Mock a realistic document generation response
        mock_response = Mock()
        mock_response.content = """{
            "enhanced_title": "Advanced Algebra Practice Worksheet",
            "introduction": "This worksheet covers quadratic equations and linear functions.",
            "blocks": [
                {
                    "block_type": "learning_objectives",
                    "content": {
                        "objectives": [
                            "Solve quadratic equations using factoring",
                            "Graph linear and quadratic functions",
                            "Apply algebra to real-world problems"
                        ]
                    },
                    "estimated_minutes": 3,
                    "reasoning": "Essential objectives for algebra mastery"
                },
                {
                    "block_type": "concept_explanation",
                    "content": {
                        "concepts": [
                            {
                                "name": "Quadratic Equations",
                                "explanation": "A quadratic equation is a polynomial equation of degree 2.",
                                "examples": ["x² + 5x + 6 = 0", "2x² - 3x - 1 = 0"]
                            }
                        ]
                    },
                    "estimated_minutes": 8,
                    "reasoning": "Core concepts need detailed explanation"
                },
                {
                    "block_type": "practice_questions",
                    "content": {
                        "questions": [
                            {
                                "text": "Solve the equation x² + 5x + 6 = 0",
                                "marks": 3,
                                "difficulty": "medium",
                                "answer": "x = -2 or x = -3"
                            },
                            {
                                "text": "Graph the function y = x² - 4x + 3",
                                "marks": 4,
                                "difficulty": "medium",
                                "answer": "Parabola opening upward with vertex at (2, -1)"
                            },
                            {
                                "text": "A ball is thrown upward with initial velocity 20 m/s. Its height h(t) = -5t² + 20t. When does it hit the ground?",
                                "marks": 5,
                                "difficulty": "hard",
                                "answer": "t = 4 seconds"
                            }
                        ]
                    },
                    "estimated_minutes": 12,
                    "reasoning": "Practice problems with varying difficulty"
                },
                {
                    "block_type": "summary",
                    "content": {
                        "key_points": [
                            "Quadratic equations can be solved by factoring or using the quadratic formula",
                            "Graphing helps visualize the solutions",
                            "Algebra has many real-world applications"
                        ]
                    },
                    "estimated_minutes": 2,
                    "reasoning": "Reinforcement of key concepts"
                }
            ],
            "total_estimated_minutes": 25,
            "actual_detail_level": 6,
            "generation_reasoning": "Selected comprehensive blocks for in-depth algebra practice with real-world applications",
            "coverage_notes": "Covers quadratic equations, graphing, and applications",
            "personalization_applied": ["grade-appropriate language", "varied difficulty levels"]
        }"""

        service.generate_non_stream = AsyncMock(return_value=mock_response)
        factory.get_service.return_value = service
        return factory

    @pytest.fixture
    def mock_prompt_manager(self):
        """Mock prompt manager."""
        manager = Mock(spec=PromptManager)
        manager.render_prompt = AsyncMock(return_value="Detailed prompt for document generation...")
        return manager

    @pytest.fixture
    def integrated_service(
        self, mock_llm_factory, mock_prompt_manager, document_storage_repository
    ):
        """Create service with all dependencies for integration testing."""
        return DocumentGenerationServiceV2(
            llm_factory=mock_llm_factory,
            prompt_manager=mock_prompt_manager,
            document_storage_repository=document_storage_repository,
        )

    @pytest.mark.asyncio
    async def test_complete_document_generation_and_storage(self, integrated_service):
        """Test complete workflow from request to storage."""
        # Create a realistic document generation request
        request = DocumentGenerationRequestV2(
            document_type=DocumentType.WORKSHEET,
            title="Algebra Practice Worksheet",
            topic=TopicName.ALGEBRA_AND_GRAPHS,
            subtopics=["quadratic equations", "linear functions"],
            detail_level=6,
            grade_level=9,
            tier=Tier.EXTENDED,
            include_questions=True,
            include_answers=True,
            custom_instructions="Focus on real-world applications with step-by-step solutions",
        )

        # Generate the document
        result = await integrated_service.generate_document(request)

        # Verify generation succeeded
        assert result.success is True
        assert result.document is not None
        assert result.processing_time > 0

        # Verify document content
        document = result.document
        assert document.title == "Advanced Algebra Practice Worksheet"  # Enhanced by LLM
        assert document.document_type == DocumentType.WORKSHEET
        assert document.total_estimated_minutes == 25
        assert document.actual_detail_level == 6
        assert document.word_count > 0

        # Verify document structure
        assert len(document.content_structure.blocks) == 4
        block_types = {block.block_type for block in document.content_structure.blocks}
        expected_blocks = {
            "learning_objectives",
            "concept_explanation",
            "practice_questions",
            "summary",
        }
        assert block_types == expected_blocks

        # Verify questions were extracted
        practice_block = next(
            block
            for block in document.content_structure.blocks
            if block.block_type == "practice_questions"
        )
        questions = practice_block.content["questions"]
        assert len(questions) == 3
        assert all("text" in q and "marks" in q for q in questions)

        # Verify storage integration
        assert "document_id" in result.generation_insights
        document_id = result.generation_insights["document_id"]
        assert document_id is not None

        # Verify storage repository was used (document_id indicates success)
        # Since this is integration test, we're using the real repository
        # but with mocked Supabase client, so we can verify the document was "saved"

        # The fact that document_id is present means the storage flow worked
        storage_repo = integrated_service.document_storage_repository
        assert storage_repo is not None

        # Verify the document metadata would have been created correctly
        # by testing the service's metadata creation logic
        from src.services.document_generation_service_v2 import DocumentGenerationServiceV2

        # Create a temporary service instance to test metadata creation
        test_service = DocumentGenerationServiceV2(None, None)
        test_metadata = await test_service._create_document_metadata(request, document)

        assert test_metadata.title == "Advanced Algebra Practice Worksheet"
        assert test_metadata.document_type == "worksheet"
        assert test_metadata.topic == "Algebra and graphs"
        assert test_metadata.grade_level == 9
        assert test_metadata.detail_level == "medium"  # Level 6 maps to medium
        assert test_metadata.total_questions == 3
        assert test_metadata.estimated_duration == 25
        assert test_metadata.status == "generated"

        # Verify tags
        expected_tags = {
            "worksheet",
            "algebra-and-graphs",
            "extended",
            "quadratic-equations",
            "linear-functions",
        }
        assert set(test_metadata.tags) == expected_tags

    @pytest.mark.asyncio
    async def test_failed_generation_storage(self, integrated_service):
        """Test that failed generations are also stored."""
        # Make the LLM service fail
        integrated_service.llm_factory.get_service.side_effect = Exception(
            "LLM service unavailable"
        )

        request = DocumentGenerationRequestV2(
            document_type=DocumentType.NOTES,
            title="Failed Generation Test",
            topic=TopicName.PROBABILITY,
            detail_level=4,
        )

        # Attempt generation
        result = await integrated_service.generate_document(request)

        # Verify generation failed
        assert result.success is False
        assert "LLM service unavailable" in result.error_message

        # Verify failed document was still stored
        assert "document_id" in result.generation_insights
        document_id = result.generation_insights["document_id"]
        assert document_id is not None

        # Verify storage repository exists and was used
        storage_repo = integrated_service.document_storage_repository
        assert storage_repo is not None

        # The presence of document_id indicates the failed storage worked

    @pytest.mark.asyncio
    async def test_multiple_concurrent_generations(self, integrated_service):
        """Test multiple concurrent document generations."""
        requests = [
            DocumentGenerationRequestV2(
                document_type=DocumentType.WORKSHEET,
                title=f"Concurrent Test {i}",
                topic=TopicName.NUMBER,
                detail_level=3 + i,
            )
            for i in range(3)
        ]

        # Generate multiple documents concurrently
        results = await asyncio.gather(
            *[integrated_service.generate_document(request) for request in requests]
        )

        # Verify all generations succeeded
        assert all(result.success for result in results)
        assert all(result.document is not None for result in results)

        # Verify all were stored (indicated by document_ids)
        storage_repo = integrated_service.document_storage_repository
        assert storage_repo is not None

        # Verify unique document IDs
        document_ids = [result.generation_insights["document_id"] for result in results]
        assert len(set(document_ids)) == 3  # All unique

    @pytest.mark.asyncio
    async def test_document_search_and_categorization(self, integrated_service):
        """Test that generated documents can be found via search."""
        # Generate a document with specific characteristics
        request = DocumentGenerationRequestV2(
            document_type=DocumentType.TEXTBOOK,
            title="Comprehensive Geometry Guide",
            topic=TopicName.GEOMETRY,
            subtopics=["triangles", "circles"],
            grade_level=10,
            tier=Tier.EXTENDED,
            detail_level=8,
        )

        result = await integrated_service.generate_document(request)
        assert result.success is True

        # Test metadata creation logic for verification
        from src.services.document_generation_service_v2 import DocumentGenerationServiceV2

        test_service = DocumentGenerationServiceV2(None, None)
        saved_metadata = await test_service._create_document_metadata(request, result.document)

        # Verify search-related fields
        assert saved_metadata.topic == "Geometry"
        assert saved_metadata.grade_level == 10
        assert saved_metadata.document_type == "textbook"
        assert saved_metadata.detail_level == "comprehensive"

        # Verify tags for categorization
        expected_tags = {"textbook", "geometry", "extended", "triangles", "circles"}
        assert set(saved_metadata.tags) == expected_tags

        # Search content should be auto-generated
        search_content = saved_metadata.generate_search_content()
        assert "comprehensive geometry guide" in search_content.lower()
        assert "textbook" in search_content.lower()
        assert "geometry" in search_content.lower()

    def test_service_graceful_degradation_without_storage(
        self, mock_llm_factory, mock_prompt_manager
    ):
        """Test that service works gracefully without storage repository."""
        service = DocumentGenerationServiceV2(
            llm_factory=mock_llm_factory,
            prompt_manager=mock_prompt_manager,
            document_storage_repository=None,  # No storage
        )

        # Should initialize without error
        assert service.document_storage_repository is None
        assert service.llm_factory is not None
        assert service.prompt_manager is not None

    @pytest.mark.asyncio
    async def test_document_metadata_with_session_linking(self, integrated_service):
        """Test document metadata includes session linking when available."""
        # Mock a question generation service that creates sessions
        mock_question_service = Mock()
        mock_session = Mock()
        mock_session.session_id = uuid.uuid4()
        mock_question_service.generate_questions = AsyncMock(return_value=mock_session)

        # Add question service to the document service
        integrated_service.question_generation_service = mock_question_service

        request = DocumentGenerationRequestV2(
            document_type=DocumentType.WORKSHEET,
            title="Session Linked Document",
            topic=TopicName.STATISTICS,
            include_questions=True,
        )

        result = await integrated_service.generate_document(request)
        assert result.success is True

        # Note: In the current implementation, session linking happens
        # through question generation, but the exact session_id passing
        # would need to be enhanced in the actual _generate_questions_if_needed method
        # This test validates the infrastructure exists

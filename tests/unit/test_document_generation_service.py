"""
Unit tests for DocumentGenerationService with PromptManager integration.
Tests the refactored service that uses LLM + PromptManager instead of hardcoded templates.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.database.supabase_repository import QuestionRepository
from src.models.document_models import (
    DetailLevel,
    DocumentGenerationRequest,
    DocumentType,
)
from src.models.enums import Tier
from src.models.llm_models import LLMResponse
from src.services.document_generation_service import DocumentGenerationService
from src.services.prompt_manager import PromptManager


class TestDocumentGenerationService:
    """Test DocumentGenerationService with PromptManager integration."""

    @pytest.fixture
    def mock_question_repository(self):
        """Create mock question repository."""
        return MagicMock(spec=QuestionRepository)


    @pytest.fixture
    def mock_prompt_manager(self):
        """Create mock prompt manager."""
        return AsyncMock(spec=PromptManager)

    @pytest.fixture
    def document_service(self, mock_question_repository, mock_llm_factory, mock_prompt_manager):
        """Create DocumentGenerationService instance."""
        factory, _ = mock_llm_factory
        return DocumentGenerationService(
            question_repository=mock_question_repository,
            llm_factory=factory,
            prompt_manager=mock_prompt_manager,
        )

    @pytest.fixture
    def sample_generation_request(self):
        """Create sample document generation request."""
        return DocumentGenerationRequest(
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            title="Algebra Practice Worksheet",
            topic="linear_equations",
            tier=Tier.CORE,
            grade_level=7,
            auto_include_questions=False,
            max_questions=5,
        )

    @pytest.mark.asyncio
    async def test_template_mapping_selection(self, document_service):
        """Test that correct templates are selected for document types."""
        templates = await document_service.get_available_templates()

        assert "worksheet" in templates
        assert "notes" in templates
        assert "textbook" in templates
        assert "slides" in templates

        assert templates["worksheet"] == "worksheet_generation"
        assert templates["notes"] == "notes_generation"
        assert templates["textbook"] == "textbook_generation"
        assert templates["slides"] == "slides_generation"

    @pytest.mark.asyncio
    async def test_structure_pattern_selection(self, document_service):
        """Test that correct structure patterns are selected."""
        patterns = await document_service.get_structure_patterns()

        worksheet_patterns = patterns["worksheet"]
        assert "minimal" in worksheet_patterns
        assert "medium" in worksheet_patterns
        assert "comprehensive" in worksheet_patterns
        assert "guided" in worksheet_patterns

        # Check specific patterns
        assert worksheet_patterns["minimal"] == ["practice_questions", "answers"]
        assert "learning_objectives" in worksheet_patterns["medium"]
        assert "worked_examples" in worksheet_patterns["medium"]

    @pytest.mark.asyncio
    async def test_successful_document_generation(
        self, document_service, mock_llm_factory, mock_prompt_manager, sample_generation_request
    ):
        """Test successful document generation flow."""
        # Setup mocks
        _, llm_service = mock_llm_factory
        mock_prompt_manager.render_prompt.return_value = "Rendered prompt content"

        # Mock LLM response with valid JSON
        mock_llm_response_content = {
            "title": "Algebra Practice Worksheet",
            "document_type": "worksheet",
            "detail_level": "medium",
            "sections": [
                {
                    "title": "Learning Objectives",
                    "content_type": "learning_objectives",
                    "content_data": {"objectives_text": "• Solve linear equations"},
                    "order_index": 0,
                },
                {
                    "title": "Practice Questions",
                    "content_type": "practice_questions",
                    "content_data": {"questions": []},
                    "order_index": 1,
                },
            ],
            "estimated_duration": 30,
            "total_questions": 5,
        }

        llm_service.generate_non_stream.return_value = LLMResponse(
            content=json.dumps(mock_llm_response_content),
            model_used="gpt-4o-mini",
            latency_ms=1500,
            tokens_used=1000,
            cost_estimate=0.001,
            provider="openai",
            metadata={},
        )

        # Execute generation
        result = await document_service.generate_document(sample_generation_request)

        # Verify success
        assert result.success is True
        assert result.document is not None
        assert result.document.title == "Algebra Practice Worksheet"
        assert result.document.document_type == DocumentType.WORKSHEET
        assert result.document.detail_level == DetailLevel.MEDIUM
        assert len(result.document.sections) == 2

        # Verify LLM was called with proper template
        mock_prompt_manager.render_prompt.assert_called_once()
        call_args = mock_prompt_manager.render_prompt.call_args[0][0]
        assert call_args.template_name == "worksheet_generation"
        assert call_args.variables["title"] == "Algebra Practice Worksheet"
        assert call_args.variables["topic"] == "linear_equations"

    @pytest.mark.asyncio
    async def test_json_parsing_failure_fallback(
        self, document_service, mock_llm_factory, mock_prompt_manager, sample_generation_request
    ):
        """Test fallback when LLM returns invalid JSON."""
        # Setup mocks
        _, llm_service = mock_llm_factory
        mock_prompt_manager.render_prompt.return_value = "Rendered prompt content"

        # Mock LLM response with invalid JSON
        llm_service.generate_non_stream.return_value = LLMResponse(
            content="Invalid JSON response from LLM",
            model_used="gpt-4o-mini",
            latency_ms=1500,
            tokens_used=800,
            cost_estimate=0.0008,
            provider="openai",
            metadata={},
        )

        # Execute generation
        result = await document_service.generate_document(sample_generation_request)

        # Verify fallback works
        assert result.success is True
        assert result.document is not None
        assert result.document.title == "Algebra Practice Worksheet"
        assert len(result.document.sections) > 0  # Should have fallback sections

    @pytest.mark.asyncio
    async def test_custom_instructions_integration(
        self, document_service, mock_llm_factory, mock_prompt_manager
    ):
        """Test that custom instructions are passed to templates."""
        # Setup request with custom instructions
        request = DocumentGenerationRequest(
            document_type=DocumentType.NOTES,
            detail_level=DetailLevel.COMPREHENSIVE,
            title="Advanced Algebra Notes",
            topic="quadratic_equations",
            tier=Tier.EXTENDED,
            custom_instructions="Include visual learning aids and step-by-step examples",
            personalization_context={"learning_style": "visual"},
        )

        # Setup mocks
        _, llm_service = mock_llm_factory
        mock_prompt_manager.render_prompt.return_value = "Rendered prompt with customizations"

        llm_service.generate_non_stream.return_value = LLMResponse(
            content='{"title": "Test", "sections": [{"title": "Test Section", "content_type": "generic", "content_data": {"text": "Test content"}, "order_index": 0}]}',
            model_used="gpt-4o-mini",
            latency_ms=1500,
            tokens_used=500,
            cost_estimate=0.0005,
            provider="openai",
            metadata={},
        )

        # Execute generation
        result = await document_service.generate_document(request)

        # Verify custom instructions were passed to template
        mock_prompt_manager.render_prompt.assert_called_once()
        call_args = mock_prompt_manager.render_prompt.call_args[0][0]

        assert "custom_instructions" in call_args.variables
        assert (
            call_args.variables["custom_instructions"]
            == "Include visual learning aids and step-by-step examples"
        )
        assert "personalization_context" in call_args.variables
        assert call_args.variables["personalization_context"]["learning_style"] == "visual"

        # Verify customizations are tracked
        assert result.success is True
        assert result.customizations_applied == 2  # custom_instructions + personalization_context
        assert result.personalization_success is True

    @pytest.mark.asyncio
    async def test_different_document_types(
        self, document_service, mock_llm_factory, mock_prompt_manager
    ):
        """Test generation for different document types."""
        document_types = [
            (DocumentType.WORKSHEET, "worksheet_generation"),
            (DocumentType.NOTES, "notes_generation"),
            (DocumentType.TEXTBOOK, "textbook_generation"),
            (DocumentType.SLIDES, "slides_generation"),
        ]

        # Setup mocks
        _, llm_service = mock_llm_factory
        mock_prompt_manager.render_prompt.return_value = "Template content"
        llm_service.generate_non_stream.return_value = LLMResponse(
            content='{"title": "Test Document", "sections": [{"title": "Test Section", "content_type": "generic", "content_data": {"text": "Test content"}, "order_index": 0}]}',
            model_used="gpt-4o-mini",
            latency_ms=1500,
            tokens_used=600,
            cost_estimate=0.0006,
            provider="openai",
            metadata={},
        )

        for doc_type, expected_template in document_types:
            request = DocumentGenerationRequest(
                document_type=doc_type,
                detail_level=DetailLevel.MEDIUM,
                title=f"Test {doc_type.value}",
                topic="test_topic",
            )

            # Reset mock
            mock_prompt_manager.render_prompt.reset_mock()

            # Execute generation
            result = await document_service.generate_document(request)

            # Verify correct template was used
            assert result.success is True
            mock_prompt_manager.render_prompt.assert_called_once()
            call_args = mock_prompt_manager.render_prompt.call_args[0][0]
            assert call_args.template_name == expected_template

    @pytest.mark.asyncio
    async def test_detail_level_structure_patterns(self, document_service):
        """Test that different detail levels have appropriate structure patterns."""
        patterns = await document_service.get_structure_patterns()

        # Test worksheet patterns for different detail levels
        worksheet = patterns["worksheet"]

        # Minimal should be simplest
        minimal = worksheet["minimal"]
        assert len(minimal) <= 3
        assert "practice_questions" in minimal

        # Comprehensive should be most detailed
        comprehensive = worksheet["comprehensive"]
        assert len(comprehensive) >= len(minimal)
        assert "learning_objectives" in comprehensive
        assert "detailed_solutions" in comprehensive

        # Medium should be between minimal and comprehensive
        medium = worksheet["medium"]
        assert len(minimal) < len(medium) < len(comprehensive)

    @pytest.mark.asyncio
    async def test_error_handling(
        self, document_service, mock_llm_factory, mock_prompt_manager, sample_generation_request
    ):
        """Test error handling when LLM service fails."""
        # Setup mocks to raise an exception
        _, llm_service = mock_llm_factory
        mock_prompt_manager.render_prompt.return_value = "Template content"
        llm_service.generate_non_stream.side_effect = Exception("LLM service error")

        # Execute generation
        result = await document_service.generate_document(sample_generation_request)

        # Verify error handling
        assert result.success is False
        assert "LLM service error" in result.error_message
        assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_duration_estimation(self, document_service):
        """Test duration estimation for different document types and detail levels."""
        # Test different combinations
        test_cases = [
            (DocumentType.WORKSHEET, DetailLevel.MINIMAL, 3, 5),  # Should be quick
            (DocumentType.TEXTBOOK, DetailLevel.COMPREHENSIVE, 5, 30),  # Should be longer
            (DocumentType.SLIDES, DetailLevel.MEDIUM, 4, 15),  # Should be medium
        ]

        for doc_type, detail_level, question_count, min_expected_duration in test_cases:
            duration = document_service._estimate_duration(doc_type, detail_level, question_count)
            assert duration >= min_expected_duration
            assert isinstance(duration, int)

    def test_practice_questions_data_creation(self, document_service):
        """Test creation of practice questions data from Question objects."""
        # Mock Question objects
        from src.models.question_models import CommandWord, Question

        mock_questions = [
            MagicMock(spec=Question),
            MagicMock(spec=Question),
        ]

        # Configure mock question attributes
        mock_questions[0].question_id_global = "q1"
        mock_questions[0].raw_text_content = "Solve: x + 3 = 7"
        mock_questions[0].marks = 2
        mock_questions[0].command_word = CommandWord.CALCULATE

        mock_questions[1].question_id_global = "q2"
        mock_questions[1].raw_text_content = "Find: 2x - 1 = 9"
        mock_questions[1].marks = 3
        mock_questions[1].command_word = CommandWord.FIND

        # Test data creation
        data = document_service._create_practice_questions_data(mock_questions)

        assert "questions" in data
        assert "total_marks" in data
        assert "estimated_time" in data

        assert len(data["questions"]) == 2
        assert data["total_marks"] == 5  # 2 + 3
        assert data["estimated_time"] == 6  # 2 questions * 3 minutes

        # Check first question data
        q1_data = data["questions"][0]
        assert q1_data["question_id"] == "q1"
        assert q1_data["question_text"] == "Solve: x + 3 = 7"
        assert q1_data["marks"] == 2
        assert q1_data["command_word"] == "Calculate"

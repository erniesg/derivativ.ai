"""
Integration tests for document generation system.
Tests the interaction between services, agents, and data repositories.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.database.supabase_repository import QuestionRepository
from src.models.document_models import (
    DetailLevel,
    DocumentGenerationRequest,
    DocumentType,
)
from src.models.enums import SubjectContentReference, Tier
from src.models.question_models import Question, QuestionTaxonomy
from src.services.document_generation_service import DocumentGenerationService
from src.services.llm_factory import LLMFactory
from src.services.prompt_manager import PromptManager


class TestDocumentGenerationIntegration:
    """Integration tests for document generation workflow."""

    @pytest.fixture
    def mock_question_repository(self):
        """Create mock question repository."""
        repository = MagicMock(spec=QuestionRepository)

        # Mock question data with required fields
        from src.models.question_models import (
            FinalAnswer,
            MarkingCriterion,
            SolutionAndMarkingScheme,
            SolverAlgorithm,
            SolverStep,
        )

        sample_question = Question(
            question_id_local="1a",
            question_id_global="test_q1",
            question_number_display="1 (a)",
            marks=3,
            command_word="Calculate",
            raw_text_content="Calculate the area of a triangle with base 6cm and height 4cm.",
            taxonomy=QuestionTaxonomy(
                topic_path=["Geometry", "Area"],
                subject_content_references=[SubjectContentReference.C5_2],
                skill_tags=["area_calculation"],
            ),
            solution_and_marking_scheme=SolutionAndMarkingScheme(
                final_answers_summary=[
                    FinalAnswer(answer_text="12 cm²", value_numeric=12.0, unit="cm²")
                ],
                mark_allocation_criteria=[
                    MarkingCriterion(
                        criterion_id="1",
                        criterion_text="Correct method",
                        mark_code_display="M1",
                        marks_value=1,
                    ),
                    MarkingCriterion(
                        criterion_id="2",
                        criterion_text="Correct answer",
                        mark_code_display="A1",
                        marks_value=2,
                    ),
                ],
                total_marks_for_part=3,
            ),
            solver_algorithm=SolverAlgorithm(
                steps=[
                    SolverStep(
                        step_number=1, description_text="Use formula Area = (1/2) × base × height"
                    ),
                    SolverStep(
                        step_number=2,
                        description_text="Substitute values: Area = (1/2) × 6 × 4 = 12 cm²",
                    ),
                ]
            ),
        )

        repository.get_question = AsyncMock(return_value=sample_question)
        repository.list_questions = AsyncMock(
            return_value=[
                {
                    "content_json": sample_question.model_dump(),
                    "quality_score": 0.85,
                    "tier": "Core",
                    "marks": 3,
                }
            ]
        )

        return repository

    @pytest.fixture
    def mock_llm_factory(self):
        """Create mock LLM factory."""
        factory = MagicMock(spec=LLMFactory)

        # Mock LLM service
        llm_service = MagicMock()
        llm_service.generate_non_stream = AsyncMock(
            return_value=MagicMock(
                content="• Understand key concepts in geometry\n• Apply area formulas\n• Solve practical problems"
            )
        )

        factory.create_service = MagicMock(return_value=llm_service)
        return factory

    @pytest.fixture
    def mock_prompt_manager(self):
        """Create mock prompt manager."""
        return MagicMock(spec=PromptManager)

    @pytest.fixture
    def document_service(self, mock_question_repository, mock_llm_factory, mock_prompt_manager):
        """Create document generation service with mocks."""
        return DocumentGenerationService(
            question_repository=mock_question_repository,
            llm_factory=mock_llm_factory,
            prompt_manager=mock_prompt_manager,
        )

    @pytest.mark.asyncio
    async def test_worksheet_generation_end_to_end(self, document_service):
        """Test complete worksheet generation workflow."""
        request = DocumentGenerationRequest(
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            title="Geometry Practice Worksheet",
            topic="area_calculation",
            tier=Tier.CORE,
            grade_level=7,
            subject_content_refs=[SubjectContentReference.C5_2],
            max_questions=5,
            include_answers=True,
            include_working=True,
        )

        # WHEN: Generating document
        result = await document_service.generate_document(request)

        # THEN: Should succeed
        assert result.success is True
        assert result.document is not None
        assert result.document.title == "Geometry Practice Worksheet"
        assert result.document.document_type == DocumentType.WORKSHEET
        assert result.document.detail_level == DetailLevel.MEDIUM

        # Check sections
        assert len(result.document.sections) > 0
        section_titles = [section.title for section in result.document.sections]
        assert "Topic Overview" in section_titles or "Introduction" in section_titles

        # Check processing metrics
        assert result.processing_time > 0
        assert result.questions_processed >= 0
        assert result.sections_generated > 0

    @pytest.mark.asyncio
    async def test_notes_generation_with_llm_integration(self, document_service):
        """Test notes generation with LLM integration for objectives."""
        request = DocumentGenerationRequest(
            document_type=DocumentType.NOTES,
            detail_level=DetailLevel.GUIDED,
            title="Understanding Quadratic Equations",
            topic="quadratic_equations",
            tier=Tier.EXTENDED,
            grade_level=9,
            auto_include_questions=True,
            max_questions=3,
        )

        # WHEN: Generating notes
        result = await document_service.generate_document(request)

        # THEN: Should succeed with LLM-generated content
        assert result.success is True
        assert result.document.document_type == DocumentType.NOTES

        # Check for learning objectives section
        learning_objectives_section = None
        for section in result.document.sections:
            if section.content_type == "learning_objectives":
                learning_objectives_section = section
                break

        # Should have learning objectives (either LLM-generated or fallback)
        if learning_objectives_section:
            assert "objectives_text" in learning_objectives_section.content_data
            objectives_text = learning_objectives_section.content_data["objectives_text"]
            assert len(objectives_text) > 0

    @pytest.mark.asyncio
    async def test_textbook_generation_comprehensive(self, document_service):
        """Test comprehensive textbook generation."""
        request = DocumentGenerationRequest(
            document_type=DocumentType.TEXTBOOK,
            detail_level=DetailLevel.COMPREHENSIVE,
            title="Complete Guide to Trigonometry",
            topic="trigonometry",
            tier=Tier.EXTENDED,
            grade_level=10,
            max_questions=15,
            include_answers=True,
            include_working=True,
            include_mark_schemes=True,
            custom_instructions="Include real-world applications and advanced examples",
        )

        # WHEN: Generating textbook
        result = await document_service.generate_document(request)

        # THEN: Should create comprehensive structure
        assert result.success is True
        assert result.document.document_type == DocumentType.TEXTBOOK
        assert result.document.detail_level == DetailLevel.COMPREHENSIVE

        # Should track custom instructions
        assert "custom_instructions" in result.document.applied_customizations
        assert result.customizations_applied > 0

        # Textbook should have more sections than other document types
        assert len(result.document.sections) >= 5

        # Check estimated duration (should be higher for comprehensive textbooks)
        assert result.document.estimated_duration is not None
        # Duration should reflect document type + detail level combination
        assert (
            result.document.estimated_duration > 60
        )  # Should be substantial for comprehensive textbook

    @pytest.mark.asyncio
    async def test_slides_generation_with_detail_levels(self, document_service):
        """Test slides generation with different detail levels."""
        # Test minimal slides
        minimal_request = DocumentGenerationRequest(
            document_type=DocumentType.SLIDES,
            detail_level=DetailLevel.MINIMAL,
            title="Introduction to Statistics - Minimal",
            topic="statistics",
            tier=Tier.CORE,
            grade_level=8,
            max_questions=4,
            include_answers=False,
            include_working=False,
        )

        # Test comprehensive slides
        comprehensive_request = DocumentGenerationRequest(
            document_type=DocumentType.SLIDES,
            detail_level=DetailLevel.COMPREHENSIVE,
            title="Introduction to Statistics - Detailed",
            topic="statistics",
            tier=Tier.CORE,
            grade_level=8,
            max_questions=4,
            include_answers=True,
            include_working=True,
            custom_instructions="Include interactive elements and detailed explanations",
        )

        # WHEN: Generating both slides
        minimal_result = await document_service.generate_document(minimal_request)
        comprehensive_result = await document_service.generate_document(comprehensive_request)

        # THEN: Both should succeed but have different characteristics
        assert minimal_result.success is True
        assert comprehensive_result.success is True

        # Comprehensive slides should have more sections and longer duration
        assert len(comprehensive_result.document.sections) > len(minimal_result.document.sections)
        assert (
            comprehensive_result.document.estimated_duration
            > minimal_result.document.estimated_duration
        )

        # Custom instructions should be tracked
        assert "custom_instructions" in comprehensive_result.document.applied_customizations

    @pytest.mark.asyncio
    async def test_document_generation_with_specific_questions(self, document_service):
        """Test document generation with specific question references."""
        from src.models.document_models import QuestionReference

        question_refs = [
            QuestionReference(
                question_id="test_q1",
                include_solution=True,
                include_marking=True,
                context_note="Good example of area calculation",
                order_index=1,
            )
        ]

        request = DocumentGenerationRequest(
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            title="Custom Question Worksheet",
            topic="geometry",
            question_references=question_refs,
            auto_include_questions=False,  # Only use specified questions
            max_questions=1,
            custom_instructions="Focus on visual problem-solving approaches",
            personalization_context={"learning_style": "visual"},
        )

        # WHEN: Generating with specific questions
        result = await document_service.generate_document(request)

        # THEN: Should use specified questions and apply personalization
        assert result.success is True
        assert len(result.document.questions_used) > 0
        assert "test_q1" in result.document.questions_used
        assert "custom_instructions" in result.document.applied_customizations
        assert "personalization_context" in result.document.applied_customizations
        assert result.customizations_applied == 2  # custom_instructions + personalization_context

    @pytest.mark.asyncio
    async def test_document_generation_error_handling(self, document_service):
        """Test error handling in document generation."""
        # Mock repository to raise an error
        document_service.question_repository.list_questions = AsyncMock(
            side_effect=Exception("Database connection failed")
        )

        request = DocumentGenerationRequest(
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            title="Error Test Worksheet",
            topic="algebra",
            auto_include_questions=True,
        )

        # WHEN: Generating with error conditions
        result = await document_service.generate_document(request)

        # THEN: Should handle error gracefully
        assert result.success is False
        assert result.error_message is not None
        assert "Database connection failed" in result.error_message
        assert result.document is None
        assert result.processing_time > 0  # Should still track time

    @pytest.mark.asyncio
    async def test_template_selection_logic(self, document_service):
        """Test template selection based on document type and detail level."""
        # Test each document type with different detail levels
        test_cases = [
            (DocumentType.WORKSHEET, DetailLevel.MINIMAL, "worksheet_default"),
            (DocumentType.WORKSHEET, DetailLevel.COMPREHENSIVE, "worksheet_default"),
            (DocumentType.NOTES, DetailLevel.GUIDED, "notes_default"),
            (DocumentType.TEXTBOOK, DetailLevel.COMPREHENSIVE, "textbook_default"),
            (DocumentType.SLIDES, DetailLevel.MINIMAL, "slides_default"),
        ]

        for doc_type, detail_level, expected_template in test_cases:
            request = DocumentGenerationRequest(
                document_type=doc_type,
                detail_level=detail_level,
                title=f"Test {doc_type.value} - {detail_level.value}",
                topic="test_topic",
            )

            # WHEN: Generating document
            result = await document_service.generate_document(request)

            # THEN: Should use correct template and support the detail level
            assert result.success is True
            assert result.document.template_used == expected_template
            assert result.document.detail_level == detail_level

    @pytest.mark.asyncio
    async def test_content_formatting_integration(self, document_service):
        """Test integration with content formatting."""
        request = DocumentGenerationRequest(
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            title="Formatting Test Worksheet",
            topic="algebra",
            max_questions=2,
            include_answers=True,
            include_working=True,
        )

        # WHEN: Generating document
        result = await document_service.generate_document(request)

        # THEN: Should generate formatted content
        assert result.success is True
        assert result.document.content_html is not None
        assert result.document.content_markdown is not None

        # Check HTML content structure
        html_content = result.document.content_html
        assert "<html>" in html_content
        assert "<title>" in html_content
        assert result.document.title in html_content

        # Check Markdown content structure
        markdown_content = result.document.content_markdown
        assert f"# {result.document.title}" in markdown_content
        assert "##" in markdown_content  # Should have section headers

    @pytest.mark.asyncio
    async def test_syllabus_coverage_tracking(self, document_service):
        """Test that syllabus coverage is properly tracked."""
        request = DocumentGenerationRequest(
            document_type=DocumentType.NOTES,
            detail_level=DetailLevel.GUIDED,
            title="Syllabus Coverage Test",
            topic="geometry",
            subject_content_refs=[SubjectContentReference.C5_1, SubjectContentReference.C5_2],
            auto_include_questions=True,
            max_questions=3,
        )

        # WHEN: Generating document
        result = await document_service.generate_document(request)

        # THEN: Should track syllabus coverage
        assert result.success is True
        assert len(result.document.syllabus_coverage) == 2
        assert SubjectContentReference.C5_1 in result.document.syllabus_coverage
        assert SubjectContentReference.C5_2 in result.document.syllabus_coverage

    @pytest.mark.asyncio
    async def test_question_quality_filtering(self, document_service, mock_question_repository):
        """Test that only high-quality questions are included."""
        # Mock repository to return questions with different quality scores
        low_quality_question = {
            "content_json": {"question_id_global": "low_q1", "marks": 2},
            "quality_score": 0.6,  # Below 0.7 threshold
        }
        high_quality_question = {
            "content_json": {"question_id_global": "high_q1", "marks": 3},
            "quality_score": 0.85,  # Above 0.7 threshold
        }

        mock_question_repository.list_questions = AsyncMock(
            return_value=[low_quality_question, high_quality_question]
        )

        request = DocumentGenerationRequest(
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            title="Quality Filter Test",
            topic="algebra",
            auto_include_questions=True,
            max_questions=10,
        )

        # WHEN: Generating document
        result = await document_service.generate_document(request)

        # THEN: Should filter by quality (this test validates the service calls the repo correctly)
        assert result.success is True
        mock_question_repository.list_questions.assert_called_once()
        call_args = mock_question_repository.list_questions.call_args[1]
        assert call_args["min_quality_score"] == 0.7

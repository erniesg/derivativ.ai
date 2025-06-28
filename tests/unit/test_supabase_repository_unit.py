"""
Unit tests for Supabase question repository.
Tests repository logic with mocked Supabase client.
"""

from datetime import datetime
from unittest.mock import Mock
from uuid import uuid4

import pytest

from src.database.supabase_repository import GenerationSessionRepository, QuestionRepository
from src.models.enums import CognitiveLevel, CommandWord, Tier
from src.models.question_models import GenerationSession, GenerationStatus, Question


class TestQuestionRepository:
    """Unit tests for QuestionRepository with mocked Supabase client."""

    @pytest.fixture
    def sample_question(self):
        """Sample question for testing."""
        from src.models.question_models import (
            FinalAnswer,
            MarkingCriterion,
            QuestionTaxonomy,
            SolutionAndMarkingScheme,
            SolverAlgorithm,
            SolverStep,
        )

        return Question(
            question_id_local="1a",
            question_id_global=str(uuid4()),
            question_number_display="1 (a)",
            marks=3,
            command_word=CommandWord.CALCULATE,
            raw_text_content="Calculate the area of a circle with radius 5cm.",
            taxonomy=QuestionTaxonomy(
                topic_path=["Number", "Geometry"],
                subject_content_references=[],
                skill_tags=["circle_area", "calculation"],
                cognitive_level=CognitiveLevel.APPLICATION,
                difficulty_estimate_0_to_1=0.6,
            ),
            solution_and_marking_scheme=SolutionAndMarkingScheme(
                final_answers_summary=[
                    FinalAnswer(answer_text="78.5 cm²", value_numeric=78.5, unit="cm²")
                ],
                mark_allocation_criteria=[
                    MarkingCriterion(
                        criterion_id="1a_m1",
                        criterion_text="Correct formula π × r²",
                        mark_code_display="M1",
                        marks_value=1,
                    ),
                    MarkingCriterion(
                        criterion_id="1a_a1",
                        criterion_text="Correct substitution and calculation",
                        mark_code_display="A2",
                        marks_value=2,
                    ),
                ],
                total_marks_for_part=3,
            ),
            solver_algorithm=SolverAlgorithm(
                steps=[
                    SolverStep(
                        step_number=1,
                        description_text="Apply circle area formula",
                        mathematical_expression_latex=r"A = \pi r^2",
                        skill_applied_tag="circle_area",
                    )
                ]
            ),
        )

    def test_question_repository_init(self, mock_supabase_client):
        """Test QuestionRepository initialization."""
        client, _ = mock_supabase_client
        repo = QuestionRepository(client)
        assert repo.supabase == client

    def test_save_question_success(self, mock_supabase_client, sample_question):
        """Test successful question save to Supabase."""
        client, mock_table = mock_supabase_client

        # Mock successful insert response
        mock_response = Mock()
        mock_response.data = [
            {"id": "123", "question_id_global": sample_question.question_id_global}
        ]
        mock_table.insert.return_value.execute.return_value = mock_response

        repo = QuestionRepository(client)
        result_id = repo.save_question(sample_question)

        # Verify Supabase client was called correctly
        mock_table.insert.assert_called_once()
        call_args = mock_table.insert.call_args[0][0]

        # Check that the data includes flattened fields
        assert call_args["question_id_global"] == sample_question.question_id_global
        assert (
            call_args["tier"] == sample_question.taxonomy.tier
            if hasattr(sample_question.taxonomy, "tier")
            else Tier.CORE.value
        )
        assert call_args["marks"] == sample_question.marks
        assert call_args["command_word"] == sample_question.command_word.value
        assert "content_json" in call_args
        assert call_args["content_json"] == sample_question.model_dump()

        assert result_id == "123"

    def test_save_question_failure(self, mock_supabase_client, sample_question):
        """Test question save failure handling."""
        client, mock_table = mock_supabase_client

        # Mock failed insert
        mock_table.insert.return_value.execute.side_effect = Exception("Database error")

        repo = QuestionRepository(client)

        with pytest.raises(Exception, match="Database error"):
            repo.save_question(sample_question)

    def test_get_question_success(self, mock_supabase_client, sample_question):
        """Test successful question retrieval."""
        client, mock_table = mock_supabase_client

        # Mock successful select response
        mock_response = Mock()
        mock_response.data = [
            {
                "id": "123",
                "question_id_global": sample_question.question_id_global,
                "content_json": sample_question.model_dump(),
                "tier": Tier.CORE.value,
                "marks": 3,
                "command_word": CommandWord.CALCULATE.value,
                "subject_content_refs": [],
                "quality_score": 0.8,
                "status": GenerationStatus.CANDIDATE.value,
                "created_at": datetime.utcnow().isoformat(),
            }
        ]
        mock_table.select.return_value.eq.return_value.execute.return_value = mock_response

        repo = QuestionRepository(client)
        result = repo.get_question(sample_question.question_id_global)

        # Verify correct query was made
        mock_table.select.assert_called_once_with("*")
        mock_table.select.return_value.eq.assert_called_once_with(
            "question_id_global", sample_question.question_id_global
        )

        # Verify question was reconstructed correctly
        assert isinstance(result, Question)
        assert result.question_id_global == sample_question.question_id_global
        assert result.marks == sample_question.marks

    def test_get_question_not_found(self, mock_supabase_client):
        """Test question not found handling."""
        client, mock_table = mock_supabase_client

        # Mock empty response
        mock_response = Mock()
        mock_response.data = []
        mock_table.select.return_value.eq.return_value.execute.return_value = mock_response

        repo = QuestionRepository(client)
        result = repo.get_question("nonexistent-id")

        assert result is None

    def test_list_questions_with_filters(self, mock_supabase_client):
        """Test listing questions with filters."""
        client, mock_table = mock_supabase_client

        # Mock response with multiple questions
        mock_response = Mock()
        mock_response.data = [
            {
                "id": "1",
                "question_id_global": "q1",
                "tier": Tier.CORE.value,
                "marks": 3,
                "command_word": CommandWord.CALCULATE.value,
                "content_json": {},
                "quality_score": 0.8,
            },
            {
                "id": "2",
                "question_id_global": "q2",
                "tier": Tier.EXTENDED.value,
                "marks": 5,
                "command_word": CommandWord.EXPLAIN.value,
                "content_json": {},
                "quality_score": 0.9,
            },
        ]
        mock_table.select.return_value.eq.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value = mock_response

        repo = QuestionRepository(client)
        results = repo.list_questions(tier=Tier.CORE, min_quality_score=0.7, limit=10)

        # Verify filters were applied
        mock_table.select.assert_called_once()
        assert len(results) == 2

    def test_delete_question(self, mock_supabase_client):
        """Test question deletion."""
        client, mock_table = mock_supabase_client

        # Mock successful delete
        mock_response = Mock()
        mock_response.data = [{"id": "123"}]
        mock_table.delete.return_value.eq.return_value.execute.return_value = mock_response

        repo = QuestionRepository(client)
        success = repo.delete_question("test-id")

        # Verify delete was called correctly
        mock_table.delete.assert_called_once()
        mock_table.delete.return_value.eq.assert_called_once_with("question_id_global", "test-id")
        assert success is True


class TestGenerationSessionRepository:
    """Unit tests for GenerationSessionRepository."""

    @pytest.fixture
    def sample_session(self):
        """Sample generation session for testing."""
        from src.models.question_models import GenerationRequest

        return GenerationSession(
            session_id=uuid4(),
            request=GenerationRequest(
                topic="algebra", tier=Tier.CORE, marks=3, command_word=CommandWord.CALCULATE
            ),
            questions=[],
            quality_decisions=[],
            agent_results=[],
            status=GenerationStatus.CANDIDATE,
        )

    def test_save_session_success(self, mock_supabase_client, sample_session):
        """Test successful session save."""
        from unittest.mock import patch

        client, mock_table = mock_supabase_client

        # Mock successful insert
        mock_response = Mock()
        mock_response.data = [{"id": "session-123", "session_id": str(sample_session.session_id)}]
        mock_table.insert.return_value.execute.return_value = mock_response

        with patch("src.database.supabase_repository.get_settings") as mock_settings:
            mock_settings.return_value.table_prefix = ""
            repo = GenerationSessionRepository(client)
            result_id = repo.save_session(sample_session)

        # Verify insert was called
        mock_table.insert.assert_called_once()
        call_args = mock_table.insert.call_args[0][0]

        assert call_args["session_id"] == str(sample_session.session_id)
        assert call_args["status"] == sample_session.status.value
        assert "request_json" in call_args
        assert "questions_json" in call_args
        assert "quality_decisions_json" in call_args
        assert "agent_results_json" in call_args

        assert result_id == "session-123"

    def test_get_session_success(self, mock_supabase_client, sample_session):
        """Test successful session retrieval."""
        from unittest.mock import patch

        client, mock_table = mock_supabase_client

        # Mock successful select
        mock_response = Mock()
        mock_response.data = [
            {
                "id": "session-123",
                "session_id": str(sample_session.session_id),
                "request_json": sample_session.request.model_dump(),
                "questions_json": [q.model_dump() for q in sample_session.questions],
                "quality_decisions_json": [
                    qd.model_dump() for qd in sample_session.quality_decisions
                ],
                "agent_results_json": [ar.model_dump() for ar in sample_session.agent_results],
                "status": sample_session.status.value,
                "created_at": datetime.utcnow().isoformat(),
            }
        ]
        mock_table.select.return_value.eq.return_value.execute.return_value = mock_response

        with patch("src.database.supabase_repository.get_settings") as mock_settings:
            mock_settings.return_value.table_prefix = ""
            repo = GenerationSessionRepository(client)
            result = repo.get_session(str(sample_session.session_id))

        # Verify correct query
        mock_table.select.assert_called_once_with("*")
        mock_table.select.return_value.eq.assert_called_once_with(
            "session_id", str(sample_session.session_id)
        )

        # Verify session reconstruction
        assert isinstance(result, GenerationSession)
        assert result.session_id == sample_session.session_id

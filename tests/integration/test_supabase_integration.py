"""
Integration tests for Supabase operations with real database.
Tests actual database connections and operations.
"""

import os
from uuid import uuid4

import pytest

from src.database.supabase_repository import (
    GenerationSessionRepository,
    QuestionRepository,
    get_supabase_client,
)
from src.models.enums import CommandWord, QuestionOrigin, Tier
from src.models.question_models import (
    FinalAnswer,
    GenerationRequest,
    GenerationSession,
    MarkingCriterion,
    Question,
    QuestionTaxonomy,
    SolutionAndMarkingScheme,
    SolverAlgorithm,
    SolverStep,
)


@pytest.fixture(scope="session")
def supabase_client():
    """Create Supabase client for integration tests."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")

    if not url or not key:
        pytest.skip("Supabase credentials not available")

    return get_supabase_client(url, key)


@pytest.fixture
def question_repository(supabase_client):
    """Create QuestionRepository with real Supabase client."""
    return QuestionRepository(supabase_client)


@pytest.fixture
def session_repository(supabase_client):
    """Create GenerationSessionRepository with real Supabase client."""
    return GenerationSessionRepository(supabase_client)


@pytest.fixture
def sample_question():
    """Create a sample question for testing."""
    return Question(
        question_id_local="1a",
        question_id_global=str(uuid4()),
        question_number_display="1 (a)",
        marks=3,
        command_word=CommandWord.CALCULATE,
        raw_text_content="Calculate the area of a circle with radius 5cm.",
        taxonomy=QuestionTaxonomy(
            topic_path=["Geometry", "Circle"],
            subject_content_references=["C5.2"],  # Valid enum value for geometry
            skill_tags=["area_calculation", "circles"],
        ),
        solution_and_marking_scheme=SolutionAndMarkingScheme(
            final_answers_summary=[FinalAnswer(answer_text="78.54 cm²", value_numeric=78.54)],
            mark_allocation_criteria=[
                MarkingCriterion(
                    criterion_id="1a_m1",
                    criterion_text="Correct formula π × r²",
                    mark_code_display="M1",
                    marks_value=1,
                ),
                MarkingCriterion(
                    criterion_id="1a_m2",
                    criterion_text="Correct substitution",
                    mark_code_display="M2",
                    marks_value=1,
                ),
                MarkingCriterion(
                    criterion_id="1a_m3",
                    criterion_text="Correct calculation",
                    mark_code_display="A1",
                    marks_value=1,
                ),
            ],
            total_marks_for_part=3,
        ),
        solver_algorithm=SolverAlgorithm(
            steps=[
                SolverStep(
                    step_number=1,
                    description_text="Apply formula A = π × r²",
                    mathematical_expression_latex="A = \\pi \\times 5^2",
                ),
                SolverStep(
                    step_number=2,
                    description_text="Calculate the result",
                    mathematical_expression_latex="A = 25\\pi \\approx 78.54 \\text{ cm}^2",
                ),
            ]
        ),
    )


@pytest.fixture
def sample_generation_session(sample_question):
    """Create a sample generation session for testing."""
    return GenerationSession(
        session_id=uuid4(),
        request=GenerationRequest(
            topic="geometry",
            tier=Tier.CORE,
            marks=3,
            command_word=CommandWord.CALCULATE,
        ),
        questions=[sample_question],
        quality_decisions=[],
        agent_results=[],
    )


class TestQuestionRepositoryIntegration:
    """Integration tests for QuestionRepository with real Supabase."""

    def test_save_and_get_question(self, question_repository, sample_question):
        """Test saving and retrieving a question from Supabase."""
        # Save question
        question_id = question_repository.save_question(
            sample_question, origin=QuestionOrigin.GENERATED
        )

        assert question_id is not None
        assert len(question_id) > 0

        # Retrieve question
        retrieved_question = question_repository.get_question(sample_question.question_id_global)

        assert retrieved_question is not None
        assert retrieved_question.question_id_global == sample_question.question_id_global
        assert retrieved_question.marks == sample_question.marks
        assert retrieved_question.command_word == sample_question.command_word
        assert retrieved_question.raw_text_content == sample_question.raw_text_content

        # Cleanup
        question_repository.delete_question(sample_question.question_id_global)

    def test_list_questions_with_filters(self, question_repository, sample_question):
        """Test listing questions with various filters."""
        # Save question first
        question_id = question_repository.save_question(sample_question)

        try:
            # Test filtering by tier
            questions = question_repository.list_questions(tier=Tier.CORE, limit=10)

            assert len(questions) >= 1
            # Should find our question
            our_question = next(
                (
                    q
                    for q in questions
                    if q["question_id_global"] == sample_question.question_id_global
                ),
                None,
            )
            assert our_question is not None
            assert our_question["tier"] == "Core"

            # Test filtering by command word
            questions = question_repository.list_questions(
                command_word=CommandWord.CALCULATE, limit=10
            )

            our_question = next(
                (
                    q
                    for q in questions
                    if q["question_id_global"] == sample_question.question_id_global
                ),
                None,
            )
            assert our_question is not None
            assert our_question["command_word"] == "Calculate"

        finally:
            # Cleanup
            question_repository.delete_question(sample_question.question_id_global)

    def test_update_question_quality_score(self, question_repository, sample_question):
        """Test updating a question's quality score."""
        # Save question
        question_id = question_repository.save_question(sample_question)

        try:
            # Update quality score
            success = question_repository.update_quality_score(
                sample_question.question_id_global, 0.85
            )

            assert success is True

            # Verify update
            retrieved_question = question_repository.get_question(
                sample_question.question_id_global
            )

            # Note: Quality score might be stored in database metadata
            # The exact assertion depends on implementation
            assert retrieved_question is not None

        finally:
            # Cleanup
            question_repository.delete_question(sample_question.question_id_global)

    def test_delete_question(self, question_repository, sample_question):
        """Test deleting a question from Supabase."""
        # Save question
        question_id = question_repository.save_question(sample_question)

        # Verify it exists
        retrieved_question = question_repository.get_question(sample_question.question_id_global)
        assert retrieved_question is not None

        # Delete question
        success = question_repository.delete_question(sample_question.question_id_global)
        assert success is True

        # Verify it's gone
        deleted_question = question_repository.get_question(sample_question.question_id_global)
        assert deleted_question is None


class TestGenerationSessionRepositoryIntegration:
    """Integration tests for GenerationSessionRepository with real Supabase."""

    def test_save_and_get_session(self, session_repository, sample_generation_session):
        """Test saving and retrieving a generation session from Supabase."""
        # Save session
        session_id = session_repository.save_session(sample_generation_session)

        assert session_id is not None
        assert len(session_id) > 0

        # Retrieve session
        retrieved_session = session_repository.get_session(
            str(sample_generation_session.session_id)
        )

        assert retrieved_session is not None
        assert retrieved_session.session_id == sample_generation_session.session_id
        assert retrieved_session.request.topic == sample_generation_session.request.topic
        assert retrieved_session.request.tier == sample_generation_session.request.tier
        assert len(retrieved_session.questions) == len(sample_generation_session.questions)

        # Cleanup
        session_repository.delete_session(str(sample_generation_session.session_id))

    def test_list_sessions_with_filters(self, session_repository, sample_generation_session):
        """Test listing sessions with status filter."""
        # Save session
        session_id = session_repository.save_session(sample_generation_session)

        try:
            # List sessions
            sessions = session_repository.list_sessions(limit=10)

            assert len(sessions) >= 1
            # Should find our session
            our_session = next(
                (
                    s
                    for s in sessions
                    if s["session_id"] == str(sample_generation_session.session_id)
                ),
                None,
            )
            assert our_session is not None
            assert our_session["topic"] == "geometry"
            assert our_session["tier"] == "Core"

        finally:
            # Cleanup
            session_repository.delete_session(str(sample_generation_session.session_id))

    def test_update_session_status(self, session_repository, sample_generation_session):
        """Test updating session status and completion time."""
        # Save session
        session_id = session_repository.save_session(sample_generation_session)

        try:
            # Update session
            sample_generation_session.status = sample_generation_session.status.APPROVED
            success = session_repository.update_session(sample_generation_session)

            assert success is True

            # Verify update
            retrieved_session = session_repository.get_session(
                str(sample_generation_session.session_id)
            )

            assert retrieved_session.status.value == "approved"

        finally:
            # Cleanup
            session_repository.delete_session(str(sample_generation_session.session_id))


class TestSupabaseConnectionIntegration:
    """Integration tests for Supabase connection and basic operations."""

    def test_supabase_connection(self, supabase_client):
        """Test that we can connect to Supabase and query basic info."""
        # Test basic connection by querying enum tables
        response = supabase_client.table("tiers").select("*").execute()

        assert response.data is not None
        assert len(response.data) >= 2  # Should have Core and Extended

        tier_values = {tier["value"] for tier in response.data}
        assert "Core" in tier_values
        assert "Extended" in tier_values

    def test_enum_tables_populated(self, supabase_client):
        """Test that enum tables are properly populated."""
        # Test command_words table
        response = supabase_client.table("command_words").select("*").execute()
        assert len(response.data) >= 5  # Should have multiple command words

        command_words = {cw["value"] for cw in response.data}
        assert "Calculate" in command_words
        assert "Solve" in command_words

        # Test calculator_policies table
        response = supabase_client.table("calculator_policies").select("*").execute()
        assert len(response.data) == 3  # required, allowed, not_allowed

        policies = {cp["value"] for cp in response.data}
        assert "required" in policies
        assert "allowed" in policies
        assert "not_allowed" in policies

    def test_database_schema_exists(self, supabase_client):
        """Test that our main tables exist with correct schema."""
        # Test generated_questions table exists and has correct columns
        try:
            response = supabase_client.table("generated_questions").select("id").limit(1).execute()
            # If this doesn't raise an exception, table exists
            assert response is not None
        except Exception as e:
            pytest.fail(f"Questions table not accessible: {e}")

        # Test generation_sessions table exists
        try:
            response = supabase_client.table("generation_sessions").select("id").limit(1).execute()
            assert response is not None
        except Exception as e:
            pytest.fail(f"Generation sessions table not accessible: {e}")


# Test configuration for pytest
def pytest_configure(config):
    """Configure pytest for integration tests."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require Supabase)"
    )


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

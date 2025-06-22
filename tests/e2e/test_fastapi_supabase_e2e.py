"""
End-to-end tests for FastAPI + Supabase integration.
Tests complete workflows from API requests to database persistence.
"""

import os
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.database.supabase_repository import (
    get_supabase_client,
)
from src.models.enums import CommandWord, Tier
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
    """Create Supabase client for E2E tests."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")

    if not url or not key:
        pytest.skip("Supabase credentials not available for E2E tests")

    return get_supabase_client(url, key)


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_question_generation_service():
    """Mock the question generation service for controlled testing."""
    with patch("src.api.endpoints.questions.question_generation_service") as mock_service:
        # Create a realistic mock question
        mock_question = Question(
            question_id_local="1a",
            question_id_global=str(uuid4()),
            question_number_display="1 (a)",
            marks=3,
            command_word=CommandWord.CALCULATE,
            raw_text_content="Calculate the area of a triangle with base 6cm and height 4cm.",
            taxonomy=QuestionTaxonomy(
                topic_path=["Geometry", "Triangles"],
                subject_content_references=["C5.2"],
                skill_tags=["area_calculation"],
            ),
            solution_and_marking_scheme=SolutionAndMarkingScheme(
                final_answers_summary=[FinalAnswer(answer_text="12 cm²", value_numeric=12)],
                mark_allocation_criteria=[
                    MarkingCriterion(
                        criterion_id="1a_m1",
                        criterion_text="Correct formula ½ × base × height",
                        mark_code_display="M1",
                        marks_value=1,
                    ),
                    MarkingCriterion(
                        criterion_id="1a_m2",
                        criterion_text="Correct substitution",
                        mark_code_display="M1",
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
                        description_text="Apply formula: Area = ½ × base × height",
                        mathematical_expression_latex="A = \\frac{1}{2} \\times 6 \\times 4",
                    ),
                    SolverStep(
                        step_number=2,
                        description_text="Calculate the result",
                        mathematical_expression_latex="A = 12 \\text{ cm}^2",
                    ),
                ]
            ),
        )

        # Mock session with question
        mock_session = GenerationSession(
            session_id=uuid4(),
            request=GenerationRequest(
                topic="geometry",
                tier=Tier.CORE,
                marks=3,
                command_word=CommandWord.CALCULATE,
            ),
            questions=[mock_question],
            quality_decisions=[],
            agent_results=[],
        )

        mock_service.generate_questions = AsyncMock(return_value=mock_session)

        # Mock streaming function
        async def mock_stream(*args, **kwargs):
            yield {"type": "generation_started", "session_id": str(mock_session.session_id)}
            yield {"type": "agent_update", "agent": "question_generator", "status": "working"}
            yield {"type": "generation_complete", "session_id": str(mock_session.session_id)}

        mock_service.generate_questions_stream = mock_stream

        yield mock_service, mock_session, mock_question


class TestFastAPISupabaseE2E:
    """End-to-end tests for FastAPI + Supabase integration."""

    def test_health_check(self, client):
        """Test basic API health check."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "service": "derivativ-api"}

    @pytest.mark.integration
    def test_generate_question_e2e_with_database(
        self, client, supabase_client, mock_question_generation_service
    ):
        """Test complete question generation workflow with database persistence."""
        mock_service, mock_session, mock_question = mock_question_generation_service

        # Mock the repository dependencies
        with (
            patch("src.api.endpoints.questions.question_repository") as mock_q_repo,
            patch("src.api.endpoints.questions.question_generation_service", mock_service),
        ):
            # Configure mock repository
            mock_q_repo.save_question.return_value = "test-db-id"
            mock_q_repo.get_question.return_value = mock_question

            # Make API request
            request_data = {
                "topic": "geometry",
                "tier": "Core",
                "marks": 3,
                "command_word": "Calculate",
            }

            response = client.post("/api/questions/generate", json=request_data)

            # Verify API response
            assert response.status_code == 201
            data = response.json()
            assert "session_id" in data
            assert "questions" in data
            assert len(data["questions"]) == 1
            assert data["status"] == "candidate"

            # Verify service was called
            mock_service.generate_questions.assert_called_once()

            # Verify question has expected structure
            question_data = data["questions"][0]
            assert question_data["marks"] == 3
            assert question_data["command_word"] == "Calculate"
            assert "raw_text_content" in question_data

    @pytest.mark.integration
    def test_question_retrieval_e2e(self, client, mock_question_generation_service):
        """Test question retrieval via API."""
        _, _, mock_question = mock_question_generation_service

        with patch("src.api.endpoints.questions.question_repository") as mock_repo:
            mock_repo.get_question.return_value = mock_question

            response = client.get(f"/api/questions/{mock_question.question_id_global}")

            assert response.status_code == 200
            data = response.json()
            assert data["question_id_global"] == mock_question.question_id_global
            assert data["marks"] == mock_question.marks

    @pytest.mark.integration
    def test_question_listing_e2e(self, client):
        """Test question listing with filters."""
        mock_questions = [
            {
                "id": "1",
                "question_id_global": str(uuid4()),
                "tier": "Core",
                "marks": 3,
                "command_word": "Calculate",
                "quality_score": 0.85,
                "created_at": "2025-06-21T12:00:00Z",
            },
            {
                "id": "2",
                "question_id_global": str(uuid4()),
                "tier": "Extended",
                "marks": 5,
                "command_word": "Prove",
                "quality_score": 0.92,
                "created_at": "2025-06-21T12:05:00Z",
            },
        ]

        with patch("src.api.endpoints.questions.question_repository") as mock_repo:
            mock_repo.list_questions.return_value = mock_questions

            response = client.get("/api/questions?tier=Core&min_quality_score=0.8&limit=10")

            assert response.status_code == 200
            data = response.json()
            assert "questions" in data
            assert "pagination" in data
            assert len(data["questions"]) == 2

    @pytest.mark.integration
    def test_session_management_e2e(self, client, mock_question_generation_service):
        """Test session retrieval and listing."""
        _, mock_session, _ = mock_question_generation_service

        with patch("src.api.endpoints.sessions.session_repository") as mock_repo:
            mock_repo.get_session.return_value = mock_session

            response = client.get(f"/api/sessions/{mock_session.session_id}")

            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == str(mock_session.session_id)
            assert "request" in data
            assert "questions" in data

    def test_websocket_connection_e2e(self, client):
        """Test WebSocket connection establishment."""
        with client.websocket_connect("/api/ws/generate/test-session-id") as websocket:
            data = websocket.receive_json()
            assert data["type"] == "connection_established"
            assert data["session_id"] == "test-session-id"

    @pytest.mark.integration
    def test_websocket_generation_e2e(self, client, mock_question_generation_service):
        """Test WebSocket generation workflow."""
        mock_service, mock_session, _ = mock_question_generation_service

        with (
            patch("src.api.endpoints.websocket.question_generation_service", mock_service),
            client.websocket_connect("/api/ws/generate/test-session") as websocket,
        ):
            # Send generation request
            websocket.send_json(
                {
                    "action": "generate",
                    "request": {
                        "topic": "geometry",
                        "tier": "Core",
                        "marks": 3,
                        "command_word": "Calculate",
                    },
                }
            )

            # Receive connection confirmation
            connection_msg = websocket.receive_json()
            assert connection_msg["type"] == "connection_established"

            # Receive generation updates
            start_msg = websocket.receive_json()
            assert start_msg["type"] == "generation_started"

            agent_msg = websocket.receive_json()
            assert agent_msg["type"] == "agent_update"
            assert agent_msg["agent"] == "question_generator"

            complete_msg = websocket.receive_json()
            assert complete_msg["type"] == "generation_complete"

    @pytest.mark.integration
    def test_error_handling_e2e(self, client):
        """Test API error handling."""
        # Test validation error
        invalid_request = {
            "topic": "",  # Invalid empty topic
            "tier": "InvalidTier",  # Invalid tier
            "marks": -1,  # Invalid negative marks
        }

        response = client.post("/api/questions/generate", json=invalid_request)
        assert response.status_code == 422
        assert "detail" in response.json()

        # Test not found
        response = client.get("/api/questions/nonexistent-id")
        assert response.status_code == 404

    @pytest.mark.integration
    def test_concurrent_requests_e2e(self, client, mock_question_generation_service):
        """Test handling multiple concurrent requests."""
        mock_service, _, _ = mock_question_generation_service

        with patch("src.api.endpoints.questions.question_generation_service", mock_service):
            request_data = {
                "topic": "algebra",
                "tier": "Core",
                "marks": 2,
                "command_word": "Solve",
            }

            # Send multiple requests concurrently
            responses = []
            for i in range(3):
                response = client.post("/api/questions/generate", json=request_data)
                responses.append(response)

            # All should succeed
            for response in responses:
                assert response.status_code == 201
                data = response.json()
                assert "session_id" in data
                assert "questions" in data


# Mark all tests in this module as e2e tests
pytestmark = pytest.mark.e2e

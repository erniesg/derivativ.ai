"""
Unit tests for FastAPI endpoints.
Tests API logic with mocked dependencies.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import (
    get_question_generation_service,
    get_question_repository,
)
from src.api.main import app
from src.models.enums import CommandWord, Tier
from src.models.question_models import GenerationRequest, GenerationSession, Question


class TestQuestionGenerationAPI:
    """Unit tests for question generation endpoints."""

    @pytest.fixture
    def client(self):
        """FastAPI test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_generation_request(self):
        """Sample generation request."""
        return {
            "topic": "algebra",
            "tier": "Core",
            "grade_level": 8,
            "marks": 3,
            "count": 1,
            "calculator_policy": "not_allowed",
            "command_word": "Calculate",
        }

    @pytest.fixture
    def sample_question(self):
        """Sample generated question."""
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
            raw_text_content="Calculate the value of x when 2x + 5 = 17",
            taxonomy=QuestionTaxonomy(
                topic_path=["Algebra"],
                subject_content_references=[],
                skill_tags=["linear_equations"],
            ),
            solution_and_marking_scheme=SolutionAndMarkingScheme(
                final_answers_summary=[FinalAnswer(answer_text="x = 6", value_numeric=6)],
                mark_allocation_criteria=[
                    MarkingCriterion(
                        criterion_id="1a_m1",
                        criterion_text="Correct rearrangement",
                        mark_code_display="M1",
                        marks_value=1,
                    )
                ],
                total_marks_for_part=3,
            ),
            solver_algorithm=SolverAlgorithm(
                steps=[
                    SolverStep(
                        step_number=1,
                        description_text="Subtract 5 from both sides",
                        mathematical_expression_latex="2x = 12",
                    )
                ]
            ),
        )

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "service" in data
        assert data["service"] == "derivativ-api"
        # Status may be healthy or unhealthy depending on configuration

    def test_generate_question_success(self, client, sample_generation_request, sample_question):
        """Test successful question generation."""
        # Mock successful generation
        mock_session = GenerationSession(
            session_id=uuid4(),
            request=GenerationRequest(**sample_generation_request),
            questions=[sample_question],
            quality_decisions=[],
            agent_results=[],
        )
        mock_service = AsyncMock()
        mock_service.generate_questions = AsyncMock(return_value=mock_session)

        # Override the dependency
        app.dependency_overrides[get_question_generation_service] = lambda: mock_service

        try:
            response = client.post("/api/questions/generate", json=sample_generation_request)

            assert response.status_code == 201
            data = response.json()
            assert "session_id" in data
            assert "questions" in data
            assert len(data["questions"]) == 1
            assert data["status"] == "candidate"

            # Verify service was called correctly
            mock_service.generate_questions.assert_called_once()
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()

    def test_generate_question_validation_error(self, client):
        """Test generation with invalid request data."""
        # Mock service (even though it shouldn't be called)
        mock_service = AsyncMock()
        app.dependency_overrides[get_question_generation_service] = lambda: mock_service

        try:
            invalid_request = {
                "topic": "",  # Invalid: empty topic
                "marks": -1,  # Invalid: negative marks
            }

            response = client.post("/api/questions/generate", json=invalid_request)

            assert response.status_code == 422
            assert "detail" in response.json()

            # Service should not be called for invalid requests
            mock_service.generate_questions.assert_not_called()
        finally:
            app.dependency_overrides.clear()

    def test_generate_question_service_error(self, client, sample_generation_request):
        """Test generation when service fails."""
        # Mock service failure
        mock_service = AsyncMock()
        mock_service.generate_questions = AsyncMock(side_effect=Exception("Generation failed"))
        app.dependency_overrides[get_question_generation_service] = lambda: mock_service

        try:
            response = client.post("/api/questions/generate", json=sample_generation_request)

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Generation failed" in data["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_get_question_success(self, client, sample_question):
        """Test successful question retrieval."""
        # Mock successful retrieval
        mock_repository = Mock()
        mock_repository.get_question = Mock(return_value=sample_question)
        app.dependency_overrides[get_question_repository] = lambda: mock_repository

        try:
            response = client.get(f"/api/questions/{sample_question.question_id_global}")

            assert response.status_code == 200
            data = response.json()
            assert data["question_id_global"] == sample_question.question_id_global
            assert data["marks"] == sample_question.marks
        finally:
            app.dependency_overrides.clear()

    def test_get_question_not_found(self, client):
        """Test question not found."""
        # Mock question not found
        mock_repository = Mock()
        mock_repository.get_question = Mock(return_value=None)
        app.dependency_overrides[get_question_repository] = lambda: mock_repository

        try:
            response = client.get("/api/questions/nonexistent-id")

            assert response.status_code == 404
            data = response.json()
            assert "not found" in data["detail"].lower()
        finally:
            app.dependency_overrides.clear()

    @patch("src.api.endpoints.questions.question_repository")
    def test_list_questions_success(self, mock_repository, client):
        """Test successful question listing."""
        # Mock question list
        mock_questions = [
            {
                "id": "1",
                "question_id_global": "q1",
                "tier": "core",
                "marks": 3,
                "command_word": "calculate",
                "quality_score": 0.8,
                "created_at": datetime.utcnow().isoformat(),
            },
            {
                "id": "2",
                "question_id_global": "q2",
                "tier": "extended",
                "marks": 5,
                "command_word": "explain",
                "quality_score": 0.9,
                "created_at": datetime.utcnow().isoformat(),
            },
        ]
        mock_repository.list_questions = Mock(return_value=mock_questions)

        response = client.get("/api/questions?tier=Core&min_quality_score=0.7&limit=10")

        assert response.status_code == 200
        data = response.json()
        assert "questions" in data
        assert len(data["questions"]) == 2
        assert "pagination" in data

        # Verify repository was called with correct filters
        mock_repository.list_questions.assert_called_once_with(
            tier=Tier.CORE, min_quality_score=0.7, command_word=None, limit=10, offset=0
        )

    @patch("src.api.endpoints.questions.question_repository")
    def test_delete_question_success(self, mock_repository, client):
        """Test successful question deletion."""
        # Mock successful deletion
        mock_repository.delete_question = Mock(return_value=True)

        response = client.delete("/api/questions/test-id")

        assert response.status_code == 204

        # Verify repository was called
        mock_repository.delete_question.assert_called_once_with("test-id")

    @patch("src.api.endpoints.questions.question_repository")
    def test_delete_question_not_found(self, mock_repository, client):
        """Test deletion of non-existent question."""
        # Mock question not found
        mock_repository.delete_question = Mock(return_value=False)

        response = client.delete("/api/questions/nonexistent-id")

        assert response.status_code == 404


class TestSessionAPI:
    """Unit tests for session management endpoints."""

    @pytest.fixture
    def client(self):
        """FastAPI test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_session(self):
        """Sample generation session."""
        return GenerationSession(
            session_id=uuid4(),
            request=GenerationRequest(topic="algebra", tier=Tier.CORE, marks=3),
            questions=[],
            quality_decisions=[],
            agent_results=[],
        )

    @patch("src.api.endpoints.sessions.session_repository")
    def test_get_session_success(self, mock_repository, client, sample_session):
        """Test successful session retrieval."""
        # Mock successful retrieval
        mock_repository.get_session = Mock(return_value=sample_session)

        response = client.get(f"/api/sessions/{sample_session.session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == str(sample_session.session_id)
        assert "request" in data
        assert "questions" in data
        assert "agent_results" in data

    @patch("src.api.endpoints.sessions.session_repository")
    def test_get_session_not_found(self, mock_repository, client):
        """Test session not found."""
        # Mock session not found
        mock_repository.get_session = Mock(return_value=None)

        response = client.get("/api/sessions/nonexistent-id")

        assert response.status_code == 404

    @patch("src.api.endpoints.sessions.session_repository")
    def test_list_sessions_success(self, mock_repository, client):
        """Test successful session listing."""
        # Mock session list
        mock_sessions = [
            {
                "id": "1",
                "session_id": str(uuid4()),
                "status": "candidate",
                "total_processing_time": 25.5,
                "created_at": datetime.utcnow().isoformat(),
            }
        ]
        mock_repository.list_sessions = Mock(return_value=mock_sessions)

        response = client.get("/api/sessions?status=candidate&limit=10")

        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert len(data["sessions"]) == 1


class TestWebSocketAPI:
    """Unit tests for WebSocket endpoints."""

    @pytest.fixture
    def client(self):
        """FastAPI test client."""
        return TestClient(app)

    def test_websocket_connection(self, client):
        """Test WebSocket connection establishment."""
        with client.websocket_connect("/api/ws/generate/test-session-id") as websocket:
            # Test connection is established
            data = websocket.receive_json()
            assert data["type"] == "connection_established"
            assert data["session_id"] == "test-session-id"

    @patch("src.api.endpoints.websocket.question_generation_service")
    def test_websocket_generation_updates(self, mock_service, client):
        """Test WebSocket receives generation updates."""

        # Mock generation service with updates
        async def mock_generate_with_updates(*args, **kwargs):
            # Simulate agent updates
            yield {"type": "agent_update", "agent": "question_generator", "status": "working"}
            yield {"type": "agent_update", "agent": "reviewer", "status": "complete"}
            yield {"type": "generation_complete", "session_id": "test-session"}

        mock_service.generate_questions_stream = mock_generate_with_updates

        with client.websocket_connect("/api/ws/generate/test-session") as websocket:
            # Send generation request
            websocket.send_json(
                {"action": "generate", "request": {"topic": "algebra", "tier": "core", "marks": 3}}
            )

            # First receive connection confirmation
            connection = websocket.receive_json()
            assert connection["type"] == "connection_established"

            # Then receive updates
            update1 = websocket.receive_json()
            assert update1["type"] == "agent_update"
            assert update1["agent"] == "question_generator"

            update2 = websocket.receive_json()
            assert update2["type"] == "agent_update"
            assert update2["agent"] == "reviewer"

            final = websocket.receive_json()
            assert final["type"] == "generation_complete"

"""
Performance tests for FastAPI + Supabase integration.
Tests generation speed, concurrent handling, and database performance.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_question_generation_service, get_question_repository
from src.api.main import app
from src.models.enums import CommandWord, Tier
from src.models.question_models import GenerationRequest, GenerationSession


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_fast_generation_service():
    """Mock service with fast generation for performance testing."""
    with patch("src.api.endpoints.questions.question_generation_service") as mock_service:
        # Mock a fast generation process
        mock_session = GenerationSession(
            session_id=uuid4(),
            request=GenerationRequest(
                topic="algebra",
                tier=Tier.CORE,
                marks=3,
                command_word=CommandWord.CALCULATE,
            ),
            questions=[],
            quality_decisions=[],
            agent_results=[],
        )

        async def fast_generate(*args, **kwargs):
            # Simulate fast generation (100ms)
            await asyncio.sleep(0.1)
            return mock_session

        mock_service.generate_questions = AsyncMock(side_effect=fast_generate)

        async def fast_stream(*args, **kwargs):
            yield {"type": "generation_started", "session_id": str(mock_session.session_id)}
            await asyncio.sleep(0.05)  # 50ms delay
            yield {"type": "generation_complete", "session_id": str(mock_session.session_id)}

        mock_service.generate_questions_stream = fast_stream

        yield mock_service


@pytest.fixture
def mock_slow_generation_service():
    """Mock service with slow generation for timeout testing."""
    with patch("src.api.endpoints.questions.question_generation_service") as mock_service:

        async def slow_generate(*args, **kwargs):
            # Simulate slow generation (35 seconds - over target)
            await asyncio.sleep(35.0)
            return GenerationSession(
                session_id=uuid4(),
                request=GenerationRequest(topic="geometry", tier=Tier.CORE, marks=5),
                questions=[],
                quality_decisions=[],
                agent_results=[],
            )

        mock_service.generate_questions = AsyncMock(side_effect=slow_generate)
        yield mock_service


class TestAPIPerformance:
    """Performance tests for API endpoints."""

    @pytest.mark.performance
    def test_health_check_response_time(self, client):
        """Test health check responds quickly."""
        start_time = time.time()
        response = client.get("/health")
        response_time = time.time() - start_time

        assert response.status_code == 200
        assert response_time < 0.1  # Should respond in under 100ms

    @pytest.mark.performance
    def test_question_generation_speed(self, client, mock_fast_generation_service):
        """Test question generation completes within target time."""
        # Mock repository
        mock_repo = Mock()
        mock_repo.save_question.return_value = "test-id"

        # Override dependencies
        app.dependency_overrides[get_question_generation_service] = (
            lambda: mock_fast_generation_service
        )
        app.dependency_overrides[get_question_repository] = lambda: mock_repo

        try:
            request_data = {
                "topic": "algebra",
                "tier": "Core",
                "marks": 3,
                "command_word": "Calculate",
            }

            start_time = time.time()
            response = client.post("/api/questions/generate", json=request_data)
            generation_time = time.time() - start_time

            assert response.status_code == 201
            assert generation_time < 30.0  # Target: sub-30 second generation
            assert generation_time > 0.05  # Should take some time (not mocked completely)
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.performance
    def test_concurrent_request_handling(self, client, mock_fast_generation_service):
        """Test API can handle multiple concurrent requests."""
        # Mock repository
        mock_repo = Mock()
        mock_repo.save_question.return_value = "test-id"

        # Override dependencies
        app.dependency_overrides[get_question_generation_service] = (
            lambda: mock_fast_generation_service
        )
        app.dependency_overrides[get_question_repository] = lambda: mock_repo

        try:
            request_data = {
                "topic": "geometry",
                "tier": "Core",
                "marks": 2,
                "command_word": "Solve",
            }

            # Test with 5 concurrent requests
            num_requests = 5
            start_time = time.time()

            with ThreadPoolExecutor(max_workers=num_requests) as executor:
                futures = [
                    executor.submit(client.post, "/api/questions/generate", json=request_data)
                    for _ in range(num_requests)
                ]

                responses = [future.result() for future in as_completed(futures)]

            total_time = time.time() - start_time

            # All requests should succeed
            for response in responses:
                assert response.status_code == 201

            # Concurrent handling should be faster than sequential
            # (5 requests * 0.1s each = 0.5s sequential vs ~0.2s concurrent)
            assert total_time < 1.0
            assert len(responses) == num_requests
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.performance
    def test_question_listing_performance(self, client):
        """Test question listing endpoint performance."""
        # Mock a large dataset
        mock_questions = [
            {
                "id": str(i),
                "question_id_global": str(uuid4()),
                "tier": "Core" if i % 2 == 0 else "Extended",
                "marks": (i % 5) + 1,
                "command_word": "Calculate",
                "quality_score": 0.8 + (i % 20) * 0.01,
                "created_at": f"2025-06-21T12:{i:02d}:00Z",
            }
            for i in range(100)  # 100 questions
        ]

        # Mock repository
        mock_repo = Mock()
        mock_repo.list_questions.return_value = mock_questions
        app.dependency_overrides[get_question_repository] = lambda: mock_repo

        try:
            start_time = time.time()
            response = client.get("/api/questions?limit=50")
            response_time = time.time() - start_time

            assert response.status_code == 200
            assert response_time < 0.5  # Should respond quickly even with large dataset

            data = response.json()
            assert len(data["questions"]) == 100
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.performance
    def test_websocket_connection_speed(self, client):
        """Test WebSocket connection establishment speed."""
        # Mock service for WebSocket
        mock_service = Mock()
        app.dependency_overrides[get_question_generation_service] = lambda: mock_service

        try:
            start_time = time.time()

            with client.websocket_connect("/api/ws/generate/perf-test-session") as websocket:
                connection_time = time.time() - start_time

                # Receive connection confirmation
                data = websocket.receive_json()
                confirmation_time = time.time() - start_time

                assert data["type"] == "connection_established"
                assert connection_time < 0.1  # Connection should be fast
                assert confirmation_time < 0.2  # Confirmation should be quick
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.performance
    def test_websocket_streaming_performance(self, client, mock_fast_generation_service):
        """Test WebSocket streaming performance."""
        # Override dependency for WebSocket endpoint
        app.dependency_overrides[get_question_generation_service] = (
            lambda: mock_fast_generation_service
        )

        try:
            start_time = time.time()

            with client.websocket_connect("/api/ws/generate/stream-perf-test") as websocket:
                # Send generation request
                websocket.send_json(
                    {
                        "action": "generate",
                        "request": {"topic": "algebra", "tier": "Core", "marks": 2},
                    }
                )

                # Receive all messages
                messages = []
                connection_msg = websocket.receive_json()
                messages.append(connection_msg)

                start_msg = websocket.receive_json()
                messages.append(start_msg)

                complete_msg = websocket.receive_json()
                messages.append(complete_msg)

                total_time = time.time() - start_time

                # Verify message order and timing
                assert connection_msg["type"] == "connection_established"
                assert start_msg["type"] == "generation_started"
                assert complete_msg["type"] == "generation_complete"

                # Should complete quickly with mocked service
                assert total_time < 1.0
                assert len(messages) == 3
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.performance
    def test_memory_usage_during_generation(self, client, mock_fast_generation_service):
        """Test memory usage doesn't grow excessively during generation."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Mock dependencies
        mock_repo = Mock()
        mock_repo.save_question.return_value = "test-id"
        app.dependency_overrides[get_question_generation_service] = (
            lambda: mock_fast_generation_service
        )
        app.dependency_overrides[get_question_repository] = lambda: mock_repo

        try:
            request_data = {
                "topic": "statistics",
                "tier": "Extended",
                "marks": 4,
                "command_word": "Analyze",
            }

            # Make multiple requests
            for i in range(10):
                response = client.post("/api/questions/generate", json=request_data)
                assert response.status_code == 201

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = final_memory - initial_memory

            # Memory growth should be reasonable (less than 50MB for 10 requests)
            assert memory_growth < 50
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.performance
    def test_error_response_performance(self, client):
        """Test error responses are fast - REAL FastAPI validation performance."""
        # Override dependencies to prevent 503 errors, but still test real validation
        from unittest.mock import Mock

        mock_service = Mock()
        mock_repo = Mock()
        app.dependency_overrides[get_question_generation_service] = lambda: mock_service
        app.dependency_overrides[get_question_repository] = lambda: mock_repo

        try:
            start_time = time.time()
            # Send invalid request to test REAL FastAPI validation speed
            response = client.post("/api/questions/generate", json={"invalid": "data"})
            error_time = time.time() - start_time

            # Should get validation error (422), not service error (503)
            assert response.status_code == 422
            assert error_time < 0.1  # Error responses should be very fast
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.performance
    @pytest.mark.slow
    def test_generation_timeout_handling(self, client, mock_slow_generation_service):
        """Test handling of slow generation (should complete or timeout gracefully)."""
        # Mock dependencies
        mock_repo = Mock()
        mock_repo.save_question.return_value = "test-id"
        app.dependency_overrides[get_question_generation_service] = (
            lambda: mock_slow_generation_service
        )
        app.dependency_overrides[get_question_repository] = lambda: mock_repo

        try:
            request_data = {
                "topic": "calculus",
                "tier": "Extended",
                "marks": 8,
                "command_word": "Derive",
            }

            # This should either complete quickly (if mocked) or handle timeout
            start_time = time.time()

            try:
                response = client.post("/api/questions/generate", json=request_data, timeout=32.0)
                generation_time = time.time() - start_time

                # If it completes, it should be either very fast (mocked) or reasonable time
                if response.status_code == 201:
                    assert generation_time < 35.0
                else:
                    # Should handle timeout/error gracefully
                    assert response.status_code in [408, 500, 503]

            except Exception as e:
                # Timeout exceptions should be handled gracefully
                generation_time = time.time() - start_time
                assert generation_time >= 30.0  # Should have tried for reasonable time
        finally:
            app.dependency_overrides.clear()


class TestDatabasePerformance:
    """Performance tests for database operations."""

    @pytest.mark.performance
    @pytest.mark.integration
    def test_question_save_performance(self):
        """Test question saving performance with mocked database."""
        from unittest.mock import Mock

        from src.database.supabase_repository import QuestionRepository
        from tests.conftest import create_test_question

        # Mock Supabase client with fast response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [{"id": "test-id"}]
        mock_client.table.return_value.insert.return_value.execute.return_value = mock_response

        repository = QuestionRepository(mock_client)

        # Create test question using helper
        question = create_test_question()

        start_time = time.time()
        result = repository.save_question(question)
        save_time = time.time() - start_time

        assert result == "test-id"
        assert save_time < 0.1  # Should be fast with mocked client

    @pytest.mark.performance
    @pytest.mark.integration
    def test_question_retrieval_performance(self):
        """Test question retrieval performance."""
        from unittest.mock import Mock

        from src.database.supabase_repository import QuestionRepository
        from tests.conftest import create_test_question

        # Create a test question to get the proper structure
        test_question = create_test_question()

        # Mock Supabase client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            {
                "question_id_global": "test-id",
                "content_json": test_question.model_dump(),
            }
        ]
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = (
            mock_response
        )

        repository = QuestionRepository(mock_client)

        start_time = time.time()
        result = repository.get_question("test-id")
        retrieval_time = time.time() - start_time

        assert result is not None
        assert retrieval_time < 0.1  # Should be fast


# Mark all tests in this module as performance tests
pytestmark = pytest.mark.performance

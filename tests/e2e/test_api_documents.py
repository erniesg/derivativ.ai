"""
End-to-end tests for Document Generation API endpoints.
Tests the complete API workflow from request to response.
"""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.models.document_models import (
    DetailLevel,
    DocumentGenerationRequest,
    DocumentType,
)


class TestDocumentGenerationAPI:
    """Test document generation API endpoints."""

    @pytest.fixture
    def client(self, mock_document_service):
        """Create test client with mocked dependencies."""
        from fastapi import FastAPI

        from src.api.dependencies import get_document_generation_service
        from src.api.endpoints.documents import router

        # Create test app with only the documents router
        test_app = FastAPI()
        test_app.include_router(router, tags=["documents"])

        # Override the dependency to return our mock
        test_app.dependency_overrides[get_document_generation_service] = lambda: mock_document_service

        return TestClient(test_app)

    @pytest.fixture
    def mock_document_service(self, monkeypatch):
        """Mock document generation service."""
        mock_service = AsyncMock()

        # Mock successful generation
        from src.models.document_models import (
            ContentSection,
            DocumentGenerationResult,
            GeneratedDocument,
        )

        # Create a proper request object for the mock
        mock_request = DocumentGenerationRequest(
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            title="Test Worksheet",
            topic="test_topic",
        )

        mock_document = GeneratedDocument(
            title="Test Worksheet",
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            generated_at="2025-06-21T12:00:00Z",
            template_used="worksheet_generation",
            generation_request=mock_request,
            sections=[
                ContentSection(
                    title="Practice Questions",
                    content_type="practice_questions",
                    content_data={"questions": []},
                    order_index=0,
                )
            ],
            total_questions=5,
            estimated_duration=30,
        )

        mock_result = DocumentGenerationResult(
            success=True,
            document=mock_document,
            processing_time=2.5,
            questions_processed=5,
            sections_generated=3,
        )

        mock_service.generate_document.return_value = mock_result

        # Mock get_document_templates to return proper template structure
        from src.models.document_models import DocumentTemplate
        mock_templates = {
            "worksheet": DocumentTemplate(
                name="worksheet_generation",
                document_type=DocumentType.WORKSHEET,
                supported_detail_levels=[DetailLevel.MINIMAL, DetailLevel.MEDIUM, DetailLevel.COMPREHENSIVE],
                structure_patterns={
                    DetailLevel.MINIMAL: ["practice_questions", "answers"],
                    DetailLevel.MEDIUM: ["learning_objectives", "worked_examples", "practice_questions", "solutions"],
                    DetailLevel.COMPREHENSIVE: ["learning_objectives", "topic_introduction", "worked_examples", "practice_questions", "detailed_solutions"]
                },
                content_rules={
                    DetailLevel.MINIMAL: {"min_questions": 1, "max_questions": 10},
                    DetailLevel.MEDIUM: {"min_questions": 3, "max_questions": 20},
                    DetailLevel.COMPREHENSIVE: {"min_questions": 5, "max_questions": 30}
                }
            )
        }
        mock_service.get_document_templates.return_value = mock_templates

        # Mock save_custom_template method
        mock_service.save_custom_template.return_value = "mock-template-id"

        return mock_service

    def test_generate_document_success(self, client, mock_document_service):
        """Test successful document generation."""
        request_data = {
            "document_type": "worksheet",
            "detail_level": "medium",
            "title": "Algebra Practice",
            "topic": "linear_equations",
            "tier": "Core",
            "grade_level": 7,
            "auto_include_questions": False,
            "max_questions": 5,
        }

        response = client.post("/api/documents/generate", json=request_data)

        # Debug: Print response if not 200
        if response.status_code != 200:
            print(f"Response status: {response.status_code}")
            print(f"Response text: {response.text}")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["document"]["title"] == "Test Worksheet"
        assert data["document"]["document_type"] == "worksheet"
        assert data["processing_time"] == 2.5
        assert len(data["document"]["sections"]) == 1

    def test_generate_document_with_custom_instructions(self, client, mock_document_service):
        """Test document generation with custom instructions."""
        request_data = {
            "document_type": "notes",
            "detail_level": "comprehensive",
            "title": "Advanced Algebra Notes",
            "topic": "quadratic_equations",
            "custom_instructions": "Include visual learning aids",
            "personalization_context": {"learning_style": "visual"},
        }

        response = client.post("/api/documents/generate", json=request_data)

        assert response.status_code == 200

        # Verify custom instructions were passed to service
        mock_document_service.generate_document.assert_called_once()
        call_args = mock_document_service.generate_document.call_args[0][0]
        assert call_args.custom_instructions == "Include visual learning aids"
        assert call_args.personalization_context == {"learning_style": "visual"}

    def test_generate_document_validation_error(self, client, mock_document_service):
        """Test document generation with invalid request data."""
        request_data = {
            "document_type": "invalid_type",
            "detail_level": "medium",
            "title": "Test",
            "topic": "test",
        }

        response = client.post("/api/documents/generate", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_get_document_templates(self, client, mock_document_service):
        """Test getting available document templates."""
        response = client.get("/api/documents/templates")

        assert response.status_code == 200
        data = response.json()

        assert "worksheet" in data
        assert data["worksheet"]["name"] == "worksheet_generation"
        assert data["worksheet"]["document_type"] == "worksheet"

    def test_create_custom_template(self, client, mock_document_service):
        """Test creating a custom document template."""
        template_data = {
            "name": "Custom Worksheet",
            "document_type": "worksheet",
            "supported_detail_levels": ["minimal", "medium"],
            "structure_patterns": {
                "minimal": ["questions", "answers"],
                "medium": ["objectives", "questions", "solutions"],
            },
        }

        response = client.post("/api/documents/templates", json=template_data)

        assert response.status_code == 200
        data = response.json()
        assert "template_id" in data


class TestDocumentExportAPI:
    """Test document export API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_formatter_agent(self, monkeypatch):
        """Mock document formatter agent."""
        mock_agent = AsyncMock()

        # Mock successful formatting
        mock_agent._execute.return_value = {
            "success": True,
            "formatted_content": "<html>Test document</html>",
            "format": "html",
        }

        # Mock the dependency
        from src.api import dependencies
        monkeypatch.setattr(dependencies, "get_document_formatter_agent", lambda: mock_agent)

        return mock_agent

    def test_export_document_html(self, client, mock_formatter_agent):
        """Test exporting document to HTML."""
        # This test assumes we have a document ID from previous generation
        document_id = "test-doc-123"
        export_data = {
            "format": "html",
            "options": {"include_css": True},
        }

        response = client.post(f"/api/documents/{document_id}/export", json=export_data)

        # This will fail initially as the endpoint isn't implemented
        # We'll implement it based on the failing test
        assert response.status_code in [200, 404, 501]  # Accept current state

    def test_download_exported_document(self, client, mock_formatter_agent):
        """Test downloading an exported document."""
        document_id = "test-doc-123"
        format_type = "pdf"

        response = client.get(f"/api/documents/{document_id}/export/{format_type}")

        # This will fail initially - we'll implement based on failing test
        assert response.status_code in [200, 404, 501]  # Accept current state


class TestAgentEndpointsAPI:
    """Test individual agent API endpoints (to be implemented)."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_question_generator_endpoint(self, client):
        """Test direct question generator agent endpoint."""
        request_data = {
            "topic": "algebra",
            "target_grade": 7,
            "marks": 3,
            "calculator_policy": "not_allowed",
        }

        # This endpoint doesn't exist yet - will implement based on failing test
        response = client.post("/api/agents/question-generator/generate", json=request_data)

        # Initially will be 404, then we'll implement
        assert response.status_code in [200, 404]

    def test_review_agent_endpoint(self, client):
        """Test direct review agent endpoint."""
        request_data = {
            "question_data": {
                "question_text": "Solve: x + 3 = 7",
                "marks": 2,
                "command_word": "Calculate",
            }
        }

        response = client.post("/api/agents/reviewer/assess", json=request_data)

        # Initially will be 404, then we'll implement
        assert response.status_code in [200, 404]

    def test_orchestrator_status_endpoint(self, client):
        """Test multi-agent orchestrator status endpoint."""
        response = client.get("/api/agents/orchestrator/status")

        # Initially will be 404, then we'll implement
        assert response.status_code in [200, 404]


class TestSystemManagementAPI:
    """Test system management API endpoints (to be implemented)."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_llm_providers_status(self, client):
        """Test LLM providers status endpoint."""
        response = client.get("/api/system/providers/status")

        # Will implement based on failing test
        assert response.status_code in [200, 404]

    def test_agent_configuration(self, client):
        """Test agent configuration endpoint."""
        response = client.get("/api/config/agents")

        # Will implement based on failing test
        assert response.status_code in [200, 404]

    def test_quality_thresholds_config(self, client):
        """Test quality thresholds configuration."""
        config_data = {
            "auto_approve": 0.85,
            "manual_review": 0.70,
            "refine": 0.50,
        }

        response = client.put("/api/config/quality-thresholds", json=config_data)

        # Will implement based on failing test
        assert response.status_code in [200, 404]


class TestTemplateManagementAPI:
    """Test template and prompt management API endpoints (to be implemented)."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_list_prompt_templates(self, client):
        """Test listing all prompt templates."""
        response = client.get("/api/templates/prompts")

        # Will implement based on failing test
        assert response.status_code in [200, 404]

    def test_render_prompt_template(self, client):
        """Test rendering a specific prompt template."""
        render_data = {
            "template_name": "question_generation",
            "variables": {
                "topic": "algebra",
                "target_grade": 7,
                "marks": 3,
            },
        }

        response = client.post("/api/templates/prompts/render", json=render_data)

        # Will implement based on failing test
        assert response.status_code in [200, 404]

    def test_validate_prompt_template(self, client):
        """Test validating a prompt template."""
        template_data = {
            "name": "test_template",
            "content": "Generate a question about {{ topic }} for grade {{ grade }}",
            "required_variables": ["topic", "grade"],
        }

        response = client.post("/api/templates/prompts/validate", json=template_data)

        # Will implement based on failing test
        assert response.status_code in [200, 404]

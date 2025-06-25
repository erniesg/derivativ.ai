"""
Integration tests for document storage API endpoints.
Tests FastAPI endpoints for document CRUD operations.
"""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


class TestDocumentAPIEndpoints:
    """Integration tests for document storage API endpoints."""

    @pytest.fixture
    def client(self):
        """Create FastAPI test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_document_request(self):
        """Sample document creation request."""
        return {
            "title": "API Test Worksheet",
            "document_type": "worksheet",
            "detail_level": "medium",
            "topic": "api_testing",
            "grade_level": 10,
            "estimated_duration": 45,
            "total_questions": 5,
            "tags": ["api", "testing", "algebra"],
            "session_id": str(uuid4()),
            "metadata": {
                "generation_request": {"document_type": "worksheet", "topic": "api_testing"}
            },
        }

    def test_create_document_endpoint_success(self, client, sample_document_request):
        """Test successful document creation via API."""
        response = client.post("/api/documents/", json=sample_document_request)

        assert response.status_code == 201

        data = response.json()
        assert "document_id" in data
        assert "status" in data
        assert data["status"] == "created"

        # Store document_id for cleanup
        document_id = data["document_id"]

        # Cleanup
        client.delete(f"/api/documents/{document_id}")

    def test_create_document_endpoint_validation_error(self, client):
        """Test document creation with invalid data."""
        invalid_request = {
            "title": "",  # Empty title should fail validation
            "document_type": "invalid_type",  # Invalid type
            "grade_level": 15,  # Invalid grade level
        }

        response = client.post("/api/documents/", json=invalid_request)

        assert response.status_code == 422  # Validation error

        error_data = response.json()
        assert "detail" in error_data

    def test_retrieve_document_endpoint_success(self, client, sample_document_request):
        """Test successful document retrieval via API."""
        # First, create a document
        create_response = client.post("/api/documents/", json=sample_document_request)
        assert create_response.status_code == 201

        document_id = create_response.json()["document_id"]

        try:
            # Retrieve the document
            response = client.get(f"/api/documents/{document_id}")

            assert response.status_code == 200

            data = response.json()
            assert data["metadata"]["id"] == document_id
            assert data["metadata"]["title"] == sample_document_request["title"]
            assert data["metadata"]["document_type"] == sample_document_request["document_type"]
            assert "files" in data
            assert "session_data" in data

        finally:
            # Cleanup
            client.delete(f"/api/documents/{document_id}")

    def test_retrieve_document_endpoint_not_found(self, client):
        """Test document retrieval with non-existent ID."""
        non_existent_id = str(uuid4())

        response = client.get(f"/api/documents/{non_existent_id}")

        assert response.status_code == 404

        error_data = response.json()
        assert "detail" in error_data
        assert "not found" in error_data["detail"].lower()

    def test_update_document_endpoint_success(self, client, sample_document_request):
        """Test successful document update via API."""
        # Create a document first
        create_response = client.post("/api/documents/", json=sample_document_request)
        document_id = create_response.json()["document_id"]

        try:
            # Update the document
            update_data = {
                "status": "exported",
                "metadata": {"export_completed_at": "2025-06-24T10:00:00Z"},
            }

            response = client.patch(f"/api/documents/{document_id}", json=update_data)

            assert response.status_code == 200

            data = response.json()
            assert data["success"] is True
            assert "updated_at" in data

            # Verify the update
            get_response = client.get(f"/api/documents/{document_id}")
            updated_doc = get_response.json()
            assert updated_doc["metadata"]["status"] == "exported"

        finally:
            # Cleanup
            client.delete(f"/api/documents/{document_id}")

    def test_delete_document_endpoint_success(self, client, sample_document_request):
        """Test successful document deletion via API."""
        # Create a document first
        create_response = client.post("/api/documents/", json=sample_document_request)
        document_id = create_response.json()["document_id"]

        # Delete the document
        response = client.delete(f"/api/documents/{document_id}")

        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["document_id"] == document_id

        # Verify deletion - document should be marked as deleted
        get_response = client.get(f"/api/documents/{document_id}")
        if get_response.status_code == 200:
            # If document still exists, it should be marked as deleted
            deleted_doc = get_response.json()
            assert deleted_doc["metadata"]["status"] == "deleted"

    def test_search_documents_endpoint_basic(self, client, sample_document_request):
        """Test basic document search via API."""
        # Create multiple documents for search testing
        documents = []
        for i in range(3):
            doc_request = sample_document_request.copy()
            doc_request["title"] = f"Search Test Document {i}"
            doc_request["topic"] = "search_testing"

            create_response = client.post("/api/documents/", json=doc_request)
            documents.append(create_response.json()["document_id"])

        try:
            # Search for documents
            search_params = {"topic": "search_testing", "document_type": "worksheet", "limit": 10}

            response = client.get("/api/documents/search", params=search_params)

            assert response.status_code == 200

            data = response.json()
            assert "documents" in data
            assert "total_count" in data
            assert "has_more" in data

            # Should find at least our test documents
            assert data["total_count"] >= 3
            assert len(data["documents"]) >= 3

            # Verify search results contain our documents
            found_titles = {doc["title"] for doc in data["documents"]}
            expected_titles = {f"Search Test Document {i}" for i in range(3)}
            assert expected_titles.issubset(found_titles)

        finally:
            # Cleanup all created documents
            for doc_id in documents:
                client.delete(f"/api/documents/{doc_id}")

    def test_search_documents_endpoint_with_filters(self, client, sample_document_request):
        """Test document search with various filters."""
        # Create documents with different attributes
        test_documents = [
            {**sample_document_request, "title": "Filter Test Worksheet", "grade_level": 9},
            {
                **sample_document_request,
                "title": "Filter Test Notes",
                "document_type": "notes",
                "grade_level": 10,
            },
            {
                **sample_document_request,
                "title": "Filter Test Textbook",
                "document_type": "textbook",
                "grade_level": 11,
            },
        ]

        document_ids = []
        for doc_request in test_documents:
            create_response = client.post("/api/documents/", json=doc_request)
            document_ids.append(create_response.json()["document_id"])

        try:
            # Test filter by document type
            response = client.get(
                "/api/documents/search",
                params={"document_type": "worksheet", "search_text": "filter test"},
            )

            assert response.status_code == 200
            data = response.json()
            worksheet_docs = [
                doc for doc in data["documents"] if doc["document_type"] == "worksheet"
            ]
            assert len(worksheet_docs) >= 1

            # Test filter by grade level
            response = client.get(
                "/api/documents/search", params={"grade_level": 10, "search_text": "filter test"}
            )

            assert response.status_code == 200
            data = response.json()
            grade_10_docs = [doc for doc in data["documents"] if doc["grade_level"] == 10]
            assert len(grade_10_docs) >= 1

            # Test text search
            response = client.get(
                "/api/documents/search", params={"search_text": "filter test notes"}
            )

            assert response.status_code == 200
            data = response.json()
            notes_docs = [doc for doc in data["documents"] if "notes" in doc["title"].lower()]
            assert len(notes_docs) >= 1

        finally:
            # Cleanup
            for doc_id in document_ids:
                client.delete(f"/api/documents/{doc_id}")

    def test_search_documents_endpoint_pagination(self, client, sample_document_request):
        """Test document search pagination."""
        # Create many documents for pagination testing
        document_ids = []
        for i in range(15):
            doc_request = sample_document_request.copy()
            doc_request["title"] = f"Pagination Test Document {i:02d}"
            doc_request["topic"] = "pagination_testing"

            create_response = client.post("/api/documents/", json=doc_request)
            document_ids.append(create_response.json()["document_id"])

        try:
            # Test first page
            response = client.get(
                "/api/documents/search",
                params={"topic": "pagination_testing", "limit": 5, "offset": 0},
            )

            assert response.status_code == 200
            page1_data = response.json()
            assert len(page1_data["documents"]) == 5
            assert page1_data["total_count"] >= 15
            assert page1_data["has_more"] is True

            # Test second page
            response = client.get(
                "/api/documents/search",
                params={"topic": "pagination_testing", "limit": 5, "offset": 5},
            )

            assert response.status_code == 200
            page2_data = response.json()
            assert len(page2_data["documents"]) == 5
            assert page2_data["has_more"] is True

            # Verify no overlap between pages
            page1_ids = {doc["id"] for doc in page1_data["documents"]}
            page2_ids = {doc["id"] for doc in page2_data["documents"]}
            assert len(page1_ids & page2_ids) == 0

            # Test third page
            response = client.get(
                "/api/documents/search",
                params={"topic": "pagination_testing", "limit": 5, "offset": 10},
            )

            assert response.status_code == 200
            page3_data = response.json()
            assert len(page3_data["documents"]) >= 5

        finally:
            # Cleanup
            for doc_id in document_ids:
                client.delete(f"/api/documents/{doc_id}")

    def test_document_files_endpoint(self, client, sample_document_request):
        """Test document files listing endpoint."""
        # Create a document with files
        create_response = client.post("/api/documents/", json=sample_document_request)
        document_id = create_response.json()["document_id"]

        try:
            # Get document files
            response = client.get(f"/api/documents/{document_id}/files")

            assert response.status_code == 200

            data = response.json()
            assert "files" in data
            assert isinstance(data["files"], list)

            # If files exist, verify their structure
            for file_info in data["files"]:
                assert "id" in file_info
                assert "file_key" in file_info
                assert "file_format" in file_info
                assert "version" in file_info
                assert "file_size" in file_info
                assert "created_at" in file_info

        finally:
            # Cleanup
            client.delete(f"/api/documents/{document_id}")

    def test_document_export_endpoint(self, client, sample_document_request):
        """Test document export functionality via API."""
        # Create a document
        create_response = client.post("/api/documents/", json=sample_document_request)
        document_id = create_response.json()["document_id"]

        try:
            # Request document export
            export_request = {
                "formats": ["pdf", "docx"],
                "create_dual_versions": True,
                "export_options": {"include_answers": True, "include_solutions": True},
            }

            response = client.post(f"/api/documents/{document_id}/export", json=export_request)

            assert response.status_code == 202  # Accepted for processing

            data = response.json()
            assert "export_id" in data
            assert "status" in data
            assert data["status"] in ["queued", "processing", "completed"]

            # If export completed immediately, check results
            if data["status"] == "completed":
                assert "files" in data
                assert len(data["files"]) == 4  # 2 formats × 2 versions

        finally:
            # Cleanup
            client.delete(f"/api/documents/{document_id}")

    def test_document_statistics_endpoint(self, client):
        """Test document statistics endpoint."""
        response = client.get("/api/documents/statistics")

        assert response.status_code == 200

        data = response.json()
        assert "total_documents" in data
        assert "total_file_size" in data
        assert "documents_by_type" in data
        assert "documents_by_status" in data

        # Verify data types
        assert isinstance(data["total_documents"], int)
        assert isinstance(data["total_file_size"], int)
        assert isinstance(data["documents_by_type"], dict)
        assert isinstance(data["documents_by_status"], dict)

    def test_document_bulk_operations_endpoint(self, client, sample_document_request):
        """Test bulk document operations."""
        # Create multiple documents
        document_ids = []
        for i in range(3):
            doc_request = sample_document_request.copy()
            doc_request["title"] = f"Bulk Test Document {i}"

            create_response = client.post("/api/documents/", json=doc_request)
            document_ids.append(create_response.json()["document_id"])

        try:
            # Test bulk status update
            bulk_update_request = {
                "document_ids": document_ids,
                "status": "archived",
                "metadata": {"bulk_operation": True},
            }

            response = client.patch("/api/documents/bulk-update", json=bulk_update_request)

            assert response.status_code == 200

            data = response.json()
            assert "updated_count" in data
            assert data["updated_count"] == 3
            assert "failed_updates" in data

            # Verify updates
            for doc_id in document_ids:
                get_response = client.get(f"/api/documents/{doc_id}")
                doc_data = get_response.json()
                assert doc_data["metadata"]["status"] == "archived"

        finally:
            # Cleanup
            for doc_id in document_ids:
                client.delete(f"/api/documents/{doc_id}")

    def test_document_download_endpoint(self, client, sample_document_request):
        """Test document file download endpoint."""
        # Create a document (assuming it creates files)
        create_response = client.post("/api/documents/", json=sample_document_request)
        document_id = create_response.json()["document_id"]

        try:
            # Get document files first
            files_response = client.get(f"/api/documents/{document_id}/files")

            if files_response.status_code == 200 and files_response.json()["files"]:
                files = files_response.json()["files"]
                file_id = files[0]["id"]

                # Test file download
                download_response = client.get(f"/api/documents/files/{file_id}/download")

                # Should either return file content or redirect/presigned URL
                assert download_response.status_code in [200, 302, 307]

                if download_response.status_code == 200:
                    # Direct file content
                    assert len(download_response.content) > 0

                    # Verify content type header
                    content_type = download_response.headers.get("content-type")
                    assert content_type is not None

        finally:
            # Cleanup
            client.delete(f"/api/documents/{document_id}")

    def test_api_error_handling(self, client):
        """Test API error handling for various scenarios."""
        # Test invalid document ID format
        response = client.get("/api/documents/invalid-uuid")
        assert response.status_code == 422  # Validation error

        # Test non-existent document update
        response = client.patch(f"/api/documents/{uuid4()}", json={"status": "updated"})
        assert response.status_code == 404

        # Test invalid search parameters
        response = client.get(
            "/api/documents/search",
            params={
                "limit": 1000,  # Too high
                "grade_level": 20,  # Invalid
            },
        )
        assert response.status_code == 422

        # Test malformed JSON
        response = client.post("/api/documents/", data="invalid json")
        assert response.status_code == 422

    def test_api_authentication_and_authorization(self, client):
        """Test API authentication and authorization (if implemented)."""
        # This test would verify authentication requirements
        # Implementation depends on the actual auth strategy

        # Example: Test without auth token
        headers = {}  # No authentication headers

        response = client.get("/api/documents/", headers=headers)

        # Adjust based on actual auth implementation
        # For now, assume endpoints are publicly accessible during testing
        assert response.status_code in [200, 401, 403]


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

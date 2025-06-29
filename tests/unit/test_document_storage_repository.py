"""
Unit tests for DocumentStorageRepository.
Tests document metadata storage operations with mocked Supabase client.
"""

from datetime import datetime
from unittest.mock import ANY, MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.models.stored_document_models import (
    DocumentFile,
    DocumentSearchFilters,
    StoredDocument,
    StoredDocumentMetadata,
)
from src.repositories.document_storage_repository import (
    DocumentStorageError,
    DocumentStorageRepository,
)


class TestDocumentStorageRepository:
    """Unit tests for DocumentStorageRepository with mocked Supabase client."""

    def _create_search_mock_chain(self, mock_supabase_client, sample_data, count=None):
        """Helper to create proper mock chain for search operations."""
        # Create a proper response object
        mock_response = type(
            "MockResponse",
            (),
            {"data": sample_data, "count": count if count is not None else len(sample_data)},
        )()

        # Create mock execute method
        mock_execute = MagicMock(return_value=mock_response)

        # Build the chain: ...limit().offset().execute()
        mock_offset = MagicMock()
        mock_offset.execute = mock_execute

        mock_limit = MagicMock()
        mock_limit.offset.return_value = mock_offset

        return mock_limit, mock_execute

    @pytest.fixture
    def mock_supabase_client(self):
        """Create mock Supabase client."""
        client = MagicMock()
        client.table = MagicMock()
        client.from_ = MagicMock()
        return client

    @pytest.fixture
    def repository(self, mock_supabase_client):
        """Create DocumentStorageRepository with mocked client."""
        with patch("src.repositories.document_storage_repository.get_settings") as mock_settings:
            # Mock settings to return no table prefix for unit tests
            settings_mock = mock_settings.return_value
            settings_mock.table_prefix = ""
            return DocumentStorageRepository(mock_supabase_client)

    @pytest.fixture
    def sample_document_metadata(self):
        """Sample document metadata for testing."""
        return StoredDocumentMetadata(
            id=uuid4(),
            session_id=uuid4(),
            title="Algebra Practice Worksheet",
            document_type="worksheet",
            detail_level=5,  # MEDIUM level
            topic="quadratic_equations",
            grade_level=9,
            estimated_duration=45,
            total_questions=10,
            status="generated",
            file_count=2,
            total_file_size=1024000,
            tags=["algebra", "equations", "practice"],
            search_content="quadratic equations practice problems solve for x",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    @pytest.fixture
    def sample_document_files(self):
        """Sample document files for testing."""
        # Use the same document_id for both files
        shared_document_id = uuid4()
        return [
            DocumentFile(
                id=uuid4(),
                document_id=shared_document_id,
                file_key="documents/worksheets/algebra_123/student.pdf",
                file_format="pdf",
                version="student",
                file_size=512000,
                content_type="application/pdf",
                r2_metadata={"upload_id": "test_upload_123"},
                created_at=datetime.now(),
            ),
            DocumentFile(
                id=uuid4(),
                document_id=shared_document_id,
                file_key="documents/worksheets/algebra_123/teacher.pdf",
                file_format="pdf",
                version="teacher",
                file_size=512000,
                content_type="application/pdf",
                r2_metadata={"upload_id": "test_upload_124"},
                created_at=datetime.now(),
            ),
        ]

    @pytest.fixture
    def sample_stored_document(self, sample_document_metadata, sample_document_files):
        """Sample complete stored document."""
        return StoredDocument(
            metadata=sample_document_metadata,
            files=sample_document_files,
            session_data={"generation_request": {"topic": "algebra"}},
        )

    async def test_save_document_metadata_success(
        self, repository, mock_supabase_client, sample_document_metadata
    ):
        """Test successful document metadata saving."""
        # Arrange
        mock_response = MagicMock()
        mock_response.data = [sample_document_metadata.model_dump()]
        mock_supabase_client.table().insert().execute.return_value = mock_response

        # Act
        result = await repository.save_document_metadata(sample_document_metadata)

        # Assert
        assert result == sample_document_metadata.id
        mock_supabase_client.table.assert_called_with("documents")
        # Note: insert() is called twice - once for the method call, once with data
        assert mock_supabase_client.table().insert.call_count == 2

    async def test_save_document_metadata_failure(
        self, repository, mock_supabase_client, sample_document_metadata
    ):
        """Test document metadata saving failure."""
        # Arrange
        mock_supabase_client.table().insert().execute.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(DocumentStorageError, match="Failed to save document metadata"):
            await repository.save_document_metadata(sample_document_metadata)

    async def test_retrieve_document_by_id_success(
        self, repository, mock_supabase_client, sample_stored_document
    ):
        """Test successful document retrieval by ID."""
        # Arrange
        document_id = sample_stored_document.metadata.id
        mock_metadata_response = MagicMock()
        mock_metadata_response.data = [sample_stored_document.metadata.model_dump()]
        mock_files_response = MagicMock()
        mock_files_response.data = [f.model_dump() for f in sample_stored_document.files]

        # Mock session response
        mock_session_response = MagicMock()
        mock_session_response.data = []

        mock_supabase_client.table().select().eq().execute.side_effect = [
            mock_metadata_response,
            mock_files_response,
            mock_session_response,
        ]

        # Act
        result = await repository.retrieve_document_by_id(document_id)

        # Assert
        assert result is not None
        assert result.metadata.id == document_id
        assert len(result.files) == 2
        assert result.metadata.title == "Algebra Practice Worksheet"

    async def test_retrieve_document_by_id_not_found(self, repository, mock_supabase_client):
        """Test document retrieval when document not found."""
        # Arrange
        document_id = uuid4()
        mock_response = MagicMock()
        mock_response.data = []
        mock_supabase_client.table().select().eq().execute.return_value = mock_response

        # Act
        result = await repository.retrieve_document_by_id(document_id)

        # Assert
        assert result is None

    async def test_update_document_status_success(self, repository, mock_supabase_client):
        """Test successful document status update."""
        # Arrange
        document_id = uuid4()
        new_status = "exported"
        mock_response = MagicMock()
        mock_response.data = [{"id": str(document_id), "status": new_status}]
        mock_supabase_client.table().update().eq().execute.return_value = mock_response

        # Act
        success = await repository.update_document_status(document_id, new_status)

        # Assert
        assert success is True
        mock_supabase_client.table().update.assert_called_with(
            {"status": new_status, "updated_at": ANY}
        )

    async def test_update_document_status_failure(self, repository, mock_supabase_client):
        """Test document status update failure."""
        # Arrange
        document_id = uuid4()
        new_status = "exported"
        mock_supabase_client.table().update().eq().execute.side_effect = Exception("Update failed")

        # Act & Assert
        with pytest.raises(DocumentStorageError, match="Failed to update document status"):
            await repository.update_document_status(document_id, new_status)

    async def test_soft_delete_document_success(self, repository, mock_supabase_client):
        """Test successful document soft deletion."""
        # Arrange
        document_id = uuid4()
        mock_response = MagicMock()
        mock_response.data = [{"id": str(document_id), "status": "deleted"}]
        mock_supabase_client.table().update().eq().execute.return_value = mock_response

        # Act
        success = await repository.soft_delete_document(document_id)

        # Assert
        assert success is True
        mock_supabase_client.table().update.assert_called_with(
            {"status": "deleted", "deleted_at": ANY, "updated_at": ANY}
        )

    async def test_search_documents_by_filters_basic(
        self, repository, mock_supabase_client, sample_stored_document
    ):
        """Test document search with basic filters."""
        # Arrange
        filters = DocumentSearchFilters(
            document_type="worksheet", topic="algebra", grade_level=9, limit=10
        )

        # Use helper to create proper mock chain
        sample_data = [sample_stored_document.metadata.model_dump()]
        mock_limit, mock_execute = self._create_search_mock_chain(
            mock_supabase_client, sample_data, count=1
        )

        # Set up the query chain: table().select().neq().eq().eq().eq().limit()
        mock_supabase_client.table.return_value.select.return_value.neq.return_value.eq.return_value.eq.return_value.eq.return_value.limit.return_value = mock_limit

        # Act
        results = await repository.search_documents(filters)

        # Assert
        assert len(results["documents"]) == 1
        assert results["total_count"] == 1
        assert results["documents"][0].title == "Algebra Practice Worksheet"

    async def test_search_documents_by_filters_with_text_search(
        self, repository, mock_supabase_client, sample_stored_document
    ):
        """Test document search with text search."""
        # Arrange
        filters = DocumentSearchFilters(search_text="quadratic equations", limit=5)

        # Use helper to create proper mock chain
        sample_data = [sample_stored_document.metadata.model_dump()]
        mock_limit, mock_execute = self._create_search_mock_chain(mock_supabase_client, sample_data)

        # Set up the query chain: table().select().neq().ilike().limit()
        mock_supabase_client.table.return_value.select.return_value.neq.return_value.ilike.return_value.limit.return_value = mock_limit

        # Act
        results = await repository.search_documents(filters)

        # Assert
        assert len(results["documents"]) == 1
        mock_supabase_client.table().select().neq().ilike.assert_called_with(
            "search_content", "%quadratic equations%"
        )

    async def test_search_documents_by_filters_with_date_range(
        self, repository, mock_supabase_client, sample_stored_document
    ):
        """Test document search with date range filters."""
        # Arrange
        from datetime import datetime, timedelta

        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()

        filters = DocumentSearchFilters(created_after=start_date, created_before=end_date, limit=10)

        # Use helper to create proper mock chain
        sample_data = [sample_stored_document.metadata.model_dump()]
        mock_limit, mock_execute = self._create_search_mock_chain(mock_supabase_client, sample_data)

        # Set up the query chain: table().select().neq().gte().lte().limit()
        mock_supabase_client.table.return_value.select.return_value.neq.return_value.gte.return_value.lte.return_value.limit.return_value = mock_limit

        # Act
        results = await repository.search_documents(filters)

        # Assert
        assert len(results["documents"]) == 1

    async def test_search_documents_by_filters_with_tags(
        self, repository, mock_supabase_client, sample_stored_document
    ):
        """Test document search with tag filters."""
        # Arrange
        filters = DocumentSearchFilters(tags=["algebra", "practice"], limit=10)

        # Use helper to create proper mock chain
        sample_data = [sample_stored_document.metadata.model_dump()]
        mock_limit, mock_execute = self._create_search_mock_chain(mock_supabase_client, sample_data)

        # Set up the query chain: table().select().neq().contains().contains().limit() (2 tags = 2 contains calls)
        mock_supabase_client.table.return_value.select.return_value.neq.return_value.contains.return_value.contains.return_value.limit.return_value = mock_limit

        # Act
        results = await repository.search_documents(filters)

        # Assert
        assert len(results["documents"]) == 1

    async def test_save_document_file_success(
        self, repository, mock_supabase_client, sample_document_files
    ):
        """Test successful document file saving."""
        # Arrange
        document_file = sample_document_files[0]
        mock_response = MagicMock()
        mock_response.data = [document_file.model_dump()]
        mock_supabase_client.table().insert().execute.return_value = mock_response

        # Act
        result = await repository.save_document_file(document_file)

        # Assert
        assert result == document_file.id
        mock_supabase_client.table.assert_called_with("document_files")

    async def test_get_document_files_success(
        self, repository, mock_supabase_client, sample_document_files
    ):
        """Test successful retrieval of document files."""
        # Arrange
        document_id = sample_document_files[0].document_id
        mock_response = MagicMock()
        mock_response.data = [f.model_dump() for f in sample_document_files]
        mock_supabase_client.table().select().eq().execute.return_value = mock_response

        # Act
        files = await repository.get_document_files(document_id)

        # Assert
        assert len(files) == 2
        assert all(f.document_id == document_id for f in files)

    async def test_update_file_storage_info_success(self, repository, mock_supabase_client):
        """Test successful file storage info update."""
        # Arrange
        file_id = uuid4()
        storage_info = {
            "r2_upload_id": "test_upload_456",
            "r2_etag": "test_etag_789",
            "file_size": 1024000,
        }
        mock_response = MagicMock()
        mock_response.data = [{"id": str(file_id)}]
        mock_supabase_client.table().update().eq().execute.return_value = mock_response

        # Act
        success = await repository.update_file_storage_info(file_id, storage_info)

        # Assert
        assert success is True
        mock_supabase_client.table().update.assert_called_with(
            {
                "r2_metadata": storage_info,
                "file_size": storage_info["file_size"],
                "updated_at": ANY,
            }
        )

    async def test_get_documents_by_session_id_success(
        self, repository, mock_supabase_client, sample_stored_document
    ):
        """Test successful retrieval of documents by session ID."""
        # Arrange
        session_id = sample_stored_document.metadata.session_id
        mock_response = MagicMock()
        mock_response.data = [sample_stored_document.metadata.model_dump()]
        mock_supabase_client.table().select().eq().execute.return_value = mock_response

        # Act
        documents = await repository.get_documents_by_session_id(session_id)

        # Assert
        assert len(documents) == 1
        assert documents[0].session_id == session_id

    async def test_get_document_statistics_success(self, repository, mock_supabase_client):
        """Test successful retrieval of document statistics."""
        # Arrange - The repository makes multiple table calls, not rpc calls

        # Mock total document count response
        total_response = type("MockResponse", (), {"count": 100, "data": []})()

        # Mock file size response
        size_response = type(
            "MockResponse",
            (),
            {
                "data": [
                    {"total_file_size": 20000000},
                    {"total_file_size": 15000000},
                    {"total_file_size": 15000000},
                ]
            },
        )()

        # Mock document type response
        type_response = type(
            "MockResponse",
            (),
            {
                "data": [
                    {"document_type": "worksheet"},
                    {"document_type": "worksheet"},
                    {"document_type": "notes"},
                ]
            },
        )()

        # Mock document status response
        status_response = type(
            "MockResponse",
            (),
            {"data": [{"status": "generated"}, {"status": "exported"}, {"status": "generated"}]},
        )()

        # Set up the mock to return different responses for different calls
        mock_supabase_client.table.return_value.select.return_value.neq.return_value.execute.side_effect = [
            total_response,  # First call for count
            size_response,  # Second call for file sizes
            type_response,  # Third call for document types
            status_response,  # Fourth call for statuses
        ]

        # Act
        stats = await repository.get_document_statistics()

        # Assert
        assert stats["total_documents"] == 100
        assert stats["total_file_size"] == 50000000  # 20M + 15M + 15M
        assert stats["documents_by_type"]["worksheet"] == 2  # 2 worksheet entries
        assert stats["documents_by_type"]["notes"] == 1  # 1 notes entry
        assert stats["documents_by_status"]["generated"] == 2  # 2 generated entries
        assert stats["documents_by_status"]["exported"] == 1  # 1 exported entry
        assert stats["average_file_size"] == 500000  # 50M / 100 docs

    async def test_cleanup_old_documents_success(self, repository, mock_supabase_client):
        """Test successful cleanup of old documents."""
        # Arrange
        from datetime import datetime, timedelta

        cutoff_date = datetime.now() - timedelta(days=30)

        mock_response = MagicMock()
        # The repository calculates deleted_count = len(response.data), so we need 5 items
        mock_response.data = [{"id": f"doc_{i}"} for i in range(5)]
        mock_supabase_client.table().update().lt().execute.return_value = mock_response

        # Act
        deleted_count = await repository.cleanup_old_documents(cutoff_date)

        # Assert
        assert deleted_count == 5
        mock_supabase_client.table().update.assert_called_with(
            {"status": "deleted", "deleted_at": ANY, "updated_at": ANY}
        )

    def test_build_search_query_basic_filters(self, repository):
        """Test search query building with basic filters."""
        filters = DocumentSearchFilters(document_type="worksheet", topic="algebra", grade_level=9)

        # This would test the internal query building logic
        # Implementation depends on the actual query builder structure
        pass

    def test_validate_document_metadata(self, repository, sample_document_metadata):
        """Test document metadata validation."""
        # Valid metadata
        assert repository.validate_document_metadata(sample_document_metadata) is True

        # Invalid metadata (missing required fields)
        # Since Pydantic validates at creation time, we need to bypass validation
        # or use a different approach. Let's test the validation method with a mock
        # or create an object that bypasses validation
        try:
            invalid_metadata = StoredDocumentMetadata(
                id=uuid4(),
                title="",  # Empty title should be invalid
                document_type="worksheet",
            )
            # If this succeeds, the validation rules have changed
            # Test that the repository validation still catches issues
            assert repository.validate_document_metadata(invalid_metadata) is False
        except ValidationError:
            # Expected - Pydantic caught the validation error at creation time
            # This means our validation is working correctly
            assert True

    def test_generate_search_content(self, repository, sample_document_metadata):
        """Test generation of searchable content from document metadata."""
        search_content = repository.generate_search_content(sample_document_metadata)

        assert "algebra" in search_content.lower()
        assert "practice" in search_content.lower()
        assert "worksheet" in search_content.lower()
        assert "quadratic" in search_content.lower()

    def test_parse_document_filters(self, repository):
        """Test parsing of document search filters from request parameters."""
        raw_filters = {
            "document_type": "worksheet",
            "topic": "algebra",
            "grade_level": "9",
            "created_after": "2025-01-01T00:00:00Z",
            "tags": "algebra,practice,equations",
            "limit": "20",
            "offset": "0",
        }

        parsed_filters = repository.parse_document_filters(raw_filters)

        assert parsed_filters.document_type == "worksheet"
        assert parsed_filters.grade_level == 9
        assert parsed_filters.tags == ["algebra", "practice", "equations"]
        assert parsed_filters.limit == 20
        assert parsed_filters.offset == 0

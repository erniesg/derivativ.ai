"""
Unit tests for stored document models.
Tests Pydantic models for document storage and retrieval.
"""

from datetime import datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from src.models.stored_document_models import (
    DocumentFile,
    DocumentSearchFilters,
    DocumentSearchResults,
    DocumentStatistics,
    StoredDocument,
    StoredDocumentMetadata,
)


class TestStoredDocumentModels:
    """Unit tests for stored document Pydantic models."""

    def test_stored_document_metadata_creation(self):
        """Test creation of StoredDocumentMetadata with valid data."""
        metadata = StoredDocumentMetadata(
            id=uuid4(),
            session_id=uuid4(),
            title="Test Worksheet",
            document_type="worksheet",
            detail_level="medium",
            topic="algebra",
            grade_level=9,
            estimated_duration=30,
            total_questions=5,
            status="generated",
            file_count=2,
            total_file_size=1024000,
            tags=["algebra", "practice"],
            search_content="algebra practice problems",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        assert isinstance(metadata.id, UUID)
        assert isinstance(metadata.session_id, UUID)
        assert metadata.title == "Test Worksheet"
        assert metadata.document_type == "worksheet"
        assert metadata.grade_level == 9
        assert metadata.file_count == 2
        assert "algebra" in metadata.tags

    def test_stored_document_metadata_validation_required_fields(self):
        """Test validation of required fields in StoredDocumentMetadata."""
        # Missing required fields should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            StoredDocumentMetadata()

        errors = exc_info.value.errors()
        required_fields = {error["loc"][0] for error in errors if error["type"] == "missing"}

        # Note: id has default_factory so it's not required
        assert "title" in required_fields
        assert "document_type" in required_fields

    def test_stored_document_metadata_validation_document_type(self):
        """Test validation of document_type enum."""
        # Valid document types
        valid_types = ["worksheet", "notes", "textbook", "slides"]

        for doc_type in valid_types:
            metadata = StoredDocumentMetadata(
                id=uuid4(), title="Test Document", document_type=doc_type
            )
            assert metadata.document_type == doc_type

        # Invalid document type should raise ValidationError
        with pytest.raises(ValidationError):
            StoredDocumentMetadata(id=uuid4(), title="Test Document", document_type="invalid_type")

    def test_stored_document_metadata_validation_status(self):
        """Test validation of status enum."""
        valid_statuses = [
            "pending",
            "generating",
            "generated",
            "exporting",
            "exported",
            "failed",
            "deleted",
        ]

        for status in valid_statuses:
            metadata = StoredDocumentMetadata(
                id=uuid4(), title="Test Document", document_type="worksheet", status=status
            )
            assert metadata.status == status

    def test_stored_document_metadata_validation_grade_level(self):
        """Test validation of grade_level range."""
        # Valid grade levels (1-12)
        for grade in range(1, 13):
            metadata = StoredDocumentMetadata(
                id=uuid4(), title="Test Document", document_type="worksheet", grade_level=grade
            )
            assert metadata.grade_level == grade

        # Invalid grade levels should raise ValidationError
        with pytest.raises(ValidationError):
            StoredDocumentMetadata(
                id=uuid4(),
                title="Test Document",
                document_type="worksheet",
                grade_level=0,  # Too low
            )

        with pytest.raises(ValidationError):
            StoredDocumentMetadata(
                id=uuid4(),
                title="Test Document",
                document_type="worksheet",
                grade_level=13,  # Too high
            )

    def test_stored_document_metadata_defaults(self):
        """Test default values in StoredDocumentMetadata."""
        metadata = StoredDocumentMetadata(
            id=uuid4(), title="Test Document", document_type="worksheet"
        )

        assert metadata.status == "pending"
        assert metadata.file_count == 0
        assert metadata.total_file_size == 0
        assert metadata.tags == []
        assert metadata.search_content == ""
        assert isinstance(metadata.created_at, datetime)
        assert isinstance(metadata.updated_at, datetime)

    def test_document_file_creation(self):
        """Test creation of DocumentFile with valid data."""
        file = DocumentFile(
            id=uuid4(),
            document_id=uuid4(),
            file_key="documents/worksheets/test_123/student.pdf",
            file_format="pdf",
            version="student",
            file_size=512000,
            content_type="application/pdf",
            r2_metadata={"upload_id": "test_upload_123", "etag": "test_etag"},
            created_at=datetime.now(),
        )

        assert isinstance(file.id, UUID)
        assert isinstance(file.document_id, UUID)
        assert file.file_key.endswith(".pdf")
        assert file.file_format == "pdf"
        assert file.version == "student"
        assert file.file_size == 512000
        assert file.r2_metadata["upload_id"] == "test_upload_123"

    def test_document_file_validation_file_format(self):
        """Test validation of file_format enum."""
        valid_formats = ["pdf", "docx", "html", "txt", "json"]

        for file_format in valid_formats:
            file = DocumentFile(
                id=uuid4(),
                document_id=uuid4(),
                file_key=f"test.{file_format}",
                file_format=file_format,
                version="student",
            )
            assert file.file_format == file_format

    def test_document_file_validation_version(self):
        """Test validation of version enum."""
        valid_versions = ["student", "teacher", "combined"]

        for version in valid_versions:
            file = DocumentFile(
                id=uuid4(),
                document_id=uuid4(),
                file_key="test.pdf",
                file_format="pdf",
                version=version,
            )
            assert file.version == version

    def test_document_file_defaults(self):
        """Test default values in DocumentFile."""
        file = DocumentFile(
            id=uuid4(),
            document_id=uuid4(),
            file_key="test.pdf",
            file_format="pdf",
            version="student",
        )

        assert file.file_size == 0
        assert file.content_type == ""
        assert file.r2_metadata == {}
        assert isinstance(file.created_at, datetime)

    def test_stored_document_creation(self):
        """Test creation of complete StoredDocument."""
        metadata = StoredDocumentMetadata(
            id=uuid4(), title="Test Document", document_type="worksheet"
        )

        files = [
            DocumentFile(
                id=uuid4(),
                document_id=metadata.id,
                file_key="test_student.pdf",
                file_format="pdf",
                version="student",
            ),
            DocumentFile(
                id=uuid4(),
                document_id=metadata.id,
                file_key="test_teacher.pdf",
                file_format="pdf",
                version="teacher",
            ),
        ]

        session_data = {"generation_request": {"topic": "algebra"}}

        document = StoredDocument(metadata=metadata, files=files, session_data=session_data)

        assert document.metadata.id == metadata.id
        assert len(document.files) == 2
        assert document.session_data["generation_request"]["topic"] == "algebra"

    def test_stored_document_defaults(self):
        """Test default values in StoredDocument."""
        metadata = StoredDocumentMetadata(
            id=uuid4(), title="Test Document", document_type="worksheet"
        )

        document = StoredDocument(metadata=metadata)

        assert document.files == []
        assert document.session_data == {}

    def test_document_search_filters_creation(self):
        """Test creation of DocumentSearchFilters."""
        filters = DocumentSearchFilters(
            document_type="worksheet",
            topic="algebra",
            grade_level=9,
            status="generated",
            search_text="quadratic equations",
            tags=["algebra", "practice"],
            created_after=datetime(2025, 1, 1),
            created_before=datetime(2025, 12, 31),
            limit=20,
            offset=0,
        )

        assert filters.document_type == "worksheet"
        assert filters.topic == "algebra"
        assert filters.grade_level == 9
        assert filters.search_text == "quadratic equations"
        assert len(filters.tags) == 2
        assert filters.limit == 20

    def test_document_search_filters_defaults(self):
        """Test default values in DocumentSearchFilters."""
        filters = DocumentSearchFilters()

        assert filters.document_type is None
        assert filters.topic is None
        assert filters.grade_level is None
        assert filters.status is None
        assert filters.search_text == ""
        assert filters.tags == []
        assert filters.created_after is None
        assert filters.created_before is None
        assert filters.limit == 50
        assert filters.offset == 0

    def test_document_search_filters_validation_limit(self):
        """Test validation of limit in DocumentSearchFilters."""
        # Valid limits
        for limit in [1, 50, 100]:
            filters = DocumentSearchFilters(limit=limit)
            assert filters.limit == limit

        # Invalid limits should raise ValidationError
        with pytest.raises(ValidationError):
            DocumentSearchFilters(limit=0)  # Too low

        with pytest.raises(ValidationError):
            DocumentSearchFilters(limit=101)  # Too high

    def test_document_search_results_creation(self):
        """Test creation of DocumentSearchResults."""
        documents = [
            StoredDocumentMetadata(id=uuid4(), title="Test Document 1", document_type="worksheet"),
            StoredDocumentMetadata(id=uuid4(), title="Test Document 2", document_type="notes"),
        ]

        results = DocumentSearchResults(
            documents=documents, total_count=2, offset=0, limit=50, has_more=False
        )

        assert len(results.documents) == 2
        assert results.total_count == 2
        assert results.has_more is False

    def test_document_search_results_has_more_calculation(self):
        """Test has_more calculation in DocumentSearchResults."""
        documents = [
            StoredDocumentMetadata(
                id=uuid4(), title=f"Test Document {i}", document_type="worksheet"
            )
            for i in range(10)
        ]

        # Case 1: No more results
        results = DocumentSearchResults(documents=documents, total_count=10, offset=0, limit=50)
        assert results.has_more is False

        # Case 2: More results available
        results = DocumentSearchResults(documents=documents, total_count=100, offset=0, limit=10)
        assert results.has_more is True

    def test_document_statistics_creation(self):
        """Test creation of DocumentStatistics."""
        stats = DocumentStatistics(
            total_documents=100,
            total_file_size=50000000,
            documents_by_type={"worksheet": 60, "notes": 30, "textbook": 10},
            documents_by_status={"generated": 80, "exported": 15, "failed": 5},
            average_file_size=500000,
            largest_document_size=2000000,
            most_popular_topics=["algebra", "geometry", "calculus"],
            generation_success_rate=0.95,
            storage_usage_percentage=0.75,
        )

        assert stats.total_documents == 100
        assert stats.documents_by_type["worksheet"] == 60
        assert stats.documents_by_status["generated"] == 80
        assert stats.generation_success_rate == 0.95
        assert "algebra" in stats.most_popular_topics

    def test_document_statistics_defaults(self):
        """Test default values in DocumentStatistics."""
        stats = DocumentStatistics()

        assert stats.total_documents == 0
        assert stats.total_file_size == 0
        assert stats.documents_by_type == {}
        assert stats.documents_by_status == {}
        assert stats.average_file_size == 0
        assert stats.largest_document_size == 0
        assert stats.most_popular_topics == []
        assert stats.generation_success_rate == 0.0
        assert stats.storage_usage_percentage == 0.0

    def test_model_serialization(self):
        """Test JSON serialization of all models."""
        # Test StoredDocumentMetadata serialization
        metadata = StoredDocumentMetadata(
            id=uuid4(), title="Test Document", document_type="worksheet", created_at=datetime.now()
        )

        metadata_dict = metadata.model_dump()
        assert "id" in metadata_dict
        assert "title" in metadata_dict
        assert "created_at" in metadata_dict

        # Test DocumentFile serialization
        file = DocumentFile(
            id=uuid4(),
            document_id=uuid4(),
            file_key="test.pdf",
            file_format="pdf",
            version="student",
        )

        file_dict = file.model_dump()
        assert "id" in file_dict
        assert "file_key" in file_dict
        assert "r2_metadata" in file_dict

    def test_model_deserialization(self):
        """Test JSON deserialization of all models."""
        # Test StoredDocumentMetadata deserialization
        metadata_data = {
            "id": str(uuid4()),
            "title": "Test Document",
            "document_type": "worksheet",
            "detail_level": "medium",
            "status": "generated",
            "file_count": 1,
            "total_file_size": 1024,
            "tags": ["test"],
            "search_content": "test content",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        metadata = StoredDocumentMetadata(**metadata_data)
        assert str(metadata.id) == metadata_data["id"]
        assert metadata.title == "Test Document"

    def test_uuid_string_conversion(self):
        """Test UUID string conversion in models."""
        doc_id = uuid4()

        # Create with UUID
        metadata = StoredDocumentMetadata(
            id=doc_id, title="Test Document", document_type="worksheet"
        )
        assert metadata.id == doc_id

        # Create with string UUID
        metadata_from_string = StoredDocumentMetadata(
            id=str(doc_id), title="Test Document", document_type="worksheet"
        )
        assert metadata_from_string.id == doc_id

    def test_tag_normalization(self):
        """Test tag normalization and validation."""
        # Test with mixed case tags
        metadata = StoredDocumentMetadata(
            id=uuid4(),
            title="Test Document",
            document_type="worksheet",
            tags=["Algebra", "PRACTICE", "equations"],
        )

        # Tags should be normalized to lowercase
        expected_tags = ["algebra", "practice", "equations"]
        assert metadata.tags == expected_tags

    def test_search_content_generation(self):
        """Test automatic search content generation."""
        metadata = StoredDocumentMetadata(
            id=uuid4(),
            title="Quadratic Equations Practice",
            document_type="worksheet",
            topic="algebra",
            tags=["equations", "practice"],
        )

        # Search content should include title, topic, and tags
        search_content = metadata.generate_search_content()
        assert "quadratic" in search_content.lower()
        assert "equations" in search_content.lower()
        assert "practice" in search_content.lower()
        assert "algebra" in search_content.lower()

    def test_file_key_validation(self):
        """Test file key format validation."""
        valid_keys = [
            "documents/worksheets/123/student.pdf",
            "documents/notes/abc/teacher.docx",
            "exports/2025/01/combined.html",
        ]

        for key in valid_keys:
            file = DocumentFile(
                id=uuid4(), document_id=uuid4(), file_key=key, file_format="pdf", version="student"
            )
            assert file.file_key == key

        # Invalid keys should be rejected
        invalid_keys = [
            "",  # Empty
            "../../../etc/passwd",  # Path traversal
            "documents//double/slash.pdf",  # Double slash
            "documents with spaces/file.pdf",  # Spaces
        ]

        for key in invalid_keys:
            with pytest.raises(ValidationError):
                DocumentFile(
                    id=uuid4(),
                    document_id=uuid4(),
                    file_key=key,
                    file_format="pdf",
                    version="student",
                )

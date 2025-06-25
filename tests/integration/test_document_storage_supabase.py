"""
Integration tests for document storage with Supabase.
Tests real database operations for document metadata and file references.
"""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from src.models.stored_document_models import (
    DocumentFile,
    DocumentSearchFilters,
    StoredDocumentMetadata,
)
from src.repositories.document_storage_repository import DocumentStorageRepository


class TestDocumentStorageSupabaseIntegration:
    """Integration tests for document storage with real Supabase database."""

    @pytest.fixture
    async def repository(self, supabase_client):
        """Create DocumentStorageRepository with real Supabase client."""
        return DocumentStorageRepository(supabase_client)

    @pytest.fixture
    def sample_document_metadata(self):
        """Create sample document metadata for testing."""
        return StoredDocumentMetadata(
            id=uuid4(),
            session_id=uuid4(),
            title="Integration Test Worksheet",
            document_type="worksheet",
            detail_level="medium",
            topic="integration_testing",
            grade_level=10,
            estimated_duration=45,
            total_questions=8,
            status="generated",
            file_count=2,
            total_file_size=1500000,
            tags=["integration", "testing", "algebra"],
            search_content="integration testing algebra worksheet practice problems",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    @pytest.fixture
    def sample_document_files(self, sample_document_metadata):
        """Create sample document files for testing."""
        return [
            DocumentFile(
                id=uuid4(),
                document_id=sample_document_metadata.id,
                file_key="test_documents/integration/student.pdf",
                file_format="pdf",
                version="student",
                file_size=750000,
                content_type="application/pdf",
                r2_metadata={"upload_id": "test_integration_student"},
                created_at=datetime.now(),
            ),
            DocumentFile(
                id=uuid4(),
                document_id=sample_document_metadata.id,
                file_key="test_documents/integration/teacher.pdf",
                file_format="pdf",
                version="teacher",
                file_size=750000,
                content_type="application/pdf",
                r2_metadata={"upload_id": "test_integration_teacher"},
                created_at=datetime.now(),
            ),
        ]

    @pytest.mark.asyncio
    async def test_save_and_retrieve_document_metadata(self, repository, sample_document_metadata):
        """Test saving and retrieving document metadata from Supabase."""
        try:
            # Save document metadata
            saved_id = await repository.save_document_metadata(sample_document_metadata)
            assert saved_id == sample_document_metadata.id

            # Retrieve document metadata
            retrieved_document = await repository.retrieve_document_by_id(saved_id)
            assert retrieved_document is not None
            assert retrieved_document.metadata.id == sample_document_metadata.id
            assert retrieved_document.metadata.title == sample_document_metadata.title
            assert (
                retrieved_document.metadata.document_type == sample_document_metadata.document_type
            )
            assert retrieved_document.metadata.topic == sample_document_metadata.topic

        finally:
            # Cleanup
            try:
                await repository.soft_delete_document(sample_document_metadata.id)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_save_and_retrieve_document_files(
        self, repository, sample_document_metadata, sample_document_files
    ):
        """Test saving and retrieving document files from Supabase."""
        try:
            # Save document metadata first
            await repository.save_document_metadata(sample_document_metadata)

            # Save document files
            file_ids = []
            for file in sample_document_files:
                file_id = await repository.save_document_file(file)
                file_ids.append(file_id)
                assert file_id == file.id

            # Retrieve document files
            retrieved_files = await repository.get_document_files(sample_document_metadata.id)
            assert len(retrieved_files) == 2

            # Verify file data
            retrieved_file_keys = {f.file_key for f in retrieved_files}
            expected_file_keys = {f.file_key for f in sample_document_files}
            assert retrieved_file_keys == expected_file_keys

            # Verify file metadata
            for retrieved_file in retrieved_files:
                assert retrieved_file.document_id == sample_document_metadata.id
                assert retrieved_file.file_format == "pdf"
                assert retrieved_file.file_size == 750000

        finally:
            # Cleanup
            try:
                await repository.soft_delete_document(sample_document_metadata.id)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_document_status_updates(self, repository, sample_document_metadata):
        """Test document status updates in Supabase."""
        try:
            # Save initial document
            await repository.save_document_metadata(sample_document_metadata)

            # Update status to exporting
            success = await repository.update_document_status(
                sample_document_metadata.id, "exporting"
            )
            assert success is True

            # Verify status update
            retrieved_document = await repository.retrieve_document_by_id(
                sample_document_metadata.id
            )
            assert retrieved_document.metadata.status == "exporting"

            # Update status to exported
            success = await repository.update_document_status(
                sample_document_metadata.id, "exported"
            )
            assert success is True

            # Verify final status
            retrieved_document = await repository.retrieve_document_by_id(
                sample_document_metadata.id
            )
            assert retrieved_document.metadata.status == "exported"

        finally:
            # Cleanup
            try:
                await repository.soft_delete_document(sample_document_metadata.id)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_document_search_functionality(self, repository):
        """Test document search functionality with Supabase."""
        # Create multiple test documents
        test_documents = []
        for i in range(5):
            doc = StoredDocumentMetadata(
                id=uuid4(),
                session_id=uuid4(),
                title=f"Search Test Document {i}",
                document_type="worksheet" if i % 2 == 0 else "notes",
                topic="search_testing",
                grade_level=9 + (i % 3),  # Mix of grades 9, 10, 11
                status="generated",
                tags=["search", "testing", f"doc_{i}"],
                search_content=f"search testing document {i} algebra geometry",
                created_at=datetime.now() - timedelta(days=i),
                updated_at=datetime.now(),
            )
            test_documents.append(doc)

        try:
            # Save all test documents
            for doc in test_documents:
                await repository.save_document_metadata(doc)

            # Test search by document type
            worksheet_filters = DocumentSearchFilters(document_type="worksheet", limit=10)
            worksheet_results = await repository.search_documents(worksheet_filters)
            assert len(worksheet_results["documents"]) >= 3  # Should find at least our 3 worksheets

            # Test search by topic
            topic_filters = DocumentSearchFilters(topic="search_testing", limit=10)
            topic_results = await repository.search_documents(topic_filters)
            assert len(topic_results["documents"]) == 5  # Should find all our test documents

            # Test search by grade level
            grade_filters = DocumentSearchFilters(grade_level=9, limit=10)
            grade_results = await repository.search_documents(grade_filters)
            assert len(grade_results["documents"]) >= 2  # Should find grade 9 documents

            # Test text search
            text_filters = DocumentSearchFilters(search_text="algebra geometry", limit=10)
            text_results = await repository.search_documents(text_filters)
            assert len(text_results["documents"]) >= 5  # Should find all documents with this text

            # Test tag search
            tag_filters = DocumentSearchFilters(tags=["search", "testing"], limit=10)
            tag_results = await repository.search_documents(tag_filters)
            assert len(tag_results["documents"]) >= 5  # Should find all documents with these tags

            # Test date range search
            date_filters = DocumentSearchFilters(
                created_after=datetime.now() - timedelta(days=3),
                created_before=datetime.now() + timedelta(days=1),
                limit=10,
            )
            date_results = await repository.search_documents(date_filters)
            assert len(date_results["documents"]) >= 3  # Should find recent documents

        finally:
            # Cleanup all test documents
            for doc in test_documents:
                try:
                    await repository.soft_delete_document(doc.id)
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_document_pagination(self, repository):
        """Test document search pagination."""
        # Create 25 test documents
        test_documents = []
        for i in range(25):
            doc = StoredDocumentMetadata(
                id=uuid4(),
                session_id=uuid4(),
                title=f"Pagination Test Document {i:02d}",
                document_type="worksheet",
                topic="pagination_testing",
                status="generated",
                tags=["pagination", "testing"],
                search_content=f"pagination testing document {i}",
                created_at=datetime.now() - timedelta(seconds=i),  # Slightly different times
                updated_at=datetime.now(),
            )
            test_documents.append(doc)

        try:
            # Save all test documents
            for doc in test_documents:
                await repository.save_document_metadata(doc)

            # Test first page
            page1_filters = DocumentSearchFilters(topic="pagination_testing", limit=10, offset=0)
            page1_results = await repository.search_documents(page1_filters)
            assert len(page1_results["documents"]) == 10
            assert page1_results["total_count"] >= 25
            assert page1_results["has_more"] is True

            # Test second page
            page2_filters = DocumentSearchFilters(topic="pagination_testing", limit=10, offset=10)
            page2_results = await repository.search_documents(page2_filters)
            assert len(page2_results["documents"]) == 10
            assert page2_results["has_more"] is True

            # Test third page
            page3_filters = DocumentSearchFilters(topic="pagination_testing", limit=10, offset=20)
            page3_results = await repository.search_documents(page3_filters)
            assert len(page3_results["documents"]) >= 5  # At least 5 remaining

            # Verify no overlap between pages
            page1_ids = {doc.id for doc in page1_results["documents"]}
            page2_ids = {doc.id for doc in page2_results["documents"]}
            page3_ids = {doc.id for doc in page3_results["documents"]}

            assert len(page1_ids & page2_ids) == 0  # No overlap
            assert len(page2_ids & page3_ids) == 0  # No overlap
            assert len(page1_ids & page3_ids) == 0  # No overlap

        finally:
            # Cleanup all test documents
            for doc in test_documents:
                try:
                    await repository.soft_delete_document(doc.id)
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_file_storage_info_updates(
        self, repository, sample_document_metadata, sample_document_files
    ):
        """Test updating file storage information."""
        try:
            # Save document metadata and files
            await repository.save_document_metadata(sample_document_metadata)

            file_id = None
            for file in sample_document_files:
                file_id = await repository.save_document_file(file)
                break  # Just test with first file

            # Update storage info
            storage_info = {
                "r2_upload_id": "updated_upload_123",
                "r2_etag": "updated_etag_456",
                "file_size": 800000,
                "upload_completed_at": datetime.now().isoformat(),
            }

            success = await repository.update_file_storage_info(file_id, storage_info)
            assert success is True

            # Verify storage info update
            retrieved_files = await repository.get_document_files(sample_document_metadata.id)
            updated_file = next(f for f in retrieved_files if f.id == file_id)

            assert updated_file.r2_metadata["r2_upload_id"] == "updated_upload_123"
            assert updated_file.r2_metadata["r2_etag"] == "updated_etag_456"
            assert updated_file.file_size == 800000

        finally:
            # Cleanup
            try:
                await repository.soft_delete_document(sample_document_metadata.id)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_documents_by_session_id(self, repository):
        """Test retrieving documents by session ID."""
        session_id = uuid4()

        # Create multiple documents for the same session
        test_documents = []
        for i in range(3):
            doc = StoredDocumentMetadata(
                id=uuid4(),
                session_id=session_id,
                title=f"Session Test Document {i}",
                document_type="worksheet",
                topic="session_testing",
                status="generated",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            test_documents.append(doc)

        try:
            # Save all documents
            for doc in test_documents:
                await repository.save_document_metadata(doc)

            # Retrieve documents by session ID
            session_documents = await repository.get_documents_by_session_id(session_id)
            assert len(session_documents) == 3

            # Verify all documents belong to the session
            for doc in session_documents:
                assert doc.session_id == session_id

        finally:
            # Cleanup
            for doc in test_documents:
                try:
                    await repository.soft_delete_document(doc.id)
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_document_statistics(self, repository):
        """Test document statistics calculation."""
        # Create documents with different types and statuses
        test_documents = [
            StoredDocumentMetadata(
                id=uuid4(),
                session_id=uuid4(),
                title="Stats Test Worksheet 1",
                document_type="worksheet",
                status="generated",
                file_count=2,
                total_file_size=1000000,
            ),
            StoredDocumentMetadata(
                id=uuid4(),
                session_id=uuid4(),
                title="Stats Test Notes 1",
                document_type="notes",
                status="exported",
                file_count=1,
                total_file_size=500000,
            ),
            StoredDocumentMetadata(
                id=uuid4(),
                session_id=uuid4(),
                title="Stats Test Worksheet 2",
                document_type="worksheet",
                status="failed",
                file_count=0,
                total_file_size=0,
            ),
        ]

        try:
            # Save test documents
            for doc in test_documents:
                await repository.save_document_metadata(doc)

            # Get statistics
            stats = await repository.get_document_statistics()

            # Verify statistics structure
            assert "total_documents" in stats
            assert "total_file_size" in stats
            assert stats["total_documents"] >= 3
            assert stats["total_file_size"] >= 1500000

        finally:
            # Cleanup
            for doc in test_documents:
                try:
                    await repository.soft_delete_document(doc.id)
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_concurrent_document_operations(self, repository):
        """Test concurrent document save and retrieve operations."""
        import asyncio

        # Create multiple documents
        test_documents = []
        for i in range(10):
            doc = StoredDocumentMetadata(
                id=uuid4(),
                session_id=uuid4(),
                title=f"Concurrent Test Document {i}",
                document_type="worksheet",
                topic="concurrent_testing",
                status="generated",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            test_documents.append(doc)

        try:
            # Concurrent saves
            save_tasks = [repository.save_document_metadata(doc) for doc in test_documents]
            saved_ids = await asyncio.gather(*save_tasks)

            assert len(saved_ids) == 10
            assert all(saved_id is not None for saved_id in saved_ids)

            # Concurrent retrieves
            retrieve_tasks = [repository.retrieve_document_by_id(doc.id) for doc in test_documents]
            retrieved_docs = await asyncio.gather(*retrieve_tasks)

            assert len(retrieved_docs) == 10
            assert all(doc is not None for doc in retrieved_docs)

        finally:
            # Cleanup
            for doc in test_documents:
                try:
                    await repository.soft_delete_document(doc.id)
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_soft_delete_and_recovery(self, repository, sample_document_metadata):
        """Test soft delete functionality and data integrity."""
        try:
            # Save document
            await repository.save_document_metadata(sample_document_metadata)

            # Verify document exists
            retrieved_doc = await repository.retrieve_document_by_id(sample_document_metadata.id)
            assert retrieved_doc is not None
            assert retrieved_doc.metadata.status == "generated"

            # Soft delete document
            success = await repository.soft_delete_document(sample_document_metadata.id)
            assert success is True

            # Verify document is marked as deleted but still exists
            deleted_doc = await repository.retrieve_document_by_id(sample_document_metadata.id)
            assert deleted_doc is not None
            assert deleted_doc.metadata.status == "deleted"
            assert deleted_doc.metadata.deleted_at is not None

        finally:
            # Final cleanup - hard delete if necessary
            pass


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

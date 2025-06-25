"""
End-to-end tests for complete document storage workflows.
Tests full pipeline from generation to storage to retrieval.
"""

from uuid import uuid4

import pytest

from src.models.document_models import DetailLevel, DocumentStructure, DocumentType
from src.models.stored_document_models import DocumentSearchFilters
from src.repositories.document_storage_repository import DocumentStorageRepository
from src.services.document_export_service import DocumentExportService
from src.services.document_storage_service import DocumentStorageService
from src.services.r2_storage_service import R2StorageService


class TestDocumentStorageE2E:
    """End-to-end tests for complete document storage workflows."""

    @pytest.fixture
    async def storage_services(self, supabase_client):
        """Create integrated storage services for testing."""
        # R2 configuration for testing
        r2_config = {
            "account_id": "test_account",
            "access_key_id": "test_access_key",
            "secret_access_key": "test_secret_key",
            "bucket_name": "test-derivativ-bucket",
            "region": "auto",
        }

        r2_service = R2StorageService(r2_config)
        repository = DocumentStorageRepository(supabase_client)
        export_service = DocumentExportService()
        storage_service = DocumentStorageService(r2_service, repository, export_service)

        return {
            "r2_service": r2_service,
            "repository": repository,
            "export_service": export_service,
            "storage_service": storage_service,
        }

    @pytest.fixture
    def sample_document_structure(self):
        """Create sample DocumentStructure for testing."""
        from src.models.document_models import DocumentSection

        return DocumentStructure(
            title="E2E Test Algebra Worksheet",
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MEDIUM,
            estimated_duration=40,
            total_questions=6,
            sections=[
                DocumentSection(
                    title="Learning Objectives",
                    content_type="learning_objectives",
                    content_data={
                        "objectives_text": "• Solve quadratic equations by factoring\\n• Apply the quadratic formula\\n• Verify solutions"
                    },
                    order_index=0,
                ),
                DocumentSection(
                    title="Practice Questions",
                    content_type="practice_questions",
                    content_data={
                        "questions": [
                            {
                                "question_id": "q1",
                                "question_text": "Solve: x² + 5x + 6 = 0",
                                "marks": 3,
                                "command_word": "Solve",
                            },
                            {
                                "question_id": "q2",
                                "question_text": "Find the roots: 2x² - 7x + 3 = 0",
                                "marks": 4,
                                "command_word": "Find",
                            },
                        ],
                        "total_marks": 7,
                        "estimated_time": 25,
                    },
                    order_index=1,
                ),
                DocumentSection(
                    title="Answers",
                    content_type="answers",
                    content_data={
                        "answers": [{"answer": "x = -2, x = -3"}, {"answer": "x = 3, x = 0.5"}]
                    },
                    order_index=2,
                ),
                DocumentSection(
                    title="Detailed Solutions",
                    content_type="detailed_solutions",
                    content_data={
                        "solutions": [
                            {"solution": "x² + 5x + 6 = 0\\n(x + 2)(x + 3) = 0\\nx = -2 or x = -3"},
                            {
                                "solution": "2x² - 7x + 3 = 0\\nUsing quadratic formula:\\nx = (7 ± √(49-24))/4 = (7 ± 5)/4\\nx = 3 or x = 0.5"
                            },
                        ]
                    },
                    order_index=3,
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_complete_document_generation_to_storage_workflow(
        self, storage_services, sample_document_structure
    ):
        """Test complete workflow from document generation to storage and retrieval."""
        storage_service = storage_services["storage_service"]
        repository = storage_services["repository"]

        session_id = uuid4()
        generation_metadata = {
            "topic": "quadratic_equations",
            "grade_level": 10,
            "generation_request": {
                "document_type": "worksheet",
                "detail_level": "medium",
                "max_questions": 6,
            },
        }

        try:
            # Step 1: Store generated document with dual version export
            storage_result = await storage_service.store_generated_document(
                document=sample_document_structure,
                session_id=session_id,
                metadata=generation_metadata,
                export_formats=["pdf", "docx"],
                create_dual_versions=True,
            )

            assert storage_result["success"] is True
            assert "document_id" in storage_result
            assert "file_count" in storage_result
            assert storage_result["file_count"] == 4  # 2 formats × 2 versions

            document_id = storage_result["document_id"]

            # Step 2: Verify document metadata was saved
            stored_document = await repository.retrieve_document_by_id(document_id)
            assert stored_document is not None
            assert stored_document.metadata.title == sample_document_structure.title
            assert stored_document.metadata.document_type == "worksheet"
            assert stored_document.metadata.session_id == session_id
            assert stored_document.metadata.status == "exported"

            # Step 3: Verify files were created and stored
            assert len(stored_document.files) == 4
            file_versions = {f.version for f in stored_document.files}
            file_formats = {f.file_format for f in stored_document.files}
            assert file_versions == {"student", "teacher"}
            assert file_formats == {"pdf", "docx"}

            # Step 4: Test file retrieval
            for file_info in stored_document.files:
                file_content = await storage_service.retrieve_document_file(file_info.file_key)
                assert file_content["success"] is True
                assert len(file_content["content"]) > 0

                # Verify content type matches format
                if file_info.file_format == "pdf":
                    assert file_content["content"].startswith(b"%PDF")
                elif file_info.file_format == "docx":
                    assert file_content["content"].startswith(b"PK")  # ZIP header

            # Step 5: Test document search and discovery
            search_filters = DocumentSearchFilters(
                topic="quadratic_equations", document_type="worksheet", grade_level=10
            )
            search_results = await repository.search_documents(search_filters)

            assert search_results["total_count"] >= 1
            found_document = next(
                (doc for doc in search_results["documents"] if doc.id == document_id), None
            )
            assert found_document is not None

            # Step 6: Test document update workflow
            update_success = await storage_service.update_document_status(
                document_id, "archived", {"archived_reason": "test_completion"}
            )
            assert update_success is True

            # Verify status update
            updated_document = await repository.retrieve_document_by_id(document_id)
            assert updated_document.metadata.status == "archived"

        finally:
            # Cleanup: Delete document and files
            try:
                await storage_service.delete_document_and_files(document_id)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_bulk_document_storage_and_retrieval(
        self, storage_services, sample_document_structure
    ):
        """Test bulk operations for multiple documents."""
        storage_service = storage_services["storage_service"]
        repository = storage_services["repository"]

        session_id = uuid4()
        document_count = 5
        stored_document_ids = []

        try:
            # Step 1: Store multiple documents
            for i in range(document_count):
                # Create variations of the document
                document_variation = sample_document_structure.model_copy()
                document_variation.title = f"Bulk Test Document {i+1}"

                metadata = {
                    "topic": f"bulk_testing_{i}",
                    "grade_level": 9 + (i % 3),  # Vary grade levels
                    "generation_request": {"batch_id": f"bulk_test_{i}"},
                }

                storage_result = await storage_service.store_generated_document(
                    document=document_variation,
                    session_id=session_id,
                    metadata=metadata,
                    export_formats=["pdf"],
                    create_dual_versions=False,  # Single version for speed
                )

                assert storage_result["success"] is True
                stored_document_ids.append(storage_result["document_id"])

            # Step 2: Verify all documents were stored
            assert len(stored_document_ids) == document_count

            # Step 3: Test batch retrieval by session
            session_documents = await repository.get_documents_by_session_id(session_id)
            assert len(session_documents) == document_count

            # Step 4: Test search across all stored documents
            search_filters = DocumentSearchFilters(search_text="bulk test", limit=10)
            search_results = await repository.search_documents(search_filters)

            bulk_test_docs = [
                doc for doc in search_results["documents"] if "Bulk Test" in doc.title
            ]
            assert len(bulk_test_docs) >= document_count

            # Step 5: Test bulk status updates
            update_tasks = []
            for doc_id in stored_document_ids:
                update_tasks.append(storage_service.update_document_status(doc_id, "processed"))

            import asyncio

            update_results = await asyncio.gather(*update_tasks)
            assert all(result is True for result in update_results)

            # Step 6: Verify batch updates
            for doc_id in stored_document_ids:
                updated_doc = await repository.retrieve_document_by_id(doc_id)
                assert updated_doc.metadata.status == "processed"

        finally:
            # Cleanup all documents
            for doc_id in stored_document_ids:
                try:
                    await storage_service.delete_document_and_files(doc_id)
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_document_version_comparison_workflow(
        self, storage_services, sample_document_structure
    ):
        """Test workflow for comparing student vs teacher document versions."""
        storage_service = storage_services["storage_service"]
        repository = storage_services["repository"]

        session_id = uuid4()

        try:
            # Step 1: Store document with dual versions
            storage_result = await storage_service.store_generated_document(
                document=sample_document_structure,
                session_id=session_id,
                metadata={"topic": "version_comparison"},
                export_formats=["html"],  # Use HTML for easy content comparison
                create_dual_versions=True,
            )

            document_id = storage_result["document_id"]

            # Step 2: Retrieve stored document with files
            stored_document = await repository.retrieve_document_by_id(document_id)

            # Step 3: Compare student vs teacher versions
            student_file = next(f for f in stored_document.files if f.version == "student")
            teacher_file = next(f for f in stored_document.files if f.version == "teacher")

            # Retrieve file contents
            student_content = await storage_service.retrieve_document_file(student_file.file_key)
            teacher_content = await storage_service.retrieve_document_file(teacher_file.file_key)

            student_html = student_content["content"].decode("utf-8")
            teacher_html = teacher_content["content"].decode("utf-8")

            # Step 4: Verify version differences
            # Student version should NOT contain answers or detailed solutions
            assert "Detailed Solutions" not in student_html
            assert "x = -2, x = -3" not in student_html  # Answer content

            # Teacher version should contain everything
            assert "Detailed Solutions" in teacher_html
            assert "x = -2, x = -3" in teacher_html  # Answer content
            assert "Teaching Notes" in teacher_html

            # Both versions should contain questions
            assert "x² + 5x + 6 = 0" in student_html
            assert "x² + 5x + 6 = 0" in teacher_html

            # Step 5: Test version metadata
            assert student_file.version == "student"
            assert teacher_file.version == "teacher"
            assert (
                teacher_file.file_size > student_file.file_size
            )  # Teacher version should be larger

        finally:
            # Cleanup
            try:
                await storage_service.delete_document_and_files(document_id)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_document_export_format_consistency(
        self, storage_services, sample_document_structure
    ):
        """Test consistency across different export formats."""
        storage_service = storage_services["storage_service"]
        repository = storage_services["repository"]

        session_id = uuid4()

        try:
            # Step 1: Store document in multiple formats
            storage_result = await storage_service.store_generated_document(
                document=sample_document_structure,
                session_id=session_id,
                metadata={"topic": "format_consistency"},
                export_formats=["pdf", "docx", "html"],
                create_dual_versions=True,
            )

            document_id = storage_result["document_id"]
            assert storage_result["file_count"] == 6  # 3 formats × 2 versions

            # Step 2: Retrieve all file versions
            stored_document = await repository.retrieve_document_by_id(document_id)

            # Group files by version
            student_files = [f for f in stored_document.files if f.version == "student"]
            teacher_files = [f for f in stored_document.files if f.version == "teacher"]

            assert len(student_files) == 3  # PDF, DOCX, HTML
            assert len(teacher_files) == 3

            # Step 3: Verify all formats were created
            student_formats = {f.file_format for f in student_files}
            teacher_formats = {f.file_format for f in teacher_files}

            expected_formats = {"pdf", "docx", "html"}
            assert student_formats == expected_formats
            assert teacher_formats == expected_formats

            # Step 4: Test format-specific properties
            for file_info in stored_document.files:
                file_content = await storage_service.retrieve_document_file(file_info.file_key)
                content = file_content["content"]

                if file_info.file_format == "pdf":
                    assert content.startswith(b"%PDF")
                    assert file_info.content_type == "application/pdf"
                elif file_info.file_format == "docx":
                    assert content.startswith(b"PK")  # ZIP header
                    assert "vnd.openxmlformats" in file_info.content_type
                elif file_info.file_format == "html":
                    html_content = content.decode("utf-8")
                    assert "<!DOCTYPE html>" in html_content or "<html" in html_content
                    assert file_info.content_type == "text/html"

                # All formats should have reasonable file sizes
                assert file_info.file_size > 0
                assert file_info.file_size < 10_000_000  # Less than 10MB

        finally:
            # Cleanup
            try:
                await storage_service.delete_document_and_files(document_id)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery_workflow(
        self, storage_services, sample_document_structure
    ):
        """Test error handling and recovery in storage workflows."""
        storage_service = storage_services["storage_service"]
        repository = storage_services["repository"]

        session_id = uuid4()

        # Test 1: Invalid document structure
        invalid_document = sample_document_structure.model_copy()
        invalid_document.title = ""  # Invalid empty title

        with pytest.raises(Exception):  # Should raise validation error
            await storage_service.store_generated_document(
                document=invalid_document, session_id=session_id, metadata={"topic": "error_test"}
            )

        # Test 2: Invalid export format
        with pytest.raises(Exception):  # Should raise unsupported format error
            await storage_service.store_generated_document(
                document=sample_document_structure,
                session_id=session_id,
                metadata={"topic": "error_test"},
                export_formats=["invalid_format"],
            )

        # Test 3: Retrieval of non-existent document
        non_existent_id = uuid4()
        retrieved_doc = await repository.retrieve_document_by_id(non_existent_id)
        assert retrieved_doc is None

        # Test 4: File retrieval with invalid key
        with pytest.raises(Exception):
            await storage_service.retrieve_document_file("invalid/file/key.pdf")

        # Test 5: Recovery after partial failure
        document_id = None
        try:
            # Store a valid document first
            storage_result = await storage_service.store_generated_document(
                document=sample_document_structure,
                session_id=session_id,
                metadata={"topic": "recovery_test"},
                export_formats=["pdf"],
            )

            document_id = storage_result["document_id"]

            # Attempt invalid update
            with pytest.raises(Exception):
                await storage_service.update_document_status(document_id, "invalid_status")

            # Verify document is still in valid state
            recovered_doc = await repository.retrieve_document_by_id(document_id)
            assert recovered_doc is not None
            assert recovered_doc.metadata.status != "invalid_status"

        finally:
            # Cleanup
            if document_id:
                try:
                    await storage_service.delete_document_and_files(document_id)
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_concurrent_access_and_consistency(
        self, storage_services, sample_document_structure
    ):
        """Test concurrent access patterns and data consistency."""
        storage_service = storage_services["storage_service"]
        repository = storage_services["repository"]

        session_id = uuid4()
        document_id = None

        try:
            # Step 1: Store initial document
            storage_result = await storage_service.store_generated_document(
                document=sample_document_structure,
                session_id=session_id,
                metadata={"topic": "concurrency_test"},
                export_formats=["pdf"],
            )

            document_id = storage_result["document_id"]

            # Step 2: Concurrent reads (should all succeed)
            import asyncio

            read_tasks = [repository.retrieve_document_by_id(document_id) for _ in range(10)]

            read_results = await asyncio.gather(*read_tasks)
            assert all(result is not None for result in read_results)
            assert all(result.metadata.id == document_id for result in read_results)

            # Step 3: Concurrent updates (test serialization)
            update_tasks = [
                storage_service.update_document_status(document_id, "processing", {"step": i})
                for i in range(5)
            ]

            update_results = await asyncio.gather(*update_tasks, return_exceptions=True)

            # At least some updates should succeed
            successful_updates = [r for r in update_results if r is True]
            assert len(successful_updates) > 0

            # Step 4: Verify final consistency
            final_doc = await repository.retrieve_document_by_id(document_id)
            assert final_doc is not None

        finally:
            # Cleanup
            if document_id:
                try:
                    await storage_service.delete_document_and_files(document_id)
                except Exception:
                    pass


# Mark all tests in this module as e2e tests
pytestmark = pytest.mark.e2e

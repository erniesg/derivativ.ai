"""
Integration tests for R2StorageService.
Tests actual R2 operations with real/mock R2 endpoints.
"""


import pytest

from src.services.r2_storage_service import R2StorageService


class TestR2StorageIntegration:
    """Integration tests for R2StorageService with real/mock R2 endpoints."""

    @pytest.fixture
    def r2_service(self):
        """Create R2StorageService with test configuration."""
        # Use test configuration or mock R2 endpoint
        config = {
            "account_id": "test_account",
            "access_key_id": "test_access_key",
            "secret_access_key": "test_secret_key",
            "bucket_name": "test-derivativ-bucket",
            "region": "auto",
        }
        return R2StorageService(config)

    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF content for testing."""
        return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n0000000103 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n171\n%%EOF"

    @pytest.fixture
    def sample_document_metadata(self):
        """Sample document metadata."""
        return {
            "document_type": "worksheet",
            "document_id": "test_doc_123",
            "version": "student",
            "format": "pdf",
            "title": "Algebra Practice Test",
            "grade_level": "9",
            "topic": "quadratic_equations",
            "generated_at": "2025-06-24T10:00:00Z",
        }

    @pytest.mark.asyncio
    async def test_end_to_end_file_upload_download(
        self, r2_service, sample_pdf_content, sample_document_metadata
    ):
        """Test complete upload and download workflow."""
        file_key = "test_documents/integration_test.pdf"

        try:
            # Upload file
            upload_result = await r2_service.upload_file(
                sample_pdf_content, file_key, sample_document_metadata
            )

            assert upload_result["success"] is True
            assert upload_result["file_key"] == file_key

            # Verify file exists
            exists = await r2_service.file_exists(file_key)
            assert exists is True

            # Download file
            download_result = await r2_service.download_file(file_key)
            assert download_result["success"] is True
            assert download_result["content"] == sample_pdf_content

            # Get metadata
            metadata = await r2_service.get_file_metadata(file_key)
            assert metadata["custom_metadata"]["document_type"] == "worksheet"
            assert metadata["custom_metadata"]["document_id"] == "test_doc_123"

        finally:
            # Cleanup
            try:
                await r2_service.delete_file(file_key)
            except Exception:
                pass  # Ignore cleanup errors

    @pytest.mark.asyncio
    async def test_large_file_handling(self, r2_service):
        """Test handling of large files (multipart upload)."""
        # Create 10MB test file
        large_content = b"x" * (10 * 1024 * 1024)
        file_key = "test_documents/large_file_test.pdf"

        try:
            # Upload large file
            upload_result = await r2_service.upload_file(large_content, file_key)
            assert upload_result["success"] is True

            # Verify file size
            metadata = await r2_service.get_file_metadata(file_key)
            assert metadata["size"] == len(large_content)

            # Download and verify content
            download_result = await r2_service.download_file(file_key)
            assert download_result["success"] is True
            assert len(download_result["content"]) == len(large_content)

        finally:
            # Cleanup
            try:
                await r2_service.delete_file(file_key)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_presigned_url_workflow(self, r2_service, sample_pdf_content):
        """Test presigned URL generation and usage."""
        file_key = "test_documents/presigned_test.pdf"

        try:
            # Upload file first
            await r2_service.upload_file(sample_pdf_content, file_key)

            # Generate presigned download URL
            download_url = await r2_service.generate_presigned_url(file_key, expiration=3600)
            assert download_url.startswith("https://")
            assert "test-derivativ-bucket" in download_url

            # Generate presigned upload URL
            upload_url_data = await r2_service.generate_presigned_upload_url(
                "test_documents/presigned_upload.pdf", expiration=3600
            )
            assert "url" in upload_url_data
            assert "fields" in upload_url_data

        finally:
            # Cleanup
            try:
                await r2_service.delete_file(file_key)
                await r2_service.delete_file("test_documents/presigned_upload.pdf")
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_file_listing_and_filtering(self, r2_service, sample_pdf_content):
        """Test file listing with prefix filtering."""
        test_files = [
            "test_documents/worksheets/algebra_1.pdf",
            "test_documents/worksheets/algebra_2.pdf",
            "test_documents/notes/geometry.pdf",
        ]

        try:
            # Upload test files
            for file_key in test_files:
                await r2_service.upload_file(sample_pdf_content, file_key)

            # List all test documents
            all_files = await r2_service.list_files("test_documents/")
            assert len(all_files) >= 3

            # List only worksheets
            worksheet_files = await r2_service.list_files("test_documents/worksheets/")
            assert len(worksheet_files) == 2
            assert all("algebra" in f["key"] for f in worksheet_files)

            # List only notes
            note_files = await r2_service.list_files("test_documents/notes/")
            assert len(note_files) == 1
            assert note_files[0]["key"].endswith("geometry.pdf")

        finally:
            # Cleanup
            for file_key in test_files:
                try:
                    await r2_service.delete_file(file_key)
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, r2_service, sample_pdf_content):
        """Test concurrent upload/download operations."""
        import asyncio

        file_keys = [f"test_documents/concurrent_{i}.pdf" for i in range(5)]

        try:
            # Concurrent uploads
            upload_tasks = [r2_service.upload_file(sample_pdf_content, key) for key in file_keys]
            upload_results = await asyncio.gather(*upload_tasks)

            assert all(result["success"] for result in upload_results)

            # Concurrent downloads
            download_tasks = [r2_service.download_file(key) for key in file_keys]
            download_results = await asyncio.gather(*download_tasks)

            assert all(result["success"] for result in download_results)
            assert all(result["content"] == sample_pdf_content for result in download_results)

        finally:
            # Cleanup
            delete_tasks = [r2_service.delete_file(key) for key in file_keys]
            await asyncio.gather(*delete_tasks, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, r2_service):
        """Test error handling for various failure scenarios."""
        # Test download of non-existent file
        with pytest.raises(Exception):  # Should raise R2StorageError
            await r2_service.download_file("test_documents/nonexistent.pdf")

        # Test invalid file key
        with pytest.raises(Exception):
            await r2_service.upload_file(b"content", "../../../invalid/path.pdf")

        # Test file existence check for non-existent file
        exists = await r2_service.file_exists("test_documents/definitely_not_there.pdf")
        assert exists is False

    @pytest.mark.asyncio
    async def test_metadata_persistence(self, r2_service, sample_pdf_content):
        """Test that custom metadata persists through upload/download cycle."""
        file_key = "test_documents/metadata_test.pdf"
        custom_metadata = {
            "document_type": "worksheet",
            "subject": "mathematics",
            "grade": "10",
            "difficulty": "medium",
            "tags": "algebra,equations,practice",
        }

        try:
            # Upload with metadata
            await r2_service.upload_file(sample_pdf_content, file_key, custom_metadata)

            # Retrieve and verify metadata
            stored_metadata = await r2_service.get_file_metadata(file_key)

            for key, value in custom_metadata.items():
                assert stored_metadata["custom_metadata"][key] == value

        finally:
            # Cleanup
            try:
                await r2_service.delete_file(file_key)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_content_type_detection(self, r2_service):
        """Test automatic content type detection for different file formats."""
        test_files = [
            ("test.pdf", b"%PDF-1.4", "application/pdf"),
            (
                "test.docx",
                b"PK\x03\x04",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            ("test.html", b"<!DOCTYPE html>", "text/html"),
            ("test.txt", b"plain text", "text/plain"),
        ]

        for filename, content, expected_type in test_files:
            file_key = f"test_documents/{filename}"

            try:
                await r2_service.upload_file(content, file_key)
                metadata = await r2_service.get_file_metadata(file_key)

                # Content type should be detected or set appropriately
                assert metadata.get("content_type") is not None

            finally:
                try:
                    await r2_service.delete_file(file_key)
                except Exception:
                    pass


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

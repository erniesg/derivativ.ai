"""
Unit tests for R2StorageService.
Tests Cloudflare R2 blob storage operations with mocked boto3 client.
"""

from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import BotoCoreError, ClientError

from src.services.r2_storage_service import R2StorageError, R2StorageService


class TestR2StorageService:
    """Unit tests for R2StorageService with mocked dependencies."""

    @pytest.fixture
    def mock_boto3_client(self):
        """Create mock boto3 S3 client."""
        client = Mock()
        client.upload_fileobj = Mock()
        client.download_fileobj = Mock()
        client.delete_object = Mock()
        client.head_object = Mock()
        client.generate_presigned_url = Mock()
        client.generate_presigned_post = Mock()
        return client

    @pytest.fixture
    def r2_config(self):
        """R2 configuration for testing."""
        return {
            "account_id": "test_account_id",
            "access_key_id": "test_access_key",
            "secret_access_key": "test_secret_key",
            "bucket_name": "test-bucket",
            "region": "auto",
        }

    @pytest.fixture
    def r2_service(self, mock_boto3_client, r2_config):
        """Create R2StorageService with mocked client."""
        with patch("src.services.r2_storage_service.boto3.client", return_value=mock_boto3_client):
            service = R2StorageService(r2_config)
            return service

    def test_initialize_r2_client(self, r2_config):
        """Test R2 client initialization with correct configuration."""
        with patch("src.services.r2_storage_service.boto3.client") as mock_boto3:
            R2StorageService(r2_config)

            mock_boto3.assert_called_once_with(
                "s3",
                endpoint_url=f"https://{r2_config['account_id']}.r2.cloudflarestorage.com",
                aws_access_key_id=r2_config["access_key_id"],
                aws_secret_access_key=r2_config["secret_access_key"],
                region_name=r2_config["region"],
            )

    def test_initialize_r2_client_with_invalid_config(self):
        """Test R2 client initialization fails with invalid configuration."""
        invalid_config = {"account_id": "test"}  # Missing required fields

        with pytest.raises(R2StorageError, match="Missing required R2 configuration"):
            R2StorageService(invalid_config)

    async def test_upload_document_file_success(self, r2_service, mock_boto3_client):
        """Test successful file upload to R2."""
        # Arrange
        file_content = b"test document content"
        file_key = "documents/test-document.pdf"
        metadata = {"document_type": "worksheet", "version": "student"}

        mock_boto3_client.upload_fileobj.return_value = None

        # Act
        result = await r2_service.upload_file(file_content, file_key, metadata)

        # Assert
        assert result["success"] is True
        assert result["file_key"] == file_key
        assert result["bucket"] == "test-bucket"
        assert "upload_id" in result

        mock_boto3_client.upload_fileobj.assert_called_once()
        call_args = mock_boto3_client.upload_fileobj.call_args
        assert call_args[1]["Bucket"] == "test-bucket"
        assert call_args[1]["Key"] == file_key
        assert call_args[1]["Metadata"] == metadata

    async def test_upload_document_file_failure(self, r2_service, mock_boto3_client):
        """Test file upload failure handling."""
        # Arrange
        file_content = b"test content"
        file_key = "documents/test-document.pdf"

        mock_boto3_client.upload_fileobj.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": "Bucket not found"}}, "upload_fileobj"
        )

        # Act & Assert
        with pytest.raises(R2StorageError, match="Bucket not found"):
            await r2_service.upload_file(file_content, file_key)

    async def test_download_document_file_success(self, r2_service, mock_boto3_client):
        """Test successful file download from R2."""
        # Arrange
        file_key = "documents/test-document.pdf"
        expected_content = b"test document content"

        # Mock get_object response
        mock_body = type("MockBody", (), {"read": lambda self: expected_content})()
        mock_response = {"Body": mock_body}
        mock_boto3_client.get_object.return_value = mock_response

        # Act
        result = await r2_service.download_file(file_key)

        # Assert
        assert result["success"] is True
        assert result["content"] == expected_content
        assert result["file_key"] == file_key

        mock_boto3_client.get_object.assert_called_once_with(Bucket="test-bucket", Key=file_key)

    async def test_download_document_file_not_found(self, r2_service, mock_boto3_client):
        """Test file download with file not found."""
        # Arrange
        file_key = "documents/nonexistent.pdf"

        mock_boto3_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Key not found"}}, "get_object"
        )

        # Act & Assert
        with pytest.raises(R2StorageError, match="File not found"):
            await r2_service.download_file(file_key)

    async def test_delete_document_file_success(self, r2_service, mock_boto3_client):
        """Test successful file deletion from R2."""
        # Arrange
        file_key = "documents/test-document.pdf"
        mock_boto3_client.delete_object.return_value = {"DeleteMarker": True}

        # Act
        result = await r2_service.delete_file(file_key)

        # Assert
        assert result["success"] is True
        assert result["file_key"] == file_key

        mock_boto3_client.delete_object.assert_called_once_with(Bucket="test-bucket", Key=file_key)

    async def test_delete_document_file_failure(self, r2_service, mock_boto3_client):
        """Test file deletion failure handling."""
        # Arrange
        file_key = "documents/test-document.pdf"

        mock_boto3_client.delete_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access denied"}}, "delete_object"
        )

        # Act & Assert
        with pytest.raises(R2StorageError, match="Failed to delete file"):
            await r2_service.delete_file(file_key)

    async def test_file_exists_true(self, r2_service, mock_boto3_client):
        """Test checking if file exists - file found."""
        # Arrange
        file_key = "documents/test-document.pdf"
        mock_boto3_client.head_object.return_value = {
            "ContentLength": 1024,
            "LastModified": "2025-01-01T00:00:00Z",
        }

        # Act
        exists = await r2_service.file_exists(file_key)

        # Assert
        assert exists is True
        mock_boto3_client.head_object.assert_called_once_with(Bucket="test-bucket", Key=file_key)

    async def test_file_exists_false(self, r2_service, mock_boto3_client):
        """Test checking if file exists - file not found."""
        # Arrange
        file_key = "documents/nonexistent.pdf"
        mock_boto3_client.head_object.side_effect = ClientError(
            {"Error": {"Code": "NotFound", "Message": "Not found"}}, "head_object"
        )

        # Act
        exists = await r2_service.file_exists(file_key)

        # Assert
        assert exists is False

    async def test_generate_presigned_download_url(self, r2_service, mock_boto3_client):
        """Test generating presigned URL for file download."""
        # Arrange
        file_key = "documents/test-document.pdf"
        expected_url = "https://test-bucket.r2.cloudflarestorage.com/presigned-url"
        expiration = 3600

        mock_boto3_client.generate_presigned_url.return_value = expected_url

        # Act
        url = await r2_service.generate_presigned_url(file_key, expiration)

        # Assert
        assert url == expected_url
        mock_boto3_client.generate_presigned_url.assert_called_once_with(
            "get_object", Params={"Bucket": "test-bucket", "Key": file_key}, ExpiresIn=expiration
        )

    async def test_generate_presigned_upload_url(self, r2_service, mock_boto3_client):
        """Test generating presigned URL for file upload."""
        # Arrange
        file_key = "documents/test-document.pdf"
        expected_response = {
            "url": "https://test-bucket.r2.cloudflarestorage.com/upload",
            "fields": {"key": file_key, "policy": "encoded_policy"},
        }
        expiration = 3600

        mock_boto3_client.generate_presigned_post.return_value = expected_response

        # Act
        result = await r2_service.generate_presigned_upload_url(file_key, expiration)

        # Assert
        assert result == expected_response
        mock_boto3_client.generate_presigned_post.assert_called_once_with(
            Bucket="test-bucket", Key=file_key, ExpiresIn=expiration
        )

    async def test_list_files_with_prefix(self, r2_service, mock_boto3_client):
        """Test listing files with specific prefix."""
        # Arrange
        prefix = "documents/worksheets/"
        from datetime import datetime

        mock_response = {
            "Contents": [
                {
                    "Key": "documents/worksheets/algebra.pdf",
                    "Size": 1024,
                    "LastModified": datetime.now(),
                    "ETag": '"abc123"',
                },
                {
                    "Key": "documents/worksheets/geometry.pdf",
                    "Size": 2048,
                    "LastModified": datetime.now(),
                    "ETag": '"def456"',
                },
            ]
        }
        mock_boto3_client.list_objects_v2.return_value = mock_response

        # Act
        files = await r2_service.list_files(prefix)

        # Assert
        assert len(files) == 2
        assert files[0]["key"] == "documents/worksheets/algebra.pdf"
        assert files[0]["size"] == 1024
        assert files[1]["key"] == "documents/worksheets/geometry.pdf"
        assert files[1]["size"] == 2048

        mock_boto3_client.list_objects_v2.assert_called_once_with(
            Bucket="test-bucket", Prefix=prefix, MaxKeys=1000
        )

    async def test_get_file_metadata(self, r2_service, mock_boto3_client):
        """Test retrieving file metadata."""
        # Arrange
        file_key = "documents/test-document.pdf"
        mock_response = {
            "ContentLength": 1024,
            "LastModified": "2025-01-01T00:00:00Z",
            "ContentType": "application/pdf",
            "Metadata": {"document_type": "worksheet", "version": "student"},
        }
        mock_boto3_client.head_object.return_value = mock_response

        # Act
        metadata = await r2_service.get_file_metadata(file_key)

        # Assert
        assert metadata["size"] == 1024
        assert metadata["content_type"] == "application/pdf"
        assert metadata["custom_metadata"]["document_type"] == "worksheet"
        assert metadata["custom_metadata"]["version"] == "student"

    async def test_handle_r2_connection_errors(self, r2_config):
        """Test handling R2 connection errors during initialization."""
        # Arrange
        with patch("src.services.r2_storage_service.boto3.client") as mock_boto3:
            mock_boto3.side_effect = BotoCoreError()

            # Act & Assert
            with pytest.raises(R2StorageError, match="Failed to initialize R2 client"):
                R2StorageService(r2_config)

    def test_generate_file_key(self, r2_service):
        """Test file key generation for documents."""
        # Arrange
        document_id = "doc_123"
        document_type = "worksheet"
        file_format = "pdf"
        version = "student"

        # Act
        file_key = r2_service.generate_file_key(document_id, document_type, file_format, version)

        # Assert
        expected_key = f"documents/{document_type}/{document_id}/{version}.{file_format}"
        assert file_key == expected_key

    def test_validate_file_key(self, r2_service):
        """Test file key validation."""
        # Valid keys
        assert r2_service.validate_file_key("documents/worksheet/123/student.pdf") is True
        assert r2_service.validate_file_key("documents/notes/abc/teacher.docx") is True

        # Invalid keys
        assert r2_service.validate_file_key("") is False
        assert r2_service.validate_file_key("invalid//double/slash.pdf") is False
        assert r2_service.validate_file_key("../../../etc/passwd") is False

    def test_get_supported_formats(self, r2_service):
        """Test getting supported file formats."""
        formats = r2_service.get_supported_formats()

        assert "pdf" in formats
        assert "docx" in formats
        assert "html" in formats
        assert len(formats) >= 3

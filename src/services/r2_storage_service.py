"""
Cloudflare R2 Storage Service.
Handles file upload, download, and management operations using R2 object storage.
"""

import asyncio
import io
import logging
import mimetypes
import re
from typing import Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class R2StorageError(Exception):
    """Raised when R2 storage operations fail."""

    pass


class R2StorageService:
    """Service for Cloudflare R2 object storage operations."""

    def __init__(self, config: dict[str, str]):
        """
        Initialize R2 storage service.

        Args:
            config: R2 configuration dictionary with required keys:
                - account_id: Cloudflare account ID
                - access_key_id: R2 access key ID
                - secret_access_key: R2 secret access key
                - bucket_name: R2 bucket name
                - region: R2 region (usually 'auto')

        Raises:
            R2StorageError: If configuration is invalid or client setup fails
        """
        self.config = config
        self._validate_config()
        self._initialize_client()

    def _validate_config(self) -> None:
        """Validate R2 configuration."""
        required_keys = [
            "account_id",
            "access_key_id",
            "secret_access_key",
            "bucket_name",
            "region",
        ]
        missing_keys = [key for key in required_keys if not self.config.get(key)]

        if missing_keys:
            raise R2StorageError(f"Missing required R2 configuration: {missing_keys}")

    def _initialize_client(self) -> None:
        """Initialize boto3 S3 client for R2."""
        try:
            endpoint_url = f"https://{self.config['account_id']}.r2.cloudflarestorage.com"

            self.client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=self.config["access_key_id"],
                aws_secret_access_key=self.config["secret_access_key"],
                region_name=self.config["region"],
            )

            logger.info(f"R2 client initialized for bucket: {self.config['bucket_name']}")

        except Exception as e:
            raise R2StorageError(f"Failed to initialize R2 client: {e}")

    async def upload_file(
        self, file_content: bytes, file_key: str, metadata: Optional[dict[str, str]] = None
    ) -> dict[str, any]:
        """
        Upload file to R2 storage.

        Args:
            file_content: File content as bytes
            file_key: Key/path for the file in R2
            metadata: Optional metadata to store with file

        Returns:
            Dict with upload result information

        Raises:
            R2StorageError: If upload fails
        """
        try:
            if not self.validate_file_key(file_key):
                raise R2StorageError(f"Invalid file key: {file_key}")

            # Prepare upload arguments for upload_fileobj
            fileobj = io.BytesIO(file_content)

            # Prepare extra arguments
            extra_args = {}

            # Add metadata if provided
            if metadata:
                extra_args["Metadata"] = metadata

            # Detect and set content type
            content_type = self._detect_content_type(file_key, file_content)
            if content_type:
                extra_args["ContentType"] = content_type

            # Perform upload using executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.upload_fileobj(
                    Fileobj=fileobj,
                    Bucket=self.config["bucket_name"],
                    Key=file_key,
                    ExtraArgs=extra_args if extra_args else None,
                ),
            )

            # Generate upload ID for tracking
            upload_id = f"upload_{file_key.replace('/', '_')}_{len(file_content)}"

            logger.info(f"Successfully uploaded file to R2: {file_key}")

            return {
                "success": True,
                "file_key": file_key,
                "bucket": self.config["bucket_name"],
                "upload_id": upload_id,
                "file_size": len(file_content),
            }

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchBucket":
                raise R2StorageError(f"Bucket not found: {self.config['bucket_name']}")
            else:
                raise R2StorageError(f"Failed to upload file: {e}")
        except Exception as e:
            raise R2StorageError(f"Failed to upload file: {e}")

    async def download_file(self, file_key: str) -> dict[str, any]:
        """
        Download file from R2 storage.

        Args:
            file_key: Key/path of the file in R2

        Returns:
            Dict with file content and metadata

        Raises:
            R2StorageError: If download fails or file not found
        """
        try:
            # Download file using executor
            loop = asyncio.get_event_loop()

            def _download():
                response = self.client.get_object(Bucket=self.config["bucket_name"], Key=file_key)
                return response["Body"].read()

            content = await loop.run_in_executor(None, _download)

            logger.info(f"Successfully downloaded file from R2: {file_key}")

            return {
                "success": True,
                "file_key": file_key,
                "content": content,
                "file_size": len(content),
            }

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ["NoSuchKey", "NotFound"]:
                raise R2StorageError(f"File not found: {file_key}")
            else:
                raise R2StorageError(f"Failed to download file: {e}")
        except Exception as e:
            raise R2StorageError(f"Failed to download file: {e}")

    async def delete_file(self, file_key: str) -> dict[str, any]:
        """
        Delete file from R2 storage.

        Args:
            file_key: Key/path of the file to delete

        Returns:
            Dict with deletion result

        Raises:
            R2StorageError: If deletion fails
        """
        try:
            loop = asyncio.get_event_loop()

            def _delete():
                return self.client.delete_object(Bucket=self.config["bucket_name"], Key=file_key)

            result = await loop.run_in_executor(None, _delete)

            logger.info(f"Successfully deleted file from R2: {file_key}")

            return {
                "success": True,
                "file_key": file_key,
                "delete_marker": result.get("DeleteMarker", False),
            }

        except ClientError as e:
            raise R2StorageError(f"Failed to delete file: {e}")
        except Exception as e:
            raise R2StorageError(f"Failed to delete file: {e}")

    async def file_exists(self, file_key: str) -> bool:
        """
        Check if file exists in R2 storage.

        Args:
            file_key: Key/path of the file to check

        Returns:
            True if file exists, False otherwise
        """
        try:
            loop = asyncio.get_event_loop()

            def _head_object():
                return self.client.head_object(Bucket=self.config["bucket_name"], Key=file_key)

            await loop.run_in_executor(None, _head_object)
            return True

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ["NoSuchKey", "NotFound", "404"]:
                return False
            else:
                # Re-raise for other errors
                raise R2StorageError(f"Error checking file existence: {e}")

    async def get_file_metadata(self, file_key: str) -> dict[str, any]:
        """
        Get file metadata from R2 storage.

        Args:
            file_key: Key/path of the file

        Returns:
            Dict with file metadata

        Raises:
            R2StorageError: If file not found or operation fails
        """
        try:
            loop = asyncio.get_event_loop()

            def _head_object():
                return self.client.head_object(Bucket=self.config["bucket_name"], Key=file_key)

            response = await loop.run_in_executor(None, _head_object)

            return {
                "file_key": file_key,
                "size": response.get("ContentLength", 0),
                "last_modified": response.get("LastModified", ""),
                "content_type": response.get("ContentType", ""),
                "custom_metadata": response.get("Metadata", {}),
                "etag": response.get("ETag", "").strip('"'),
            }

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ["NoSuchKey", "NotFound"]:
                raise R2StorageError(f"File not found: {file_key}")
            else:
                raise R2StorageError(f"Failed to get file metadata: {e}")

    async def list_files(self, prefix: str = "", max_keys: int = 1000) -> list[dict[str, any]]:
        """
        List files in R2 storage with optional prefix filter.

        Args:
            prefix: Prefix to filter files
            max_keys: Maximum number of files to return

        Returns:
            List of file information dictionaries
        """
        try:
            loop = asyncio.get_event_loop()

            def _list_objects():
                params = {"Bucket": self.config["bucket_name"], "MaxKeys": max_keys}
                if prefix:
                    params["Prefix"] = prefix

                return self.client.list_objects_v2(**params)

            response = await loop.run_in_executor(None, _list_objects)

            files = []
            for obj in response.get("Contents", []):
                files.append(
                    {
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "last_modified": obj["LastModified"],
                        "etag": obj["ETag"].strip('"'),
                    }
                )

            return files

        except ClientError as e:
            raise R2StorageError(f"Failed to list files: {e}")

    async def generate_presigned_url(
        self, file_key: str, expiration: int = 3600, method: str = "get_object"
    ) -> str:
        """
        Generate presigned URL for file access.

        Args:
            file_key: Key/path of the file
            expiration: URL expiration time in seconds
            method: HTTP method ('get_object' or 'put_object')

        Returns:
            Presigned URL string

        Raises:
            R2StorageError: If URL generation fails
        """
        try:
            loop = asyncio.get_event_loop()

            def _generate_url():
                return self.client.generate_presigned_url(
                    method,
                    Params={"Bucket": self.config["bucket_name"], "Key": file_key},
                    ExpiresIn=expiration,
                )

            url = await loop.run_in_executor(None, _generate_url)

            logger.info(f"Generated presigned URL for {file_key}, expires in {expiration}s")
            return url

        except ClientError as e:
            raise R2StorageError(f"Failed to generate presigned URL: {e}")

    async def generate_presigned_upload_url(
        self, file_key: str, expiration: int = 3600
    ) -> dict[str, any]:
        """
        Generate presigned upload URL and form data.

        Args:
            file_key: Key/path for the file to be uploaded
            expiration: URL expiration time in seconds

        Returns:
            Dict with URL and form fields for browser upload

        Raises:
            R2StorageError: If URL generation fails
        """
        try:
            loop = asyncio.get_event_loop()

            def _generate_post():
                return self.client.generate_presigned_post(
                    Bucket=self.config["bucket_name"], Key=file_key, ExpiresIn=expiration
                )

            post_data = await loop.run_in_executor(None, _generate_post)

            logger.info(f"Generated presigned upload URL for {file_key}")
            return post_data

        except ClientError as e:
            raise R2StorageError(f"Failed to generate presigned upload URL: {e}")

    def generate_file_key(
        self, document_id: str, document_type: str, file_format: str, version: str = "combined"
    ) -> str:
        """
        Generate standardized file key for document storage.

        Args:
            document_id: Unique document identifier
            document_type: Type of document (worksheet, notes, etc.)
            file_format: File format extension (pdf, docx, etc.)
            version: Document version (student, teacher, combined)

        Returns:
            Standardized file key string
        """
        # Sanitize inputs
        document_id = re.sub(r"[^a-zA-Z0-9_-]", "_", str(document_id))
        document_type = re.sub(r"[^a-zA-Z0-9_-]", "_", str(document_type))
        version = re.sub(r"[^a-zA-Z0-9_-]", "_", str(version))

        return f"documents/{document_id}/{version}.{file_format}"

    def validate_file_key(self, file_key: str) -> bool:
        """
        Validate file key format and security.

        Args:
            file_key: File key to validate

        Returns:
            True if valid, False otherwise
        """
        if not file_key or not isinstance(file_key, str):
            return False

        # Check for path traversal attempts
        if "../" in file_key or ".." in file_key:
            return False

        # Check for double slashes
        if "//" in file_key:
            return False

        # Check for spaces (not allowed in R2 keys)
        if " " in file_key:
            return False

        # Must not be empty and have reasonable length
        return not (len(file_key) == 0 or len(file_key) > 1024)

    def get_supported_formats(self) -> list[str]:
        """
        Get list of supported file formats.

        Returns:
            List of supported file format extensions
        """
        return ["pdf", "docx", "html", "txt", "json", "png", "jpg", "svg"]

    def _detect_content_type(self, file_key: str, content: bytes) -> Optional[str]:
        """
        Detect content type from file extension and content.

        Args:
            file_key: File key/path
            content: File content bytes

        Returns:
            MIME type string or None
        """
        # Try to detect from file extension
        content_type, _ = mimetypes.guess_type(file_key)

        if content_type:
            return content_type

        # Fallback: detect from content signatures
        content_detectors = [
            (lambda c: c.startswith(b"%PDF"), "application/pdf"),
            (
                lambda c: c.startswith(b"PK\x03\x04") and file_key.endswith(".docx"),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            (lambda c: c.startswith(b"PK\x03\x04"), "application/zip"),
            (lambda c: c.startswith(b"<!DOCTYPE html>") or c.startswith(b"<html"), "text/html"),
            (lambda c: c.startswith(b"{") or c.startswith(b"["), "application/json"),
        ]

        for detector, mime_type in content_detectors:
            if detector(content):
                return mime_type

        return "application/octet-stream"

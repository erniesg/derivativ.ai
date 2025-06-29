"""
Live R2 Connection Integration Tests
Tests real Cloudflare R2 operations with actual credentials
"""

import asyncio
import json
import logging
from datetime import datetime

import pytest

from src.api.dependencies import get_r2_storage_service
from src.core.config import get_settings

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_r2_live_connection():
    """Test basic R2 connection with real credentials"""
    logger.info("🔗 Testing R2 live connection...")

    settings = get_settings()
    logger.info(f"📋 Using bucket: {settings.cloudflare_r2_bucket_name}")
    logger.info(f"📋 Account ID: {settings.cloudflare_account_id}")
    logger.info(f"📋 Region: {settings.cloudflare_r2_region}")

    # Get R2 service
    r2_service = get_r2_storage_service()
    logger.info(f"✅ R2 service initialized: {type(r2_service).__name__}")

    # Test connection with bucket listing (if permissions allow)
    try:
        # Create a test file to verify connection
        test_key = f"test/connection_test_{datetime.now().isoformat()}.txt"
        test_content = f"R2 connection test at {datetime.now()}"

        logger.info(f"📤 Uploading test file: {test_key}")
        upload_result = await r2_service.upload_file(
            file_content=test_content.encode(),
            file_key=test_key,
            metadata={"test": "connection", "timestamp": datetime.now().isoformat()},
        )

        logger.info(f"✅ Upload successful: {upload_result}")
        assert upload_result["success"] is True

        # Verify file exists
        logger.info(f"🔍 Checking if file exists: {test_key}")
        exists = await r2_service.file_exists(test_key)
        logger.info(f"✅ File exists: {exists}")
        assert exists is True

        # Download and verify content
        logger.info(f"📥 Downloading file: {test_key}")
        download_result = await r2_service.download_file(test_key)
        logger.info(f"✅ Downloaded {download_result['file_size']} bytes")
        assert download_result["content"].decode() == test_content

        # Clean up test file
        logger.info(f"🗑️ Cleaning up test file: {test_key}")
        delete_result = await r2_service.delete_file(test_key)
        logger.info(f"✅ Delete successful: {delete_result}")
        assert delete_result["success"] is True

        logger.info("🎉 R2 live connection test PASSED!")

    except Exception as e:
        logger.error(f"❌ R2 connection test FAILED: {e}")
        raise


@pytest.mark.asyncio
async def test_r2_live_presigned_urls():
    """Test presigned URL generation and usage"""
    logger.info("🔗 Testing R2 presigned URLs...")

    settings = get_settings()
    r2_service = get_r2_storage_service()

    # Upload a test file first
    test_key = f"test/presigned_test_{datetime.now().isoformat()}.json"
    test_data = {
        "message": "Test data for presigned URL",
        "timestamp": datetime.now().isoformat(),
        "test_type": "presigned_url",
    }
    test_content = json.dumps(test_data, indent=2)

    logger.info(f"📤 Uploading test file for presigned URL test: {test_key}")
    await r2_service.upload_file(file_content=test_content.encode(), file_key=test_key)

    # Generate presigned download URL
    logger.info("🔗 Generating presigned download URL...")
    download_url = await r2_service.generate_presigned_url(
        file_key=test_key, expiration=3600, method="get_object"
    )

    logger.info(f"✅ Generated presigned URL: {download_url[:100]}...")
    assert download_url.startswith("https://")
    assert settings.cloudflare_account_id in download_url
    assert test_key in download_url

    # Test the presigned URL with HTTP client
    import aiohttp

    async with aiohttp.ClientSession() as session:
        logger.info("📥 Testing presigned download URL...")
        async with session.get(download_url) as response:
            logger.info(f"✅ Presigned URL response: {response.status}")
            assert response.status == 200

            downloaded_data = await response.text()
            parsed_data = json.loads(downloaded_data)
            assert parsed_data["message"] == test_data["message"]
            logger.info("✅ Presigned URL content verified")

    # Clean up
    logger.info("🗑️ Cleaning up presigned URL test file")
    await r2_service.delete_file(test_key)

    logger.info("🎉 R2 presigned URL test PASSED!")


@pytest.mark.asyncio
async def test_r2_live_file_formats():
    """Test R2 with different file formats (PDF, DOCX, HTML)"""
    logger.info("📋 Testing R2 with different file formats...")

    r2_service = get_r2_storage_service()
    test_files = []

    try:
        # Test different file formats
        file_tests = [
            {
                "key": f"test/formats/test_{datetime.now().isoformat()}.pdf",
                "content": b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\nxref\n0 3\n0000000000 65535 f \ntrailer\n<<\n/Size 3\n/Root 1 0 R\n>>\nstartxref\n0\n%%EOF",
                "content_type": "application/pdf",
            },
            {
                "key": f"test/formats/test_{datetime.now().isoformat()}.html",
                "content": b"<!DOCTYPE html><html><head><title>Test</title></head><body><h1>R2 Test HTML</h1></body></html>",
                "content_type": "text/html",
            },
            {
                "key": f"test/formats/test_{datetime.now().isoformat()}.json",
                "content": json.dumps(
                    {"test": "data", "timestamp": datetime.now().isoformat()}
                ).encode(),
                "content_type": "application/json",
            },
        ]

        for file_test in file_tests:
            logger.info(f"📤 Testing upload: {file_test['key']}")

            # Upload
            result = await r2_service.upload_file(
                file_content=file_test["content"],
                file_key=file_test["key"],
                metadata={"test": "file_format", "format": file_test["content_type"]},
            )
            assert result["success"] is True
            test_files.append(file_test["key"])

            # Verify exists
            exists = await r2_service.file_exists(file_test["key"])
            assert exists is True

            # Download and verify
            download_result = await r2_service.download_file(file_test["key"])
            assert download_result["content"] == file_test["content"]

            logger.info(f"✅ Format test passed: {file_test['content_type']}")

    finally:
        # Clean up all test files
        for key in test_files:
            logger.info(f"🗑️ Cleaning up: {key}")
            await r2_service.delete_file(key)

    logger.info("🎉 R2 file formats test PASSED!")


@pytest.mark.asyncio
async def test_r2_live_error_handling():
    """Test R2 error handling with real service"""
    logger.info("⚠️ Testing R2 error handling...")

    r2_service = get_r2_storage_service()

    # Test download non-existent file
    non_existent_key = f"test/non_existent_{datetime.now().isoformat()}.txt"
    logger.info(f"🔍 Testing download of non-existent file: {non_existent_key}")

    try:
        download_result = await r2_service.download_file(non_existent_key)
        raise AssertionError("Should have raised an exception for non-existent file")
    except Exception as e:
        logger.info(f"✅ Correctly raised exception: {type(e).__name__}: {e}")
        assert "not found" in str(e).lower() or "404" in str(e)

    # Test file_exists for non-existent file
    logger.info("🔍 Testing exists check for non-existent file")
    exists = await r2_service.file_exists(non_existent_key)
    assert exists is False
    logger.info("✅ Correctly returned False for non-existent file")

    # Test delete non-existent file
    logger.info("🗑️ Testing delete of non-existent file")
    delete_result = await r2_service.delete_file(non_existent_key)
    # This should not raise an error - idempotent delete
    assert delete_result in [True, False]  # Either is acceptable
    logger.info(f"✅ Delete non-existent file handled gracefully: {delete_result}")

    logger.info("🎉 R2 error handling test PASSED!")


if __name__ == "__main__":
    asyncio.run(test_r2_live_connection())
    asyncio.run(test_r2_live_presigned_urls())
    asyncio.run(test_r2_live_file_formats())
    asyncio.run(test_r2_live_error_handling())
    print("🎉 All R2 live tests PASSED!")

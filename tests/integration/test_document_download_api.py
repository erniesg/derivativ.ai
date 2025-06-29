"""
Document Download API Integration Tests
Tests FastAPI endpoints that serve R2 presigned URLs for frontend downloads
"""

import asyncio
import logging

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.config import get_settings

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = TestClient(app)


@pytest.mark.asyncio
async def test_document_generation_creates_r2_files():
    """Test that document generation creates files in R2 that can be downloaded"""
    logger.info("🧪 Testing document generation → R2 storage → download URLs")

    # Create a simple document generation request
    generation_request = {
        "document_type": "worksheet",
        "detail_level": "minimal",
        "title": "Test R2 Download Worksheet",
        "topic": "linear_equations",
        "tier": "Core",
        "grade_level": 7,
        "auto_include_questions": True,
        "max_questions": 2,
        "include_answers": True,
        "include_working": True,
        "custom_instructions": "Simple test worksheet for R2 download testing",
    }

    logger.info(f"📤 Generating document: {generation_request['title']}")

    # Generate document via API
    response = client.post("/api/generation/documents/generate", json=generation_request)

    logger.info(f"✅ Generation response status: {response.status_code}")

    if response.status_code != 200:
        logger.error(f"❌ Response error: {response.text}")

    assert response.status_code == 200
    generation_result = response.json()

    assert generation_result["success"] is True
    assert "document" in generation_result

    document = generation_result["document"]
    document_id = document["document_id"]

    logger.info(f"📄 Generated document ID: {document_id}")
    return document_id


@pytest.mark.asyncio
async def test_document_export_to_r2():
    """Test document export creates files in R2 with correct metadata"""
    logger.info("🧪 Testing document export to R2 storage")

    # First generate a document
    document_id = await test_document_generation_creates_r2_files()

    # Test export to different formats
    export_formats = ["html", "markdown"]  # Start with lightweight formats

    for format_type in export_formats:
        logger.info(f"📤 Testing export to {format_type.upper()}")

        export_request = {"document_id": document_id, "format": format_type, "version": "student"}

        response = client.post("/api/generation/documents/export", json=export_request)

        assert response.status_code == 200
        export_result = response.json()

        assert export_result["success"] is True
        assert "content" in export_result
        assert "r2_file_key" in export_result

        logger.info(f"✅ Export successful: {export_result['r2_file_key']}")


@pytest.mark.asyncio
async def test_download_url_generation():
    """Test generation of presigned download URLs for frontend"""
    logger.info("🧪 Testing presigned URL generation for downloads")

    # Generate and export a document first
    document_id = await test_document_generation_creates_r2_files()

    # Export to HTML
    export_request = {"document_id": document_id, "format": "html", "version": "student"}

    export_response = client.post("/api/generation/documents/export", json=export_request)
    export_result = export_response.json()
    r2_file_key = export_result["r2_file_key"]

    logger.info(f"📋 R2 file key: {r2_file_key}")

    # Test download URL generation endpoint
    download_response = client.get(
        f"/api/documents/{document_id}/download", params={"format": "html", "version": "student"}
    )

    assert download_response.status_code == 200
    download_result = download_response.json()

    assert "download_url" in download_result
    assert "expires_at" in download_result
    assert download_result["format"] == "html"
    assert download_result["version"] == "student"

    download_url = download_result["download_url"]
    logger.info(f"🔗 Generated download URL: {download_url[:100]}...")

    # Verify URL contains expected components
    settings = get_settings()
    assert settings.cloudflare_account_id in download_url
    assert "derivativ-documents" in download_url  # bucket name

    return download_url


@pytest.mark.asyncio
async def test_download_url_actually_works():
    """Test that generated presigned URLs actually download the files"""
    logger.info("🧪 Testing actual file download via presigned URL")

    import aiohttp

    # Get a download URL
    download_url = await test_download_url_generation()

    # Test downloading via HTTP client
    async with aiohttp.ClientSession() as session:
        logger.info("📥 Testing download via presigned URL...")
        async with session.get(download_url) as response:
            logger.info(f"✅ Download response: {response.status}")
            assert response.status == 200

            content = await response.text()
            assert len(content) > 0
            assert "Test R2 Download Worksheet" in content

            logger.info(f"✅ Downloaded {len(content)} characters")


@pytest.mark.asyncio
async def test_list_available_documents():
    """Test API endpoint that lists available documents for download"""
    logger.info("🧪 Testing document listing API")

    # Generate a test document first
    document_id = await test_document_generation_creates_r2_files()

    # Export it to multiple formats
    for format_type in ["html", "markdown"]:
        export_request = {"document_id": document_id, "format": format_type, "version": "student"}
        client.post("/api/generation/documents/export", json=export_request)

    # Test listing available documents
    response = client.get("/api/documents/available")

    assert response.status_code == 200
    documents = response.json()

    assert "documents" in documents
    assert len(documents["documents"]) > 0

    # Find our test document
    test_doc = None
    for doc in documents["documents"]:
        if doc["document_id"] == document_id:
            test_doc = doc
            break

    assert test_doc is not None
    assert "title" in test_doc
    assert "available_formats" in test_doc
    assert "html" in test_doc["available_formats"]
    assert "markdown" in test_doc["available_formats"]

    logger.info(f"✅ Found document with {len(test_doc['available_formats'])} formats")


@pytest.mark.asyncio
async def test_search_documents_by_topic():
    """Test searching for pre-generated documents by topic"""
    logger.info("🧪 Testing document search by topic")

    # Generate documents with specific topics
    topics_to_test = ["linear_equations", "quadratic_functions"]

    for topic in topics_to_test:
        generation_request = {
            "document_type": "worksheet",
            "detail_level": "minimal",
            "title": f"Test {topic.replace('_', ' ').title()} Worksheet",
            "topic": topic,
            "tier": "Core",
            "grade_level": 7,
            "auto_include_questions": True,
            "max_questions": 2,
        }

        response = client.post("/api/generation/documents/generate", json=generation_request)
        assert response.status_code == 200

        # Export to at least one format
        result = response.json()
        document_id = result["document"]["document_id"]

        export_request = {"document_id": document_id, "format": "html", "version": "student"}
        client.post("/api/generation/documents/export", json=export_request)

    # Test search by topic
    search_response = client.get("/api/documents/search", params={"topic": "linear_equations"})

    assert search_response.status_code == 200
    search_results = search_response.json()

    assert "documents" in search_results
    assert len(search_results["documents"]) > 0

    # Verify results contain linear_equations topic
    linear_docs = [
        doc for doc in search_results["documents"] if "linear_equations" in doc.get("topic", "")
    ]
    assert len(linear_docs) > 0

    logger.info(f"✅ Found {len(linear_docs)} documents for linear_equations")


@pytest.mark.asyncio
async def test_bulk_download_preparation():
    """Test preparing multiple documents for bulk download"""
    logger.info("🧪 Testing bulk download preparation")

    # Generate multiple documents
    document_ids = []
    for i in range(2):
        generation_request = {
            "document_type": "worksheet",
            "detail_level": "minimal",
            "title": f"Bulk Test Document {i+1}",
            "topic": "algebra",
            "tier": "Core",
            "grade_level": 7,
            "max_questions": 2,
        }

        response = client.post("/api/generation/documents/generate", json=generation_request)
        result = response.json()
        document_id = result["document"]["document_id"]
        document_ids.append(document_id)

        # Export each document
        export_request = {"document_id": document_id, "format": "html", "version": "student"}
        client.post("/api/generation/documents/export", json=export_request)

    # Test bulk download preparation
    bulk_request = {"document_ids": document_ids, "format": "html", "version": "student"}

    response = client.post("/api/documents/bulk-download", json=bulk_request)

    assert response.status_code == 200
    bulk_result = response.json()

    assert "download_urls" in bulk_result
    assert len(bulk_result["download_urls"]) == len(document_ids)

    for url_info in bulk_result["download_urls"]:
        assert "document_id" in url_info
        assert "download_url" in url_info
        assert "filename" in url_info

    logger.info(f"✅ Prepared {len(bulk_result['download_urls'])} documents for bulk download")


if __name__ == "__main__":
    asyncio.run(test_document_generation_creates_r2_files())
    asyncio.run(test_document_export_to_r2())
    asyncio.run(test_download_url_generation())
    asyncio.run(test_download_url_actually_works())
    asyncio.run(test_list_available_documents())
    asyncio.run(test_search_documents_by_topic())
    asyncio.run(test_bulk_download_preparation())
    print("🎉 All R2 download API tests PASSED!")

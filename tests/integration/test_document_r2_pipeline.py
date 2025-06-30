"""
Document to R2 Pipeline Integration Tests
Tests the complete workflow: Document Generation → Export → R2 Storage → Download
"""

import asyncio
import json
import logging
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.dependencies import get_r2_storage_service
from src.core.config import get_settings
from src.database.supabase_repository import QuestionRepository
from src.models.document_models import DetailLevel, DocumentGenerationRequest, DocumentType, Tier
from src.services.document_export_service import DocumentExportService
from src.services.document_generation_service import DocumentGenerationService
from src.services.llm_factory import LLMFactory
from src.services.prompt_manager import PromptManager

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_document_generation_to_r2_storage():  # noqa: PLR0915
    """Test complete document generation → R2 storage pipeline"""
    logger.info("🚀 Testing Document Generation → R2 Storage pipeline...")

    # Get services
    settings = get_settings()
    r2_service = get_r2_storage_service()

    # Create a simple document generation request
    generation_request = DocumentGenerationRequest(
        document_type=DocumentType.WORKSHEET,
        detail_level=DetailLevel.MINIMAL,
        title="R2 Pipeline Test Worksheet",
        topic="linear_equations",
        tier=Tier.CORE,
        grade_level=7,
        auto_include_questions=True,
        max_questions=2,
        include_answers=True,
        include_working=True,
        custom_instructions="Simple test worksheet for R2 integration",
    )

    logger.info(f"📋 Document request: {generation_request.title}")
    logger.info(f"📋 Topic: {generation_request.topic}")
    logger.info(f"📋 Detail level: {generation_request.detail_level}")

    try:
        # Initialize document generation service with proper dependencies
        mock_question_repository = MagicMock(spec=QuestionRepository)
        mock_question_repository.search_questions = AsyncMock(return_value=[])

        mock_llm_factory = MagicMock(spec=LLMFactory)
        mock_llm_service = AsyncMock()
        mock_llm_service.generate.return_value = json.dumps(
            {
                "title": "Test Worksheet",
                "document_type": "worksheet",
                "detail_level": 1,
                "sections": [
                    {"title": "Practice Questions", "content": "Sample content"},
                    {"title": "Solutions", "content": "Sample solutions"},
                ],
                "estimated_duration": "30 minutes",
                "total_questions": 2,
            }
        )
        mock_llm_factory.get_service.return_value = mock_llm_service

        prompt_manager = PromptManager()

        doc_gen_service = DocumentGenerationService(
            question_repository=mock_question_repository,
            llm_factory=mock_llm_factory,
            prompt_manager=prompt_manager,
        )

        # Generate document
        logger.info("📝 Generating document...")
        generation_result = await doc_gen_service.generate_document(generation_request)

        logger.info(f"✅ Document generated: {generation_result.success}")
        assert generation_result.success is True
        assert generation_result.document is not None

        generated_doc = generation_result.document
        logger.info(f"📄 Generated document ID: {generated_doc.document_id}")
        logger.info(f"📄 Sections: {len(generated_doc.sections)}")
        logger.info(f"📄 Questions: {generated_doc.total_questions}")

        # Test export to different formats
        export_service = DocumentExportService()
        export_formats = ["html", "markdown"]  # Start with lightweight formats

        uploaded_files = []

        for format_type in export_formats:
            logger.info(f"📤 Exporting to {format_type.upper()}...")

            # Convert GeneratedDocument to dict for export service
            document_dict = {
                "document_id": generated_doc.document_id,
                "title": generated_doc.title,
                "document_type": generated_doc.document_type,
                "sections": [
                    {"title": section.title, "content": section.content_data}
                    for section in generated_doc.sections
                ],
            }

            # Export document
            export_result = await export_service.export_document(
                document=document_dict,
                format_type=format_type,
                version="student",  # Test student version
            )

            assert export_result["success"] is True
            logger.info(f"✅ Export successful: {len(export_result['content'])} characters")

            # Upload to R2
            file_key = (
                f"test/pipeline/{generated_doc.document_id}_{format_type}_student.{format_type}"
            )
            logger.info(f"📤 Uploading to R2: {file_key}")

            upload_result = await r2_service.upload_file(
                file_content=export_result["content"].encode("utf-8"),
                file_key=file_key,
                metadata={
                    "document_id": generated_doc.document_id,
                    "format": format_type,
                    "version": "student",
                    "title": generated_doc.title,
                    "test": "pipeline_integration",
                    "timestamp": datetime.now().isoformat(),
                },
            )

            assert upload_result["success"] is True
            logger.info(f"✅ Upload to R2 successful: {upload_result['file_key']}")
            uploaded_files.append(file_key)

            # Verify file exists in R2
            exists = await r2_service.file_exists(file_key)
            assert exists is True
            logger.info(f"✅ File verified in R2: {file_key}")

            # Test download from R2
            logger.info("📥 Testing download from R2...")
            download_result = await r2_service.download_file(file_key)
            assert download_result["success"] is True

            downloaded_content = download_result["content"].decode("utf-8")
            assert len(downloaded_content) == len(export_result["content"])
            logger.info(f"✅ Download successful: {download_result['file_size']} bytes")

            # Test presigned URL generation
            logger.info("🔗 Testing presigned URL...")
            presigned_url = await r2_service.generate_presigned_url(
                file_key=file_key, expiration=3600, method="get_object"
            )

            assert presigned_url.startswith("https://")
            assert settings.cloudflare_account_id in presigned_url
            logger.info(f"✅ Presigned URL generated: {presigned_url[:100]}...")

        logger.info("🎉 Document Generation → R2 Storage pipeline test PASSED!")
        return uploaded_files

    except Exception as e:
        logger.error(f"❌ Pipeline test FAILED: {e}")
        raise

    finally:
        # Clean up uploaded files
        logger.info("🗑️ Cleaning up test files...")
        for file_key in uploaded_files:
            try:
                delete_result = await r2_service.delete_file(file_key)
                logger.info(f"✅ Cleaned up: {file_key}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to clean up {file_key}: {e}")


@pytest.mark.asyncio
async def test_student_teacher_version_storage():
    """Test storing both student and teacher versions of the same document"""
    logger.info("👥 Testing Student/Teacher version storage...")

    r2_service = get_r2_storage_service()

    # Create a simple test document data
    test_document_id = f"test_doc_{datetime.now().isoformat()}"
    base_content = {
        "title": "Test Worksheet",
        "questions": [
            {"question": "What is 2 + 2?", "answer": "4"},
            {"question": "What is 5 × 3?", "answer": "15"},
        ],
    }

    uploaded_files = []

    try:
        # Create student version (questions only)
        student_content = {
            "title": base_content["title"],
            "questions": [{"question": q["question"]} for q in base_content["questions"]],
        }

        # Create teacher version (questions + answers)
        teacher_content = base_content.copy()

        versions = [
            {"version": "student", "content": student_content},
            {"version": "teacher", "content": teacher_content},
        ]

        for version_data in versions:
            version = version_data["version"]
            content = version_data["content"]

            logger.info(f"📤 Uploading {version} version...")

            file_key = f"test/versions/{test_document_id}_{version}.json"
            content_str = json.dumps(content, indent=2)

            upload_result = await r2_service.upload_file(
                file_content=content_str.encode("utf-8"),
                file_key=file_key,
                metadata={
                    "document_id": test_document_id,
                    "version": version,
                    "format": "json",
                    "test": "version_storage",
                    "timestamp": datetime.now().isoformat(),
                },
            )

            assert upload_result["success"] is True
            uploaded_files.append(file_key)
            logger.info(f"✅ {version.capitalize()} version uploaded: {upload_result['file_key']}")

        # Verify both versions exist and have different content
        student_key = f"test/versions/{test_document_id}_student.json"
        teacher_key = f"test/versions/{test_document_id}_teacher.json"

        # Download both versions
        student_download = await r2_service.download_file(student_key)
        teacher_download = await r2_service.download_file(teacher_key)

        student_data = json.loads(student_download["content"].decode("utf-8"))
        teacher_data = json.loads(teacher_download["content"].decode("utf-8"))

        # Verify student version doesn't have answers
        assert "answer" not in str(student_data)
        logger.info("✅ Student version correctly excludes answers")

        # Verify teacher version has answers
        assert "answer" in str(teacher_data)
        logger.info("✅ Teacher version correctly includes answers")

        # Verify both have same number of questions
        assert len(student_data["questions"]) == len(teacher_data["questions"])
        logger.info("✅ Both versions have same number of questions")

        logger.info("🎉 Student/Teacher version storage test PASSED!")

    except Exception as e:
        logger.error(f"❌ Version storage test FAILED: {e}")
        raise

    finally:
        # Clean up
        logger.info("🗑️ Cleaning up version test files...")
        for file_key in uploaded_files:
            try:
                await r2_service.delete_file(file_key)
                logger.info(f"✅ Cleaned up: {file_key}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to clean up {file_key}: {e}")


@pytest.mark.asyncio
async def test_bulk_document_operations():
    """Test bulk upload and download operations"""
    logger.info("📦 Testing bulk document operations...")

    r2_service = get_r2_storage_service()

    # Create multiple test documents
    test_docs = []
    uploaded_files = []

    try:
        for i in range(3):
            doc_id = f"bulk_test_{i}_{datetime.now().isoformat()}"
            content = {
                "id": doc_id,
                "title": f"Bulk Test Document {i+1}",
                "content": f"This is test document number {i+1}",
                "created": datetime.now().isoformat(),
            }
            test_docs.append({"id": doc_id, "content": content})

        logger.info(f"📤 Uploading {len(test_docs)} documents...")

        # Upload all documents
        for doc in test_docs:
            file_key = f"test/bulk/{doc['id']}.json"
            content_str = json.dumps(doc["content"], indent=2)

            upload_result = await r2_service.upload_file(
                file_content=content_str.encode("utf-8"),
                file_key=file_key,
                metadata={
                    "document_id": doc["id"],
                    "test": "bulk_operations",
                    "timestamp": datetime.now().isoformat(),
                },
            )

            assert upload_result["success"] is True
            uploaded_files.append(file_key)
            logger.info(f"✅ Uploaded: {file_key}")

        # Verify all files exist
        logger.info("🔍 Verifying all files exist...")
        for file_key in uploaded_files:
            exists = await r2_service.file_exists(file_key)
            assert exists is True

        logger.info("✅ All bulk uploads verified")

        # Test concurrent downloads
        logger.info("📥 Testing concurrent downloads...")
        download_tasks = [r2_service.download_file(file_key) for file_key in uploaded_files]

        download_results = await asyncio.gather(*download_tasks)

        assert len(download_results) == len(uploaded_files)
        for result in download_results:
            assert result["success"] is True

        logger.info("✅ Concurrent downloads successful")
        logger.info("🎉 Bulk document operations test PASSED!")

    except Exception as e:
        logger.error(f"❌ Bulk operations test FAILED: {e}")
        raise

    finally:
        # Clean up
        logger.info("🗑️ Cleaning up bulk test files...")
        for file_key in uploaded_files:
            try:
                await r2_service.delete_file(file_key)
                logger.info(f"✅ Cleaned up: {file_key}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to clean up {file_key}: {e}")


if __name__ == "__main__":
    asyncio.run(test_document_generation_to_r2_storage())
    asyncio.run(test_student_teacher_version_storage())
    asyncio.run(test_bulk_document_operations())
    print("🎉 All Document → R2 pipeline tests PASSED!")

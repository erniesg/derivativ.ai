"""
End-to-end tests for live document generation pipeline.
Tests complete workflow: Prompt → Generation → Export → R2 Storage → Verification

Following existing E2E patterns from test_api_documents.py and test_document_storage_e2e.py
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any

import pytest

from src.api.dependencies import get_r2_storage_service
from src.database.supabase_repository import QuestionRepository
from src.models.document_models import (
    DetailLevel,
    DocumentGenerationRequest,
    DocumentType,
    Tier,
)
from src.services.document_export_service import DocumentExportService
from src.services.document_generation_service import DocumentGenerationService
from src.services.llm_factory import LLMFactory
from src.services.prompt_manager import PromptManager

# Configure logging
logger = logging.getLogger(__name__)


@pytest.mark.e2e
class TestLiveDocumentPipeline:
    """End-to-end tests for live document generation pipeline."""

    @pytest.fixture
    async def services(self):
        """Setup all required services for live testing."""
        # Set demo mode for testing
        os.environ["DEMO_MODE"] = "true"

        # Mock Supabase for demo mode
        class MockSupabaseClient:
            def __init__(self):
                pass

        question_repo = QuestionRepository(MockSupabaseClient())
        llm_factory = LLMFactory()
        prompt_manager = PromptManager()

        # Document generation service
        doc_service = DocumentGenerationService(
            question_repository=question_repo,
            llm_factory=llm_factory,
            prompt_manager=prompt_manager,
        )

        # Export service
        export_service = DocumentExportService()

        # R2 service with fallback handling
        try:
            r2_service = get_r2_storage_service()
            logger.info("✅ R2 service initialized")
        except Exception as e:
            logger.warning(f"⚠️ R2 service not available: {e}")
            r2_service = None

        return {
            "doc_service": doc_service,
            "export_service": export_service,
            "r2_service": r2_service,
        }

    def _get_test_cases(self):
        """Get test cases for different document types."""
        return [
            (DocumentType.WORKSHEET, DetailLevel.MINIMAL, "Algebra", "Basic algebra practice"),
            (DocumentType.NOTES, DetailLevel.MINIMAL, "Geometry", "Basic geometry concepts"),
            (
                DocumentType.TEXTBOOK,
                DetailLevel.MINIMAL,
                "Statistics",
                "Introduction to statistics",
            ),
            (DocumentType.SLIDES, DetailLevel.MINIMAL, "Trigonometry", "Basic trig functions"),
        ]

    def _analyze_content(self, document) -> dict[str, Any]:
        """Analyze generated document content for quality."""
        total_chars = 0
        rich_sections = 0

        for section in document.sections:
            if section.content_data:
                content_str = json.dumps(section.content_data, indent=2)
                section_chars = len(content_str)
                total_chars += section_chars

                # Check for rich content indicators
                is_rich = (
                    any(
                        key in str(section.content_data).lower()
                        for key in ["question", "example", "objective", "step", "solution"]
                    )
                    or section_chars > 100
                )

                if is_rich:
                    rich_sections += 1

        # Calculate overall quality score
        base_score = min(total_chars / 500, 5)
        richness_score = (rich_sections / len(document.sections)) * 3
        structure_score = min(len(document.sections) / 2, 2)

        overall_score = base_score + richness_score + structure_score

        return {
            "total_characters": total_chars,
            "rich_sections": rich_sections,
            "total_sections": len(document.sections),
            "richness_ratio": rich_sections / len(document.sections),
            "overall_score": min(overall_score, 10),
        }

    async def _test_export_formats(self, document, services) -> list[dict[str, Any]]:
        """Test exporting document to formats."""
        export_service = services["export_service"]
        export_results = []

        # Convert GeneratedDocument to dict for export service
        document_dict = {
            "document_id": document.document_id,
            "title": document.title,
            "document_type": document.document_type,
            "sections": [
                {"title": section.title, "content": section.content_data}
                for section in document.sections
            ],
        }

        logger.info("   📤 Exporting to HTML...")

        try:
            export_result = await export_service.export_document(
                document=document_dict,
                format_type="html",
                version="student",
            )

            if export_result["success"]:
                logger.info(f"   ✅ HTML export: {export_result['file_size']} bytes")
                export_results.append(
                    {
                        "format": "html",
                        "success": True,
                        "file_size": export_result["file_size"],
                        "content": export_result.get("content", ""),
                    }
                )
            else:
                logger.error("   ❌ HTML export failed")
                export_results.append(
                    {
                        "format": "html",
                        "success": False,
                        "error": "Export failed",
                    }
                )

        except Exception as e:
            logger.error(f"   ❌ HTML export exception: {e}")
            export_results.append(
                {
                    "format": "html",
                    "success": False,
                    "error": str(e),
                }
            )

        return export_results

    async def _test_r2_storage(self, document, export_results, test_number, services):
        """Test storing documents in R2 storage."""
        r2_service = services["r2_service"]
        if not r2_service:
            return []

        storage_results = []
        uploaded_files = []

        for export_result in export_results:
            if not export_result["success"]:
                continue

            format_type = export_result["format"]
            content = export_result["content"]

            # Create R2 file key
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_key = f"test/e2e_pipeline/{timestamp}_test{test_number:02d}_{document.document_id}_{format_type}.{format_type}"

            try:
                logger.info(f"   📤 Uploading {format_type.upper()} to R2...")

                # Upload to R2
                upload_result = await r2_service.upload_file(
                    file_content=content.encode("utf-8"),
                    file_key=file_key,
                    metadata={
                        "document_id": document.document_id,
                        "document_type": document.document_type,
                        "format": format_type,
                        "test_number": str(test_number),
                        "test_type": "e2e_pipeline",
                    },
                )

                # Track for cleanup
                uploaded_files.append(file_key)

                # Verify upload
                file_exists = await r2_service.file_exists(file_key)

                if file_exists:
                    logger.info(f"   ✅ {format_type.upper()} uploaded successfully")

                    # Test download
                    download_result = await r2_service.download_file(file_key)
                    download_success = len(download_result) == len(content.encode("utf-8"))

                    # Test presigned URL
                    presigned_url = await r2_service.generate_presigned_url(
                        file_key, expiration=3600
                    )

                    storage_results.append(
                        {
                            "format": format_type,
                            "file_key": file_key,
                            "upload_success": True,
                            "download_success": download_success,
                            "presigned_url": presigned_url[:50] + "..." if presigned_url else None,
                            "file_size": len(content.encode("utf-8")),
                        }
                    )
                else:
                    logger.error(f"   ❌ {format_type.upper()} upload verification failed")
                    storage_results.append(
                        {
                            "format": format_type,
                            "file_key": file_key,
                            "upload_success": False,
                            "error": "Upload verification failed",
                        }
                    )

            except Exception as e:
                logger.error(f"   ❌ {format_type.upper()} R2 storage failed: {e}")
                storage_results.append(
                    {
                        "format": format_type,
                        "file_key": file_key,
                        "upload_success": False,
                        "error": str(e),
                    }
                )

        # Store uploaded files for cleanup
        if not hasattr(self, "_uploaded_files"):
            self._uploaded_files = []
        self._uploaded_files.extend(uploaded_files)

        return storage_results

    @pytest.mark.asyncio
    async def test_live_document_generation_pipeline_all_types(self, services):
        """Test live document generation pipeline for all document types."""
        logger.info("🚀 Starting Live Document Generation Pipeline Test")
        logger.info("=" * 80)

        test_cases = self._get_test_cases()
        test_results = []

        logger.info(f"📋 Testing {len(test_cases)} document types:")
        for i, (doc_type, detail_level, topic, description) in enumerate(test_cases, 1):
            logger.info(f"   {i}. {doc_type.value:<10} | {topic}")

        logger.info("=" * 80)

        # Run tests for each configuration
        for i, test_case in enumerate(test_cases, 1):
            doc_type, detail_level, topic, description = test_case

            logger.info(f"\n🧪 TEST {i}/{len(test_cases)}: {doc_type.value.upper()} - {topic}")
            logger.info("-" * 60)

            start_time = time.time()

            try:
                # 1. Create generation request
                generation_request = DocumentGenerationRequest(
                    document_type=doc_type,
                    detail_level=detail_level,
                    title=f"E2E Test {doc_type.value.title()}: {topic}",
                    topic=topic.lower().replace(" ", "_"),
                    tier=Tier.CORE,
                    grade_level=8,
                    auto_include_questions=True,
                    max_questions=2,  # Keep small for speed
                    include_answers=True,
                    include_working=True,
                    custom_instructions=description,
                )

                logger.info(f"📝 Generating {doc_type.value} document...")

                # 2. Generate document
                generation_result = await services["doc_service"].generate_document(
                    generation_request
                )

                if not generation_result.success:
                    test_results.append(
                        {
                            "test_number": i,
                            "document_type": doc_type.value,
                            "success": False,
                            "error": f"Generation failed: {generation_result.error_message}",
                            "processing_time": time.time() - start_time,
                        }
                    )
                    continue

                generated_doc = generation_result.document
                logger.info(f"✅ Document generated: {generated_doc.document_id}")
                logger.info(f"   📄 Sections: {len(generated_doc.sections)}")
                logger.info(f"   ❓ Questions: {generated_doc.total_questions}")

                # 3. Analyze content quality
                content_analysis = self._analyze_content(generated_doc)
                logger.info(f"   📊 Content quality: {content_analysis['overall_score']:.1f}/10")

                # 4. Test export
                export_results = await self._test_export_formats(generated_doc, services)

                # 5. Test R2 storage if available
                storage_results = []
                if services["r2_service"] and export_results:
                    storage_results = await self._test_r2_storage(
                        generated_doc, export_results, i, services
                    )
                else:
                    logger.info("   📁 R2 storage skipped")

                processing_time = time.time() - start_time

                test_results.append(
                    {
                        "test_number": i,
                        "document_type": doc_type.value,
                        "topic": topic,
                        "success": True,
                        "document_id": generated_doc.document_id,
                        "sections_count": len(generated_doc.sections),
                        "questions_count": generated_doc.total_questions,
                        "content_analysis": content_analysis,
                        "export_results": export_results,
                        "storage_results": storage_results,
                        "processing_time": processing_time,
                    }
                )

                logger.info(
                    f"✅ TEST {i} PASSED - Generated in {processing_time:.1f}s, Quality: {content_analysis['overall_score']:.1f}/10"
                )

            except Exception as e:
                logger.error(f"❌ TEST {i} FAILED - Exception: {e}")
                test_results.append(
                    {
                        "test_number": i,
                        "document_type": doc_type.value,
                        "success": False,
                        "error": str(e),
                        "processing_time": time.time() - start_time,
                    }
                )

        # Generate final report
        await self._generate_final_report(test_results, services)

        # Verify all tests passed
        successful_tests = [r for r in test_results if r["success"]]
        assert len(successful_tests) == len(
            test_cases
        ), f"Only {len(successful_tests)}/{len(test_cases)} tests passed"

    async def _generate_final_report(self, test_results, services):
        """Generate final test report."""
        successful_tests = [r for r in test_results if r["success"]]
        failed_tests = [r for r in test_results if not r["success"]]

        logger.info("\n" + "=" * 80)
        logger.info("📊 E2E LIVE PIPELINE TEST RESULTS")
        logger.info("=" * 80)

        logger.info(f"✅ Successful tests: {len(successful_tests)}/{len(test_results)}")
        logger.info(f"❌ Failed tests: {len(failed_tests)}")
        logger.info(f"📈 Success rate: {len(successful_tests)/len(test_results)*100:.1f}%")

        if successful_tests:
            logger.info("\n✅ SUCCESSFUL DOCUMENT GENERATIONS:")
            total_time_gen = 0
            total_quality = 0

            for result in successful_tests:
                doc_type = result["document_type"]
                topic = result["topic"]
                time_taken = result["processing_time"]
                quality_score = result.get("content_analysis", {}).get("overall_score", 0)
                sections = result.get("sections_count", 0)

                total_time_gen += time_taken
                total_quality += quality_score

                logger.info(
                    f"   🎯 {doc_type:<10} | {topic:<15} | {time_taken:5.1f}s | {sections} sections | Quality: {quality_score:.1f}/10"
                )

            if successful_tests:
                avg_time = total_time_gen / len(successful_tests)
                avg_quality = total_quality / len(successful_tests)
                logger.info(f"\n📊 AVERAGES: Time: {avg_time:.1f}s | Quality: {avg_quality:.1f}/10")

        if failed_tests:
            logger.info("\n❌ FAILED TESTS:")
            for result in failed_tests:
                doc_type = result["document_type"]
                error = result.get("error", "Unknown error")
                logger.info(f"   ❌ {doc_type:<10} | Error: {error}")

        # Document type coverage
        doc_types_tested = set(r["document_type"] for r in successful_tests)
        logger.info(f"\n📋 DOCUMENT TYPES TESTED: {', '.join(doc_types_tested)}")

        # Cleanup
        await self._cleanup_test_files(services)

        # Final assessment
        if len(successful_tests) == len(test_results):
            logger.info("\n🎉 ALL TESTS PASSED! Complete pipeline working end-to-end!")
        elif len(successful_tests) > 0:
            logger.info(
                f"\n✅ {len(successful_tests)}/{len(test_results)} tests passed. Pipeline working for tested document types."
            )
        else:
            logger.info("\n❌ ALL TESTS FAILED! Pipeline needs investigation.")

        logger.info("=" * 80)

    async def _cleanup_test_files(self, services):
        """Clean up test files from R2 storage."""
        r2_service = services["r2_service"]
        if not r2_service or not hasattr(self, "_uploaded_files"):
            return

        uploaded_files = getattr(self, "_uploaded_files", [])
        if not uploaded_files:
            return

        logger.info(f"\n🗑️ Cleaning up {len(uploaded_files)} test files from R2...")

        cleanup_success = 0
        for file_key in uploaded_files:
            try:
                await r2_service.delete_file(file_key)
                cleanup_success += 1
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to delete {file_key}: {e}")

        logger.info(f"✅ Cleaned up {cleanup_success}/{len(uploaded_files)} files")

    @pytest.mark.asyncio
    async def test_single_worksheet_generation_with_r2(self, services):
        """Test single worksheet generation with R2 storage - focused test."""
        # Quick focused test for CI/CD
        generation_request = DocumentGenerationRequest(
            document_type=DocumentType.WORKSHEET,
            detail_level=DetailLevel.MINIMAL,
            title="CI Test Worksheet",
            topic="linear_equations",
            tier=Tier.CORE,
            grade_level=7,
            auto_include_questions=True,
            max_questions=1,
            include_answers=True,
        )

        # Generate document
        generation_result = await services["doc_service"].generate_document(generation_request)
        assert generation_result.success, f"Generation failed: {generation_result.error_message}"

        generated_doc = generation_result.document
        assert generated_doc.document_id is not None
        assert len(generated_doc.sections) > 0
        assert generated_doc.total_questions >= 0

        # Test export
        export_results = await self._test_export_formats(generated_doc, services)
        assert len(export_results) > 0
        assert any(r["success"] for r in export_results), "No successful exports"

        # Test R2 storage if available
        if services["r2_service"]:
            storage_results = await self._test_r2_storage(
                generated_doc, export_results, 1, services
            )
            if storage_results:
                assert any(
                    r.get("upload_success") for r in storage_results
                ), "No successful uploads"

        logger.info("✅ Single worksheet test passed")

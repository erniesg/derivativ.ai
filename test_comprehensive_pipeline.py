#!/usr/bin/env python3
"""
Comprehensive End-to-End Pipeline Test

Tests document generation across all types and detail levels,
stores results in R2 for manual review and verification.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import get_settings  # noqa: E402
from src.database.supabase_repository import QuestionRepository  # noqa: E402
from src.models.document_models import (  # noqa: E402
    DetailLevel,
    DocumentGenerationRequest,
    DocumentType,
)
from src.services.document_generation_service import DocumentGenerationService  # noqa: E402
from src.services.llm_factory import LLMFactory  # noqa: E402
from src.services.prompt_manager import PromptManager  # noqa: E402
from src.services.r2_storage_service import R2StorageService  # noqa: E402

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)8s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensivePipelineTest:
    """Comprehensive test suite for document generation pipeline."""

    def __init__(self):
        self.settings = get_settings()
        self.setup_services()
        self.test_results = []
        self.stored_documents = []

    def setup_services(self):
        """Initialize all required services."""
        # Mock supabase client for demo mode
        class MockSupabaseClient:
            def __init__(self):
                pass

        self.question_repo = QuestionRepository(MockSupabaseClient())
        self.llm_factory = LLMFactory()
        self.prompt_manager = PromptManager()

        self.doc_service = DocumentGenerationService(
            question_repository=self.question_repo,
            llm_factory=self.llm_factory,
            prompt_manager=self.prompt_manager
        )

        # Initialize R2 storage service
        self.r2_service = self._setup_r2_service()

        # Also create local backup directory
        self.output_dir = Path("test_output")
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.output_dir / "documents").mkdir(exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)

    def _setup_r2_service(self):
        """Set up R2 storage service with proper configuration."""
        try:
            r2_config = {
                "account_id": self.settings.cloudflare_account_id,
                "access_key_id": self.settings.cloudflare_r2_access_key_id,
                "secret_access_key": self.settings.cloudflare_r2_secret_access_key,
                "bucket_name": self.settings.cloudflare_r2_bucket_name,
                "region": self.settings.cloudflare_r2_region,
            }

            # Check if R2 is properly configured
            missing_keys = [k for k, v in r2_config.items() if not v]
            if missing_keys:
                logger.warning(f"R2 not configured (missing: {missing_keys}). Documents will be stored locally only.")
                return None

            return R2StorageService(r2_config)

        except Exception as e:
            logger.warning(f"Failed to initialize R2 service: {e}. Documents will be stored locally only.")
            return None

    async def run_comprehensive_test(self):
        """Run comprehensive test across all document types and detail levels."""
        print("🧪 Starting Comprehensive End-to-End Pipeline Test")
        print("=" * 60)

        # Test combinations
        test_cases = [
            # Worksheets - all detail levels
            (DocumentType.WORKSHEET, DetailLevel.MINIMAL, "Linear Equations", "Basic linear equation solving"),
            (DocumentType.WORKSHEET, DetailLevel.MEDIUM, "Quadratic Functions", "Quadratic function analysis"),
            (DocumentType.WORKSHEET, DetailLevel.COMPREHENSIVE, "Trigonometry", "Advanced trigonometric identities"),

            # Notes - different detail levels
            (DocumentType.NOTES, DetailLevel.BASIC, "Algebra Basics", "Introduction to algebraic concepts"),
            (DocumentType.NOTES, DetailLevel.MEDIUM, "Coordinate Geometry", "Points, lines, and circles"),
            (DocumentType.NOTES, DetailLevel.COMPREHENSIVE, "Calculus Introduction", "Limits and derivatives"),

            # Textbook - different detail levels
            (DocumentType.TEXTBOOK, DetailLevel.MEDIUM, "Statistics", "Data analysis and probability"),
            (DocumentType.TEXTBOOK, DetailLevel.COMPREHENSIVE, "Vectors", "Vector operations and applications"),

            # Slides - different detail levels
            (DocumentType.SLIDES, DetailLevel.BASIC, "Percentages", "Percentage calculations"),
            (DocumentType.SLIDES, DetailLevel.MEDIUM, "Sequences", "Arithmetic and geometric sequences"),
        ]

        successful_tests = 0
        total_tests = len(test_cases)

        for i, (doc_type, detail_level, topic, description) in enumerate(test_cases, 1):
            print(f"\n📄 Test {i}/{total_tests}: {doc_type.value.title()} - {detail_level.name} - {topic}")
            print("-" * 50)

            try:
                # Create request
                request = DocumentGenerationRequest(
                    title=f"{topic} {doc_type.value.title()}",
                    document_type=doc_type,
                    detail_level=detail_level,
                    topic=topic,
                    grade_level=8,
                    max_questions=3,
                    auto_include_questions=False,
                    custom_instructions=f"Focus on {description}"
                )

                # Generate document
                start_time = datetime.now()
                result = await self.doc_service.generate_document(request)
                generation_time = (datetime.now() - start_time).total_seconds()

                if result.success:
                    # Analyze content
                    content_analysis = self.analyze_content(result.document)

                    # Store in R2 and locally
                    storage_info = await self.store_document(
                        result.document, doc_type, detail_level, topic
                    )

                    print(f"✅ SUCCESS - Generated in {generation_time:.2f}s")
                    print(f"   Sections: {len(result.document.sections)}")
                    print(f"   Content size: {content_analysis['total_chars']} chars")
                    print(f"   Rich sections: {content_analysis['rich_sections']}")
                    if storage_info["r2_url"]:
                        print(f"   🌐 R2 URL: {storage_info['r2_url']}")
                    print(f"   📁 Local: {storage_info['local_path']}")

                    # Record success
                    test_result = {
                        "test_id": f"{doc_type.value}_{detail_level.name}_{topic.replace(' ', '_')}",
                        "document_type": doc_type.value,
                        "detail_level": detail_level.name,
                        "detail_level_int": detail_level.value,
                        "topic": topic,
                        "success": True,
                        "generation_time": generation_time,
                        "sections_count": len(result.document.sections),
                        "content_analysis": content_analysis,
                        "r2_url": storage_info["r2_url"],
                        "local_path": str(storage_info["local_path"]),
                        "html_path": str(storage_info["html_path"]),
                        "timestamp": datetime.now().isoformat()
                    }

                    self.test_results.append(test_result)
                    self.stored_documents.append({
                        "title": result.document.title,
                        "r2_url": storage_info["r2_url"],
                        "local_path": str(storage_info["local_path"]),
                        "html_path": str(storage_info["html_path"]),
                        "type": doc_type.value,
                        "detail": detail_level.name
                    })

                    successful_tests += 1

                else:
                    print(f"❌ FAILED - {result.error_message}")

                    # Record failure
                    test_result = {
                        "test_id": f"{doc_type.value}_{detail_level.name}_{topic.replace(' ', '_')}",
                        "document_type": doc_type.value,
                        "detail_level": detail_level.name,
                        "topic": topic,
                        "success": False,
                        "error": result.error_message,
                        "generation_time": generation_time,
                        "timestamp": datetime.now().isoformat()
                    }

                    self.test_results.append(test_result)

            except Exception as e:
                print(f"❌ EXCEPTION - {e!s}")
                logger.exception(f"Test failed for {doc_type.value} {detail_level.name}")

        # Generate summary
        await self.generate_test_summary(successful_tests, total_tests)

        return successful_tests, total_tests

    def analyze_content(self, document):
        """Analyze generated document content for richness and quality."""
        total_chars = 0
        rich_sections = 0
        section_analysis = []

        for section in document.sections:
            section_chars = 0

            if section.content_data:
                content_str = json.dumps(section.content_data, indent=2)
                section_chars = len(content_str)
                total_chars += section_chars

                # Check for rich content indicators
                is_rich = any(
                    key in section.content_data
                    for key in ["questions", "examples", "objectives", "steps", "solution"]
                ) or section_chars > 200

                if is_rich:
                    rich_sections += 1

                section_analysis.append({
                    "title": section.title,
                    "type": section.content_type,
                    "chars": section_chars,
                    "is_rich": is_rich
                })

        return {
            "total_chars": total_chars,
            "rich_sections": rich_sections,
            "total_sections": len(document.sections),
            "section_details": section_analysis,
            "avg_chars_per_section": total_chars // max(len(document.sections), 1)
        }

    async def store_document(self, document, doc_type, detail_level, topic):
        """Store generated document in R2 and locally for manual review."""
        try:
            # Create document content for storage
            document_content = {
                "title": document.title,
                "document_type": doc_type.value,
                "detail_level": detail_level.name,
                "detail_level_int": detail_level.value,
                "topic": topic,
                "generated_at": document.generated_at,
                "estimated_duration": document.estimated_duration,
                "sections": []
            }

            # Add section content
            for section in document.sections:
                section_data = {
                    "title": section.title,
                    "content_type": section.content_type,
                    "order_index": section.order_index,
                    "content_data": section.content_data
                }
                document_content["sections"].append(section_data)

            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_topic = topic.replace(" ", "_").replace("/", "_")
            filename = f"{timestamp}_{doc_type.value}_{detail_level.name}_{safe_topic}.json"

            # Store locally
            file_path = self.output_dir / "documents" / filename

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(document_content, f, indent=2, ensure_ascii=False)

            # Create readable HTML version
            html_filename = filename.replace('.json', '.html')
            html_path = self.output_dir / "documents" / html_filename

            self.create_html_version(document_content, html_path)

            # Store in R2 if available
            r2_url = None
            if self.r2_service:
                try:
                    content_bytes = json.dumps(document_content, indent=2, ensure_ascii=False).encode('utf-8')

                    r2_result = await self.r2_service.upload_file(
                        file_content=content_bytes,
                        file_key=f"test_documents/{filename}",
                        metadata={
                            "test_type": "comprehensive_pipeline",
                            "document_type": doc_type.value,
                            "detail_level": detail_level.name,
                            "topic": topic,
                            "generated_at": document.generated_at
                        }
                    )

                    if r2_result.get("success"):
                        # Generate R2 URL manually since it's not returned
                        account_id = self.settings.cloudflare_account_id
                        bucket_name = self.settings.cloudflare_r2_bucket_name
                        r2_url = f"https://{account_id}.r2.cloudflarestorage.com/{bucket_name}/test_documents/{filename}"
                        logger.info(f"Document stored in R2: {r2_url}")
                    else:
                        logger.error(f"Failed to store in R2: {r2_result}")

                except Exception as e:
                    logger.error(f"R2 upload error: {e}")

            return {
                "local_path": file_path,
                "html_path": html_path,
                "r2_url": r2_url
            }

        except Exception as e:
            logger.exception(f"Error storing document: {e}")
            return {
                "local_path": None,
                "html_path": None,
                "r2_url": None
            }

    def create_html_version(self, document_content, html_path):
        """Create a readable HTML version of the document."""
        try:
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{document_content['title']}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
        .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #e9ecef; border-radius: 8px; }}
        .section-title {{ color: #495057; font-size: 1.5em; font-weight: bold; margin-bottom: 15px; }}
        .content {{ background: #f8f9fa; padding: 15px; border-radius: 5px; }}
        .meta {{ color: #6c757d; font-size: 0.9em; }}
        pre {{ background: #f1f3f4; padding: 10px; border-radius: 5px; white-space: pre-wrap; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{document_content['title']}</h1>
        <div class="meta">
            <strong>Type:</strong> {document_content['document_type'].title()} |
            <strong>Detail Level:</strong> {document_content['detail_level']} ({document_content['detail_level_int']}/10) |
            <strong>Topic:</strong> {document_content['topic']} |
            <strong>Duration:</strong> {document_content['estimated_duration']} min
        </div>
        <div class="meta">
            <strong>Generated:</strong> {document_content['generated_at']}
        </div>
    </div>
"""

            for section in document_content['sections']:
                html_content += f"""
    <div class="section">
        <div class="section-title">{section['title']}</div>
        <div class="meta">Type: {section['content_type']} | Order: {section['order_index']}</div>
        <div class="content">
            <pre>{json.dumps(section['content_data'], indent=2, ensure_ascii=False)}</pre>
        </div>
    </div>
"""

            html_content += """
</body>
</html>
"""

            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

        except Exception as e:
            logger.error(f"Failed to create HTML version: {e}")

    async def generate_test_summary(self, successful_tests, total_tests):
        """Generate and store comprehensive test summary."""
        print("\n📊 COMPREHENSIVE TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")

        # Calculate statistics
        generation_times = [r['generation_time'] for r in self.test_results if r['success']]
        content_sizes = [r['content_analysis']['total_chars'] for r in self.test_results if r['success']]

        if generation_times:
            print("\n⏱️  PERFORMANCE METRICS:")
            print(f"Average Generation Time: {sum(generation_times)/len(generation_times):.2f}s")
            print(f"Fastest: {min(generation_times):.2f}s")
            print(f"Slowest: {max(generation_times):.2f}s")

            print("\n📄 CONTENT METRICS:")
            print(f"Average Content Size: {sum(content_sizes)/len(content_sizes):.0f} chars")
            print(f"Largest Document: {max(content_sizes):,} chars")
            print(f"Smallest Document: {min(content_sizes):,} chars")

        # Show stored documents
        print(f"\n📁 STORED DOCUMENTS ({len(self.stored_documents)} files):")
        for doc in self.stored_documents:
            print(f"   • {doc['title']} ({doc['type']}, {doc['detail']})")
            if doc.get('r2_url'):
                print(f"     R2: {doc['r2_url']}")
            print(f"     Local: {doc['local_path']}")
            print(f"     HTML: {doc['html_path']}")

        # Create summary file
        summary_data = {
            "test_run": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": (successful_tests/total_tests)*100,
                "performance_metrics": {
                    "avg_generation_time": sum(generation_times)/len(generation_times) if generation_times else 0,
                    "min_generation_time": min(generation_times) if generation_times else 0,
                    "max_generation_time": max(generation_times) if generation_times else 0,
                },
                "content_metrics": {
                    "avg_content_size": sum(content_sizes)/len(content_sizes) if content_sizes else 0,
                    "min_content_size": min(content_sizes) if content_sizes else 0,
                    "max_content_size": max(content_sizes) if content_sizes else 0,
                }
            },
            "test_results": self.test_results,
            "stored_documents": self.stored_documents
        }

        # Store summary in R2 if available
        if self.r2_service:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                summary_filename = f"test_summaries/comprehensive_test_{timestamp}.json"

                summary_bytes = json.dumps(summary_data, indent=2, ensure_ascii=False).encode('utf-8')

                result = await self.r2_service.upload_file(
                    file_content=summary_bytes,
                    file_key=summary_filename,
                    metadata={
                        "test_type": "comprehensive_summary",
                        "total_tests": str(total_tests),
                        "successful_tests": str(successful_tests)
                    }
                )

                if result.get("success"):
                    account_id = self.settings.cloudflare_account_id
                    bucket_name = self.settings.cloudflare_r2_bucket_name
                    summary_url = f"https://{account_id}.r2.cloudflarestorage.com/{bucket_name}/{summary_filename}"
                    print(f"\n📋 Test Summary Stored: {summary_url}")
                else:
                    print(f"\n⚠️  Failed to store summary in R2: {result}")
            except Exception as e:
                print(f"\n⚠️  Error storing summary in R2: {e}")

        # Also store summary locally
        local_summary_path = self.output_dir / "summaries" / f"comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(local_summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"📋 Local Summary: {local_summary_path}")

        return summary_data


async def main():
    """Run comprehensive pipeline test."""
    # Set demo mode
    os.environ["DEMO_MODE"] = "true"

    try:
        tester = ComprehensivePipelineTest()
        successful, total = await tester.run_comprehensive_test()

        print(f"\n🎯 FINAL RESULT: {successful}/{total} tests passed")

        if successful == total:
            print("🎉 ALL TESTS PASSED - Pipeline is production ready!")
            sys.exit(0)
        else:
            print("⚠️  Some tests failed - review results for issues")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        logging.exception("Test suite exception")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

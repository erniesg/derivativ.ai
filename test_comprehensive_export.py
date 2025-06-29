#!/usr/bin/env python3
"""
Comprehensive Document Export Test

Tests document generation and export to all formats (PDF, DOCX, HTML, Markdown)
with both student and teacher versions, storing all results in R2.
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
    DocumentVersion,
    ExportFormat,
)
from src.services.document_export_service import DocumentExportService  # noqa: E402
from src.services.document_generation_service import DocumentGenerationService  # noqa: E402
from src.services.llm_factory import LLMFactory  # noqa: E402
from src.services.prompt_manager import PromptManager  # noqa: E402
from src.services.r2_storage_service import R2StorageService  # noqa: E402

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)8s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class ComprehensiveExportTest:
    """Comprehensive test for document generation and export with R2 storage."""

    def __init__(self):
        self.settings = get_settings()
        self.setup_services()
        self.test_results = []
        self.exported_files = []

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
            prompt_manager=self.prompt_manager,
        )

        # Initialize export service
        self.export_service = DocumentExportService()

        # Initialize R2 storage service
        self.r2_service = self._setup_r2_service()

        # Create local output directory
        self.output_dir = Path("test_exports")
        self.output_dir.mkdir(exist_ok=True)

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
                logger.warning(
                    f"R2 not configured (missing: {missing_keys}). Documents will be stored locally only."
                )
                return None

            return R2StorageService(r2_config)

        except Exception as e:
            logger.warning(f"Failed to initialize R2 service: {e}. Documents will be stored locally only.")
            return None

    async def run_comprehensive_export_test(self):
        """Run comprehensive test of document generation and export."""
        print("🧪 Starting Comprehensive Document Export Test")
        print("=" * 60)

        # Test cases for different document types
        test_cases = [
            (DocumentType.WORKSHEET, DetailLevel.MEDIUM, "Linear Equations", "Solving linear equations"),
            (DocumentType.NOTES, DetailLevel.COMPREHENSIVE, "Quadratic Functions", "Quadratic function properties"),
        ]

        successful_tests = 0
        total_operations = 0

        for doc_type, detail_level, topic, description in test_cases:
            print(f"\n📄 Testing: {doc_type.value.title()} - {detail_level.name} - {topic}")
            print("-" * 50)

            try:
                # Step 1: Generate document
                print("🚀 Generating document...")
                document_result = await self._generate_document(doc_type, detail_level, topic, description)

                if not document_result["success"]:
                    print(f"❌ Document generation failed: {document_result['error']}")
                    continue

                document = document_result["document"]
                print(f"✅ Document generated in {document_result['generation_time']:.2f}s")

                # Step 2: Export to all formats and versions
                export_formats = [ExportFormat.PDF, ExportFormat.DOCX, ExportFormat.HTML, ExportFormat.MARKDOWN]
                versions = [DocumentVersion.STUDENT, DocumentVersion.TEACHER]

                for export_format in export_formats:
                    for version in versions:
                        total_operations += 1
                        print(f"\n📤 Exporting {export_format.value.upper()} ({version.value})...")

                        export_result = await self.export_service.export_document(
                            document=document,
                            format_type=export_format.value,
                            version=version.value,
                            store_in_r2=True,
                            r2_service=self.r2_service,
                        )

                        if export_result["success"]:
                            successful_tests += 1
                            print(f"✅ Exported {export_format.value.upper()} ({version.value})")

                            # Store export info
                            export_info = {
                                "document_type": doc_type.value,
                                "detail_level": detail_level.name,
                                "topic": topic,
                                "format": export_format.value,
                                "version": version.value,
                                "file_size": export_result.get("file_size", 0),
                                "local_path": export_result.get("local_file_path"),
                                "r2_key": export_result.get("r2_file_key"),
                                "timestamp": datetime.now().isoformat(),
                            }

                            if export_result.get("r2_file_key"):
                                account_id = self.settings.cloudflare_account_id
                                bucket_name = self.settings.cloudflare_r2_bucket_name
                                r2_url = f"https://{account_id}.r2.cloudflarestorage.com/{bucket_name}/{export_result['r2_file_key']}"
                                export_info["r2_url"] = r2_url
                                print(f"   🌐 R2: {r2_url}")

                            if export_result.get("local_file_path"):
                                print(f"   📁 Local: {export_result['local_file_path']}")

                            self.exported_files.append(export_info)

                        else:
                            print(f"❌ Export failed: {export_result.get('error', 'Unknown error')}")

            except Exception as e:
                print(f"❌ Test failed: {e}")
                logger.exception(f"Test failed for {doc_type.value} {detail_level.name}")

        # Generate summary
        await self.generate_export_summary(successful_tests, total_operations)

        return successful_tests, total_operations

    async def _generate_document(self, doc_type, detail_level, topic, description):
        """Generate a document for testing."""
        request = DocumentGenerationRequest(
            title=f"{topic} {doc_type.value.title()}",
            document_type=doc_type,
            detail_level=detail_level,
            topic=topic,
            grade_level=8,
            max_questions=3,
            auto_include_questions=False,
            custom_instructions=f"Focus on {description}",
            generate_versions=[DocumentVersion.STUDENT, DocumentVersion.TEACHER],
            export_formats=[ExportFormat.PDF, ExportFormat.DOCX, ExportFormat.HTML, ExportFormat.MARKDOWN],
        )

        start_time = datetime.now()
        result = await self.doc_service.generate_document(request)
        generation_time = (datetime.now() - start_time).total_seconds()

        if result.success:
            # Convert GeneratedDocument to dict for export service
            document_dict = {
                "document_id": result.document.document_id,
                "title": result.document.title,
                "document_type": result.document.document_type.value,
                "generated_at": result.document.generated_at,
                "total_estimated_minutes": result.document.estimated_duration,
                "actual_detail_level": result.document.detail_level.value,
                "generation_request": request,
                "content_structure": {
                    "blocks": [
                        {
                            "block_type": section.content_type,
                            "content": section.content_data,
                            "estimated_minutes": 5,  # Default
                        }
                        for section in result.document.sections
                    ]
                },
            }

            return {"success": True, "document": document_dict, "generation_time": generation_time}
        else:
            return {"success": False, "error": result.error_message, "generation_time": generation_time}

    async def generate_export_summary(self, successful_exports, total_operations):
        """Generate and store comprehensive export summary."""
        print("\n📊 COMPREHENSIVE EXPORT SUMMARY")
        print("=" * 60)
        print(f"Total Export Operations: {total_operations}")
        print(f"Successful Exports: {successful_exports}")
        print(f"Failed Exports: {total_operations - successful_exports}")
        print(f"Success Rate: {(successful_exports/total_operations)*100:.1f}%")

        # Analyze export results by format and version
        format_stats = {}
        version_stats = {}

        for export_info in self.exported_files:
            format_name = export_info["format"]
            version_name = export_info["version"]

            if format_name not in format_stats:
                format_stats[format_name] = {"count": 0, "total_size": 0}
            if version_name not in version_stats:
                version_stats[version_name] = {"count": 0, "formats": set()}

            format_stats[format_name]["count"] += 1
            format_stats[format_name]["total_size"] += export_info.get("file_size", 0)

            version_stats[version_name]["count"] += 1
            version_stats[version_name]["formats"].add(format_name)

        print("\n📄 FORMAT STATISTICS:")
        for format_name, stats in format_stats.items():
            avg_size = stats["total_size"] / stats["count"] if stats["count"] > 0 else 0
            print(f"   {format_name.upper()}: {stats['count']} files, avg size: {avg_size:.0f} bytes")

        print("\n👥 VERSION STATISTICS:")
        for version_name, stats in version_stats.items():
            formats_list = ", ".join(sorted(stats["formats"]))
            print(f"   {version_name.upper()}: {stats['count']} files, formats: {formats_list}")

        # Show exported files
        print(f"\n📁 EXPORTED FILES ({len(self.exported_files)} files):")
        for export_info in self.exported_files:
            print(
                f"   • {export_info['topic']} ({export_info['format'].upper()}, {export_info['version']})"
            )
            if export_info.get("r2_url"):
                print(f"     R2: {export_info['r2_url']}")
            if export_info.get("local_path"):
                print(f"     Local: {export_info['local_path']}")

        # Create summary file
        summary_data = {
            "export_test_run": {
                "timestamp": datetime.now().isoformat(),
                "total_operations": total_operations,
                "successful_exports": successful_exports,
                "success_rate": (successful_exports / total_operations) * 100 if total_operations > 0 else 0,
                "format_statistics": format_stats,
                "version_statistics": {
                    k: {"count": v["count"], "formats": list(v["formats"])} for k, v in version_stats.items()
                },
            },
            "exported_files": self.exported_files,
        }

        # Store summary locally
        summary_path = self.output_dir / f"export_test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"\n📋 Summary stored: {summary_path}")

        # Store summary in R2 if available
        if self.r2_service:
            try:
                summary_bytes = json.dumps(summary_data, indent=2, ensure_ascii=False).encode("utf-8")

                result = await self.r2_service.upload_file(
                    file_content=summary_bytes,
                    file_key=f"export_test_summaries/comprehensive_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    metadata={
                        "test_type": "comprehensive_export",
                        "total_operations": str(total_operations),
                        "successful_exports": str(successful_exports),
                    },
                )

                if result.get("success"):
                    account_id = self.settings.cloudflare_account_id
                    bucket_name = self.settings.cloudflare_r2_bucket_name
                    summary_r2_key = f"export_test_summaries/comprehensive_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    summary_url = f"https://{account_id}.r2.cloudflarestorage.com/{bucket_name}/{summary_r2_key}"
                    print(f"📋 R2 Summary: {summary_url}")

            except Exception as e:
                print(f"⚠️  Error storing summary in R2: {e}")

        return summary_data


async def main():
    """Run comprehensive export test."""
    # Set demo mode
    os.environ["DEMO_MODE"] = "true"

    try:
        tester = ComprehensiveExportTest()
        successful, total = await tester.run_comprehensive_export_test()

        print(f"\n🎯 FINAL RESULT: {successful}/{total} export operations passed")

        if successful == total and total > 0:
            print("🎉 ALL EXPORTS SUCCESSFUL - System ready for teacher workflow!")
            sys.exit(0)
        elif successful > 0:
            print(f"⚠️  {successful}/{total} exports successful - Some formats may need attention")
            sys.exit(0)
        else:
            print("❌ All exports failed - System needs debugging")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        logging.exception("Export test suite exception")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
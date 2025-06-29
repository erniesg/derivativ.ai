#!/usr/bin/env python3
"""
Test script to generate a worksheet and export it to PDF for verification.
"""

import asyncio
import logging
import os
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set demo mode
os.environ["DEMO_MODE"] = "false"

from src.models.document_generation_v2 import DocumentGenerationRequestV2  # noqa: E402
from src.models.document_models import DocumentType  # noqa: E402
from src.models.enums import Tier, TopicName  # noqa: E402
from src.services.document_export_service import DocumentExportService  # noqa: E402
from src.services.document_generation_service_v2 import DocumentGenerationServiceV2  # noqa: E402
from src.services.llm_factory import LLMFactory  # noqa: E402
from src.services.prompt_manager import PromptManager  # noqa: E402


async def test_worksheet_generation_and_export():
    """Generate a worksheet and export it to PDF."""
    print("🧪 Testing worksheet generation and PDF export...")
    print("=" * 60)

    try:
        # 1. Create generation request
        request = DocumentGenerationRequestV2(
            title="Linear Equations Practice Worksheet",
            document_type=DocumentType.WORKSHEET,
            topic=TopicName.ALGEBRA_AND_GRAPHS,
            subtopics=["Linear equations", "Graphing", "Solving systems"],
            detail_level=6,
            target_duration_minutes=45,
            grade_level=9,
            difficulty="medium",
            tier=Tier.CORE,
            num_questions=6,
            include_questions=True,
            custom_instructions="Focus on real-world applications",
        )

        print("📝 Generation request created:")
        print(f"   Title: {request.title}")
        print(f"   Type: {request.document_type.value}")
        print(f"   Topic: {request.topic.value}")
        print(f"   Detail Level: {request.detail_level}")
        print(f"   Duration: {request.target_duration_minutes} minutes")

        # 2. Create document generation service
        llm_factory = LLMFactory()
        prompt_manager = PromptManager()
        doc_gen_service = DocumentGenerationServiceV2(
            llm_factory=llm_factory, prompt_manager=prompt_manager
        )

        # 3. Generate document
        print("\n🔧 Generating document...")
        start_time = datetime.now()

        result = await doc_gen_service.generate_document(request)

        generation_time = (datetime.now() - start_time).total_seconds()
        print(f"⏱️ Generation completed in {generation_time:.2f}s")

        if not result.success:
            print(f"❌ Generation failed: {result.error_message}")
            return False

        document = result.document
        print("✅ Document generated successfully!")
        print(f"   Title: {document.title}")
        print(f"   Blocks: {len(document.content_structure.blocks)}")
        print(f"   Estimated duration: {document.total_estimated_minutes} minutes")

        # 4. Show content blocks summary
        print("\n📋 Content blocks:")
        for i, block in enumerate(document.content_structure.blocks, 1):
            print(f"   {i}. {block.block_type} ({block.estimated_minutes} min)")

            # Show questions count if it's a practice block
            if block.block_type == "practice_questions":
                questions = block.content.get("questions", [])
                print(f"      → {len(questions)} questions")

        # 5. Export to PDF
        print("\n📄 Exporting to PDF...")

        # Create export service with current directory for easy access
        export_service = DocumentExportService(output_directory="./exported_documents")

        # Convert document to dictionary for export
        document_dict = {
            "title": document.title,
            "content_structure": {
                "blocks": [
                    {
                        "block_type": block.block_type,
                        "content": block.content,
                        "estimated_minutes": block.estimated_minutes,
                    }
                    for block in document.content_structure.blocks
                ]
            },
            "total_estimated_minutes": document.total_estimated_minutes,
            "actual_detail_level": document.actual_detail_level,
            "generation_request": request,
        }

        # Export to both PDF and DOCX (both student and teacher versions)
        exported_files = await export_service.export_document(
            document=document_dict,
            formats=["pdf", "docx"],
            create_versions=True,
            output_prefix=f"algebra_worksheet_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        # 6. Show export results
        print("✅ Export completed!")

        for format_type, file_paths in exported_files.items():
            print(f"\n📁 {format_type.upper()} files:")
            for file_path in file_paths:
                print(f"   → {file_path}")

                # Check file size
                try:
                    file_size = os.path.getsize(file_path)
                    print(f"     Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
                except Exception as e:
                    print(f"     Error checking size: {e}")

        # 7. Show export directory
        export_dir = export_service.get_export_directory()
        print(f"\n📂 Export directory: {export_dir}")
        print("   You can now open the PDF files to view the generated worksheets!")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    asyncio.run(test_worksheet_generation_and_export())

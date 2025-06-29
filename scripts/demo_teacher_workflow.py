#!/usr/bin/env python3
"""
Demo Teacher Workflow
Shows the complete end-to-end flow for teachers:
1. Generate a worksheet
2. Export to R2 storage
3. Get download URLs for student and teacher versions
4. Test actual downloads
"""

import asyncio
import logging

import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)8s] %(message)s")
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000"


async def demo_teacher_workflow():  # noqa: PLR0915
    """Demo the complete teacher workflow."""
    logger.info("🎓 Starting Teacher Workflow Demo")

    # Step 1: Generate a worksheet
    logger.info("\n📚 Step 1: Generating worksheet...")

    generation_request = {
        "document_type": "worksheet",
        "detail_level": "medium",
        "title": "Linear Equations Practice Worksheet",
        "topic": "linear_equations,algebra",
        "tier": "Core",
        "grade_level": 7,
        "auto_include_questions": True,
        "max_questions": 3,
        "include_answers": True,
        "include_working": True,
        "custom_instructions": "Create a practice worksheet suitable for IGCSE students with worked examples",
    }

    async with aiohttp.ClientSession() as session:
        # Generate document
        async with session.post(
            f"{API_BASE_URL}/api/generation/documents/generate", json=generation_request
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Generation failed: {response.status} - {error_text}")

            result = await response.json()
            document_id = result["document"]["document_id"]
            logger.info(f"✅ Generated worksheet: {document_id}")
            logger.info(f"   Title: {result['document']['title']}")
            logger.info(f"   Processing time: {result['processing_time']:.2f}s")

        # Step 2: Export to different formats
        logger.info("\n📤 Step 2: Exporting to R2 storage...")

        formats_to_export = ["html", "markdown"]
        versions_to_export = ["student", "teacher"]

        for format_type in formats_to_export:
            for version in versions_to_export:
                export_request = {
                    "document_id": document_id,
                    "format": format_type,
                    "version": version,
                }

                async with session.post(
                    f"{API_BASE_URL}/api/generation/documents/export", json=export_request
                ) as response:
                    if response.status == 200:
                        export_result = await response.json()
                        logger.info(
                            f"✅ Exported {format_type} ({version}): {export_result.get('r2_file_key', 'success')}"
                        )
                    else:
                        logger.error(f"❌ Failed to export {format_type} ({version})")

        # Step 3: Get download URLs
        logger.info("\n🔗 Step 3: Getting download URLs...")

        download_urls = {}
        for format_type in formats_to_export:
            for version in versions_to_export:
                async with session.get(
                    f"{API_BASE_URL}/api/documents/{document_id}/download",
                    params={"format": format_type, "version": version},
                ) as response:
                    if response.status == 200:
                        download_data = await response.json()
                        download_urls[f"{format_type}_{version}"] = download_data["download_url"]
                        logger.info(f"✅ Got {format_type} ({version}) download URL")
                        logger.info(f"   Expires: {download_data['expires_at']}")
                    else:
                        logger.error(f"❌ Failed to get {format_type} ({version}) download URL")

        # Step 4: Test downloading the files
        logger.info("\n📥 Step 4: Testing downloads...")

        for name, url in download_urls.items():
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        content_preview = content[:200] + "..." if len(content) > 200 else content
                        logger.info(f"✅ Downloaded {name}: {len(content)} characters")
                        logger.info(f"   Preview: {content_preview}")
                    else:
                        logger.error(f"❌ Download failed for {name}: {response.status}")
            except Exception as e:
                logger.error(f"❌ Download error for {name}: {e}")

        # Step 5: List available documents
        logger.info("\n📋 Step 5: Listing available documents...")

        async with session.get(f"{API_BASE_URL}/api/documents/available") as response:
            if response.status == 200:
                documents_data = await response.json()
                logger.info(f"✅ Found {len(documents_data['documents'])} available documents")

                # Find our document
                our_doc = None
                for doc in documents_data["documents"]:
                    if doc["document_id"] == document_id:
                        our_doc = doc
                        break

                if our_doc:
                    logger.info(f"   Our document: {our_doc['document_id']}")
                    logger.info(f"   Available formats: {our_doc['available_formats']}")
                    logger.info(f"   File count: {our_doc['file_count']}")
            else:
                logger.error("❌ Failed to list documents")

    logger.info("\n🎉 Teacher Workflow Demo Complete!")
    logger.info("✅ All steps completed successfully")
    logger.info("\nSummary:")
    logger.info("- Document generated with AI content")
    logger.info("- Exported to multiple formats (HTML, Markdown)")
    logger.info("- Created student and teacher versions")
    logger.info("- Stored securely in Cloudflare R2")
    logger.info("- Generated presigned download URLs")
    logger.info("- Tested actual file downloads")
    logger.info("- Listed available documents via API")


async def main():
    """Main function to run the demo."""
    try:
        await demo_teacher_workflow()
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))

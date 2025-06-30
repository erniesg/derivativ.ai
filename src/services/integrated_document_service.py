"""
Integrated Document Service

Combines markdown generation, pandoc conversion, and R2 storage into a
single pipeline that eliminates all the complex JSON structure issues.

Pipeline: Generate MD → Pandoc Convert → Store in R2 → Return URLs
"""

import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.core.config import get_settings
from src.models.markdown_generation_models import MarkdownGenerationRequest
from src.services.markdown_document_service import MarkdownDocumentService
from src.services.pandoc_service import PandocError, PandocService
from src.services.r2_storage_service import R2StorageService

logger = logging.getLogger(__name__)


class IntegratedDocumentService:
    """Service that generates documents as markdown and immediately converts/stores all formats."""

    def __init__(
        self,
        markdown_service: MarkdownDocumentService,
        pandoc_service: PandocService,
        r2_service: R2StorageService
    ):
        self.markdown_service = markdown_service
        self.pandoc_service = pandoc_service
        self.r2_service = r2_service
        self.settings = get_settings()

    async def generate_and_store_all_formats(
        self,
        request: MarkdownGenerationRequest,
        custom_instructions: Optional[str] = None
    ) -> dict[str, Any]:
        """Generate document and immediately create + store all formats in R2.

        Returns:
            {
                "success": bool,
                "document_id": str,
                "formats": {
                    "markdown": {"success": bool, "r2_url": str, "file_key": str},
                    "html": {"success": bool, "r2_url": str, "file_key": str},
                    "pdf": {"success": bool, "r2_url": str, "file_key": str},
                    "docx": {"success": bool, "r2_url": str, "file_key": str}
                },
                "metadata": dict,
                "generation_info": dict
            }
        """
        logger.info(f"🚀 Starting integrated document generation: {request.document_type} - {request.topic}")

        try:
            # Step 1: Generate clean markdown
            markdown_result = await self.markdown_service.generate_markdown_document(
                request, custom_instructions
            )

            if not markdown_result["success"]:
                return {
                    "success": False,
                    "error": f"Markdown generation failed: {markdown_result.get('error')}",
                    "formats": {}
                }

            markdown_content = markdown_result["markdown_content"]
            metadata = markdown_result["metadata"]
            generation_info = markdown_result["generation_info"]

            # Generate document ID for consistent file keys
            document_id = self._generate_document_id(request)

            logger.info(f"✅ Generated {len(markdown_content)} chars of markdown")

            # Step 2: Convert to all formats using pandoc
            conversion_results = await self._convert_to_all_formats(
                markdown_content,
                metadata["title"],
                metadata
            )

            # Step 3: Store all formats in R2
            storage_results = await self._store_all_formats_in_r2(
                conversion_results,
                document_id,
                metadata
            )

            # Step 4: Generate presigned URLs for downloads
            final_results = await self._generate_download_urls(storage_results)

            return {
                "success": True,
                "document_id": document_id,
                "formats": final_results,
                "metadata": metadata,
                "generation_info": generation_info,
                "markdown_content": markdown_content  # For frontend display
            }

        except Exception as e:
            logger.error(f"❌ Integrated document generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "formats": {}
            }

    async def _convert_to_all_formats(
        self,
        markdown_content: str,
        title: str,
        metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert markdown to all supported formats."""

        results = {
            "markdown": {"success": True, "content": markdown_content},
            "html": {"success": False},
            "pdf": {"success": False},
            "docx": {"success": False}
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Convert to HTML
                html_path = await self.pandoc_service.convert_markdown_to_html(
                    markdown_content,
                    temp_path / "document.html"
                )

                with open(html_path, encoding="utf-8") as f:
                    html_content = f.read()

                results["html"] = {
                    "success": True,
                    "content": html_content,
                    "file_path": str(html_path)
                }

                logger.info("✅ Converted to HTML")

            except PandocError as e:
                logger.warning(f"⚠️ HTML conversion failed: {e}")
                results["html"]["error"] = str(e)

            try:
                # Convert to PDF
                pdf_path = await self.pandoc_service.convert_markdown_to_pdf(
                    markdown_content,
                    temp_path / "document.pdf",
                    template_options={
                        "fontsize": "11pt",
                        "geometry": "margin=2cm",
                        "linestretch": "1.25"
                    }
                )

                with open(pdf_path, "rb") as f:
                    pdf_content = f.read()

                results["pdf"] = {
                    "success": True,
                    "content": pdf_content,
                    "file_path": str(pdf_path)
                }

                logger.info("✅ Converted to PDF")

            except PandocError as e:
                logger.warning(f"⚠️ PDF conversion failed: {e}")
                results["pdf"]["error"] = str(e)

            try:
                # Convert to DOCX
                docx_path = await self.pandoc_service.convert_markdown_to_docx(
                    markdown_content,
                    temp_path / "document.docx"
                )

                with open(docx_path, "rb") as f:
                    docx_content = f.read()

                results["docx"] = {
                    "success": True,
                    "content": docx_content,
                    "file_path": str(docx_path)
                }

                logger.info("✅ Converted to DOCX")

            except PandocError as e:
                logger.warning(f"⚠️ DOCX conversion failed: {e}")
                results["docx"]["error"] = str(e)

        return results

    async def _store_all_formats_in_r2(
        self,
        conversion_results: dict[str, Any],
        document_id: str,
        metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Store all successful conversions in R2."""

        storage_results = {}

        for format_name, conversion_result in conversion_results.items():
            if not conversion_result["success"]:
                storage_results[format_name] = conversion_result
                continue

            try:
                # Generate R2 file key
                file_extension = self._get_file_extension(format_name)
                file_key = self.r2_service.generate_file_key(
                    document_id=document_id,
                    document_type="document",
                    file_format=file_extension,
                    version="student"  # Default to student version
                )

                # Prepare content for upload
                content = conversion_result["content"]
                file_content = content.encode("utf-8") if isinstance(content, str) else content

                # Upload to R2
                upload_result = await self.r2_service.upload_file(
                    file_content=file_content,
                    file_key=file_key,
                    metadata={
                        "document_id": document_id,
                        "format": format_name,
                        "title": metadata.get("title", ""),
                        "topic": metadata.get("topic", ""),
                        "generated_at": datetime.now().isoformat(),
                        "content_type": self._get_content_type(format_name)
                    }
                )

                if upload_result["success"]:
                    storage_results[format_name] = {
                        "success": True,
                        "file_key": file_key,
                        "upload_id": upload_result["upload_id"],
                        "size": len(file_content)
                    }

                    logger.info(f"✅ Stored {format_name} in R2: {file_key}")
                else:
                    storage_results[format_name] = {
                        "success": False,
                        "error": f"R2 upload failed: {upload_result}"
                    }

            except Exception as e:
                logger.error(f"❌ Failed to store {format_name} in R2: {e}")
                storage_results[format_name] = {
                    "success": False,
                    "error": str(e)
                }

        return storage_results

    async def _generate_download_urls(self, storage_results: dict[str, Any]) -> dict[str, Any]:
        """Generate presigned URLs for all successfully stored files."""

        final_results = {}

        for format_name, storage_result in storage_results.items():
            if not storage_result["success"]:
                final_results[format_name] = storage_result
                continue

            try:
                # Generate presigned URL (24 hour expiration)
                presigned_url = await self.r2_service.generate_presigned_url(
                    storage_result["file_key"],
                    expiration=86400  # 24 hours
                )

                final_results[format_name] = {
                    "success": True,
                    "r2_url": presigned_url,
                    "file_key": storage_result["file_key"],
                    "size": storage_result["size"]
                }

                logger.info(f"✅ Generated download URL for {format_name}")

            except Exception as e:
                logger.error(f"❌ Failed to generate URL for {format_name}: {e}")
                final_results[format_name] = {
                    "success": False,
                    "error": str(e)
                }

        return final_results

    def _generate_document_id(self, request: MarkdownGenerationRequest) -> str:
        """Generate consistent document ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{request.document_type.value}_{request.topic.value}_{timestamp}".replace(" ", "_").lower()

    def _get_file_extension(self, format_name: str) -> str:
        """Get file extension for format."""
        extensions = {
            "markdown": "md",
            "html": "html",
            "pdf": "pdf",
            "docx": "docx"
        }
        return extensions.get(format_name, format_name)

    def _get_content_type(self, format_name: str) -> str:
        """Get MIME type for format."""
        content_types = {
            "markdown": "text/markdown",
            "html": "text/html",
            "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        }
        return content_types.get(format_name, "application/octet-stream")

    async def get_document_status(self, document_id: str) -> dict[str, Any]:
        """Get status of a generated document."""

        # Check if files exist in R2
        formats = ["markdown", "html", "pdf", "docx"]
        status = {}

        for format_name in formats:
            file_extension = self._get_file_extension(format_name)
            file_key = self.r2_service.generate_file_key(
                document_id=document_id,
                document_type="document",
                file_format=file_extension,
                version="student"
            )

            try:
                exists = await self.r2_service.file_exists(file_key)
                if exists:
                    presigned_url = await self.r2_service.generate_presigned_url(
                        file_key, expiration=86400
                    )
                    status[format_name] = {
                        "available": True,
                        "download_url": presigned_url,
                        "file_key": file_key
                    }
                else:
                    status[format_name] = {"available": False}

            except Exception as e:
                status[format_name] = {"available": False, "error": str(e)}

        return status

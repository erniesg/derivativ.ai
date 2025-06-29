"""
Document generation API endpoints.

Provides REST API for generating, managing, and exporting educational documents
including worksheets, notes, textbooks, and slides.
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from src.api.dependencies import (
    get_document_generation_service,
    get_document_generation_service_v2,
    get_r2_storage_service,
)
from src.models.document_generation_v2 import (
    DocumentGenerationRequestV2,
    DocumentGenerationResultV2,
)
from src.models.document_models import (
    DocumentGenerationRequest,
    DocumentGenerationResult,
    DocumentTemplate,
    DocumentType,
    ExportFormat,
    ExportRequest,
    GeneratedDocument,
)
from src.services.document_export_service import DocumentExportService
from src.services.document_generation_service import DocumentGenerationService
from src.services.document_generation_service_v2 import DocumentGenerationServiceV2
from src.services.r2_storage_service import R2StorageService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/generation/documents", tags=["Document Generation"])


@router.post("/generate", response_model=DocumentGenerationResult)
async def generate_document(
    request: DocumentGenerationRequest,
    service: DocumentGenerationService = Depends(get_document_generation_service),
) -> DocumentGenerationResult:
    """
    Generate an educational document (worksheet, notes, textbook, or slides).

    Creates a structured document by combining questions from the database
    with appropriate content templates and formatting.
    """
    try:
        logger.info(f"Generating {request.document_type.value}: {request.title}")

        result = await service.generate_document(request)

        if result.success:
            logger.info(f"Document generated successfully: {result.document.document_id}")
        else:
            logger.error(f"Document generation failed: {result.error_message}")

        return result

    except Exception as e:
        logger.error(f"Document generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Document generation failed: {e!s}")


@router.post("/generate-v2", response_model=DocumentGenerationResultV2)
async def generate_document_v2(
    request: DocumentGenerationRequestV2,
    service: DocumentGenerationServiceV2 = Depends(get_document_generation_service_v2),
) -> DocumentGenerationResultV2:
    """
    Generate an educational document using V2 service with blocks-based architecture.

    Creates a structured document using flexible content blocks, with database
    question retrieval and fallback to on-the-fly generation.
    """
    try:
        logger.info(f"Generating V2 document: {request.title} ({request.document_type})")

        result = await service.generate_document(request)

        if result.success:
            logger.info(f"V2 Document generated successfully: {result.document.document_id}")
            # Check if document was saved to storage
            storage_id = result.generation_insights.get("document_id")
            if storage_id:
                logger.info(f"Document saved to storage with ID: {storage_id}")
        else:
            logger.error(f"V2 Document generation failed: {result.error_message}")

        return result

    except Exception as e:
        logger.error(f"V2 Document generation error: {e}")
        raise HTTPException(status_code=500, detail=f"V2 Document generation failed: {e!s}")


@router.get("/templates", response_model=dict[str, DocumentTemplate])
async def get_document_templates(
    service: DocumentGenerationService = Depends(get_document_generation_service),
) -> dict[str, DocumentTemplate]:
    """
    Get all available document templates.

    Returns the available templates for each document type with their
    structure patterns and content rules.
    """
    try:
        templates = await service.get_document_templates()
        return templates
    except Exception as e:
        logger.error(f"Failed to retrieve templates: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve templates")


@router.post("/templates", response_model=dict[str, str])
async def create_custom_template(
    template: DocumentTemplate,
    service: DocumentGenerationService = Depends(get_document_generation_service),
) -> dict[str, str]:
    """
    Create a custom document template.

    Allows users to define custom document structures and content rules
    for specialized educational materials.
    """
    try:
        template_id = await service.save_custom_template(template)
        return {"template_id": template_id, "message": "Template created successfully"}
    except Exception as e:
        logger.error(f"Failed to create template: {e}")
        raise HTTPException(status_code=500, detail="Failed to create template")


@router.get("/{document_id}", response_model=GeneratedDocument)
async def get_document(
    document_id: str,
    service: DocumentGenerationService = Depends(get_document_generation_service),
) -> GeneratedDocument:
    """
    Retrieve a specific generated document by ID.

    Returns the complete document structure including all sections,
    content, and metadata.
    """
    try:
        # In a full implementation, this would retrieve from database
        # For now, return placeholder
        raise HTTPException(status_code=501, detail="Document retrieval not yet implemented")
    except Exception as e:
        logger.error(f"Failed to retrieve document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document")


@router.get("/", response_model=list[dict[str, str]])
async def list_documents(
    document_type: Optional[DocumentType] = None,
    limit: int = 20,
    offset: int = 0,
    service: DocumentGenerationService = Depends(get_document_generation_service),
) -> list[dict[str, str]]:
    """
    List generated documents with optional filtering.

    Returns a paginated list of documents with basic metadata.
    Can be filtered by document type.
    """
    try:
        # In a full implementation, this would query database
        # For now, return placeholder
        return [
            {
                "document_id": "placeholder-1",
                "title": "Sample Worksheet",
                "document_type": "worksheet",
                "created_at": "2025-06-21T12:00:00Z",
            }
        ]
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to list documents")


@router.post("/export")
async def export_document(
    export_request: ExportRequest,
    service: DocumentGenerationService = Depends(get_document_generation_service),
    r2_service: R2StorageService = Depends(get_r2_storage_service),
) -> dict[str, Any]:
    """
    Export a document to a specific format with R2 storage.

    Converts the structured document data into the requested format
    and stores it in Cloudflare R2 for download.
    """
    try:
        logger.info(
            f"📤 Exporting document {export_request.document_id} to {export_request.format}"
        )

        # For now, we'll use a mock document since document retrieval isn't implemented
        # In production, you would: document = await service.get_document(export_request.document_id)
        mock_document = {
            "document_id": export_request.document_id,
            "title": "Sample Generated Document",
            "content_structure": {
                "blocks": [
                    {
                        "block_type": "practice_questions",
                        "content": {
                            "questions": [
                                {"text": "Solve for x: 2x + 5 = 13", "answer": "x = 4", "marks": 2},
                                {
                                    "text": "Find the gradient of the line passing through (2,3) and (5,9)",
                                    "answer": "Gradient = 2",
                                    "marks": 3,
                                },
                            ]
                        },
                    }
                ]
            },
        }

        # Initialize export service
        export_service = DocumentExportService()

        # Export and store in R2
        export_result = await export_service.export_document(
            document=mock_document,
            format_type=export_request.format.value,
            version=export_request.version,
            store_in_r2=True,
            r2_service=r2_service,
        )

        if export_result["success"]:
            logger.info(f"✅ Export successful: {export_result.get('r2_file_key', 'local file')}")
            return {
                "success": True,
                "document_id": export_request.document_id,
                "format": export_request.format.value,
                "r2_file_key": export_result.get("r2_file_key"),
                "content": export_result.get("content"),  # For HTML/markdown
                "file_size": export_result.get("file_size", 0),
                "message": "Document exported successfully",
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Export failed: {export_result.get('error', 'Unknown error')}",
            )

    except Exception as e:
        logger.error(f"❌ Document export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document export failed: {e}")


@router.get("/{document_id}/export/{format}", response_class=FileResponse)
async def download_document(
    document_id: str,
    format: ExportFormat,
    service: DocumentGenerationService = Depends(get_document_generation_service),
) -> FileResponse:
    """
    Download a document in a specific format.

    Streams the formatted document file directly to the client
    with appropriate headers for download.
    """
    try:
        # In a full implementation, this would:
        # 1. Retrieve document from database
        # 2. Format to requested type
        # 3. Return as file download

        if format == ExportFormat.HTML:
            media_type = "text/html"
            filename = f"document_{document_id}.html"
        elif format == ExportFormat.PDF:
            media_type = "application/pdf"
            filename = f"document_{document_id}.pdf"
        elif format == ExportFormat.DOCX:
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            filename = f"document_{document_id}.docx"
        elif format == ExportFormat.MARKDOWN:
            media_type = "text/markdown"
            filename = f"document_{document_id}.md"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

        # Placeholder - would return actual file
        raise HTTPException(status_code=501, detail="Document download not yet implemented")

    except HTTPException:
        # Re-raise HTTPExceptions without wrapping them
        raise
    except Exception as e:
        logger.error(f"Document download failed: {e}")
        raise HTTPException(status_code=500, detail="Document download failed")


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    service: DocumentGenerationService = Depends(get_document_generation_service),
) -> dict[str, str]:
    """
    Delete a generated document.

    Removes the document and associated files from storage.
    """
    try:
        # In a full implementation, this would:
        # 1. Check if document exists
        # 2. Remove from database
        # 3. Clean up any associated files

        return {"message": f"Document {document_id} deletion not yet implemented"}

    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")


# Document generation status and progress endpoints


@router.get("/{document_id}/status")
async def get_generation_status(document_id: str) -> dict[str, str]:
    """
    Get the generation status of a document.

    For long-running document generation processes, this endpoint
    allows clients to check progress.
    """
    try:
        # In a full implementation, this would check generation status
        return {"document_id": document_id, "status": "not_implemented", "progress": "0%"}
    except Exception as e:
        logger.error(f"Failed to get status for {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get generation status")


# Document analytics and insights


@router.get("/{document_id}/analytics")
async def get_document_analytics(document_id: str) -> dict[str, Any]:
    """
    Get analytics and insights for a document.

    Returns usage statistics, difficulty analysis, and educational metrics.
    """
    try:
        # In a full implementation, this would return:
        # - Difficulty distribution of questions
        # - Syllabus coverage analysis
        # - Estimated completion times
        # - Question type breakdown

        return {"document_id": document_id, "analytics": "not_implemented"}
    except Exception as e:
        logger.error(f"Failed to get analytics for {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get document analytics")


# Batch operations


@router.post("/batch/generate")
async def batch_generate_documents(
    requests: list[DocumentGenerationRequest],
    service: DocumentGenerationService = Depends(get_document_generation_service),
) -> list[DocumentGenerationResult]:
    """
    Generate multiple documents in batch.

    Useful for creating complete course materials or assessment sets.
    """
    try:
        if len(requests) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 documents per batch")

        # In a full implementation, this would:
        # 1. Process requests concurrently
        # 2. Handle partial failures gracefully
        # 3. Return results for each request

        results = []
        for request in requests:
            result = await service.generate_document(request)
            results.append(result)

        return results

    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail="Batch generation failed")

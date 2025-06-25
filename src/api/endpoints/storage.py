"""
Document Storage API endpoints.

Provides REST API for storing, retrieving, and managing generated documents
with Cloudflare R2 storage and Supabase metadata persistence.
"""

import logging
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.dependencies import (
    get_document_storage_repository,
    get_document_storage_service,
)
from src.models.stored_document_models import (
    DocumentExportRequest,
    DocumentExportResult,
    DocumentFile,
    DocumentSearchFilters,
    StoredDocument,
    StoredDocumentMetadata,
)
from src.repositories.document_storage_repository import DocumentStorageRepository
from src.services.document_storage_service import DocumentStorageService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["Document Storage"])


# Request/Response Models
class CreateDocumentRequest(BaseModel):
    """Request model for creating a stored document."""

    title: str = Field(..., min_length=1, description="Document title")
    document_type: str = Field(..., description="Document type")
    detail_level: Optional[str] = Field(None, description="Detail level")
    topic: Optional[str] = Field(None, description="Topic")
    grade_level: Optional[int] = Field(None, ge=1, le=12, description="Grade level")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in minutes")
    total_questions: Optional[int] = Field(None, ge=0, description="Total questions")
    tags: list[str] = Field(default_factory=list, description="Document tags")
    session_id: Optional[str] = Field(None, description="Generation session ID")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CreateDocumentResponse(BaseModel):
    """Response model for document creation."""

    document_id: str = Field(..., description="Created document ID")
    status: str = Field(..., description="Creation status")
    message: str = Field(..., description="Status message")


class UpdateDocumentRequest(BaseModel):
    """Request model for updating a document."""

    status: Optional[str] = Field(None, description="New status")
    metadata: Optional[dict[str, Any]] = Field(None, description="Metadata to update")


class UpdateDocumentResponse(BaseModel):
    """Response model for document updates."""

    success: bool = Field(..., description="Update success")
    updated_at: str = Field(..., description="Update timestamp")
    message: str = Field(..., description="Update message")


class FileDownloadResponse(BaseModel):
    """Response model for file download info."""

    download_url: str = Field(..., description="File download URL")
    expires_in: int = Field(..., description="URL expiration time in seconds")
    file_info: DocumentFile = Field(..., description="File metadata")


# Search and listing endpoints (must come before /{document_id} route)


@router.get("/statistics", response_model=dict[str, Any])
async def get_document_statistics(
    storage_service: DocumentStorageService = Depends(get_document_storage_service),
) -> dict[str, Any]:
    """
    Get document storage statistics and metrics.

    Returns comprehensive statistics about stored documents,
    including counts by type, status, file sizes, and success rates.
    """
    try:
        stats = await storage_service.get_document_storage_stats()
        return stats

    except Exception as e:
        logger.error(f"Failed to get document statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {e}")


@router.get("/search", response_model=dict[str, Any])
async def search_documents(
    document_type: Optional[str] = Query(None, description="Filter by document type"),
    topic: Optional[str] = Query(None, description="Filter by topic"),
    grade_level: Optional[int] = Query(None, ge=1, le=12, description="Filter by grade level"),
    status: Optional[str] = Query(None, description="Filter by status"),
    search_text: str = Query("", description="Full-text search query"),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    limit: int = Query(50, ge=1, le=100, description="Results limit"),
    offset: int = Query(0, ge=0, description="Results offset"),
    repository: DocumentStorageRepository = Depends(get_document_storage_repository),
) -> dict[str, Any]:
    """
    Search stored documents with filters and pagination.

    Supports filtering by document type, topic, grade level, status,
    full-text search, tags, and date ranges.
    """
    try:
        # Parse tags from comma-separated string
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

        # Build search filters
        filters = DocumentSearchFilters(
            document_type=document_type,
            topic=topic,
            grade_level=grade_level,
            status=status,
            search_text=search_text,
            tags=tag_list,
            limit=limit,
            offset=offset,
        )

        # Execute search
        results = await repository.search_documents(filters)

        return results

    except Exception as e:
        logger.error(f"Failed to search documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search documents: {e}")


# Document CRUD endpoints


@router.post("/", response_model=CreateDocumentResponse, status_code=201)
async def create_document(
    request: CreateDocumentRequest,
    storage_service: DocumentStorageService = Depends(get_document_storage_service),
) -> CreateDocumentResponse:
    """
    Create a new stored document metadata record.

    Creates a document metadata entry in the database without generating files.
    Files can be added later via the storage workflow.
    """
    try:
        logger.info(f"Creating document: {request.title}")

        # Convert request to StoredDocumentMetadata
        metadata = StoredDocumentMetadata(
            title=request.title,
            document_type=request.document_type,
            detail_level=request.detail_level,
            topic=request.topic,
            grade_level=request.grade_level,
            estimated_duration=request.estimated_duration,
            total_questions=request.total_questions,
            tags=request.tags,
            session_id=UUID(request.session_id) if request.session_id else None,
            status="pending",
        )

        # Save to database
        repository = storage_service.repository
        document_id = await repository.save_document_metadata(metadata)

        logger.info(f"Successfully created document: {document_id}")

        return CreateDocumentResponse(
            document_id=str(document_id),
            status="created",
            message="Document metadata created successfully",
        )

    except Exception as e:
        logger.error(f"Failed to create document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create document: {e}")


@router.get("/{document_id}", response_model=StoredDocument)
async def get_document(
    document_id: str,
    storage_service: DocumentStorageService = Depends(get_document_storage_service),
) -> StoredDocument:
    """
    Retrieve a stored document by ID.

    Returns complete document information including metadata,
    associated files, and generation session data.
    """
    try:
        document_uuid = UUID(document_id)
        repository = storage_service.repository

        document = await repository.retrieve_document_by_id(document_uuid)

        if not document:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

        return document

    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid document ID format")
    except Exception as e:
        logger.error(f"Failed to retrieve document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document: {e}")


@router.patch("/{document_id}", response_model=UpdateDocumentResponse)
async def update_document(
    document_id: str,
    request: UpdateDocumentRequest,
    storage_service: DocumentStorageService = Depends(get_document_storage_service),
) -> UpdateDocumentResponse:
    """
    Update document status and metadata.

    Allows updating the document status (pending, generating, exported, etc.)
    and merging additional metadata.
    """
    try:
        document_uuid = UUID(document_id)

        success = await storage_service.update_document_status(
            document_uuid, request.status, request.metadata
        )

        if not success:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

        from datetime import datetime

        return UpdateDocumentResponse(
            success=True,
            updated_at=datetime.now().isoformat(),
            message="Document updated successfully",
        )

    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid document ID format")
    except Exception as e:
        logger.error(f"Failed to update document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update document: {e}")


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    storage_service: DocumentStorageService = Depends(get_document_storage_service),
) -> dict[str, Any]:
    """
    Delete a document and all associated files.

    Removes the document from R2 storage and marks it as deleted
    in the database (soft delete).
    """
    try:
        document_uuid = UUID(document_id)

        result = await storage_service.delete_document_and_files(document_uuid)

        return result

    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid document ID format")
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}")


@router.get("/{document_id}/files", response_model=dict[str, list[DocumentFile]])
async def get_document_files(
    document_id: str,
    repository: DocumentStorageRepository = Depends(get_document_storage_repository),
) -> dict[str, list[DocumentFile]]:
    """
    Get all files associated with a document.

    Returns a list of files stored in R2 for the specified document,
    including file metadata and access information.
    """
    try:
        document_uuid = UUID(document_id)

        files = await repository.get_document_files(document_uuid)

        return {"files": files}

    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid document ID format")
    except Exception as e:
        logger.error(f"Failed to get document files {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document files: {e}")


# File access endpoints


@router.get("/files/{file_id}/download", response_model=FileDownloadResponse)
async def get_file_download_url(
    file_id: str,
    expiration: int = Query(3600, ge=300, le=86400, description="URL expiration in seconds"),
    storage_service: DocumentStorageService = Depends(get_document_storage_service),
) -> FileDownloadResponse:
    """
    Generate a presigned download URL for a document file.

    Creates a temporary, secure URL that allows direct download
    of the file from R2 storage.
    """
    try:
        file_uuid = UUID(file_id)
        repository = storage_service.repository

        # Get all files and find the one with matching ID
        # In a more optimized implementation, we'd have a direct file lookup
        # For now, we'll need to implement a file-by-id method

        # This is a placeholder - we need to add get_file_by_id to repository
        raise HTTPException(status_code=501, detail="File download not yet implemented")

    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid file ID format")
    except Exception as e:
        logger.error(f"Failed to generate download URL for file {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate download URL: {e}")


# Export endpoints


@router.post("/{document_id}/export", response_model=DocumentExportResult, status_code=202)
async def export_document(
    document_id: str,
    export_request: DocumentExportRequest,
    storage_service: DocumentStorageService = Depends(get_document_storage_service),
) -> DocumentExportResult:
    """
    Export an existing document to additional formats.

    Converts the stored document to the requested formats
    and uploads the new files to R2 storage.
    """
    try:
        document_uuid = UUID(document_id)

        result = await storage_service.export_existing_document(
            document_uuid, export_request.formats, export_request.create_dual_versions
        )

        from datetime import datetime
        from uuid import uuid4

        return DocumentExportResult(
            export_id=uuid4(),
            document_id=document_uuid,
            status="completed" if result["success"] else "failed",
            files=[],  # Would be populated with actual file objects
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )

    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid document ID format")
    except Exception as e:
        logger.error(f"Failed to export document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export document: {e}")


# Bulk operations endpoints


@router.patch("/bulk-update", response_model=dict[str, Any])
async def bulk_update_documents(
    request: dict[str, Any],  # Simplified for now
    storage_service: DocumentStorageService = Depends(get_document_storage_service),
) -> dict[str, Any]:
    """
    Perform bulk updates on multiple documents.

    Allows updating status, metadata, or performing other operations
    on multiple documents simultaneously.
    """
    try:
        document_ids = request.get("document_ids", [])
        status = request.get("status")
        metadata = request.get("metadata", {})

        if not document_ids:
            raise HTTPException(status_code=400, detail="No document IDs provided")

        if len(document_ids) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 documents per bulk operation")

        updated_count = 0
        failed_updates = []

        # Process each document
        for doc_id_str in document_ids:
            try:
                doc_id = UUID(doc_id_str)
                success = await storage_service.update_document_status(doc_id, status, metadata)
                if success:
                    updated_count += 1
                else:
                    failed_updates.append(
                        {"document_id": doc_id_str, "error": "Document not found"}
                    )
            except Exception as e:
                failed_updates.append({"document_id": doc_id_str, "error": str(e)})

        return {
            "updated_count": updated_count,
            "failed_updates": failed_updates,
            "total_requested": len(document_ids),
        }

    except Exception as e:
        logger.error(f"Failed to perform bulk update: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to perform bulk update: {e}")

"""
Document Download API Endpoints
Provides R2 presigned URLs for frontend document downloads
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.api.dependencies import get_r2_storage_service
from src.services.r2_storage_service import R2StorageError, R2StorageService

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models
class BulkDownloadRequest(BaseModel):
    document_ids: list[str]
    format: str = "html"
    version: str = "student"


class DownloadUrlResponse(BaseModel):
    download_url: str
    expires_at: str
    format: str
    version: str
    filename: str


class BulkDownloadResponse(BaseModel):
    download_urls: list[dict]
    expires_at: str


class AvailableDocumentsResponse(BaseModel):
    documents: list[dict]
    total_count: int


class DocumentSearchResponse(BaseModel):
    documents: list[dict]
    total_count: int
    search_query: str


@router.get("/documents/{document_id}/download", response_model=DownloadUrlResponse)
async def get_document_download_url(
    document_id: str,
    format: str = Query("html", description="Export format"),
    version: str = Query("student", description="Document version"),
    expiration_hours: int = Query(1, description="URL expiration in hours"),
    r2_service: R2StorageService = Depends(get_r2_storage_service),
):
    """
    Generate presigned download URL for a specific document.

    This endpoint creates a temporary, secure URL that the frontend can use
    to download the document directly from Cloudflare R2.
    """
    try:
        logger.info(f"🔗 Generating download URL for {document_id} ({format}, {version})")

        # Generate file key using R2 service pattern
        file_key = r2_service.generate_file_key(
            document_id=document_id,
            document_type="document",  # Generic type for downloads
            file_format=format,
            version=version,
        )

        logger.info(f"📋 Generated file key: {file_key}")

        # Check if file exists in R2
        exists = await r2_service.file_exists(file_key)
        if not exists:
            logger.warning(f"❌ File not found in R2: {file_key}")
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found in {format} format ({version} version)",
            )

        # Generate presigned URL
        expiration_seconds = expiration_hours * 3600
        download_url = await r2_service.generate_presigned_url(
            file_key=file_key, expiration=expiration_seconds, method="get_object"
        )

        # Calculate expiration time
        expires_at = (datetime.now() + timedelta(hours=expiration_hours)).isoformat()

        # Generate filename
        filename = f"{document_id}_{version}.{format}"

        logger.info(f"✅ Generated download URL (expires in {expiration_hours}h)")

        return DownloadUrlResponse(
            download_url=download_url,
            expires_at=expires_at,
            format=format,
            version=version,
            filename=filename,
        )

    except R2StorageError as e:
        logger.error(f"❌ R2 storage error: {e}")
        raise HTTPException(status_code=500, detail=f"Storage error: {e}")
    except Exception as e:
        logger.error(f"❌ Unexpected error generating download URL: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate download URL")


@router.post("/documents/bulk-download", response_model=BulkDownloadResponse)
async def prepare_bulk_download(
    request: BulkDownloadRequest,
    expiration_hours: int = Query(1, description="URL expiration in hours"),
    r2_service: R2StorageService = Depends(get_r2_storage_service),
):
    """
    Prepare multiple documents for bulk download.

    Returns presigned URLs for all requested documents.
    """
    try:
        logger.info(f"📦 Preparing bulk download for {len(request.document_ids)} documents")

        download_urls = []
        expiration_seconds = expiration_hours * 3600
        expires_at = (datetime.now() + timedelta(hours=expiration_hours)).isoformat()

        for document_id in request.document_ids:
            try:
                # Generate file key
                file_key = r2_service.generate_file_key(
                    document_id=document_id,
                    document_type="document",
                    file_format=request.format,
                    version=request.version,
                )

                # Check if file exists
                exists = await r2_service.file_exists(file_key)
                if not exists:
                    logger.warning(f"⚠️ File not found: {file_key}")
                    continue

                # Generate presigned URL
                download_url = await r2_service.generate_presigned_url(
                    file_key=file_key, expiration=expiration_seconds, method="get_object"
                )

                download_urls.append(
                    {
                        "document_id": document_id,
                        "download_url": download_url,
                        "filename": f"{document_id}_{request.version}.{request.format}",
                        "file_key": file_key,
                    }
                )

            except Exception as e:
                logger.error(f"❌ Failed to prepare download for {document_id}: {e}")
                continue

        logger.info(f"✅ Prepared {len(download_urls)} documents for bulk download")

        return BulkDownloadResponse(download_urls=download_urls, expires_at=expires_at)

    except Exception as e:
        logger.error(f"❌ Bulk download preparation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to prepare bulk download")


@router.get("/documents/available", response_model=AvailableDocumentsResponse)
async def list_available_documents(
    limit: int = Query(50, description="Maximum documents to return"),
    offset: int = Query(0, description="Number of documents to skip"),
    r2_service: R2StorageService = Depends(get_r2_storage_service),
):
    """
    List all documents available for download.

    Scans R2 storage to find available documents and their formats.
    """
    try:
        logger.info(f"📋 Listing available documents (limit: {limit}, offset: {offset})")

        # List files in the documents directory
        files = await r2_service.list_files(prefix="documents/", max_keys=limit + offset)

        # Group files by document ID
        documents_dict = {}

        for file_info in files[offset : offset + limit]:
            file_key = file_info["key"]

            # Parse file key: documents/document_type/document_id/version.format
            try:
                parts = file_key.split("/")
                if len(parts) >= 4:
                    document_type = parts[1]
                    document_id = parts[2]
                    version_and_format = parts[3]

                    # Extract version and format
                    if "." in version_and_format:
                        version, format_ext = version_and_format.rsplit(".", 1)
                    else:
                        continue

                    if document_id not in documents_dict:
                        documents_dict[document_id] = {
                            "document_id": document_id,
                            "document_type": document_type,
                            "available_formats": [],
                            "available_versions": [],
                            "last_modified": file_info["last_modified"],
                            "file_count": 0,
                        }

                    doc = documents_dict[document_id]

                    if format_ext not in doc["available_formats"]:
                        doc["available_formats"].append(format_ext)

                    if version not in doc["available_versions"]:
                        doc["available_versions"].append(version)

                    doc["file_count"] += 1

                    # Update last modified to latest
                    if file_info["last_modified"] > doc["last_modified"]:
                        doc["last_modified"] = file_info["last_modified"]

            except Exception as e:
                logger.warning(f"⚠️ Failed to parse file key {file_key}: {e}")
                continue

        documents = list(documents_dict.values())

        logger.info(f"✅ Found {len(documents)} available documents")

        return AvailableDocumentsResponse(documents=documents, total_count=len(documents))

    except Exception as e:
        logger.error(f"❌ Failed to list available documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to list documents")


@router.get("/documents/search", response_model=DocumentSearchResponse)
async def search_documents(
    topic: Optional[str] = Query(None, description="Filter by topic"),
    document_type: Optional[str] = Query(None, description="Filter by document type"),
    format: Optional[str] = Query(None, description="Filter by available format"),
    limit: int = Query(20, description="Maximum documents to return"),
    r2_service: R2StorageService = Depends(get_r2_storage_service),
):
    """
    Search for documents by topic, type, or format.

    This enables teachers to find pre-generated worksheets for specific topics.
    """
    try:
        search_query = f"topic:{topic}, type:{document_type}, format:{format}"
        logger.info(f"🔍 Searching documents: {search_query}")

        # Get all available documents
        all_docs_response = await list_available_documents(
            limit=1000, offset=0, r2_service=r2_service
        )
        all_documents = all_docs_response.documents

        # Apply filters
        filtered_documents = []

        for doc in all_documents:
            # Filter by topic (check if topic is in document_id or type)
            if topic and topic.lower() not in doc["document_id"].lower():
                continue

            # Filter by document type
            if document_type and document_type.lower() != doc["document_type"].lower():
                continue

            # Filter by format availability
            if format and format.lower() not in [f.lower() for f in doc["available_formats"]]:
                continue

            filtered_documents.append(doc)

            if len(filtered_documents) >= limit:
                break

        logger.info(f"✅ Found {len(filtered_documents)} documents matching search criteria")

        return DocumentSearchResponse(
            documents=filtered_documents,
            total_count=len(filtered_documents),
            search_query=search_query,
        )

    except Exception as e:
        logger.error(f"❌ Document search failed: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


@router.get("/documents/{document_id}/metadata")
async def get_document_metadata(
    document_id: str,
    r2_service: R2StorageService = Depends(get_r2_storage_service),
):
    """
    Get metadata for a specific document including available formats and versions.
    """
    try:
        logger.info(f"📋 Getting metadata for document: {document_id}")

        # List all files for this document
        prefix = f"documents/document/{document_id}/"
        files = await r2_service.list_files(prefix=prefix)

        if not files:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

        metadata = {
            "document_id": document_id,
            "available_formats": [],
            "available_versions": [],
            "files": [],
            "total_size": 0,
            "last_modified": None,
        }

        for file_info in files:
            file_key = file_info["key"]
            filename = file_key.split("/")[-1]  # Get just the filename

            if "." in filename:
                version, format_ext = filename.rsplit(".", 1)

                if format_ext not in metadata["available_formats"]:
                    metadata["available_formats"].append(format_ext)

                if version not in metadata["available_versions"]:
                    metadata["available_versions"].append(version)

            metadata["files"].append(
                {
                    "file_key": file_key,
                    "filename": filename,
                    "size": file_info["size"],
                    "last_modified": file_info["last_modified"],
                }
            )

            metadata["total_size"] += file_info["size"]

            if (
                not metadata["last_modified"]
                or file_info["last_modified"] > metadata["last_modified"]
            ):
                metadata["last_modified"] = file_info["last_modified"]

        logger.info(
            f"✅ Retrieved metadata: {len(metadata['files'])} files, {len(metadata['available_formats'])} formats"
        )

        return metadata

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get document metadata: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metadata")

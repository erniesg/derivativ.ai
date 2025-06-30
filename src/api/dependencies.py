"""
FastAPI dependency injection for repositories and services.
Provides singleton instances of database clients and services.
"""

import os
from functools import lru_cache
from typing import Any

from fastapi import Depends, HTTPException

from src.agents.document_formatter_agent import DocumentFormatterAgent
from src.database.supabase_repository import (
    GenerationSessionRepository,
    QuestionRepository,
    get_supabase_client,
)
from src.repositories.document_storage_repository import DocumentStorageRepository
from src.services.document_export_service import DocumentExportService
from src.services.document_generation_service import DocumentGenerationService
from src.services.document_generation_service_v2 import DocumentGenerationServiceV2
from src.services.document_storage_service import DocumentStorageService
from src.services.integrated_document_service import IntegratedDocumentService
from src.services.llm_factory import LLMFactory
from src.services.markdown_document_service import MarkdownDocumentService
from src.services.pandoc_service import PandocService
from src.services.prompt_manager import PromptManager
from src.services.question_generation_service import QuestionGenerationService
from src.services.r2_storage_service import R2StorageService
from src.supabase_realtime.supabase_realtime import get_realtime_client


@lru_cache
def get_supabase_credentials() -> tuple[str, str]:
    """
    Get Supabase credentials from environment.

    Returns:
        Tuple of (url, key)

    Raises:
        HTTPException: If credentials are not configured
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")

    if not url or not key:
        raise HTTPException(
            status_code=503,
            detail="Supabase not configured. Please set SUPABASE_URL and SUPABASE_ANON_KEY environment variables.",
        )

    return url, key


@lru_cache
def is_demo_mode() -> bool:
    """Check if running in demo mode (no database required)."""
    return os.getenv("DEMO_MODE", "false").lower() in ("true", "1", "yes")


@lru_cache
def get_database_client():
    """
    Get singleton Supabase client instance.

    Returns:
        Configured Supabase client
    """
    url, key = get_supabase_credentials()
    return get_supabase_client(url, key)


def get_question_repository(client=Depends(get_database_client)) -> QuestionRepository:
    """
    Get QuestionRepository instance.

    Args:
        client: Supabase client (injected)

    Returns:
        Configured QuestionRepository
    """
    return QuestionRepository(client)


def get_session_repository(client=Depends(get_database_client)) -> GenerationSessionRepository:
    """
    Get GenerationSessionRepository instance.

    Args:
        client: Supabase client (injected)

    Returns:
        Configured GenerationSessionRepository
    """
    return GenerationSessionRepository(client)


def get_question_generation_service(
    question_repo: QuestionRepository = Depends(get_question_repository),
    session_repo: GenerationSessionRepository = Depends(get_session_repository),
) -> QuestionGenerationService:
    """
    Get QuestionGenerationService instance.

    Args:
        question_repo: Question repository (injected)
        session_repo: Session repository (injected)

    Returns:
        Configured QuestionGenerationService
    """
    return QuestionGenerationService(question_repo, session_repo)


@lru_cache
def get_realtime_client_instance():
    """
    Get singleton Realtime client instance.

    Returns:
        Configured Realtime client, or None if not configured
    """
    try:
        url, key = get_supabase_credentials()
        return get_realtime_client(url, key)
    except HTTPException:
        # Realtime is optional - return None if not configured
        return None


def get_optional_realtime_client():
    """
    Get optional Realtime client instance.

    Returns:
        Configured Realtime client, or None if not available
    """
    return get_realtime_client_instance()


@lru_cache
def get_llm_factory() -> LLMFactory:
    """
    Get singleton LLMFactory instance.

    Returns:
        Configured LLMFactory
    """
    return LLMFactory()


@lru_cache
def get_prompt_manager() -> PromptManager:
    """
    Get singleton PromptManager instance.

    Returns:
        Configured PromptManager
    """
    return PromptManager()


def get_document_generation_service(
    llm_factory: LLMFactory = Depends(get_llm_factory),
    prompt_manager: PromptManager = Depends(get_prompt_manager),
) -> DocumentGenerationService:
    """
    Get DocumentGenerationService instance.

    Args:
        llm_factory: LLM factory (injected)
        prompt_manager: Prompt manager (injected)

    Returns:
        Configured DocumentGenerationService
    """
    if is_demo_mode():
        # Use mock repository for demo mode
        from unittest.mock import AsyncMock, MagicMock

        from src.models.enums import SubjectContentReference
        from src.models.question_models import (
            FinalAnswer,
            MarkingCriterion,
            Question,
            QuestionTaxonomy,
            SolutionAndMarkingScheme,
            SolverAlgorithm,
            SolverStep,
        )

        mock_repo = MagicMock()

        # Create sample question for demo
        sample_question = Question(
            question_id_local="1a",
            question_id_global="demo_q1",
            question_number_display="1 (a)",
            marks=3,
            command_word="Calculate",
            raw_text_content="Calculate the area of a triangle with base 6cm and height 4cm.",
            taxonomy=QuestionTaxonomy(
                topic_path=["Geometry", "Area"],
                subject_content_references=[SubjectContentReference.C5_2],
                skill_tags=["area_calculation"],
            ),
            solution_and_marking_scheme=SolutionAndMarkingScheme(
                final_answers_summary=[
                    FinalAnswer(answer_text="12 cm²", value_numeric=12.0, unit="cm²")
                ],
                mark_allocation_criteria=[
                    MarkingCriterion(
                        criterion_id="1",
                        criterion_text="Correct method",
                        mark_code_display="M1",
                        marks_value=1,
                    ),
                    MarkingCriterion(
                        criterion_id="2",
                        criterion_text="Correct answer",
                        mark_code_display="A1",
                        marks_value=2,
                    ),
                ],
                total_marks_for_part=3,
            ),
            solver_algorithm=SolverAlgorithm(
                steps=[
                    SolverStep(
                        step_number=1, description_text="Use formula Area = (1/2) × base × height"
                    ),
                    SolverStep(
                        step_number=2,
                        description_text="Substitute values: Area = (1/2) × 6 × 4 = 12 cm²",
                    ),
                ]
            ),
        )

        mock_repo.list_questions = AsyncMock(
            return_value=[
                {
                    "content_json": sample_question.model_dump(),
                    "quality_score": 0.85,
                    "tier": "Core",
                    "marks": 3,
                }
            ]
        )

        return DocumentGenerationService(mock_repo, llm_factory, prompt_manager)
    else:
        # Get real repository - this will fail if Supabase not configured
        try:
            client = get_database_client()
            question_repo = QuestionRepository(client)
            return DocumentGenerationService(question_repo, llm_factory, prompt_manager)
        except HTTPException:
            # Fallback to mock for demo if database not available
            os.environ["DEMO_MODE"] = "true"
            # Clear cache and retry with demo mode
            is_demo_mode.cache_clear()
            return get_document_generation_service(llm_factory, prompt_manager)


def get_document_formatter_agent(
    llm_factory: LLMFactory = Depends(get_llm_factory),
) -> DocumentFormatterAgent:
    """
    Get DocumentFormatterAgent instance.

    Args:
        llm_factory: LLM factory (injected)

    Returns:
        Configured DocumentFormatterAgent
    """
    # Create LLM service for the agent
    llm_service = llm_factory.get_service("openai")
    return DocumentFormatterAgent(llm_service=llm_service)


@lru_cache
def get_demo_document_storage() -> Any:
    """Get singleton mock document storage for demo mode."""
    from unittest.mock import AsyncMock, MagicMock
    from uuid import uuid4

    from src.models.stored_document_models import StoredDocument

    mock_repo = MagicMock(spec=DocumentStorageRepository)

    # Store created documents in memory for demo mode
    mock_repo._documents = {}

    # Mock the save_document_metadata method
    async def mock_save_document_metadata(metadata):
        doc_id = metadata.id
        mock_repo._documents[doc_id] = metadata
        return doc_id

    mock_repo.save_document_metadata = AsyncMock(side_effect=mock_save_document_metadata)

    # Mock the retrieve_document_by_id method
    async def mock_retrieve_document(doc_id):
        if doc_id in mock_repo._documents:
            metadata = mock_repo._documents[doc_id]
            return StoredDocument(metadata=metadata, files=[], session_data={})
        return None

    mock_repo.retrieve_document_by_id = AsyncMock(side_effect=mock_retrieve_document)

    # Mock update document status
    async def mock_update_document_status(doc_id, status, metadata=None):
        if doc_id in mock_repo._documents:
            mock_repo._documents[doc_id].status = status
            if metadata:
                # Merge metadata (simplified)
                pass
            return True
        return False

    mock_repo.update_document_status = AsyncMock(side_effect=mock_update_document_status)

    # Mock soft delete
    async def mock_soft_delete_document(doc_id):
        if doc_id in mock_repo._documents:
            mock_repo._documents[doc_id].status = "deleted"
            return True
        return False

    mock_repo.soft_delete_document = AsyncMock(side_effect=mock_soft_delete_document)

    # Mock search documents
    async def mock_search_documents(filters):
        # Simple search logic for demo mode
        matching_docs = []
        for doc_id, metadata in mock_repo._documents.items():
            # Skip deleted documents
            if metadata.status == "deleted":
                continue

            # Apply filters
            if filters.document_type and metadata.document_type != filters.document_type:
                continue
            if filters.topic and metadata.topic != filters.topic:
                continue
            if filters.grade_level and metadata.grade_level != filters.grade_level:
                continue
            if filters.status and metadata.status != filters.status:
                continue
            if (
                filters.search_text
                and filters.search_text.lower() not in (metadata.title or "").lower()
            ):
                continue

            matching_docs.append(metadata)

        # Apply pagination
        total_count = len(matching_docs)
        offset = filters.offset
        limit = filters.limit

        paginated_docs = matching_docs[offset : offset + limit]
        has_more = (offset + limit) < total_count

        return {
            "documents": paginated_docs,
            "total_count": total_count,
            "offset": offset,
            "limit": limit,
            "has_more": has_more,
        }

    mock_repo.search_documents = AsyncMock(side_effect=mock_search_documents)
    mock_repo.get_document_files = AsyncMock(return_value=[])
    mock_repo.save_document_file = AsyncMock(return_value=uuid4())
    mock_repo.get_document_statistics = AsyncMock(
        return_value={
            "total_documents": 0,
            "total_file_size": 0,
            "documents_by_type": {},
            "documents_by_status": {},
            "average_file_size": 0,
            "generation_success_rate": 0.0,
        }
    )

    return mock_repo


@lru_cache
def get_r2_storage_config() -> dict[str, str]:
    """
    Get R2 storage configuration from environment.

    Returns:
        R2 configuration dictionary

    Raises:
        HTTPException: If R2 credentials are not configured
    """
    config = {
        "account_id": os.getenv("CLOUDFLARE_ACCOUNT_ID"),
        "access_key_id": os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID"),
        "secret_access_key": os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY"),
        "bucket_name": os.getenv("CLOUDFLARE_R2_BUCKET_NAME"),
        "region": os.getenv("CLOUDFLARE_R2_REGION", "auto"),
    }

    missing_keys = [key for key, value in config.items() if not value and key != "region"]

    if missing_keys and not is_demo_mode():
        raise HTTPException(
            status_code=503,
            detail=f"R2 storage not configured. Missing: {missing_keys}",
        )

    return config


def get_r2_storage_service() -> R2StorageService:
    """
    Get R2StorageService instance.

    Returns:
        Configured R2StorageService or None in demo mode
    """
    if is_demo_mode():
        # Return a mock R2 service for demo mode
        from unittest.mock import AsyncMock, MagicMock

        mock_r2_service = MagicMock(spec=R2StorageService)
        mock_r2_service.upload_file = AsyncMock(
            return_value={
                "success": True,
                "file_key": "demo/file.pdf",
                "bucket": "demo-bucket",
                "upload_id": "demo-upload-123",
                "file_size": 1024,
            }
        )
        mock_r2_service.download_file = AsyncMock(
            return_value={
                "success": True,
                "file_key": "demo/file.pdf",
                "content": b"demo file content",
                "file_size": 1024,
            }
        )
        mock_r2_service.delete_file = AsyncMock(
            return_value={"success": True, "file_key": "demo/file.pdf"}
        )
        mock_r2_service.generate_presigned_url = AsyncMock(
            return_value="https://demo.r2.url/file.pdf"
        )
        mock_r2_service.generate_file_key = MagicMock(
            return_value="demo/documents/worksheet/doc-id/combined.pdf"
        )

        return mock_r2_service
    else:
        config = get_r2_storage_config()
        return R2StorageService(config)


def get_document_storage_repository() -> DocumentStorageRepository:
    """
    Get DocumentStorageRepository instance.

    Returns:
        Configured DocumentStorageRepository or mock in demo mode
    """
    if is_demo_mode():
        return get_demo_document_storage()
    else:
        client = get_database_client()
        return DocumentStorageRepository(client)


def get_document_storage_service(
    r2_service: R2StorageService = Depends(get_r2_storage_service),
    repository: DocumentStorageRepository = Depends(get_document_storage_repository),
) -> DocumentStorageService:
    """
    Get DocumentStorageService instance.

    Args:
        r2_service: R2 storage service (injected)
        repository: Document storage repository (injected)

    Returns:
        Configured DocumentStorageService
    """
    # Create export service for document format conversion
    export_service = DocumentExportService()

    return DocumentStorageService(r2_service, repository, export_service)


def get_document_generation_service_v2(
    llm_factory: LLMFactory = Depends(get_llm_factory),
    prompt_manager: PromptManager = Depends(get_prompt_manager),
    question_repo: QuestionRepository = Depends(get_question_repository),
    question_service: QuestionGenerationService = Depends(get_question_generation_service),
    document_storage_repo: DocumentStorageRepository = Depends(get_document_storage_repository),
) -> DocumentGenerationServiceV2:
    """
    Get DocumentGenerationServiceV2 instance with full dependency injection.

    Args:
        llm_factory: LLM factory (injected)
        prompt_manager: Prompt manager (injected)
        question_repo: Question repository (injected)
        question_service: Question generation service (injected)
        document_storage_repo: Document storage repository (injected)

    Returns:
        Configured DocumentGenerationServiceV2
    """
    return DocumentGenerationServiceV2(
        llm_factory=llm_factory,
        prompt_manager=prompt_manager,
        question_repository=question_repo,
        question_generation_service=question_service,
        document_storage_repository=document_storage_repo,
    )


# Global service instances for backwards compatibility
# These will be replaced by dependency injection


def initialize_global_services():
    """
    Initialize global service instances for endpoints that haven't been updated
    to use dependency injection yet.

    This is a transitional function while we migrate to full dependency injection.
    """
    try:
        # Initialize repositories
        client = get_database_client()
        question_repo = QuestionRepository(client)
        session_repo = GenerationSessionRepository(client)

        # Initialize services
        question_service = QuestionGenerationService(question_repo, session_repo)

        # Initialize realtime client
        realtime_client = get_realtime_client_instance()

        # Update global references in endpoint modules
        from src.api.endpoints import questions, sessions, websocket

        questions.question_repository = question_repo
        questions.question_generation_service = question_service

        sessions.session_repository = session_repo

        if realtime_client:
            websocket.initialize_realtime_client(*get_supabase_credentials())
        websocket.question_generation_service = question_service

        return True

    except Exception as e:
        # Log error but don't fail startup - endpoints will show 503 if needed
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to initialize services: {e}")
        return False


def check_database_health() -> dict:
    """
    Check database connectivity and health.

    Returns:
        Health status dictionary
    """
    try:
        client = get_database_client()

        # Test basic connectivity by querying a simple table
        response = client.table("tiers").select("value").limit(1).execute()

        return {
            "database": "healthy",
            "supabase": "connected",
            "tables_accessible": len(response.data) >= 0,
        }

    except Exception as e:
        return {
            "database": "unhealthy",
            "supabase": "disconnected",
            "error": str(e),
        }


def check_realtime_health() -> dict:
    """
    Check Realtime connectivity and health.

    Returns:
        Realtime health status dictionary
    """
    try:
        realtime_client = get_realtime_client_instance()

        if realtime_client is None:
            return {
                "realtime": "not_configured",
                "websocket": "unavailable",
            }

        return {
            "realtime": "configured",
            "websocket": "available",
        }

    except Exception as e:
        return {
            "realtime": "error",
            "websocket": "unavailable",
            "error": str(e),
        }


def get_integrated_document_service(
    llm_factory: LLMFactory = Depends(get_llm_factory),
    prompt_manager: PromptManager = Depends(get_prompt_manager),
    r2_service: R2StorageService = Depends(get_r2_storage_service),
) -> IntegratedDocumentService:
    """
    Get IntegratedDocumentService instance with markdown + pandoc + R2 pipeline.

    This service provides the clean markdown-first document generation approach
    that eliminates complex JSON structure issues.
    """
    try:
        # Create markdown service
        markdown_service = MarkdownDocumentService(
            llm_service=llm_factory.get_service("openai"),  # Use OpenAI as default
            prompt_manager=prompt_manager,
        )

        # Create pandoc service
        pandoc_service = PandocService()

        # Create integrated service
        return IntegratedDocumentService(
            markdown_service=markdown_service, pandoc_service=pandoc_service, r2_service=r2_service
        )

    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Failed to initialize integrated document service: {e}"
        )


def get_system_health() -> dict:
    """
    Get comprehensive system health status.

    Returns:
        Complete health status dictionary
    """
    if is_demo_mode():
        # In demo mode, return healthy status without checking external dependencies
        return {
            "status": "healthy",
            "service": "derivativ-api",
            "database": "demo_mode",
            "realtime": "demo_mode",
            "mode": "demo",
        }

    database_health = check_database_health()
    realtime_health = check_realtime_health()

    overall_status = "healthy"
    if database_health.get("database") != "healthy":
        overall_status = "degraded"
    if "error" in database_health or "error" in realtime_health:
        overall_status = "unhealthy"

    return {
        "status": overall_status,
        "service": "derivativ-api",
        **database_health,
        **realtime_health,
    }

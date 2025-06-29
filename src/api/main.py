"""
Main FastAPI application with Supabase integration.
Provides REST API and WebSocket endpoints for question generation.
"""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Load environment variables before importing anything else
load_dotenv()

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from src.api.dependencies import get_system_health, initialize_global_services  # noqa: E402
from src.api.endpoints import (  # noqa: E402
    agents,
    documents,
    questions,
    sessions,
    storage,
    system,
    templates,
    websocket,
)
from src.core.config import get_settings  # noqa: E402

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Derivativ API...")

    # Initialize services and dependencies
    services_initialized = initialize_global_services()
    if services_initialized:
        logger.info("Services initialized successfully")
    else:
        logger.warning("Some services failed to initialize - API may have limited functionality")

    yield

    # Shutdown
    logger.info("Shutting down Derivativ API...")

    # Cleanup realtime connections
    try:
        from src.supabase_realtime.supabase_realtime import cleanup_realtime_client

        await cleanup_realtime_client()
        logger.info("Realtime connections cleaned up")
    except Exception as e:
        logger.warning(f"Error cleaning up realtime connections: {e}")


# Create FastAPI app
app = FastAPI(
    title="Derivativ AI API",
    description="AI-powered Cambridge IGCSE Mathematics question generation platform",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint with system status."""
    return get_system_health()


# Include routers
app.include_router(agents.router, tags=["agents"])
app.include_router(documents.router, tags=["documents"])
app.include_router(storage.router, tags=["storage"])
app.include_router(questions.router, prefix="/api", tags=["questions"])
app.include_router(sessions.router, prefix="/api", tags=["sessions"])
app.include_router(system.router, tags=["system"])
app.include_router(templates.router, tags=["templates"])
app.include_router(websocket.router, prefix="/api", tags=["websocket"])


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug",
    )

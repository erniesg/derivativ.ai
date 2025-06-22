"""
WebSocket endpoints for real-time question generation with Supabase Realtime integration.
"""

import asyncio
import contextlib
import logging
from typing import Optional

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from src.api.dependencies import get_question_generation_service
from src.realtime.supabase_realtime import get_realtime_client
from src.services.question_generation_service import QuestionGenerationService

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependencies (will be injected via dependency injection in production)
question_generation_service: Optional[QuestionGenerationService] = None
realtime_client = None  # Will be initialized with Supabase credentials


def initialize_realtime_client(supabase_url: str, supabase_key: str) -> None:
    """
    Initialize the global Realtime client.

    Args:
        supabase_url: Supabase project URL
        supabase_key: Supabase API key
    """
    global realtime_client  # noqa: PLW0603
    realtime_client = get_realtime_client(supabase_url, supabase_key)
    logger.info("Realtime client initialized")


@router.websocket("/ws/questions/{question_id}")
async def websocket_question_updates(websocket: WebSocket, question_id: str):
    """
    WebSocket endpoint for real-time question updates.

    Args:
        websocket: WebSocket connection
        question_id: Question ID to monitor
    """
    await websocket.accept()

    try:
        # Send connection confirmation
        await websocket.send_json({"type": "connection_established", "question_id": question_id})

        if realtime_client:
            # Stream question updates from Supabase Realtime
            async for update in realtime_client.stream_question_updates([question_id]):
                await websocket.send_json(update)
        else:
            await websocket.send_json({"type": "error", "message": "Realtime not configured"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for question {question_id}")
    except Exception as e:
        logger.error(f"WebSocket error for question {question_id}: {e}")
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "message": "Internal server error"})


@router.websocket("/ws/generate/{session_id}")
async def websocket_generate(
    websocket: WebSocket,
    session_id: str,
    service: QuestionGenerationService = Depends(get_question_generation_service),
):
    """
    WebSocket endpoint for real-time question generation with agent updates and Supabase Realtime.

    Args:
        websocket: WebSocket connection
        session_id: Generation session identifier
        service: Question generation service (injected)
    """
    await websocket.accept()

    try:
        # Send connection confirmation
        await websocket.send_json({"type": "connection_established", "session_id": session_id})

        # Start Supabase Realtime streaming for database updates
        realtime_task = None
        if realtime_client:
            realtime_task = asyncio.create_task(_stream_realtime_updates(websocket, session_id))

        while True:
            # Receive message from client
            data = await websocket.receive_json()

            if data.get("action") == "generate":
                # Start generation with real-time updates
                request_data = data.get("request", {})

                try:
                    # Use injected service, fallback to global for backwards compatibility
                    generation_service = service or question_generation_service
                    if not generation_service:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "Question generation service not available",
                            }
                        )
                        continue

                    # Stream generation updates from service
                    async for update in generation_service.generate_questions_stream(
                        request_data, session_id
                    ):
                        await websocket.send_json(update)

                except Exception as e:
                    await websocket.send_json(
                        {"type": "error", "message": f"Generation failed: {e!s}"}
                    )

            elif data.get("action") == "cancel":
                # Cancel ongoing generation
                await websocket.send_json(
                    {"type": "generation_cancelled", "session_id": session_id}
                )
                break

            else:
                await websocket.send_json({"type": "error", "message": "Unknown action"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "message": "Internal server error"})
    finally:
        # Cleanup realtime task
        if realtime_task:
            realtime_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await realtime_task


async def _stream_realtime_updates(websocket: WebSocket, session_id: str) -> None:
    """
    Stream Supabase Realtime database updates to WebSocket client.

    Args:
        websocket: WebSocket connection
        session_id: Generation session to monitor
    """
    try:
        # Stream database updates for this session
        async for update in realtime_client.stream_generation_updates(session_id):
            await websocket.send_json(update)

    except asyncio.CancelledError:
        logger.info(f"Realtime streaming cancelled for session {session_id}")
    except Exception as e:
        logger.error(f"Realtime streaming error for session {session_id}: {e}")
        with contextlib.suppress(Exception):
            await websocket.send_json(
                {"type": "realtime_error", "session_id": session_id, "error": str(e)}
            )

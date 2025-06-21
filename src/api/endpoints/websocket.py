"""
WebSocket endpoints for real-time question generation.
"""

import contextlib
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.services.question_generation_service import QuestionGenerationService

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependencies (will be injected via dependency injection in production)
question_generation_service: Optional[QuestionGenerationService] = None


@router.websocket("/ws/generate/{session_id}")
async def websocket_generate(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time question generation with agent updates.

    Args:
        websocket: WebSocket connection
        session_id: Generation session identifier
    """
    await websocket.accept()

    try:
        # Send connection confirmation
        await websocket.send_json({"type": "connection_established", "session_id": session_id})

        while True:
            # Receive message from client
            data = await websocket.receive_json()

            if data.get("action") == "generate":
                # Start generation with real-time updates
                request_data = data.get("request", {})

                try:
                    # Stream generation updates
                    async for update in question_generation_service.generate_questions_stream(
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

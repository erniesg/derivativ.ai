"""
Supabase Realtime client for live database updates and WebSocket streaming.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any, Optional

from realtime import AsyncRealtimeChannel, AsyncRealtimeClient

logger = logging.getLogger(__name__)


class SupabaseRealtimeClient:
    """Client for Supabase Realtime subscriptions and live updates."""

    def __init__(self, url: str, api_key: str):
        """
        Initialize Realtime client.

        Args:
            url: Supabase project URL
            api_key: Supabase API key
        """
        # Convert HTTP URL to WebSocket URL for Realtime
        ws_url = url.replace("https://", "wss://").replace("http://", "ws://")
        self.realtime_url = f"{ws_url}/realtime/v1/websocket"
        self.api_key = api_key
        self.client: Optional[AsyncRealtimeClient] = None
        self.channels: dict[str, AsyncRealtimeChannel] = {}

    async def connect(self) -> None:
        """Establish WebSocket connection to Supabase Realtime."""
        if self.client is None:
            self.client = AsyncRealtimeClient(
                url=self.realtime_url,
                api_key=self.api_key,
                auto_reconnect=True,
                heartbeat_interval=30,
            )

        await self.client.connect()
        logger.info("Connected to Supabase Realtime")

    async def disconnect(self) -> None:
        """Close WebSocket connection and cleanup channels."""
        if self.client:
            # Remove all channels first
            for channel_name in list(self.channels.keys()):
                await self.unsubscribe(channel_name)

            await self.client.disconnect()
            self.client = None
            logger.info("Disconnected from Supabase Realtime")

    async def subscribe_to_table(
        self,
        table: str,
        event: str = "*",
        schema: str = "public",
        filter_condition: Optional[str] = None,
    ) -> AsyncRealtimeChannel:
        """
        Subscribe to table changes.

        Args:
            table: Table name to monitor
            event: Event type ('INSERT', 'UPDATE', 'DELETE', or '*' for all)
            schema: Database schema name
            filter_condition: Optional filter like 'status=eq.in_progress'

        Returns:
            Realtime channel for the subscription
        """
        if not self.client:
            await self.connect()

        # Create channel name
        channel_name = f"{schema}:{table}"
        if filter_condition:
            channel_name += f":{filter_condition}"

        # Create channel if it doesn't exist
        if channel_name not in self.channels:
            channel = self.client.channel(channel_name)

            # Configure subscription
            subscription_config = {
                "event": event,
                "schema": schema,
                "table": table,
            }
            if filter_condition:
                subscription_config["filter"] = filter_condition

            channel.on_postgres_changes(**subscription_config)
            self.channels[channel_name] = channel

            # Subscribe to the channel
            await channel.subscribe()
            logger.info(f"Subscribed to {channel_name}")

        return self.channels[channel_name]

    async def unsubscribe(self, channel_name: str) -> None:
        """
        Unsubscribe from a channel.

        Args:
            channel_name: Name of the channel to unsubscribe from
        """
        if channel_name in self.channels:
            channel = self.channels[channel_name]
            await channel.unsubscribe()
            del self.channels[channel_name]
            logger.info(f"Unsubscribed from {channel_name}")

    async def stream_generation_updates(
        self, session_id: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream real-time updates for a generation session.

        Args:
            session_id: Generation session UUID to monitor

        Yields:
            Real-time database updates for the session
        """
        try:
            # Subscribe to generation_sessions table updates for this session
            channel = await self.subscribe_to_table(
                table="generation_sessions",
                event="UPDATE",
                filter_condition=f"session_id=eq.{session_id}",
            )

            # Create async queue for updates
            update_queue: asyncio.Queue = asyncio.Queue()

            def handle_update(payload: dict[str, Any]) -> None:
                """Handle incoming database updates."""
                try:
                    update_queue.put_nowait(payload)
                except asyncio.QueueFull:
                    logger.warning("Update queue full, dropping update")

            # Register update handler
            channel.on_postgres_changes(callback=handle_update)

            # Initial status message
            yield {
                "type": "realtime_connected",
                "session_id": session_id,
                "message": "Connected to live updates",
            }

            # Stream updates as they arrive
            while True:
                try:
                    # Wait for next update with timeout
                    payload = await asyncio.wait_for(update_queue.get(), timeout=30.0)

                    # Parse the update
                    event_type = payload.get("eventType", "UPDATE")
                    new_record = payload.get("new", {})
                    old_record = payload.get("old", {})

                    # Extract relevant fields
                    update = {
                        "type": "database_update",
                        "event": event_type.lower(),
                        "session_id": session_id,
                        "changes": {},
                    }

                    # Compare old and new records for specific changes
                    if "status" in new_record and new_record.get("status") != old_record.get(
                        "status"
                    ):
                        update["changes"]["status"] = {
                            "old": old_record.get("status"),
                            "new": new_record.get("status"),
                        }

                    if "questions_generated" in new_record:
                        update["changes"]["questions_count"] = new_record["questions_generated"]

                    if "total_processing_time" in new_record:
                        update["changes"]["processing_time"] = new_record["total_processing_time"]

                    # Parse agent results if available
                    if "agent_results_json" in new_record:
                        try:
                            agent_results = json.loads(new_record["agent_results_json"])
                            if agent_results:
                                latest_result = agent_results[-1]
                                update["changes"]["latest_agent"] = {
                                    "name": latest_result.get("agent_name"),
                                    "status": "success"
                                    if latest_result.get("success")
                                    else "error",
                                    "reasoning": latest_result.get("reasoning_steps", []),
                                }
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass

                    yield update

                except asyncio.TimeoutError:
                    # Send keepalive message
                    yield {
                        "type": "keepalive",
                        "session_id": session_id,
                        "timestamp": asyncio.get_event_loop().time(),
                    }

        except Exception as e:
            logger.error(f"Error in generation updates stream: {e}")
            yield {
                "type": "error",
                "session_id": session_id,
                "error": str(e),
            }

        finally:
            # Cleanup subscription
            channel_name = f"public:generation_sessions:session_id=eq.{session_id}"
            await self.unsubscribe(channel_name)

    async def stream_question_updates(
        self, question_ids: list[str]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream real-time updates for specific questions.

        Args:
            question_ids: List of question IDs to monitor

        Yields:
            Real-time database updates for the questions
        """
        try:
            # Subscribe to questions table updates
            channel = await self.subscribe_to_table(
                table="generated_questions",
                event="*",  # Monitor all events
            )

            # Create async queue for updates
            update_queue: asyncio.Queue = asyncio.Queue()

            def handle_update(payload: dict[str, Any]) -> None:
                """Handle incoming question updates."""
                try:
                    # Filter for our specific questions
                    record = payload.get("new") or payload.get("old", {})
                    question_id = record.get("question_id_global")

                    if question_id in question_ids:
                        update_queue.put_nowait(payload)
                except asyncio.QueueFull:
                    logger.warning("Question update queue full, dropping update")

            # Register update handler
            channel.on_postgres_changes(callback=handle_update)

            # Stream filtered updates
            while True:
                try:
                    payload = await asyncio.wait_for(update_queue.get(), timeout=60.0)

                    event_type = payload.get("eventType", "UPDATE")
                    record = payload.get("new") or payload.get("old", {})

                    yield {
                        "type": "question_update",
                        "event": event_type.lower(),
                        "question_id": record.get("question_id_global"),
                        "quality_score": record.get("quality_score"),
                        "updated_at": record.get("updated_at"),
                    }

                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {"type": "keepalive", "timestamp": asyncio.get_event_loop().time()}

        except Exception as e:
            logger.error(f"Error in question updates stream: {e}")
            yield {"type": "error", "error": str(e)}

        finally:
            # Cleanup subscription
            await self.unsubscribe("public:questions")


# Global realtime client instance
_realtime_client: Optional[SupabaseRealtimeClient] = None


def get_realtime_client(url: str, api_key: str) -> SupabaseRealtimeClient:
    """
    Get singleton Realtime client instance.

    Args:
        url: Supabase project URL
        api_key: Supabase API key

    Returns:
        Configured Realtime client
    """
    global _realtime_client  # noqa: PLW0603

    if _realtime_client is None:
        _realtime_client = SupabaseRealtimeClient(url, api_key)

    return _realtime_client


async def cleanup_realtime_client() -> None:
    """Cleanup global Realtime client on shutdown."""
    global _realtime_client  # noqa: PLW0603

    if _realtime_client:
        await _realtime_client.disconnect()
        _realtime_client = None

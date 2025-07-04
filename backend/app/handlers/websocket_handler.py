"""
WebSocket handler for real-time communication.
"""
import json
import logging
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

from app.websocket_manager import connection_manager

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """Handles WebSocket connections and message processing."""
    
    def __init__(self):
        self.connection_manager = connection_manager
    
    async def handle_connection(self, websocket: WebSocket):
        """
        Handle a WebSocket connection lifecycle.
        
        Args:
            websocket: The WebSocket connection
        """
        session_id = None
        try:
            # Accept connection and get session ID
            session_id = await self.connection_manager.connect(websocket)

            # Send session established message to client
            await self.connection_manager.send_message(session_id, {
                "type": "session_established",
                "data": {"session_id": session_id},
                "timestamp": datetime.now().isoformat()
            })

            # Keep connection alive and handle incoming messages
            while True:
                try:
                    # Receive message from client
                    data = await websocket.receive_text()
                    logger.info(f"Received WebSocket message from {session_id}: {data}")

                    # Process the message
                    await self._process_message(session_id, data)

                except WebSocketDisconnect:
                    logger.info(f"WebSocket client disconnected: {session_id}")
                    break

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            if session_id:
                await self.connection_manager.disconnect(session_id)
    
    async def _process_message(self, session_id: str, data: str):
        """
        Process incoming WebSocket message.
        
        Args:
            session_id: The session ID
            data: Raw message data
        """
        try:
            message = json.loads(data)
            message_type = message.get("type", "unknown")

            if message_type == "ping":
                await self._handle_ping(session_id, message)
            elif message_type == "status_request":
                await self._handle_status_request(session_id)
            else:
                await self._handle_echo(session_id, message)

        except json.JSONDecodeError:
            await self._handle_invalid_json(session_id)
    
    async def _handle_ping(self, session_id: str, message: dict):
        """Handle ping message."""
        await self.connection_manager.send_message(session_id, {
            "type": "pong",
            "data": {"timestamp": message.get("timestamp")},
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def _handle_status_request(self, session_id: str):
        """Handle status request message."""
        status = self.connection_manager.get_session_status(session_id)
        await self.connection_manager.send_message(session_id, {
            "type": "status_response",
            "data": status or {"status": "connected"},
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def _handle_echo(self, session_id: str, message: dict):
        """Handle echo message."""
        await self.connection_manager.send_message(session_id, {
            "type": "echo",
            "data": message,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def _handle_invalid_json(self, session_id: str):
        """Handle invalid JSON message."""
        await self.connection_manager.send_message(session_id, {
            "type": "error",
            "data": {"message": "Invalid JSON format"},
            "timestamp": datetime.utcnow().isoformat()
        })


# Singleton instance
websocket_handler = WebSocketHandler() 
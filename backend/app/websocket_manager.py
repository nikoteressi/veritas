"""
WebSocket connection manager for real-time updates.
"""
import logging
import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from redis.asyncio.client import Redis

from app.json_utils import json_dumps
from app.redis_client import redis_manager
from app.schemas import ProgressEvent

logger = logging.getLogger(__name__)

SESSION_KEY_PREFIX = "ws:session:"
ACTIVE_CONNECTIONS_KEY = "ws:active_connections"


class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime
    session_id: Optional[str] = None


class ConnectionManager:
    """Manages WebSocket connections and message broadcasting."""
    
    def __init__(self):
        # Store active connections by session ID
        self.active_connections: Dict[str, WebSocket] = {}
    
    @property
    def redis(self) -> Redis:
        """Get a redis client from the redis_manager singleton."""
        return redis_manager.get_client()

    async def connect(self, websocket: WebSocket) -> str:
        """
        Accept a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            
        Returns:
            Session ID for the connection
        """
        await websocket.accept()
        session_id = str(uuid4())
        self.active_connections[session_id] = websocket
        
        # Store active connection in Redis
        await self.redis.hset(ACTIVE_CONNECTIONS_KEY, session_id, "active")

        logger.info(f"WebSocket connected: {session_id}")
        
        # Send welcome message
        await self.send_message(session_id, {
            "type": "connection_established",
            "data": {"session_id": session_id},
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return session_id
    
    async def disconnect(self, session_id: str):
        """
        Remove a WebSocket connection.
        
        Args:
            session_id: Session ID to disconnect
        """
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        
        # Remove from Redis
        await self.redis.hdel(ACTIVE_CONNECTIONS_KEY, session_id)
        await self.redis.delete(f"{SESSION_KEY_PREFIX}{session_id}")
        
        logger.info(f"WebSocket disconnected: {session_id}")
    
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """
        Send a message to a specific WebSocket connection.
        
        Args:
            session_id: Target session ID
            message: Message to send
        """
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                json_message = json_dumps(message)
                await websocket.send_text(json_message)
            except TypeError as e:
                logger.error(f"Failed to serialize message for {session_id}: {e}\\nMessage: {message}")
                # Try to send a fallback error message
                try:
                    await websocket.send_text(json_dumps({
                        "type": "error",
                        "data": {
                            "message": "Internal server error: could not serialize response.",
                            "code": "SERIALIZATION_ERROR"
                        }
                    }))
                except Exception:
                    pass  # Can't do much if this fails
            except Exception as e:
                logger.error(f"Failed to send message to {session_id}: {e}")
                # Remove broken connection
                await self.disconnect(session_id)
    
    async def broadcast_message(self, message: Dict[str, Any]):
        """
        Broadcast a message to all connected clients.
        
        Args:
            message: Message to broadcast
        """
        if not self.active_connections:
            return

        try:
            json_message = json_dumps(message)
        except TypeError as e:
            logger.error(f"Failed to serialize broadcast message: {e}\\nMessage: {message}")
            return  # Cannot send to anyone
        
        # Send to all connections
        disconnected = []
        for session_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json_message)
            except Exception as e:
                logger.error(f"Failed to broadcast to {session_id}: {e}")
                disconnected.append(session_id)
        
        # Clean up disconnected sessions
        for session_id in disconnected:
            await self.disconnect(session_id)
    
    async def send_event(
        self, 
        session_id: str, 
        event: ProgressEvent
    ):
        """
        Send a progress event to a specific session.
        
        Args:
            session_id: Target session ID
            event: Progress event to send
        """
        message = {
            "type": "progress_event",
            "data": event.model_dump(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await self.send_message(session_id, message)
        
        # Update session state in Redis
        session_key = f"{SESSION_KEY_PREFIX}{session_id}"
        await self.redis.hset(session_key, mapping={
            "current_event": event.event_name,
            "last_update": datetime.now(timezone.utc).isoformat()
        })

    async def send_progress_update(
        self, 
        session_id: str, 
        step: str, 
        progress: int,
        details: Optional[str] = None
    ):
        """
        Send a progress update to a specific session.
        
        Args:
            session_id: Target session ID
            step: Current processing step
            progress: Progress percentage (0-100)
            details: Optional additional details
        """
        message = {
            "type": "progress_update",
            "data": {
                "step": step,
                "progress": progress,
                "details": details,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        
        await self.send_message(session_id, message)
        
        # Update session state in Redis
        session_key = f"{SESSION_KEY_PREFIX}{session_id}"
        await self.redis.hset(session_key, mapping={
            "current_step": step,
            "progress": progress,
            "last_update": datetime.now(timezone.utc).isoformat()
        })
    
    async def send_verification_result(
        self, 
        session_id: str, 
        result: Dict[str, Any]
    ):
        """
        Send verification result to a specific session.
        
        Args:
            session_id: Target session ID
            result: Verification result data
        """
        message = {
            "type": "verification_result",
            "data": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await self.send_message(session_id, message)
        
        # Mark session as completed in Redis
        session_key = f"{SESSION_KEY_PREFIX}{session_id}"
        await self.redis.hset(session_key, mapping={
            "status": "completed",
            "result": json_dumps(result),
            "completed_at": datetime.now(timezone.utc).isoformat()
        })
    
    async def send_error(
        self, 
        session_id: str, 
        error_message: str,
        error_code: Optional[str] = None
    ):
        """
        Send an error message to a specific session.
        
        Args:
            session_id: Target session ID
            error_message: Error message
            error_code: Optional error code
        """
        message = {
            "type": "error",
            "data": {
                "message": error_message,
                "code": error_code,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        
        await self.send_message(session_id, message)
        
        # Mark session as failed in Redis
        session_key = f"{SESSION_KEY_PREFIX}{session_id}"
        await self.redis.hset(session_key, mapping={
            "status": "failed",
            "error": error_message,
            "failed_at": datetime.now(timezone.utc).isoformat()
        })
    
    async def start_verification_session(
        self, 
        session_id: str, 
        verification_data: Dict[str, Any]
    ):
        """
        Start a new verification session.
        
        Args:
            session_id: Session ID
            verification_data: Initial verification data
        """
        session_key = f"{SESSION_KEY_PREFIX}{session_id}"
        await self.redis.hset(session_key, mapping={
            "status": "started",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "current_step": "initializing",
            "progress": 0,
            "data": json.dumps(verification_data)
        })
        
        logger.info(f"Started verification session: {session_id}")
    
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a verification session from Redis.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session status dictionary or None if not found
        """
        session_key = f"{SESSION_KEY_PREFIX}{session_id}"
        return await self.redis.hgetall(session_key)
    
    async def get_active_sessions(self) -> List[str]:
        """Get a list of active session IDs from Redis."""
        return list(await self.redis.hkeys(ACTIVE_CONNECTIONS_KEY))
    
    async def get_verification_sessions(self) -> List[Dict[str, Any]]:
        """
        Get all verification session details from Redis.
        
        Note: This can be a very expensive operation if there are many sessions.
        Use with caution in a production environment.
        """
        session_keys = await self.redis.keys(f"{SESSION_KEY_PREFIX}*")
        
        if not session_keys:
            return []
            
        sessions = []
        for key in session_keys:
            session_data = await self.redis.hgetall(key)
            if session_data:
                sessions.append(session_data)
                
        return sessions


class EventProgressTracker:
    """Event-driven progress tracker for verification operations."""
    
    def __init__(self, connection_manager: ConnectionManager, session_id: str):
        self.connection_manager = connection_manager
        self.session_id = session_id
    
    async def emit_event(self, event: ProgressEvent):
        """
        Emit a progress event to the WebSocket connection.
        
        Args:
            event: Progress event to emit
        """
        await self.connection_manager.send_event(self.session_id, event)
        logger.info(f"Progress event [{self.session_id}]: {event.event_name}")
    
    async def complete(self, result: Dict[str, Any]):
        """
        Mark progress as complete and send result.
        
        Args:
            result: Final verification result
        """
        await self.connection_manager.send_verification_result(
            self.session_id, result
        )
        
        logger.info(f"Verification completed [{self.session_id}]")
    
    async def error(self, error_message: str, error_code: Optional[str] = None):
        """
        Mark progress as failed and send error.
        
        Args:
            error_message: Error message
            error_code: Optional error code
        """
        await self.connection_manager.send_error(
            self.session_id, error_message, error_code
        )
        
        logger.error(f"Verification failed [{self.session_id}]: {error_message}")


class ProgressTracker:
    """Legacy progress tracker for backward compatibility."""
    
    def __init__(self, connection_manager: ConnectionManager, session_id: str):
        self.connection_manager = connection_manager
        self.session_id = session_id
        self.current_progress = 0
    
    async def update(self, step: str, progress: int, details: Optional[str] = None):
        """
        Update progress and send to WebSocket.
        
        Args:
            step: Current processing step
            progress: Progress percentage (0-100)
            details: Optional additional details
        """
        self.current_progress = progress
        await self.connection_manager.send_progress_update(
            self.session_id, step, progress, details
        )
        
        logger.info(f"Progress update [{self.session_id}]: {step} ({progress}%)")
    
    async def complete(self, result: Dict[str, Any]):
        """
        Mark progress as complete and send result.
        
        Args:
            result: Final verification result
        """
        await self.connection_manager.send_verification_result(
            self.session_id, result
        )
        
        logger.info(f"Verification completed [{self.session_id}]")
    
    async def error(self, error_message: str, error_code: Optional[str] = None):
        """
        Mark progress as failed and send error.
        
        Args:
            error_message: Error message
            error_code: Optional error code
        """
        await self.connection_manager.send_error(
            self.session_id, error_message, error_code
        )
        
        logger.error(f"Verification failed [{self.session_id}]: {error_message}")


# Global connection manager instance
connection_manager = ConnectionManager()

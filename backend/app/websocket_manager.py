"""
WebSocket connection manager for real-time updates.
"""
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.json_utils import json_dumps

logger = logging.getLogger(__name__)


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
        # Store verification sessions
        self.verification_sessions: Dict[str, Dict[str, Any]] = {}
    
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
        
        logger.info(f"WebSocket connected: {session_id}")
        
        # Send welcome message
        await self.send_message(session_id, {
            "type": "connection_established",
            "data": {"session_id": session_id},
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return session_id
    
    def disconnect(self, session_id: str):
        """
        Remove a WebSocket connection.
        
        Args:
            session_id: Session ID to disconnect
        """
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        
        if session_id in self.verification_sessions:
            del self.verification_sessions[session_id]
        
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
                self.disconnect(session_id)
    
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
            self.disconnect(session_id)
    
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
        
        # Update session state
        if session_id in self.verification_sessions:
            self.verification_sessions[session_id].update({
                "current_step": step,
                "progress": progress,
                "last_update": datetime.now(timezone.utc)
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
        
        # Mark session as completed
        if session_id in self.verification_sessions:
            self.verification_sessions[session_id].update({
                "status": "completed",
                "result": result,
                "completed_at": datetime.now(timezone.utc)
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
        
        # Mark session as failed
        if session_id in self.verification_sessions:
            self.verification_sessions[session_id].update({
                "status": "failed",
                "error": error_message,
                "failed_at": datetime.now(timezone.utc)
            })
    
    def start_verification_session(
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
        self.verification_sessions[session_id] = {
            "status": "started",
            "started_at": datetime.now(timezone.utc),
            "current_step": "initializing",
            "progress": 0,
            "data": verification_data
        }
        
        logger.info(f"Started verification session: {session_id}")
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a verification session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session status data or None if not found
        """
        return self.verification_sessions.get(session_id)
    
    def get_active_sessions(self) -> List[str]:
        """
        Get list of active session IDs.
        
        Returns:
            List of active session IDs
        """
        return list(self.active_connections.keys())
    
    def get_verification_sessions(self) -> List[Dict[str, Any]]:
        """
        Get list of all verification sessions.
        
        Returns:
            List of verification session data
        """
        sessions = []
        for session_id, data in self.verification_sessions.items():
            session_info = data.copy()
            session_info["session_id"] = session_id
            sessions.append(session_info)
        
        return sessions


class ProgressTracker:
    """Tracks progress for verification operations."""
    
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

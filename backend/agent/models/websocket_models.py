"""
Pydantic models for WebSocket communication.
"""
from typing import Optional, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field


class ProgressEventPayload(BaseModel):
    """Progress event payload."""
    step_name: Optional[str] = Field(None, description="Step name")
    progress: Optional[float] = Field(None, description="Progress (0-1)")
    total_steps: Optional[int] = Field(None, description="Total number of steps")
    current_step: Optional[int] = Field(None, description="Current step")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data")


class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: Optional[str] = Field(None, description="Message timestamp")

class WebSocketResponse(BaseModel):
    """WebSocket response model."""
    status: str = Field(..., description="Response status")
    message: str = Field(..., description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
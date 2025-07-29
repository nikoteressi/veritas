"""
Pydantic schemas for WebSocket communication models.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ProgressEvent(BaseModel):
    """Event model for progress tracking in the verification pipeline."""

    event_name: str = Field(..., description="The unique name of the event, e.g., 'CLAIMS_EXTRACTED'.")
    payload: dict[str, Any] = Field(default_factory=dict, description="A dictionary containing event-specific data.")


class WebSocketMessage(BaseModel):
    """WebSocket message model."""

    type: str  # status_update, progress, result, error
    data: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)

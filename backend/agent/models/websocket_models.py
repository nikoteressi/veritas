"""
from __future__ import annotations

Pydantic models for WebSocket communication.
"""

from typing import Any

from pydantic import BaseModel, Field


class WebSocketMessage(BaseModel):
    """WebSocket message model."""

    type: str = Field(..., description="Message type")
    data: dict[str, Any] = Field(..., description="Message data")
    timestamp: str | None = Field(None, description="Message timestamp")


class WebSocketResponse(BaseModel):
    """WebSocket response model."""

    status: str = Field(..., description="Response status")
    message: str = Field(..., description="Response message")
    data: dict[str, Any] | None = Field(None, description="Response data")

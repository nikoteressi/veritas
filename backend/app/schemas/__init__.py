"""
This package defines the data schemas for the application's public interfaces.

- `api.py`: Defines the public REST API request/response models.
- `websocket.py`: Defines models for WebSocket communication.
"""
from .api import (
    ErrorResponse,
    ReputationStatsResponse,
    UserReputation,
    VerificationRequest,
    VerificationResponse,
    VerificationStatusResponse,
)
from .websocket import ProgressEvent, WebSocketMessage

__all__ = [
    "ErrorResponse",
    "ReputationStatsResponse",
    "UserReputation",
    "VerificationRequest",
    "VerificationResponse",
    "VerificationStatusResponse",
    "ProgressEvent",
    "WebSocketMessage",
]

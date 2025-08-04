"""
Pydantic schemas for WebSocket communication models.
"""

from datetime import datetime
from typing import Any, Union, Literal
from pydantic import BaseModel, Field

# Import new progress models
from app.models.progress import StepsDefinitionData, ProgressUpdateData, StepUpdateData


class ProgressEvent(BaseModel):
    """Event model for progress tracking in the verification pipeline."""

    event_name: str = Field(
        ..., description="The unique name of the event, e.g., 'CLAIMS_EXTRACTED'.")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="A dictionary containing event-specific data.")


class WebSocketMessage(BaseModel):
    """WebSocket message model."""

    type: str  # status_update, progress, result, error
    data: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Enhanced progress-specific WebSocket messages
class EnhancedProgressEvent(ProgressEvent):
    """Enhanced progress event with type-safe payload"""
    payload: Union[StepsDefinitionData, ProgressUpdateData, StepUpdateData]

    class Config:
        use_enum_values = True


class StepsDefinitionMessage(BaseModel):
    """WebSocket message for steps definition."""

    type: Literal["steps_definition"] = "steps_definition"
    data: StepsDefinitionData
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ProgressUpdateMessage(BaseModel):
    """WebSocket message for progress updates."""

    type: Literal["progress_update"] = "progress_update"
    data: ProgressUpdateData
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StepUpdateMessage(BaseModel):
    """WebSocket message for step updates."""

    type: Literal["step_update"] = "step_update"
    data: StepUpdateData
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ProgressWebSocketMessage(BaseModel):
    """WebSocket message specifically for progress tracking"""
    type: Literal["steps_definition", "progress_update", "step_update"]
    data: Union[StepsDefinitionData, ProgressUpdateData, StepUpdateData]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def validate_data_type(self, v, values):
        """Ensure data type matches message type"""
        msg_type = values.get('type')
        if msg_type == "steps_definition" and not isinstance(v, StepsDefinitionData):
            raise ValueError(
                "Data must be StepsDefinitionData for steps_definition type")
        elif msg_type == "progress_update" and not isinstance(v, ProgressUpdateData):
            raise ValueError(
                "Data must be ProgressUpdateData for progress_update type")
        elif msg_type == "step_update" and not isinstance(v, StepUpdateData):
            raise ValueError(
                "Data must be StepUpdateData for step_update type")
        return v


# Union type for all progress-related WebSocket messages
ProgressWebSocketMessageUnion = Union[
    StepsDefinitionMessage,
    ProgressUpdateMessage,
    StepUpdateMessage
]

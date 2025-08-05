"""
Pydantic schemas for WebSocket communication models.
"""

from datetime import datetime
from typing import Any, Union, Literal
from pydantic import BaseModel, Field

# Import new progress models
from app.models.progress import StepsDefinitionData, StepUpdateData


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
    payload: Union[StepsDefinitionData, StepUpdateData]

    class Config:
        use_enum_values = True


class StepsDefinitionMessage(BaseModel):
    """WebSocket message for steps definition."""

    type: Literal["steps_definition"] = "steps_definition"
    data: StepsDefinitionData
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StepUpdateMessage(BaseModel):
    """WebSocket message for step updates."""

    type: Literal["step_update"] = "step_update"
    data: StepUpdateData
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ProgressWebSocketMessage(BaseModel):
    """Union type for all progress-related WebSocket messages"""
    type: Literal["steps_definition", "step_update"]
    data: Union[StepsDefinitionData, StepUpdateData]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def validate_data_type(cls, v, info):
        """Validate that data type matches message type"""
        msg_type = info.data.get('type')
        if msg_type == "steps_definition" and not isinstance(v, StepsDefinitionData):
            raise ValueError(
                "Data must be StepsDefinitionData for steps_definition type")
        elif msg_type == "step_update" and not isinstance(v, StepUpdateData):
            raise ValueError(
                "Data must be StepUpdateData for step_update type")
        return v


# Union type for all progress-related WebSocket messages
ProgressWebSocketMessageUnion = Union[
    StepsDefinitionMessage,
    StepUpdateMessage
]

"""
Enhanced Progress Tracking Models

This module defines Pydantic models for the new progress tracking system
that centralizes step management and enables smooth progress animations.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from enum import Enum


class StepStatus(str, Enum):
    """Enum for step status values."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProgressStep(BaseModel):
    """Enhanced Pydantic model for progress step definition."""

    id: str = Field(..., description="Unique step identifier")
    name: str = Field(..., description="Human-readable step name")
    description: str = Field(..., description="Step description")
    estimated_duration: int = Field(..., ge=0,
                                    description="Estimated duration in seconds")
    weight: float = Field(..., ge=0.1, le=100.0,
                          description="Step weight for progress calculation")
    status: StepStatus = Field(
        default=StepStatus.PENDING, description="Current step status")
    substeps: Optional[List['ProgressStep']] = Field(
        default=None, description="Optional substeps")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional step metadata")
    started_at: Optional[datetime] = Field(
        default=None, description="Step start timestamp")
    completed_at: Optional[datetime] = Field(
        default=None, description="Step completion timestamp")
    progress_percentage: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Current step progress")

    def validate_weight(self, v):
        if v <= 0:
            raise ValueError('Weight must be positive')
        return v

    def validate_duration(self, v):
        if v < 0:
            raise ValueError('Duration cannot be negative')
        return v

    class Config:
        # Enable forward references for self-referencing substeps
        validate_by_name = True
        use_enum_values = True


# Update forward reference
ProgressStep.model_rebuild()


class PipelineConfiguration(BaseModel):
    """Configuration for the entire pipeline."""

    steps: List[ProgressStep] = Field(...,
                                      description="List of pipeline steps")
    session_id: str = Field(..., description="Session identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def validate_steps_not_empty(self, v):
        if not v:
            raise ValueError('Steps list cannot be empty')
        return v

    def calculate_total_weight(self) -> float:
        """Calculate total weight of all steps."""
        return sum(step.weight for step in self.steps)

    def get_step_by_id(self, step_id: str) -> Optional[ProgressStep]:
        """Get step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None


class SmartProgressCalculator(BaseModel):
    """Type-safe progress calculator using Pydantic models."""

    pipeline_config: PipelineConfiguration
    start_time: datetime = Field(default_factory=datetime.utcnow)

    def calculate_weighted_progress(self, current_step_progress: float = 0.0) -> float:
        """
        Calculate progress based on step weights and completion.

        Args:
            current_step_progress: Progress of current step (0.0-1.0)

        Returns:
            Overall progress percentage (0.0-100.0)
        """
        total_weight = self.pipeline_config.calculate_total_weight()
        completed_weight = 0.0
        current_step_weight = 0.0

        for step in self.pipeline_config.steps:
            if step.status == StepStatus.COMPLETED:
                completed_weight += step.weight
            elif step.status == StepStatus.IN_PROGRESS:
                current_step_weight = step.weight * current_step_progress

        return min(100.0, ((completed_weight + current_step_weight) / total_weight) * 100.0)

    def get_smooth_progress_updates(self,
                                    start_progress: float,
                                    target_progress: float,
                                    duration_ms: int = 300,
                                    fps: int = 30) -> List[Tuple[float, int]]:
        """
        Generate intermediate progress values for smooth animation.

        Args:
            start_progress: Starting progress percentage
            target_progress: Target progress percentage
            duration_ms: Animation duration in milliseconds
            fps: Target frames per second

        Returns:
            List of (progress, timestamp_offset) tuples
        """
        # Ensure minimum duration for smooth animation
        duration_ms = max(duration_ms, 200)

        frame_count = int((duration_ms / 1000.0) * fps)
        if frame_count <= 1:
            return [(target_progress, duration_ms)]

        progress_diff = target_progress - start_progress
        frame_duration = duration_ms / frame_count

        updates = []
        for i in range(frame_count + 1):
            # Apply easing function (ease-out cubic)
            t = i / frame_count
            eased_t = 1 - pow(1 - t, 3)

            current_progress = start_progress + (progress_diff * eased_t)
            timestamp_offset = int(i * frame_duration)

            updates.append((current_progress, timestamp_offset))

        return updates

    def get_current_step(self) -> Optional[ProgressStep]:
        """Get the currently active step."""
        for step in self.pipeline_config.steps:
            if step.status == StepStatus.IN_PROGRESS:
                return step
        return None

    def update_step_status(self, step_id: str, status: StepStatus, progress: float = 0.0) -> bool:
        """
        Update step status and progress.

        Args:
            step_id: Step identifier
            status: New step status
            progress: Step progress (0.0-1.0)

        Returns:
            True if step was found and updated, False otherwise
        """
        step = self.pipeline_config.get_step_by_id(step_id)
        if not step:
            return False

        step.status = status
        step.progress_percentage = progress * 100.0

        if status == StepStatus.IN_PROGRESS and not step.started_at:
            step.started_at = datetime.now()
        elif status in [StepStatus.COMPLETED, StepStatus.FAILED] and not step.completed_at:
            step.completed_at = datetime.now()

        return True


# WebSocket Message Data Models
class StepsDefinitionData(BaseModel):
    """Data model for steps definition message."""

    steps: List[ProgressStep] = Field(...,
                                      description="List of pipeline steps")
    session_id: str = Field(..., description="Session identifier")


class StepUpdateData(BaseModel):
    """Data model for step update message."""

    step_id: str = Field(..., description="Step identifier")
    status: StepStatus = Field(..., description="New step status")
    progress: float = Field(..., ge=0, le=100,
                            description="Step progress percentage")
    message: str = Field(..., description="Step status message")
    duration: Optional[int] = Field(
        None, ge=0, description="Step duration in ms if completed")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional step metadata")

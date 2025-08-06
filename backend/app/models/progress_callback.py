"""
Progress callback interfaces for reporting progress during pipeline execution.
"""

from abc import ABC, abstractmethod

from app.models.progress import StepStatus


class ProgressCallback(ABC):
    """Interface for progress reporting callbacks."""

    @abstractmethod
    async def update_progress(
        self,
        current: float,
        total: float,
        message: str | None = None,
        use_smooth_animation: bool = True
    ) -> None:
        """Update progress with current/total values."""

    @abstractmethod
    async def update_substep(
        self,
        substep_name: str,
        progress: float,
        message: str | None = None
    ) -> None:
        """Update progress for a substep."""


class PipelineProgressCallback(ProgressCallback):
    """Implementation for pipeline steps."""

    def __init__(self, progress_manager, session_id: str, step_id: str, last_progress: float = 0.0):
        self.progress_manager = progress_manager
        self.session_id = session_id
        self.step_id = step_id
        self.last_progress = last_progress

    async def update_progress(
        self,
        current: float,
        total: float,
        message: str | None = None,
        use_smooth_animation: bool = True
    ) -> None:
        """Update progress with smooth animations for significant changes."""
        progress = min(1.0, current / total) if total > 0 else 0.0

        # Determine status based on progress
        status = StepStatus.COMPLETED if progress >= 1.0 else StepStatus.IN_PROGRESS

        if use_smooth_animation and (abs(progress - self.last_progress) > 0.1 or progress >= 1.0):
            # Use smooth animation for significant progress changes OR step completion
            await self.progress_manager.update_smooth_progress(
                session_id=self.session_id,
                step_id=self.step_id,
                start_progress=self.last_progress,
                target_progress=progress,
                duration_ms=500,
                message=message or f"Processing {current:.0f}/{total:.0f}..."
            )

            # Note: Step completion is handled by update_smooth_progress when target_progress >= 1.0
            # No need to send additional completion status here to avoid duplicates
        else:
            # Use regular update for small changes
            await self.progress_manager.update_step_status(
                session_id=self.session_id,
                step_id=self.step_id,
                status=status,
                progress=progress,
                message=message or f"Processing {current:.0f}/{total:.0f}..."
            )

        self.last_progress = progress

    async def update_substep(
        self,
        substep_name: str,
        progress: float,
        message: str | None = None
    ) -> None:
        """Update progress for a substep."""
        # Convert progress from percentage (0-100) to fraction (0.0-1.0) if needed
        normalized_progress = progress / 100.0 if progress > 1.0 else progress

        # Determine status based on normalized progress
        status = StepStatus.COMPLETED if normalized_progress >= 1.0 else StepStatus.IN_PROGRESS

        # Format message to include substep information
        formatted_message = message or f"{substep_name}: {normalized_progress:.1%}"

        await self.progress_manager.update_step_status(
            session_id=self.session_id,
            step_id=self.step_id,
            status=status,
            progress=normalized_progress,
            message=formatted_message
        )

        # Update last_progress to track changes
        self.last_progress = normalized_progress


class NoOpProgressCallback(ProgressCallback):
    """No-operation progress callback for cases where progress reporting is not needed."""

    async def update_progress(
        self,
        current: float,
        total: float,
        message: str | None = None,
        use_smooth_animation: bool = True
    ) -> None:
        """No-op implementation."""

    async def update_substep(
        self,
        substep_name: str,
        progress: float,
        message: str | None = None
    ) -> None:
        """No-op implementation."""

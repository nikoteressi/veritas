"""
Base pipeline step for the verification workflow.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from agent.models.verification_context import VerificationContext
from app.exceptions import AgentError
from app.models.progress_callback import ProgressCallback, NoOpProgressCallback


class BasePipelineStep(ABC):
    """Base class for verification pipeline steps."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.progress_callback: ProgressCallback = NoOpProgressCallback()

    def set_progress_callback(self, callback: Optional[ProgressCallback]) -> None:
        """Set the progress callback for this step."""
        self.progress_callback = callback or NoOpProgressCallback()

    async def update_progress(self, progress: float, message: str = "") -> None:
        """Update the progress for this step."""
        await self.progress_callback.update_progress(progress, 1.0, message)

    async def update_substep(self, substep: str, progress: float = 0.0) -> None:
        """Update the current substep being executed."""
        await self.progress_callback.update_substep(substep, progress)

    @abstractmethod
    async def execute(self, context: VerificationContext) -> VerificationContext:
        """
        Execute the verification step.

        Args:
            context: Verification context containing all necessary data

        Returns:
            Updated verification context
        """

    async def safe_execute(self, context: VerificationContext) -> VerificationContext:
        """
        Safely execute the step with error handling and progress tracking.

        Args:
            context: The verification context

        Returns:
            Updated verification context

        Raises:
            AgentError: If step execution fails
        """
        try:
            self.logger.info("🚀 Starting step: %s", self.name)

            # Update progress to 0% at start
            if self.progress_callback:
                await self.progress_callback.update_progress(0, 100, f"Starting {self.name}")

            # Execute the step
            result_context = await self.execute(context)

            # Note: Step completion is handled automatically by PipelineProgressCallback
            # when progress reaches 100%, so we don't call update_progress(100%) here
            # to avoid duplicate completion messages
            return result_context

        except Exception as e:
            self.logger.error("Step %s failed: %s",
                              self.name, e, exc_info=True)

            # Update progress to indicate failure
            if self.progress_callback:
                self.logger.info(
                    f"BasePipelineStep.safe_execute: Updating progress to 0% for failed step '{self.name}'")
                await self.progress_callback.update_progress(0, 100, f"Failed: {str(e)}")

            raise AgentError(f"Step '{self.name}' failed: {e}") from e

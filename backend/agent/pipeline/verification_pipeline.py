"""
Configurable verification pipeline service for orchestrating verification steps.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from agent.models.verification_context import VerificationContext
from agent.pipeline.pipeline_steps import BasePipelineStep, step_registry
from app.config import VerificationSteps, settings
from app.exceptions import AgentError
from app.models.progress import StepStatus
from app.models.progress_callback import PipelineProgressCallback
from app.services.progress_manager import progress_manager

from ..services.infrastructure.event_emission import EventEmissionService
from ..services.infrastructure.storage import storage_service
from ..services.output.result_compiler import ResultCompiler

logger = logging.getLogger(__name__)


class VerificationPipeline:
    """
    Configurable pipeline for executing verification steps in sequence.

    This class manages the execution of individual verification steps using the
    VerificationContext model, providing better separation of concerns, type safety,
    and configurability.
    """

    def __init__(self, step_names: list[str] | None = None):
        """
        Initialize the verification pipeline.

        Args:
            step_names: Optional list of step names to use. If None, uses default configuration.
        """
        if step_names is None:
            step_names = settings.get_pipeline_steps()

        try:
            self.steps = step_registry.create_pipeline_steps(step_names)
            self.step_names = step_names
            logger.info(
                "Initialized verification pipeline with steps: %s",
                step_names,
            )
        except Exception as e:
            logger.error(
                "Failed to initialize pipeline with steps %s: %s",
                step_names,
                e,
            )
            raise AgentError(f"Pipeline initialization failed: {e}") from e

    async def execute(
        self,
        image_bytes: bytes,
        user_prompt: str,
        db: AsyncSession,
        session_id: str,
        progress_callback: Callable | None = None,
        event_callback: Callable | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute the complete verification pipeline.

        Args:
            image_bytes: Image content to analyze
            user_prompt: User's question/prompt
            db: Database session
            session_id: Session identifier
            progress_callback: Optional progress callback
            filename: Optional filename for display

        Returns:
            Complete verification result
        """
        # Initialize verification context
        try:
            context = VerificationContext(
                image_bytes=image_bytes,
                user_prompt=user_prompt,
                session_id=session_id,
                filename=filename,
                db=db,
                event_service=(EventEmissionService(
                    event_callback) if event_callback else None),
                result_compiler=ResultCompiler(),
                progress_manager=progress_manager,
            )
        except Exception as e:
            logger.error("Failed to create verification context: %s", e)
            raise AgentError(f"Context initialization failed: {e}") from e

        # Start timing
        if context.result_compiler:
            result_compiler = context.result_compiler
            result_compiler.start_timing()

        # Initialize progress tracking session
        try:
            await progress_manager.start_session(session_id)
            logger.info(f"Started progress tracking for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to start progress tracking: {e}")
            # Continue without progress tracking

        # Emit verification started event
        if context.event_service:
            await context.event_service.emit_verification_started()

        try:
            # Execute each step in sequence with progress tracking
            for i, step in enumerate(self.steps):
                # Map step name to VerificationSteps enum
                verification_step = self._map_step_to_enum(step.name)

                if verification_step:
                    # Get current progress for this step
                    current_progress = progress_manager.get_step_progress(
                        session_id, verification_step.value
                    )

                    # Set up progress callback for this step
                    progress_callback = PipelineProgressCallback(
                        progress_manager,
                        session_id,
                        verification_step.value,
                        last_progress=current_progress
                    )
                    step.set_progress_callback(progress_callback)

                    # Update step status to IN_PROGRESS
                    await progress_manager.update_step_status(
                        session_id=session_id,
                        step_id=verification_step.value,
                        status=StepStatus.IN_PROGRESS,
                        progress=0.0,
                        message=f"Starting {step.name}..."
                    )

                # Execute step safely
                context = await step.safe_execute(context)

                # Note: Step completion is now handled automatically by PipelineProgressCallback
                # when progress reaches 100%, so no need to manually update status here

            # Compile final result
            final_result = await self._compile_final_result(context)

            # Store in vector database
            await storage_service.store_in_vector_db(final_result)

            # Complete progress tracking session
            await progress_manager.complete_session(session_id)

            # Emit verification completed event
            if context.event_service:
                await context.event_service.emit_verification_completed()

            processing_time = context.result_compiler.get_processing_time()
            logger.info(
                "Verification completed in %s: %s",
                processing_time,
                context.verdict_result.verdict,
            )

            return final_result

        except Exception as e:
            logger.error("Pipeline execution failed: %s", e, exc_info=True)

            # Mark current step as failed if we can identify it
            try:
                # Get current step enum, which may be None
                current_step = self._get_current_step_enum(
                ) if self._get_current_step_enum() is not None else None
                if current_step:
                    await progress_manager.update_step_status(
                        session_id=session_id,
                        step_id=current_step.value,
                        status=StepStatus.FAILED,
                        progress=0.0,
                        message=f"Failed: {str(e)}"
                    )
            except Exception:
                pass  # Don't let progress tracking errors mask the original error

            raise AgentError(f"Verification pipeline failed: {e}") from e

    def _map_step_to_enum(self, step_name: str) -> VerificationSteps | None:
        """Map step name to VerificationSteps enum."""
        step_mapping = {
            "Validation": VerificationSteps.VALIDATION,
            "Screenshot Parsing": VerificationSteps.SCREENSHOT_PARSING,
            "Temporal Analysis": VerificationSteps.TEMPORAL_ANALYSIS,
            "Post Analysis": VerificationSteps.POST_ANALYSIS,
            "Reputation Retrieval": VerificationSteps.REPUTATION_RETRIEVAL,
            "Fact Checking": VerificationSteps.FACT_CHECKING,
            "Summarization": VerificationSteps.SUMMARIZATION,
            "Verdict Generation": VerificationSteps.VERDICT_GENERATION,
            "Motives Analysis": VerificationSteps.MOTIVES_ANALYSIS,
            "Reputation Update": VerificationSteps.REPUTATION_UPDATE,
        }
        return step_mapping.get(step_name)

    def _get_current_step_enum(self) -> VerificationSteps | None:
        """Get the current step enum (placeholder for now)."""
        # This would need to be implemented based on current execution state
        # For now, return None to avoid errors
        return None

    async def _compile_final_result(self, context: VerificationContext) -> dict[str, Any]:
        """Compile the final verification result."""

        final_result = await context.result_compiler.compile_result(context=context)

        return final_result

    def get_step_names(self) -> list[str]:
        """Get the list of step names in this pipeline."""
        return self.step_names.copy()

    def get_steps(self) -> list[BasePipelineStep]:
        """Get the list of step instances in this pipeline."""
        return self.steps.copy()

    def add_step(self, step_name: str, position: int | None = None) -> None:
        """
        Add a step to the pipeline.

        Args:
            step_name: Name of the step to add
            position: Position to insert the step. If None, appends to the end.
        """
        try:
            step = step_registry.create_step(step_name)

            if position is None:
                self.steps.append(step)
                self.step_names.append(step_name)
            else:
                self.steps.insert(position, step)
                self.step_names.insert(position, step_name)

            logger.info(
                "Added step '%s' to pipeline at position %s",
                step_name,
                position or len(self.steps) - 1,
            )
        except Exception as e:
            logger.error(
                "Failed to add step '%s' to pipeline: %s",
                step_name,
                e,
            )
            raise AgentError(f"Failed to add step to pipeline: {e}") from e

    def remove_step(self, step_name: str) -> bool:
        """
        Remove a step from the pipeline.

        Args:
            step_name: Name of the step to remove

        Returns:
            True if the step was removed, False if it wasn't found
        """
        try:
            if step_name in self.step_names:
                index = self.step_names.index(step_name)
                self.steps.pop(index)
                self.step_names.remove(step_name)
                logger.info(
                    "Removed step '%s' from pipeline",
                    step_name,
                )
                return True
            return False
        except Exception as e:
            logger.error(
                "Failed to remove step '%s' from pipeline: %s",
                step_name,
                e,
            )
            raise AgentError(
                f"Failed to remove step from pipeline: {e}") from e

    def reorder_steps(self, new_step_names: list[str]) -> None:
        """
        Reorder pipeline steps.

        Args:
            new_step_names: New order of step names
        """
        try:
            # Validate that all step names are available
            for step_name in new_step_names:
                if step_name not in step_registry.get_available_steps():
                    raise ValueError(f"Unknown step: {step_name}")

            # Create new step instances
            new_steps = step_registry.create_pipeline_steps(new_step_names)

            self.steps = new_steps
            self.step_names = new_step_names

            logger.info(
                "Reordered pipeline steps to: %s",
                new_step_names,
            )
        except Exception as e:
            logger.error(
                "Failed to reorder pipeline steps: %s",
                e,
            )
            raise AgentError(f"Failed to reorder pipeline steps: {e}") from e

    async def close(self) -> None:
        """Close all pipeline steps and release resources."""
        logger.info("Closing verification pipeline...")

        for step in self.steps:
            try:
                if hasattr(step, "close") and callable(step.close):
                    await step.close()
                    logger.debug("Closed step: %s", step.name)
            except Exception as e:
                logger.error("Error closing step %s: %s", step.name, e)

        logger.info("Verification pipeline closed successfully")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Create default pipeline instance
def create_default_pipeline() -> VerificationPipeline:
    """Create a pipeline with default configuration."""
    return VerificationPipeline()


# Create customizable pipeline instance
def create_custom_pipeline(step_names: list[str]) -> VerificationPipeline:
    """Create a pipeline with custom step configuration."""
    return VerificationPipeline(step_names)


# Global default pipeline instance
verification_pipeline = create_default_pipeline()

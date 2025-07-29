"""
Configurable verification pipeline service for orchestrating verification steps.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from agent.models.verification_context import VerificationContext
from agent.pipeline.pipeline_steps import BasePipelineStep, step_registry
from app.config import settings
from app.exceptions import AgentError
from sqlalchemy.ext.asyncio import AsyncSession

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
                event_service=(EventEmissionService(event_callback) if event_callback else None),
                result_compiler=ResultCompiler(),
            )
        except Exception as e:
            logger.error("Failed to create verification context: %s", e)
            raise AgentError(f"Context initialization failed: {e}") from e

        # Start timing
        if context.result_compiler:
            result_compiler = context.result_compiler
            result_compiler.start_timing()

        # Emit verification started event
        if context.event_service:
            await context.event_service.emit_verification_started()

        try:
            # Execute each step in sequence
            for step in self.steps:
                # Execute step safely
                context = await step.safe_execute(context)

            # Compile final result
            final_result = await self._compile_final_result(context)

            # Store in vector database
            await storage_service.store_in_vector_db(final_result)

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
            raise AgentError(f"Verification pipeline failed: {e}") from e

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
            raise AgentError(f"Failed to remove step from pipeline: {e}") from e

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

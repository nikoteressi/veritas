"""
Lightweight coordinator for the verification workflow.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from agent.pipeline.verification_pipeline import verification_pipeline
from app.exceptions import AgentError

logger = logging.getLogger(__name__)


class WorkflowCoordinator:
    """
    Lightweight coordinator that orchestrates the verification workflow.

    This class now delegates to a modular pipeline for better separation of concerns.
    """

    def __init__(self):
        self.pipeline = verification_pipeline

    async def execute_verification(
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
        Execute the complete verification workflow.

        This method now delegates to the modular verification pipeline
        for better separation of concerns and maintainability.

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
        try:
            return await self.pipeline.execute(
                image_bytes=image_bytes,
                user_prompt=user_prompt,
                db=db,
                session_id=session_id,
                progress_callback=progress_callback,
                event_callback=event_callback,
                filename=filename,
            )
        except Exception as e:
            logger.error(f"Workflow coordination failed: {e}", exc_info=True)
            raise AgentError(f"Verification workflow failed: {e}") from e

    async def close(self) -> None:
        """Close the workflow coordinator and release resources."""
        logger.info("Closing workflow coordinator...")

        try:
            if hasattr(self.pipeline, "close") and callable(self.pipeline.close):
                await self.pipeline.close()
                logger.debug("Closed verification pipeline")
        except Exception as e:
            logger.error(f"Error closing verification pipeline: {e}")

        logger.info("Workflow coordinator closed successfully")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Singleton instance
workflow_coordinator = WorkflowCoordinator()

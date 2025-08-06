"""
from __future__ import annotations

Application service for handling verification requests.
"""

import logging
from typing import Any
from uuid import uuid4

from fastapi import BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from agent.orchestration import workflow_coordinator
from agent.services.processing.validation_service import validation_service
from app.cache.factory import get_verification_cache
from app.crud import VerificationResultCRUD
from app.database import async_session_factory
from app.websocket_manager import EventProgressTracker, websocket_manager

logger = logging.getLogger(__name__)


class VerificationService:
    """Service for handling verification requests and coordination."""

    def __init__(self):
        self.coordinator = workflow_coordinator

    async def submit_verification_request(
        self,
        background_tasks: BackgroundTasks,
        file_data: bytes,
        filename: str,
        prompt: str,
        session_id: str | None,
        db: AsyncSession,
    ) -> dict[str, Any]:
        """
        Submit a verification request for processing.

        Args:
            background_tasks: FastAPI background tasks
            file_data: Image file data
            filename: Original filename
            prompt: User prompt
            session_id: Optional WebSocket session ID
            db: Database session

        Returns:
            Verification response
        """
        # Validate the request first
        validated_data = validation_service.validate_verification_request(
            file_data=file_data, prompt=prompt, filename=filename, session_id=session_id
        )

        # Generate verification ID
        verification_id = str(uuid4())

        logger.info("Received verification request: %s, prompt: %s...",
                    filename, validated_data["prompt"][:100])

        # If session_id provided, start WebSocket session
        if session_id and session_id in websocket_manager.active_connections:
            await websocket_manager.start_verification_session(
                session_id,
                {
                    "verification_id": verification_id,
                    "filename": filename,
                    "prompt": validated_data["prompt"],
                },
            )

            # Run verification in background with WebSocket updates
            background_tasks.add_task(
                self._run_verification_with_websocket,
                verification_id,
                session_id,
                validated_data["image_data"],
                validated_data["prompt"],
                filename,
            )

            return {
                "status": "processing",
                "message": "Verification started. Check WebSocket for real-time updates.",
                "verification_id": verification_id,
                "session_id": session_id,
            }
        else:
            # Check cache first
            cache = await get_verification_cache()
            content = f"{filename}:{validated_data['prompt']}"
            verification_type = "image_verification"
            context = {"filename": filename} if filename else None

            cached_result = await cache.get_verification_result(
                content=content,
                verification_type=verification_type,
                context=context
            )
            if cached_result:
                logger.info("Cache hit for verification request: %s", content)
                return cached_result.get('result', cached_result)

            # Run verification synchronously
            logger.info("Cache miss for verification request: %s", content)
            result = await self.coordinator.execute_verification(
                image_bytes=validated_data["image_data"],
                user_prompt=validated_data["prompt"],
                db=db,
                session_id="sync",
                filename=filename,
            )

            # Cache the result
            await cache.set_verification_result(
                content=content,
                verification_type=verification_type,
                result=result,
                context=context
            )

            return result

    async def _run_verification_with_websocket(
        self,
        verification_id: str,
        session_id: str,
        image_content: bytes,
        prompt: str,
        filename: str = None,
    ):
        """
        Run verification with WebSocket progress updates.

        Args:
            verification_id: Unique verification ID
            session_id: WebSocket session ID
            image_content: Image bytes
            prompt: User prompt
            filename: Original filename
        """
        db: AsyncSession | None = None
        event_tracker = EventProgressTracker(websocket_manager, session_id)

        try:
            # Create a new database session for the background task
            async with async_session_factory() as db:
                logger.info(
                    "Starting background verification for session %s", session_id)

                # Check cache first
                cache = await get_verification_cache()
                content = f"{filename}:{prompt}"
                verification_type = "image_verification"
                context = {"filename": filename} if filename else None

                cached_result = await cache.get_verification_result(
                    content=content,
                    verification_type=verification_type,
                    context=context
                )
                if cached_result:
                    logger.info(
                        "Cache hit for WebSocket verification: %s", content)
                    await event_tracker.complete(cached_result.get('result', cached_result))
                    return

                logger.info(
                    "Cache miss for WebSocket verification: %s", content)
                # Run verification with event callback
                result = await self.coordinator.execute_verification(
                    image_bytes=image_content,
                    user_prompt=prompt,
                    db=db,
                    event_callback=event_tracker.emit_event,
                    session_id=session_id,
                    filename=filename,
                )

                # Cache the result
                await cache.set_verification_result(
                    content=content,
                    verification_type=verification_type,
                    result=result,
                    context=context
                )

                logger.info(
                    "Background verification completed for session %s, sending result", session_id)

                # Explicitly commit the database transaction
                try:
                    await db.commit()
                    logger.info(
                        "Database transaction committed for session %s", session_id)
                except Exception as commit_error:
                    logger.error(
                        "Failed to commit database transaction for session %s: %s",
                        session_id,
                        commit_error,
                        exc_info=True,
                    )
                    await db.rollback()
                    raise

                # Send final result - handle any WebSocket errors separately after DB commit
                try:
                    await event_tracker.complete(result)
                    logger.info(
                        "Final result sent successfully for session %s", session_id)
                except Exception as ws_error:
                    logger.error(
                        "Failed to send WebSocket result for session %s: %s",
                        session_id,
                        ws_error,
                        exc_info=True,
                    )
                    # Don't raise the exception - the verification succeeded even if WebSocket failed

        except Exception as e:
            logger.error(
                "WebSocket verification failed for session %s: %s",
                session_id,
                e,
                exc_info=True,
            )
            await event_tracker.error(str(e))

    async def get_verification_status(self, verification_id: str, db: AsyncSession) -> dict[str, Any]:
        """
        Get the status and result of a verification request.

        Args:
            verification_id: ID of the verification request
            db: Database session

        Returns:
            Status information and verification result if completed
        """
        crud = VerificationResultCRUD(db)
        result = await crud.get_verification_result_by_id(db=db, result_id=verification_id)

        if not result:
            return None

        return {
            "status": "completed",
            "message": "Verification result retrieved successfully.",
            "verification_id": str(result.id),
            "user_nickname": result.user_nickname,
            "extracted_text": result.extracted_text,
            "primary_topic": result.primary_topic,
            "identified_claims": result.identified_claims,
            "verdict": result.verdict,
            "justification": result.justification,
            "confidence_score": result.confidence_score,
            "processing_time_seconds": result.processing_time_seconds,
            "prompt": result.user_prompt,
            "filename": getattr(result, "filename", None),
            "file_size": getattr(result, "file_size", None),
        }

    async def close(self) -> None:
        """Close the verification service and release resources."""
        logger.info("Closing verification service...")

        try:
            if hasattr(self.coordinator, "close") and callable(self.coordinator.close):
                await self.coordinator.close()
                logger.debug("Closed workflow coordinator")
        except Exception as e:
            logger.error("Error closing workflow coordinator: %s", e)

        logger.info("Verification service closed successfully")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Singleton service instance
verification_service = VerificationService()

"""
Application service for handling verification requests.
"""
import logging
from typing import Dict, Any, Optional
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import BackgroundTasks

from app.websocket_manager import connection_manager, ProgressTracker
from app.schemas import VerificationResponse
from app.exceptions import ValidationError, ImageProcessingError
from app.database import async_session_factory
from agent.workflow_coordinator import workflow_coordinator
from app.crud import VerificationResultCRUD
from agent.services.validation_service import validation_service

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
        session_id: Optional[str],
        db: AsyncSession
    ) -> Dict[str, Any]:
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
            file_data=file_data,
            prompt=prompt,
            filename=filename,
            session_id=session_id
        )
        
        # Generate verification ID
        verification_id = str(uuid4())
        
        logger.info(f"Received verification request: {filename}, prompt: {validated_data['prompt'][:100]}...")
        
        # If session_id provided, start WebSocket session
        if session_id and session_id in connection_manager.active_connections:
            await connection_manager.start_verification_session(session_id, {
                "verification_id": verification_id,
                "filename": filename,
                "prompt": validated_data['prompt']
            })
            
            # Run verification in background with WebSocket updates
            background_tasks.add_task(
                self._run_verification_with_websocket,
                verification_id,
                session_id,
                validated_data['image_data'],
                validated_data['prompt'],
                filename,
            )
            
            return {
                "status": "processing",
                "message": "Verification started. Check WebSocket for real-time updates.",
                "verification_id": verification_id,
                "session_id": session_id
            }
        else:
            # Run verification synchronously
            result = await self.coordinator.execute_verification(
                image_bytes=validated_data['image_data'],
                user_prompt=validated_data['prompt'],
                db=db,
                session_id="sync",
                filename=filename
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
        db: Optional[AsyncSession] = None
        progress_tracker = ProgressTracker(connection_manager, session_id)
        
        try:
            # Create a new database session for the background task
            async with async_session_factory() as db:
                logger.info(f"Starting background verification for session {session_id}")
                
                # Run verification with progress callback
                result = await self.coordinator.execute_verification(
                    image_bytes=image_content,
                    user_prompt=prompt,
                    db=db,
                    progress_callback=progress_tracker.update,
                    session_id=session_id,
                    filename=filename,
                )
                
                logger.info(f"Background verification completed for session {session_id}, sending result")
                
                # Explicitly commit the database transaction
                try:
                    await db.commit()
                    logger.info(f"Database transaction committed for session {session_id}")
                except Exception as commit_error:
                    logger.error(f"Failed to commit database transaction for session {session_id}: {commit_error}", exc_info=True)
                    await db.rollback()
                    raise
                
                # Send final result - handle any WebSocket errors separately after DB commit
                try:
                    await progress_tracker.complete(result)
                    logger.info(f"Final result sent successfully for session {session_id}")
                except Exception as ws_error:
                    logger.error(f"Failed to send WebSocket result for session {session_id}: {ws_error}", exc_info=True)
                    # Don't raise the exception - the verification succeeded even if WebSocket failed
                    
        except Exception as e:
            logger.error(f"WebSocket verification failed for session {session_id}: {e}", exc_info=True)
            await progress_tracker.error(str(e))
    
    async def get_verification_status(
        self,
        verification_id: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Get the status and result of a verification request.
        
        Args:
            verification_id: ID of the verification request
            db: Database session
            
        Returns:
            Status information and verification result if completed
        """
        crud = VerificationResultCRUD(db)
        result = await crud.get_verification_result_by_uuid(verification_id)
        
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
            "temporal_analysis": result.temporal_analysis,
            "examined_sources": result.examined_sources,
            "search_queries_used": result.search_queries_used,
            "sources": result.sources,
            "user_reputation": result.user_reputation,
            "warnings": result.warnings,
            "prompt": result.prompt,
            "filename": result.filename,
            "file_size": result.file_size
        }


# Singleton service instance
verification_service = VerificationService() 
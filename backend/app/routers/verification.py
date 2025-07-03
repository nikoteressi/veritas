"""
Verification endpoints for post fact-checking.
"""
import logging
from typing import Dict, Any, Optional
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, async_session_factory
from app.websocket_manager import connection_manager, ProgressTracker
from app.schemas import VerificationResponse
from app.validators import RequestValidator
from app.exceptions import ValidationError, ImageProcessingError
from agent.orchestrator import verification_orchestrator
from app.crud import VerificationResultCRUD

logger = logging.getLogger(__name__)

router = APIRouter()


async def run_verification_with_websocket(
    verification_id: str,
    session_id: str,
    image_content: bytes,
    prompt: str,
):
    """
    Run verification with WebSocket progress updates.

    Args:
        verification_id: Unique verification ID
        session_id: WebSocket session ID
        image_content: Image bytes
        prompt: User prompt
    """
    db: Optional[AsyncSession] = None
    progress_tracker = ProgressTracker(connection_manager, session_id)
    try:
        # Create a new database session for the background task
        async with async_session_factory() as db:
            logger.info(f"Starting background verification for session {session_id}")
            # Run verification with progress callback
            result = await verification_orchestrator.verify_post(
                image_bytes=image_content,
                user_prompt=prompt,
                db=db,
                progress_callback=progress_tracker.update,
                session_id=session_id,
            )
            logger.info(f"Background verification completed for session {session_id}, sending result")
            # Send final result
            await progress_tracker.complete(result)
            logger.info(f"Final result sent successfully for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket verification failed for session {session_id}: {e}", exc_info=True)
        await progress_tracker.error(str(e))


@router.post("/verify-post", response_model=VerificationResponse)
async def verify_post(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt: str = Form(...),
    session_id: str = Form(None),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Verify a social media post for factual accuracy.

    Args:
        background_tasks: FastAPI background tasks
        file: Screenshot image of the social media post
        prompt: User's text prompt/question about the post
        session_id: Optional WebSocket session ID for real-time updates
        db: Database session

    Returns:
        Verification result with verdict and justification
    """
    try:
        # Validate request using comprehensive validator
        validated_data = await RequestValidator.validate_verification_request(
            file, prompt, session_id
        )

        # Get image content from validated data
        image_content = validated_data["image_bytes"]

        # Additional file size check after reading - This is now handled in the validator
        # if len(image_content) > 10 * 1024 * 1024:
        #     raise ValidationError(
        #         "Image file too large (max 10MB)",
        #         error_code="FILE_TOO_LARGE"
        #     )

        logger.info(f"Received verification request: {file.filename}, prompt: {validated_data['prompt'][:100]}...")

        # Generate verification ID
        verification_id = str(uuid4())

        # If session_id provided, start WebSocket session
        if session_id and session_id in connection_manager.active_connections:
            await connection_manager.start_verification_session(session_id, {
                "verification_id": verification_id,
                "filename": file.filename,
                "prompt": validated_data['prompt']
            })

            # Run verification in background with WebSocket updates
            background_tasks.add_task(
                run_verification_with_websocket,
                verification_id,
                session_id,
                image_content,
                validated_data['prompt'],
            )

            return {
                "status": "processing",
                "message": "Verification started. Check WebSocket for real-time updates.",
                "verification_id": verification_id,
                "session_id": session_id
            }
        else:
            # Run verification synchronously
            result = await verification_orchestrator.verify_post(
                image_bytes=image_content,
                user_prompt=validated_data['prompt'],
                db=db,
                session_id="sync"
            )

            return result

    except (ValidationError, ImageProcessingError):
        # These will be handled by the global exception handlers
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in verify_post: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during verification")


@router.get("/verification-status/{verification_id}", response_model=VerificationResponse)
async def get_verification_status(
    verification_id: str,
    db: AsyncSession = Depends(get_db)
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
        raise HTTPException(status_code=404, detail="Verification result not found")

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
        "fact_check_summary": result.fact_check_summary,
        "user_reputation": result.user_reputation,
        "warnings": result.warnings
    }

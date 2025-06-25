"""
Verification endpoints for post fact-checking.
"""
import logging
from typing import Dict, Any
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.image_processing import image_processor
from app.websocket_manager import connection_manager, ProgressTracker
from app.schemas import VerificationResponse
from app.validators import RequestValidator
from app.exceptions import ValidationError, ImageProcessingError
from agent.core import veritas_agent

logger = logging.getLogger(__name__)

router = APIRouter()


async def run_verification_with_websocket(
    verification_id: str,
    session_id: str,
    image_content: bytes,
    prompt: str,
    db: AsyncSession
):
    """
    Run verification with WebSocket progress updates.

    Args:
        verification_id: Unique verification ID
        session_id: WebSocket session ID
        image_content: Image bytes
        prompt: User prompt
        db: Database session
    """
    try:
        # Create progress tracker
        progress_tracker = ProgressTracker(connection_manager, session_id)

        # Run verification with progress callback
        result = await veritas_agent.verify_post(
            image_content,
            prompt,
            db,
            progress_callback=progress_tracker.update
        )

        # Send final result
        await progress_tracker.complete(result)

    except Exception as e:
        logger.error(f"WebSocket verification failed: {e}")
        progress_tracker = ProgressTracker(connection_manager, session_id)
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
        validated_data = RequestValidator.validate_verification_request(
            file, prompt, session_id
        )

        # Read and validate image content
        image_content = await file.read()

        if not image_processor.validate_image(image_content):
            raise ImageProcessingError(
                "Invalid or corrupted image file",
                error_code="INVALID_IMAGE"
            )

        # Additional file size check after reading
        if len(image_content) > 10 * 1024 * 1024:
            raise ValidationError(
                "Image file too large (max 10MB)",
                error_code="FILE_TOO_LARGE"
            )

        logger.info(f"Received verification request: {file.filename}, prompt: {validated_data['prompt'][:100]}...")

        # Generate verification ID
        verification_id = str(uuid4())

        # If session_id provided, start WebSocket session
        if session_id and session_id in connection_manager.active_connections:
            connection_manager.start_verification_session(session_id, {
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
                db
            )

            return {
                "status": "processing",
                "message": "Verification started. Check WebSocket for real-time updates.",
                "verification_id": verification_id,
                "session_id": session_id
            }
        else:
            # Run verification synchronously
            result = await veritas_agent.verify_post(
                image_content, validated_data['prompt'], db
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


@router.get("/verification-status/{verification_id}")
async def get_verification_status(
    verification_id: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get the status of a verification request.
    
    Args:
        verification_id: ID of the verification request
        db: Database session
    
    Returns:
        Status information
    """
    # TODO: Implement status checking logic
    return {
        "verification_id": verification_id,
        "status": "completed",
        "message": "Status endpoint is working"
    }

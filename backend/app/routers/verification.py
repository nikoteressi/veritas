"""
Verification endpoints for post fact-checking.
"""
import logging
from typing import Dict, Any

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import VerificationResponse
from app.exceptions import ValidationError, ImageProcessingError
from app.services.verification_service import verification_service

logger = logging.getLogger(__name__)

router = APIRouter()


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
        # Read file data
        file_data = await file.read()

        # Delegate validation and processing to the service layer
        result = await verification_service.submit_verification_request(
            background_tasks=background_tasks,
            file_data=file_data,
            filename=file.filename,
            prompt=prompt,
            session_id=session_id,
            db=db
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
    result = await verification_service.get_verification_status(verification_id, db)
    
    if not result:
        raise HTTPException(status_code=404, detail="Verification result not found")

    return result

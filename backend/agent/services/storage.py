"""
Service for handling data persistence.
"""

import hashlib
import json
import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from agent.vector_store import vector_store
from app.config import settings
from app.crud import VerificationResultCRUD

logger = logging.getLogger(__name__)


class StorageService:
    """Service to manage persistence to databases."""

    def _hash_image(self, image_bytes: bytes) -> str:
        """Create a SHA256 hash of the image bytes."""
        return hashlib.sha256(image_bytes).hexdigest()

    async def save_verification_result(
        self,
        db: AsyncSession,
        user_nickname: str,
        image_bytes: bytes,
        user_prompt: str,
        extracted_info: dict[str, Any],
        verdict_result: dict[str, Any],
        reputation_data: dict[str, Any],
    ) -> "VerificationResult":
        """Saves the complete verification result to the database."""

        # Extract claims from the new structure
        claims = extracted_info.get("claims", [])
        if not claims:
            raise ValueError("No claims found in extracted information")

        result_data = {
            "user_nickname": user_nickname,
            "verdict": verdict_result.get("verdict"),
            "justification": verdict_result.get("reasoning"),
            "identified_claims": json.dumps(claims),
            "primary_topic": extracted_info.get("primary_topic"),
            "extracted_text": extracted_info.get("extracted_text"),
            "image_hash": self._hash_image(image_bytes),
            "user_prompt": user_prompt,
            "confidence_score": verdict_result.get("confidence_score"),
            "processing_time_seconds": extracted_info.get("processing_time_seconds", 0),
            "vision_model_used": settings.vision_model_name,
            "reasoning_model_used": settings.reasoning_model_name,
            "tools_used": json.dumps(["SearxNGSearchTool"]),
        }

        return await VerificationResultCRUD.create_verification_result(
            db=db, result_data=result_data
        )

    async def store_in_vector_db(self, verification_data: dict[str, Any]) -> None:
        """
        Store the verification result in the vector database by calling
        the dedicated method in the vector_store object.
        """
        try:
            await vector_store.store_verification_result(verification_data)
            logger.info("Stored verification result in vector database.")

        except Exception as e:
            logger.error(f"Failed to store in vector DB: {e}", exc_info=True)
            # This operation should not fail the entire request, so we just log the error.


# Singleton instance
storage_service = StorageService()

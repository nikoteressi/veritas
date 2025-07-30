"""
Service for handling data persistence.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any

from app.config import settings
from app.crud import VerificationResultCRUD
from app.exceptions import AgentError
from sqlalchemy.ext.asyncio import AsyncSession

from agent.models.verification_result import VerificationResult
from agent.clients.vector_store import vector_store

logger = logging.getLogger(__name__)


class StorageService:
    """Service to manage persistence to databases."""

    def _hash_image(self, image_bytes: bytes) -> str:
        """Create a SHA256 hash of the image bytes."""
        return hashlib.sha256(image_bytes).hexdigest()

    def _serialize_for_json(self, obj: Any) -> Any:
        """Serialize object for JSON, handling datetime objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._serialize_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        else:
            return obj

    async def save_verification_result(
        self,
        db: AsyncSession,
        user_nickname: str,
        image_bytes: bytes,
        user_prompt: str,
        extracted_info: dict[str, Any],
        verdict_result: dict[str, Any],
        reputation_data: dict[str, Any],
    ) -> VerificationResult:
        """Saves the complete verification result to the database."""

        # Extract claims from the new structure
        claims = extracted_info.get("claims", [])
        if not claims:
            raise ValueError("No claims found in extracted information")

        # Serialize reputation data for JSON storage
        serialized_reputation = self._serialize_for_json(reputation_data)

        result_data = {
            "user_nickname": user_nickname,
            "reputation_data": json.dumps(serialized_reputation),
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

        return await VerificationResultCRUD.create_verification_result(db=db, result_data=result_data)

    async def store_in_vector_db(self, verification_data: dict[str, Any]) -> None:
        """
        Store the verification result in the vector database by calling
        the dedicated method in the vector_store object.
        """
        try:
            await vector_store.store_verification_result(verification_data)
            logger.info("Stored verification result in vector database.")

        except Exception as e:
            raise AgentError(
                f"Vector database storage failed: {str(e)}") from e


# Singleton instance
storage_service = StorageService()

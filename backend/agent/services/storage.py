"""
Service for handling data persistence.
"""
import logging
import json
import hashlib
from typing import Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.crud import VerificationResultCRUD
from app.config import settings
from agent.vector_store import vector_store
from app.exceptions import StorageError
from app.schemas import ImageAnalysisResult, FactCheckResult, VerdictResult, UserReputation

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
        extracted_info: Dict[str, Any],
        verdict_result: Dict[str, Any],
        reputation_data: Dict[str, Any]
    ) -> "VerificationResult":
        """Saves the complete verification result to the database."""
        
        result_data = {
            "user_nickname": user_nickname,
            "verdict": verdict_result.get("verdict"),
            "justification": verdict_result.get("reasoning"),
            "identified_claims": json.dumps([fact['description'] for fact in extracted_info.get("fact_hierarchy", {}).get("supporting_facts", [])]),
            "primary_topic": extracted_info.get("primary_topic"),
            "extracted_text": extracted_info.get("extracted_text"),
            "image_hash": self._hash_image(image_bytes),
            "user_prompt": user_prompt,
            "confidence_score": verdict_result.get("confidence_score"),
            "processing_time_seconds": extracted_info.get("processing_time_seconds", 0),
            "model_used": settings.ollama_model,
            "tools_used": json.dumps(["SearxNGSearchTool"])
        }
        
        return await VerificationResultCRUD.create_verification_result(
            db=db,
            result_data=result_data
        )

    async def store_in_vector_db(
        self,
        verification_data: Dict[str, Any]
    ) -> None:
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
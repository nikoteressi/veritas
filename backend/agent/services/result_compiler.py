"""
Service for compiling and serializing verification results.
"""
import asyncio
import logging
import time
from typing import Dict, Any, Optional

from agent.models import ImageAnalysisResult, FactCheckResult, VerdictResult
from app.schemas import VerificationResponse, UserReputation
from app.json_utils import json_dumps, prepare_for_json_serialization

logger = logging.getLogger(__name__)


class ResultCompiler:
    """Compiles and serializes verification results into the final format."""
    
    def __init__(self):
        self.start_time = None
    
    def start_timing(self):
        """Start timing the verification process."""
        self.start_time = time.time()
    
    def get_processing_time(self) -> int:
        """Get the processing time in seconds."""
        if self.start_time is None:
            return 0
        return int(time.time() - self.start_time)
    
    async def compile_result(
        self,
        context: 'VerificationContext',
    ) -> Dict[str, Any]:
        """
        Compile all verification results into a final JSON-serializable format.

        Args:
            context: The verification context containing all data.

        Returns:
            Compiled result dictionary
        """
        processing_time = self.get_processing_time()

        serializable_temporal = self._prepare_temporal_analysis(context.temporal_analysis)
        serializable_motives = self._prepare_motives_analysis(context.motives_analysis)

        summary_result = await context.summary if context.summary and asyncio.iscoroutine(context.summary) else context.summary

        final_result = {
            "status": "success",
            "message": "Verification completed successfully",
            "verification_id": context.verification_id,
            "nickname": context.screenshot_data.post_content.author,
            "extracted_text": context.screenshot_data.post_content.text_body,
            "primary_topic": context.primary_topic,
            "identified_claims": [fact.description for fact in context.fact_hierarchy.supporting_facts],
            "verdict": context.verdict_result.verdict,
            "justification": context.verdict_result.reasoning,
            "confidence_score": context.verdict_result.confidence_score,
            "processing_time_seconds": processing_time,
            "temporal_analysis": serializable_temporal,
            "motives_analysis": serializable_motives,
            "fact_check_results": {
                "examined_sources": context.fact_check_result.examined_sources,
                "search_queries_used": context.fact_check_result.search_queries_used,
                "summary": context.fact_check_result.summary.dict(),
            },
            "sources": context.verdict_result.sources or [],
            "user_reputation": prepare_for_json_serialization(context.user_reputation),
            "warnings": context.warnings,
            "prompt": context.user_prompt,
            "filename": context.filename or "uploaded_image",
            "file_size": len(context.image_bytes) if context.image_bytes else 0,
            "summary": summary_result,
        }

        return self._ensure_json_serializable(final_result, context)
    
    def _prepare_temporal_analysis(self, temporal_analysis: Any) -> Dict[str, Any]:
        """Prepare temporal analysis data for JSON serialization."""
        try:
            return prepare_for_json_serialization(temporal_analysis)
        except Exception as e:
            logger.warning(f"Failed to prepare temporal analysis for serialization: {e}")
            return {}

    def _prepare_motives_analysis(self, motives_analysis: Any) -> Dict[str, Any]:
        """Prepare motives analysis data for JSON serialization."""
        try:
            return prepare_for_json_serialization(motives_analysis)
        except Exception as e:
            logger.warning(f"Failed to prepare motives analysis for serialization: {e}")
            return {}

    def _ensure_json_serializable(
        self,
        final_result: Dict[str, Any],
        context: 'VerificationContext'
    ) -> Dict[str, Any]:
        """
        Test JSON serialization and return fallback if serialization fails.

        Args:
            final_result: The compiled result to test.
            context: The verification context for fallback data.

        Returns:
            JSON-serializable result dictionary.
        """
        try:
            # Test JSON serialization to catch issues early
            json_dumps(final_result)
            return final_result
        except Exception as serialization_error:
            logger.error(f"JSON serialization error: {serialization_error}", exc_info=True)
            return self._create_fallback_result(
                context,
                final_result.get("processing_time_seconds", 0)
            )
    
    def _create_fallback_result(
        self,
        context: 'VerificationContext',
        processing_time: int
    ) -> Dict[str, Any]:
        """Create a fallback result with minimal complexity."""
        return {
            "status": "success",
            "message": "Verification completed successfully, but some data could not be serialized.",
            "verification_id": context.verification_id,
            "nickname": context.screenshot_data.post_content.author if context.screenshot_data else "unknown",
            "extracted_text": context.screenshot_data.post_content.text_body if context.screenshot_data else "",
            "primary_topic": context.primary_topic,
            "identified_claims": [fact.description for fact in context.fact_hierarchy.supporting_facts] if context.fact_hierarchy else [],
            "verdict": context.verdict_result.verdict if context.verdict_result else "unknown",
            "justification": context.verdict_result.reasoning if context.verdict_result else "",
            "confidence_score": context.verdict_result.confidence_score if context.verdict_result else 0.0,
            "processing_time_seconds": processing_time,
            "sources": context.verdict_result.sources if context.verdict_result else [],
            "user_reputation": prepare_for_json_serialization(context.user_reputation),
            "warnings": context.warnings or [],
            "prompt": context.user_prompt,
            "filename": context.filename or "uploaded_image",
            "file_size": len(context.image_bytes) if context.image_bytes else 0,
            "summary": context.summary
        }
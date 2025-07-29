"""
Service for compiling and serializing verification results.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from app.exceptions import AnalysisError
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
        context: "VerificationContext",
    ) -> dict[str, Any]:
        """
        Compile all verification results into a final JSON-serializable format.

        Args:
            context: The verification context containing all data.

        Returns:
            Compiled result dictionary
        """
        processing_time = self.get_processing_time()

        # Use new typed methods
        temporal_analysis = context.get_temporal_analysis()
        motives_analysis = context.get_motives_analysis()
        summarization_result = context.get_summarization_result()

        serializable_temporal = self._prepare_temporal_analysis(
            temporal_analysis)
        serializable_motives = self._prepare_motives_analysis(motives_analysis)
        serializable_summarization = self._prepare_summarization_result(
            summarization_result)

        # Ensure summarization result is provided
        if not summarization_result:
            raise ValueError(
                "Summarization result is required for compilation")

        summary_text = summarization_result.summary

        final_result = {
            "status": "success",
            "message": "Verification completed successfully",
            "verification_id": context.verification_id,
            "nickname": context.screenshot_data.post_content.author,
            "extracted_text": context.screenshot_data.post_content.text_body,
            "primary_topic": context.primary_topic,
            "identified_claims": context.claims,
            "verdict": context.verdict_result.verdict,
            "justification": context.verdict_result.reasoning,
            "confidence_score": context.verdict_result.confidence_score,
            "processing_time_seconds": processing_time,
            "temporal_analysis": serializable_temporal,
            "motives_analysis": serializable_motives,
            "fact_check_results": {
                "examined_sources": context.fact_check_result.examined_sources,
                "search_queries_used": context.fact_check_result.search_queries_used,
                "summary": context.fact_check_result.summary.model_dump(),
            },
            "sources": context.verdict_result.sources or [],
            "user_reputation": prepare_for_json_serialization(context.user_reputation),
            "warnings": context.warnings,
            "prompt": context.user_prompt,
            "filename": context.filename or "uploaded_image",
            "file_size": len(context.image_bytes) if context.image_bytes else 0,
            "summary": summary_text,
            "summarization_details": serializable_summarization,
        }

        return self._ensure_json_serializable(final_result)

    def _prepare_temporal_analysis(self, temporal_analysis: Any) -> dict[str, Any]:
        """Prepare temporal analysis data for JSON serialization."""
        try:
            if temporal_analysis is None:
                return {}

            # If it's a typed object with model_dump() method, use it
            if hasattr(temporal_analysis, "model_dump"):
                return temporal_analysis.model_dump()

            # Otherwise, use the general serialization function
            return prepare_for_json_serialization(temporal_analysis)
        except Exception as e:
            raise AnalysisError(
                f"Temporal analysis serialization failed: {str(e)}") from e

    def _prepare_motives_analysis(self, motives_analysis: Any) -> dict[str, Any]:
        """Prepare motives analysis data for JSON serialization."""
        try:
            if motives_analysis is None:
                return {}

            # If it's a typed object with model_dump() method, use it
            if hasattr(motives_analysis, "model_dump"):
                return motives_analysis.model_dump()

            # Otherwise, use the general serialization function
            return prepare_for_json_serialization(motives_analysis)
        except Exception as e:
            raise AnalysisError(
                f"Motives analysis serialization failed: {str(e)}") from e

    def _prepare_summarization_result(self, summarization_result: Any) -> dict[str, Any]:
        """Prepare summarization result data for JSON serialization."""
        try:
            if summarization_result is None:
                return {}

            # If it's a typed object with model_dump() method, use it
            if hasattr(summarization_result, "model_dump"):
                return summarization_result.model_dump()

            # Otherwise, use the general serialization function
            return prepare_for_json_serialization(summarization_result)
        except Exception as e:
            raise AnalysisError(
                f"Summarization result serialization failed: {str(e)}") from e

    def _ensure_json_serializable(self, final_result: dict[str, Any]) -> dict[str, Any]:
        """
        Test JSON serialization and raise error if serialization fails.

        Args:
            final_result: The compiled result to test.
            context: The verification context.

        Returns:
            JSON-serializable result dictionary.
        """
        # Test JSON serialization to catch issues early
        json_dumps(final_result)
        return final_result

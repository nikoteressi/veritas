"""
Service for compiling and serializing verification results.
"""
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
    
    def compile_result(
        self,
        verification_id: str,
        analysis_result: ImageAnalysisResult,
        fact_check_result: FactCheckResult,
        verdict_result: VerdictResult,
        extracted_info: Dict[str, Any],
        reputation_data: Dict[str, Any],
        warnings: list,
        user_prompt: str,
        image_bytes: bytes,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compile all verification results into a final JSON-serializable format.
        
        Args:
            verification_id: Unique verification identifier
            analysis_result: Image analysis results
            fact_check_result: Fact-checking results
            verdict_result: Final verdict results
            extracted_info: Information extracted from the image
            reputation_data: User reputation data
            warnings: List of warnings
            user_prompt: Original user prompt
            image_bytes: Original image bytes
            filename: Original filename
            
        Returns:
            Compiled result dictionary
        """
        processing_time = self.get_processing_time()
        
        # Prepare serializable versions of complex objects
        serializable_temporal = self._prepare_temporal_analysis(extracted_info)
        serializable_motives = self._prepare_motives_analysis(extracted_info)
        
        final_result = {
            "status": "success",
            "message": "Verification completed successfully",
            "verification_id": verification_id,
            "nickname": extracted_info.get("username"),
            "extracted_text": analysis_result.extracted_text,
            "primary_topic": analysis_result.primary_topic,
            "identified_claims": [fact.description for fact in analysis_result.fact_hierarchy.supporting_facts],
            "verdict": verdict_result.verdict,
            "justification": verdict_result.reasoning,
            "confidence_score": verdict_result.confidence_score,
            "processing_time_seconds": processing_time,
            "temporal_analysis": serializable_temporal,
            "motives_analysis": serializable_motives,
            "fact_check_results": {
                "examined_sources": fact_check_result.examined_sources,
                "search_queries_used": fact_check_result.search_queries_used,
                "summary": fact_check_result.summary.dict(),
            },
            "sources": verdict_result.sources or [],
            "user_reputation": reputation_data,
            "warnings": warnings,
            # Frontend-expected fields
            "prompt": user_prompt,
            "filename": filename or extracted_info.get("filename", "uploaded_image"),
            "file_size": len(image_bytes) if image_bytes else 0
        }
        
        # Test JSON serialization and return fallback if needed
        return self._ensure_json_serializable(final_result, analysis_result, verdict_result, reputation_data, warnings, user_prompt, filename)
    
    def _prepare_temporal_analysis(self, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare temporal analysis data for JSON serialization."""
        try:
            temporal_analysis = extracted_info.get("temporal_analysis", {})
            return prepare_for_json_serialization(temporal_analysis)
        except Exception as e:
            logger.warning(f"Failed to prepare temporal analysis for serialization: {e}")
            return {}
    
    def _prepare_motives_analysis(self, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare motives analysis data for JSON serialization."""
        try:
            motives_analysis = extracted_info.get("motives_analysis", {})
            return prepare_for_json_serialization(motives_analysis)
        except Exception as e:
            logger.warning(f"Failed to prepare motives analysis for serialization: {e}")
            return {}
    
    def _ensure_json_serializable(
        self,
        final_result: Dict[str, Any],
        analysis_result: ImageAnalysisResult,
        verdict_result: VerdictResult,
        reputation_data: Dict[str, Any],
        warnings: list,
        user_prompt: str,
        filename: Optional[str]
    ) -> Dict[str, Any]:
        """
        Test JSON serialization and return fallback if serialization fails.
        
        Args:
            final_result: The compiled result to test
            analysis_result: Analysis result for fallback
            verdict_result: Verdict result for fallback
            reputation_data: Reputation data for fallback
            warnings: Warnings for fallback
            user_prompt: User prompt for fallback
            filename: Filename for fallback
            
        Returns:
            JSON-serializable result dictionary
        """
        try:
            # Test JSON serialization to catch issues early
            json_dumps(final_result)
            return final_result
        except Exception as serialization_error:
            logger.error(f"JSON serialization error: {serialization_error}", exc_info=True)
            # Return fallback result without complex objects
            return self._create_fallback_result(
                final_result.get("verification_id", "unknown"),
                analysis_result,
                verdict_result,
                reputation_data,
                warnings,
                user_prompt,
                filename,
                final_result.get("processing_time_seconds", 0)
            )
    
    def _create_fallback_result(
        self,
        verification_id: str,
        analysis_result: ImageAnalysisResult,
        verdict_result: VerdictResult,
        reputation_data: Dict[str, Any],
        warnings: list,
        user_prompt: str,
        filename: Optional[str],
        processing_time: int
    ) -> Dict[str, Any]:
        """Create a fallback result with minimal complexity."""
        return {
            "status": "success",
            "message": "Verification completed successfully",
            "verification_id": verification_id,
            "nickname": analysis_result.username if hasattr(analysis_result, 'username') else "unknown",
            "extracted_text": analysis_result.extracted_text,
            "primary_topic": analysis_result.primary_topic,
            "identified_claims": [fact.description for fact in analysis_result.fact_hierarchy.supporting_facts],
            "verdict": verdict_result.verdict,
            "justification": verdict_result.reasoning,
            "confidence_score": verdict_result.confidence_score,
            "processing_time_seconds": processing_time,
            "sources": verdict_result.sources or [],
            "user_reputation": reputation_data,
            "warnings": warnings or [],
            "prompt": user_prompt,
            "filename": filename or "uploaded_image",
            "file_size": 0
        } 
from __future__ import annotations

import logging
import time
from typing import Optional

from agent.llm.manager import OllamaLLMManager
from agent.models.summarization_result import SummarizationResult
from agent.models.verification_context import VerificationContext
from agent.prompts.manager import PromptManager
from app.models.progress_callback import ProgressCallback, NoOpProgressCallback

logger = logging.getLogger(__name__)


class SummarizerService:
    def __init__(self, llm_manager: OllamaLLMManager, prompt_manager: PromptManager):
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager
        self.progress_callback: ProgressCallback = NoOpProgressCallback()

    def set_progress_callback(self, callback: Optional[ProgressCallback]) -> None:
        """Set the progress callback for detailed progress reporting."""
        self.progress_callback = callback or NoOpProgressCallback()

    async def summarize(self, context: VerificationContext) -> SummarizationResult:
        """
        Generate a comprehensive summary of the verification process.

        Args:
            context: Verification context with all analysis results

        Returns:
            SummarizationResult: Structured summary with metadata
        """
        start_time = time.time()

        # Initial progress
        await self.progress_callback.update_progress(0, 100, "Starting summarization...")

        if not context.fact_check_result:
            raise ValueError(
                "Fact check result is required for summarization.")

        # Get temporal analysis using the new typed method
        temporal_analysis = context.temporal_analysis_result

        await self.progress_callback.update_progress(20, 100, "Preparing summarization prompt...")

        # Prepare prompt for LLM
        prompt_template = self.prompt_manager.get_prompt_template(
            "summarization")
        prompt = await prompt_template.aformat(
            temporal_analysis=temporal_analysis,
            research_results=context.fact_check_result.model_dump_json(
                indent=2),
        )

        await self.progress_callback.update_progress(40, 100, "Generating summary with LLM...")

        # Generate summary text
        summary_text = await self.llm_manager.invoke_text_only(prompt)

        await self.progress_callback.update_progress(60, 100, "Processing summary metadata...")

        # Extract metadata from fact check result
        sources_used = context.fact_check_result.examined_sources
        claims_analyzed = len(context.fact_check_result.claim_results)

        # Create fact check summary
        fact_check_summary = self._create_fact_check_summary(
            context.fact_check_result)

        await self.progress_callback.update_progress(80, 100, "Extracting key points and calculating confidence...")

        # Extract key points (simple implementation - can be enhanced)
        key_points = self._extract_key_points(summary_text)

        # Calculate confidence based on fact check results
        confidence_score = self._calculate_confidence(
            context.fact_check_result)

        # Check if temporal context was included
        temporal_context_included = temporal_analysis is not None

        processing_time = time.time() - start_time

        await self.progress_callback.update_progress(100, 100, "Summarization completed")

        return SummarizationResult(
            summary=summary_text,
            confidence_score=confidence_score,
            sources_used=sources_used,
            key_points=key_points,
            temporal_context_included=temporal_context_included,
            fact_check_summary=fact_check_summary,
            claims_analyzed=claims_analyzed,
            processing_time=processing_time,
        )

    def _create_fact_check_summary(self, fact_check_result) -> str:
        """Create a brief summary of fact check results."""
        total_claims = len(fact_check_result.claim_results)
        if total_claims == 0:
            return "No claims were analyzed."

        # Count assessments
        assessments = [
            claim.assessment for claim in fact_check_result.claim_results]
        true_count = assessments.count(
            "true") + assessments.count("likely_true")
        false_count = assessments.count(
            "false") + assessments.count("likely_false")
        unverified_count = assessments.count("unverified")

        return f"Analyzed {total_claims} claims: {true_count} true/likely true, {false_count} false/likely false, {unverified_count} unverified."

    def _extract_key_points(self, summary_text: str) -> list[str]:
        """Extract key points from summary text (simple implementation)."""
        # Simple implementation - split by sentences and take first few
        sentences = [s.strip() for s in summary_text.split(".") if s.strip()]
        # Return first 3 sentences as key points
        return sentences[:3] if len(sentences) >= 3 else sentences

    def _calculate_confidence(self, fact_check_result) -> float:
        """Calculate confidence score based on fact check results."""
        if not fact_check_result.claim_results:
            return 0.5  # Neutral confidence if no claims

        # Average confidence from all claim results
        total_confidence = sum(
            claim.confidence for claim in fact_check_result.claim_results)
        avg_confidence = total_confidence / \
            len(fact_check_result.claim_results)

        # Factor in number of sources
        # Max boost at 5+ sources
        source_factor = min(len(fact_check_result.examined_sources) / 5.0, 1.0)

        # Combine factors
        final_confidence = (avg_confidence * 0.8) + (source_factor * 0.2)

        return min(max(final_confidence, 0.0), 1.0)  # Ensure 0.0-1.0 range

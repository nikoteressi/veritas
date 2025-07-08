"""
Service for generating the final verdict.
"""
import logging
import json
import re
from typing import Dict, Any, List

from langchain_core.output_parsers import JsonOutputParser
from agent.llm import llm_manager
from agent.prompt_manager import prompt_manager
from app.exceptions import LLMError
from agent.models import FactCheckResult, VerdictResult

logger = logging.getLogger(__name__)


class VerdictService:
    """Service to generate the final verdict based on fact-checking results."""

    def __init__(self):
        self.llm_manager = llm_manager

    async def generate(
        self,
        fact_check_result: FactCheckResult,
        user_prompt: str,
        temporal_analysis: Dict[str, Any],
        motives_analysis: Dict[str, Any] = None
    ) -> VerdictResult:
        """
        Generate the final verdict.

        Args:
            fact_check_result: The results from the fact-checking service.
            user_prompt: The original user prompt.
            temporal_analysis: The temporal analysis context.
            motives_analysis: The motives analysis context.

        Returns:
            A VerdictResult object.
        """
        summary = self._summarize_fact_check(fact_check_result)
        motives_summary = self._summarize_motives_analysis(motives_analysis or {})

        try:
            prompt_template = prompt_manager.get_prompt_template("verdict_generation")

            prompt = await prompt_template.aformat(
                research_results=summary + "\n\n" + motives_summary,
                user_prompt=user_prompt,
                temporal_analysis=json.dumps(temporal_analysis)
            )
            response = await self.llm_manager.invoke_text_only(prompt)

            # Clean and parse the response directly with Pydantic
            clean_response = re.sub(r"```json\n|```", "", response).strip()
            verdict_result = VerdictResult.model_validate_json(clean_response)

            # Ensure motives_analysis is included in the result
            if motives_analysis:
                verdict_result.motives_analysis = motives_analysis
            
            return verdict_result

        except Exception as e:
            logger.error(f"Failed to generate verdict: {e}", exc_info=True)
            raise LLMError(f"Failed to generate verdict: {e}", error_code="VERDICT_GENERATION_FAILED")

    def _summarize_fact_check(self, fact_check_result: FactCheckResult) -> str:
        """
        Create a concise summary of the fact-checking results for the final prompt.
        """
        summary_lines = []
        all_sources = set()
        primary_thesis = None
        
        # Check if we have hierarchical structure
        for claim_result in fact_check_result.claim_results:
            if claim_result.get('primary_thesis') and primary_thesis is None:
                primary_thesis = claim_result.get('primary_thesis')
                break
        
        # Add primary thesis context if available
        if primary_thesis:
            summary_lines.append(f"**Primary Thesis Being Evaluated:** {primary_thesis}\n")
        
        for claim_result in fact_check_result.claim_results:
            # Collect sources from each claim
            claim_sources = claim_result.get('examined_sources', [])
            all_sources.update(claim_sources)
            
            # Format claim summary with sources and context
            sources_text = ""
            if claim_sources:
                sources_text = f"\nSources checked: {', '.join(claim_sources[:3])}"
                if len(claim_sources) > 3:
                    sources_text += f" and {len(claim_sources) - 3} more"
            
            # Add context information if available
            context_text = ""
            claim_context = claim_result.get('context', {})
            if claim_context:
                context_items = [f"{k}: {v}" for k, v in claim_context.items() if v]
                if context_items:
                    context_text = f"\nContext: {', '.join(context_items)}"
            
            summary_lines.append(
                f"Supporting Fact: '{claim_result.get('claim', 'N/A')}'\n"
                f"Assessment: {claim_result.get('assessment', 'unverified')} "
                f"(Confidence: {claim_result.get('confidence', 0.0):.2f})\n"
                f"Summary: {claim_result.get('summary', 'No summary available.')}{context_text}{sources_text}\n"
            )

        # Include comprehensive sources list
        sources_summary = ""
        if all_sources:
            sources_list = list(all_sources)[:10]  # Limit to top 10 sources
            sources_summary = f"\nSources Consulted:\n" + "\n".join([f"- {source}" for source in sources_list])
            if len(all_sources) > 10:
                sources_summary += f"\n... and {len(all_sources) - 10} additional sources"

        overall_summary = (
            "\nOverall Fact-Check Summary:\n"
            f"- Total Sources Found: {fact_check_result.summary.total_sources_found}\n"
            f"- Credible Sources: {fact_check_result.summary.credible_sources}\n"
            f"- Supporting Evidence Found: {fact_check_result.summary.supporting_evidence}\n"
            f"- Contradicting Evidence Found: {fact_check_result.summary.contradicting_evidence}\n"
            f"{sources_summary}"
        )
        summary_lines.append(overall_summary)
        return "\n".join(summary_lines)

    def _summarize_motives_analysis(self, motives_analysis: Dict[str, Any]) -> str:
        """
        Create a summary of the motives analysis for the final prompt.
        """
        if not motives_analysis:
            return ""
        
        primary_motive = motives_analysis.get("primary_motive", "unknown")
        confidence_score = motives_analysis.get("confidence_score", 0.0)
        manipulation_indicators = motives_analysis.get("manipulation_indicators", [])
        credibility_assessment = motives_analysis.get("credibility_assessment", "neutral")
        risk_level = motives_analysis.get("risk_level", "low")
        
        summary_lines = [
            "\n**Motives Analysis:**",
            f"Primary Motive: {primary_motive} (Confidence: {confidence_score:.2f})",
            f"Credibility Assessment: {credibility_assessment}",
            f"Risk Level: {risk_level}"
        ]
        
        if manipulation_indicators:
            summary_lines.append("Manipulation Indicators:")
            for indicator in manipulation_indicators[:5]:  # Limit to top 5
                summary_lines.append(f"- {indicator}")
        
        secondary_motives = motives_analysis.get("secondary_motives", [])
        if secondary_motives:
            summary_lines.append("Secondary Motives:")
            for motive in secondary_motives:
                summary_lines.append(f"- {motive['type']} (Confidence: {motive['confidence']:.2f})")
        
        return "\n".join(summary_lines)


# Singleton instance
verdict_service = VerdictService() 
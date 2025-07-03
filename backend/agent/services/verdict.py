"""
Service for generating the final verdict.
"""
import logging
import json
import re
from typing import Dict, Any

from agent.llm import llm_manager
from agent.prompts import VERDICT_GENERATION_PROMPT
from app.exceptions import LLMError
from app.schemas import FactCheckResult, VerdictResult

logger = logging.getLogger(__name__)


class VerdictService:
    """Service to generate the final verdict based on fact-checking results."""

    def __init__(self):
        self.llm_manager = llm_manager

    async def generate(
        self,
        fact_check_result: FactCheckResult,
        user_prompt: str,
        temporal_analysis: Dict[str, Any]
    ) -> VerdictResult:
        """
        Generate the final verdict.

        Args:
            fact_check_result: The results from the fact-checking service.
            user_prompt: The original user prompt.
            temporal_analysis: The temporal analysis context.

        Returns:
            A VerdictResult object.
        """
        summary = self._summarize_fact_check(fact_check_result)

        try:
            prompt = await VERDICT_GENERATION_PROMPT.aformat(
                research_results=summary,
                user_prompt=user_prompt,
                temporal_analysis=json.dumps(temporal_analysis)
            )
            response = await self.llm_manager.invoke_text_only(prompt)
            
            # Clean and parse the response directly with Pydantic
            clean_response = re.sub(r"```json\n|```", "", response).strip()
            return VerdictResult.model_validate_json(clean_response)

        except Exception as e:
            logger.error(f"Failed to generate verdict: {e}", exc_info=True)
            raise LLMError(f"Failed to generate verdict: {e}", error_code="VERDICT_GENERATION_FAILED")

    def _summarize_fact_check(self, fact_check_result: FactCheckResult) -> str:
        """
        Create a concise summary of the fact-checking results for the final prompt.
        """
        summary_lines = []
        for claim_result in fact_check_result.claim_results:
            summary_lines.append(
                f"Claim: '{claim_result.get('claim', 'N/A')}'\n"
                f"Assessment: {claim_result.get('assessment', 'unverified')} "
                f"(Confidence: {claim_result.get('confidence', 0.0):.2f})\n"
                f"Summary: {claim_result.get('summary', 'No summary available.')}\n"
            )

        overall_summary = (
            "\nOverall Fact-Check Summary:\n"
            f"- Total Sources Found: {fact_check_result.summary.total_sources_found}\n"
            f"- Credible Sources: {fact_check_result.summary.credible_sources}\n"
            f"- Supporting Evidence Found: {fact_check_result.summary.supporting_evidence}\n"
            f"- Contradicting Evidence Found: {fact_check_result.summary.contradicting_evidence}\n"
        )
        summary_lines.append(overall_summary)
        return "\n".join(summary_lines)


# Singleton instance
verdict_service = VerdictService() 
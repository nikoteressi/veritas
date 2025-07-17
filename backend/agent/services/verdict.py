"""
Service for generating the final verdict.
"""
import logging
import json
import re
from typing import Dict, Any, List, Optional

from langchain_core.output_parsers import PydanticOutputParser
from agent.llm import llm_manager
from agent.prompt_manager import prompt_manager
from app.exceptions import LLMError
from agent.models import FactCheckResult, VerdictResult
from agent.models.temporal_analysis import TemporalAnalysisResult
from agent.models.motives_analysis import MotivesAnalysisResult

logger = logging.getLogger(__name__)


class VerdictService:
    """Service to generate the final verdict based on fact-checking results."""

    def __init__(self):
        self.llm_manager = llm_manager

    async def generate(
        self,
        fact_check_result: FactCheckResult,
        user_prompt: str,
        temporal_analysis: Optional[TemporalAnalysisResult] = None,
        motives_analysis: Optional[MotivesAnalysisResult] = None,
        summary: Optional[str] = None
    ) -> VerdictResult:
        """
        Generate the final verdict using enhanced context from summarization and motives analysis.

        Args:
            fact_check_result: The results from the fact-checking service.
            user_prompt: The original user prompt.
            temporal_analysis: The temporal analysis result.
            motives_analysis: The motives analysis result (now available before verdict generation).
            summary: The summary from summarization service (enhanced context).

        Returns:
            A VerdictResult object.
        """
        # Use provided summary or create fallback summary
        research_summary = summary if summary else self._summarize_fact_check(fact_check_result)
        motives_summary = self._summarize_motives_analysis(motives_analysis)
        
        # Enhanced context logging
        if summary:
            logger.info(f"Using enhanced summary for verdict: {summary[:100]}...")
        if motives_analysis:
            logger.info(f"Using motives analysis for verdict: {motives_analysis.primary_motive}")
        
        parser = PydanticOutputParser(pydantic_object=VerdictResult)

        try:
            # Use enhanced prompt template that better integrates all components
            prompt_template = prompt_manager.get_prompt_template("verdict_generation_enhanced")

            # Convert temporal_analysis to dict for JSON serialization
            temporal_dict = temporal_analysis.model_dump() if temporal_analysis else {}
            
            # Create comprehensive research context
            comprehensive_context = self._create_comprehensive_context(
                research_summary, motives_summary, summary, motives_analysis
            )

            prompt = await prompt_template.aformat(
                comprehensive_context=comprehensive_context,
                user_prompt=user_prompt,
                temporal_analysis=json.dumps(temporal_dict),
                format_instructions=parser.get_format_instructions()
            )

            logger.info(f"Enhanced verdict generation prompt prepared")

            response = await self.llm_manager.invoke_text_only(prompt)
            logger.info(f"Raw response from LLM: {response}")

            # Clean and parse the response directly with Pydantic
            parsed_response = parser.parse(response)
            logger.info(f"Enhanced verdict generated: {parsed_response.verdict}")
            
            return parsed_response

        except Exception as e:
            logger.error(f"Failed to generate enhanced verdict: {e}", exc_info=True)
            # Fallback to basic verdict generation
            return await self._generate_fallback_verdict(fact_check_result, user_prompt, temporal_analysis)

    def _summarize_fact_check(self, fact_check_result: FactCheckResult) -> str:
        """
        Create a concise summary of the fact-checking results for the final prompt.
        """
        summary_lines = []
        all_sources = set()
        primary_thesis = None
        
        # Check if we have hierarchical structure - ClaimResult doesn't have primary_thesis field
        # This logic seems to be legacy code, skipping primary_thesis check
        
        for claim_result in fact_check_result.claim_results:
            # Collect sources from each claim (ClaimResult has 'sources' field)
            claim_sources = getattr(claim_result, 'sources', [])
            all_sources.update(claim_sources)
            
            # Format claim summary with sources and context
            sources_text = ""
            if claim_sources:
                sources_text = f"\nSources checked: {', '.join(claim_sources[:3])}"
                if len(claim_sources) > 3:
                    sources_text += f" and {len(claim_sources) - 3} more"
            
            # ClaimResult doesn't have 'context' field, removing context logic
            
            summary_lines.append(
                f"Supporting Fact: '{getattr(claim_result, 'claim', 'N/A')}'\n"
                f"Assessment: {getattr(claim_result, 'assessment', 'unverified')} "
                f"(Confidence: {getattr(claim_result, 'confidence', 0.0):.2f})\n"
                f"Summary: {getattr(claim_result, 'reasoning', 'No summary available.')}{sources_text}\n"
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


    def _create_comprehensive_context(
        self, 
        research_summary: str, 
        motives_summary: str, 
        summary: Optional[str], 
        motives_analysis: Optional[MotivesAnalysisResult]
    ) -> str:
        """Create a comprehensive context combining all analysis results."""
        context_parts = []
        
        # Add enhanced summary if available
        if summary:
            context_parts.append("**COMPREHENSIVE SUMMARY:**")
            context_parts.append(summary)
            context_parts.append("")
        
        # Add research summary
        context_parts.append("**FACT-CHECK ANALYSIS:**")
        context_parts.append(research_summary)
        context_parts.append("")
        
        # Add enhanced motives analysis
        if motives_analysis:
            context_parts.append("**MOTIVES & INTENT ANALYSIS:**")
            context_parts.append(motives_summary)
            
            # Add additional context from motives analysis
            if hasattr(motives_analysis, 'analysis_summary') and motives_analysis.analysis_summary:
                context_parts.append(f"**Analysis Summary:** {motives_analysis.analysis_summary}")
            
            if hasattr(motives_analysis, 'credibility_assessment') and motives_analysis.credibility_assessment:
                context_parts.append(f"**Credibility Assessment:** {motives_analysis.credibility_assessment}")
            
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    async def _generate_fallback_verdict(
        self,
        fact_check_result: FactCheckResult,
        user_prompt: str,
        temporal_analysis: Optional[TemporalAnalysisResult] = None
    ) -> VerdictResult:
        """Generate a basic verdict when enhanced generation fails."""
        try:
            logger.warning("Using fallback verdict generation")
            
            research_summary = self._summarize_fact_check(fact_check_result)
            parser = PydanticOutputParser(pydantic_object=VerdictResult)
            
            # Use basic prompt template
            prompt_template = prompt_manager.get_prompt_template("verdict_generation")
            temporal_dict = temporal_analysis.model_dump() if temporal_analysis else {}
            
            prompt = await prompt_template.aformat(
                research_results=research_summary,
                user_prompt=user_prompt,
                temporal_analysis=json.dumps(temporal_dict),
                format_instructions=parser.get_format_instructions()
            )
            
            response = await self.llm_manager.invoke_text_only(prompt)
            parsed_response = parser.parse(response)
            
            logger.info("Fallback verdict generated successfully")
            return parsed_response
            
        except Exception as e:
            logger.error(f"Even fallback verdict generation failed: {e}")
            # Return minimal safe verdict
            return VerdictResult(
                verdict="inconclusive",
                confidence_score=0.1,
                reasoning="Verdict generation failed, unable to determine accuracy",
                evidence_summary="Analysis could not be completed due to technical issues"
            )

    def _summarize_motives_analysis(self, motives_analysis: Optional[MotivesAnalysisResult]) -> str:
        """
        Create a summary of the motives analysis for the final prompt.
        """
        if not motives_analysis:
            return ""
        
        summary_lines = [
            "\n**Motives Analysis:**",
            f"Primary Motive: {motives_analysis.primary_motive} (Confidence: {motives_analysis.confidence_score:.2f})",
            f"Credibility Assessment: {motives_analysis.credibility_assessment}",
            f"Risk Level: {motives_analysis.risk_level}"
        ]
        
        if motives_analysis.manipulation_indicators:
            summary_lines.append("Manipulation Indicators:")
            for indicator in motives_analysis.manipulation_indicators[:5]:  # Limit to top 5
                summary_lines.append(f"- {indicator}")
        
        # Add analysis summary if available
        if hasattr(motives_analysis, 'analysis_summary') and motives_analysis.analysis_summary:
            summary_lines.append(f"Analysis Summary: {motives_analysis.analysis_summary}")
        
        return "\n".join(summary_lines)


# Singleton instance
verdict_service = VerdictService()
"""
Service for handling the fact-checking process.
"""
import logging
from typing import Dict, Any, List, Type, Optional

from langchain_core.output_parsers import PydanticOutputParser

from agent.llm import llm_manager
from agent.prompt_manager import prompt_manager
from agent.fact_checkers.base import BaseFactChecker
from agent.fact_checkers.general_checker import GeneralFactChecker
from agent.fact_checkers.financial_checker import FinancialFactChecker
from agent.models import FactCheckResult, FactCheckSummary, ClaimResult, FactCheckerResponse
from agent.models.fact_checking_models import QueryGenerationOutput
from agent.models.verification_context import VerificationContext
from agent.analyzers.temporal_analyzer import TemporalAnalysisResult
from agent.tools import SearxNGSearchTool

logger = logging.getLogger(__name__)


class FactCheckingService:
    """Service to manage the claim fact-checking process."""

    FACT_CHECKER_REGISTRY: Dict[str, Type[BaseFactChecker]] = {
        "general": GeneralFactChecker,
        "financial": FinancialFactChecker,
        "bitcoin": FinancialFactChecker,  # Bitcoin topics use financial fact-checker
        # Crypto topics use financial fact-checker
        "cryptocurrency": FinancialFactChecker,
        "crypto": FinancialFactChecker,  # Crypto topics use financial fact-checker
        "finance": FinancialFactChecker,  # Finance topics use financial fact-checker
        # Investment topics use financial fact-checker
        "investment": FinancialFactChecker,
        "market": FinancialFactChecker,  # Market topics use financial fact-checker
        "trading": FinancialFactChecker,  # Trading topics use financial fact-checker
    }

    def __init__(self, search_tool: SearxNGSearchTool):
        self.llm_manager = llm_manager
        self.search_tool = search_tool

    async def check(self, context: VerificationContext) -> FactCheckResult:
        """
        Fact-check all identified claims from the verification context.

        Args:
            context: The verification context containing all data.

        Returns:
            A FactCheckResult object containing the results.
        """
        # Use claims from context
        claims_list = context.post_analysis_result.fact_hierarchy.supporting_facts
        primary_thesis = context.post_analysis_result.fact_hierarchy.primary_thesis
        temporal_analysis = context.temporal_analysis_result

        if not claims_list:
            logger.warning("No claims found in context.")
            return FactCheckResult(
                claim_results=[],
                examined_sources=[],
                search_queries_used=[],
                summary=FactCheckSummary(
                    total_sources_found=0,
                    credible_sources=0,
                    supporting_evidence=0,
                    contradicting_evidence=0,
                )
            )

        primary_topic = context.post_analysis_result.primary_topic.lower(
        ) if context.post_analysis_result.primary_topic else "general"
        checker_class = self.FACT_CHECKER_REGISTRY.get(
            primary_topic, GeneralFactChecker)
        fact_checker = checker_class(self.search_tool)

        logger.info(f"Using '{primary_topic}' fact-checker.")

        all_claim_results: List[ClaimResult] = []
        all_examined_sources = set()
        all_search_queries = set()

        for claim in claims_list:
            claim_text = claim.description
            claim_context = claim.context
            claim_primary_thesis = primary_thesis

            search_queries = await self._generate_contextual_search_queries(
                claim_text,
                claim_context,
                claim_primary_thesis,
                temporal_analysis,
                primary_topic  # Pass the primary topic for domain-specific role
            )

            all_search_queries.update(search_queries)

            claim_result = await fact_checker.check(claim_text, search_queries, temporal_analysis)

            # Convert FactCheckerResponse to ClaimResult
            if claim_result.assessment == "error":
                claim_result_obj = ClaimResult(
                    claim=claim_text,
                    assessment="error",
                    confidence=0.0,
                    supporting_evidence=0,
                    contradicting_evidence=0,
                    sources=[],
                    reasoning=claim_result.summary
                )
            else:
                # Extract sources URLs from credible_sources
                sources = [
                    source.url for source in claim_result.credible_sources]

                claim_result_obj = ClaimResult(
                    claim=claim_text,
                    assessment=claim_result.assessment,
                    confidence=claim_result.confidence,
                    supporting_evidence=claim_result.supporting_evidence,
                    contradicting_evidence=claim_result.contradicting_evidence,
                    sources=sources,
                    reasoning=claim_result.summary
                )

            all_claim_results.append(claim_result_obj)
            # Extract examined sources from the claim result sources
            all_examined_sources.update(claim_result_obj.sources)

        # Compile final summary
        summary = self._compile_summary(all_claim_results)

        return FactCheckResult(
            claim_results=all_claim_results,
            examined_sources=list(all_examined_sources),
            search_queries_used=list(all_search_queries),
            summary=summary,
        )

    async def _generate_contextual_search_queries(
        self,
        claim: str,
        claim_context: Dict[str, Any],
        primary_thesis: Optional[str],
        temporal_analysis: Optional[TemporalAnalysisResult] = None,
        domain: str = "general"
    ) -> List[str]:
        """Generate enhanced search queries using hierarchical context and domain-specific expertise."""
        try:
            parser = PydanticOutputParser(
                pydantic_object=QueryGenerationOutput)

            # Get domain-specific role description
            role_description = prompt_manager.get_domain_role_description(
                domain)

            # Create an enhanced claim description for search query generation
            # enhanced_claim = claim
            # if claim_context:
            #     context_str = ", ".join(
            #         [f"{k}: {v}" for k, v in claim_context.items() if v])
            #     if context_str:
            #         enhanced_claim = f"{claim} (Context: {context_str})"

            # if primary_thesis:
            #     enhanced_claim = f"{enhanced_claim} (Supporting: {primary_thesis})"

            temporal_context = "No specific temporal context provided."
            if temporal_analysis:
                temporal_context = (
                    f"Estimated Publication Date: {temporal_analysis.post_date}. \n"
                    f"Key Dates Mentioned: {temporal_analysis.mentioned_dates}. \n"
                    f"Temporal Context: {temporal_analysis.temporal_context}. \n"
                    f"Date Relevance: {temporal_analysis.date_relevance}."
                )

            prompt_template = prompt_manager.get_prompt_template(
                "query_generation")
            prompt = prompt_template.partial(
                role_description=role_description,
                claim=claim,
                primary_thesis=primary_thesis,
                temporal_context=temporal_context
            )

            formatted_prompt = prompt.format(
                format_instructions=parser.get_format_instructions())
            logger.info(
                f"QUERY GENERATION PROMPT (Domain: {domain}): {formatted_prompt}")

            response = await self.llm_manager.invoke_text_only(formatted_prompt)
            parsed_response = parser.parse(response)
            logger.info(f"QUERY GENERATION PARSED RESULT: {parsed_response}")

            # Extract query strings from the parsed Pydantic objects
            queries = [query.query for query in parsed_response.queries]

            logger.info(
                f"Generated {len(queries)} contextual search queries for claim: '{claim}' using {domain} domain expertise")
            return queries
        except Exception as e:
            logger.error(
                f"Failed to generate contextual search queries: {e}", exc_info=True)
            return []

    def _compile_summary(self, all_claim_results: List[ClaimResult]) -> FactCheckSummary:
        """Compile a summary from individual claim check results."""
        total_sources_found = 0
        credible_sources = 0
        supporting_evidence = 0
        contradicting_evidence = 0

        for result in all_claim_results:
            # Count sources found for this claim
            total_sources_found += len(result.sources)
            # All sources in ClaimResult.sources are considered credible
            credible_sources += len(result.sources)
            supporting_evidence += result.supporting_evidence
            contradicting_evidence += result.contradicting_evidence

        return FactCheckSummary(
            total_sources_found=total_sources_found,
            credible_sources=credible_sources,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
        )

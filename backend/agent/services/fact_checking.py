"""
Service for handling the fact-checking process.
"""
import logging
import json
from typing import Dict, Any, List, Type, Optional

from langchain_core.output_parsers import PydanticOutputParser

from agent.llm import llm_manager
from agent.prompt_manager import prompt_manager
from agent.fact_checkers.base import BaseFactChecker
from agent.fact_checkers.general_checker import GeneralFactChecker
from agent.fact_checkers.financial_checker import FinancialFactChecker
from agent.models import FactCheckResult, FactCheckSummary
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
    }

    def __init__(self, search_tool: SearxNGSearchTool):
        self.llm_manager = llm_manager
        self.search_tool = search_tool

    async def check(self, context: VerificationContext) -> FactCheckResult:
        """
        Fact-check all identified claims from the extracted information.

        Args:
            extracted_info: The structured information from the image analysis.
            user_prompt: The original user prompt.
            event_callback: Optional callback for event emission.

        Returns:
            A FactCheckResult object containing the results.
        """
        fact_hierarchy = context.fact_hierarchy
        claims_to_check = []
        primary_thesis = None
        
        if not fact_hierarchy:
            logger.warning("No fact hierarchy found in context.")
            return FactCheckResult(
                claim_results=[],
                examined_sources=[],
                search_queries_used=[],
                summary=FactCheckSummary(
                    total_sources_found=0,
                    credible_sources=[],
                    supporting_evidence=0,
                    contradicting_evidence=0,
                )
            )
        
        primary_thesis = fact_hierarchy.primary_thesis
        supporting_facts = fact_hierarchy.supporting_facts
        
        # Convert supporting facts to claims with enhanced context
        for fact in supporting_facts:
            if isinstance(fact, dict):
                claims_to_check.append({
                    "claim": fact.get("description", ""),
                    "context": fact.get("context", {}),
                    "primary_thesis": primary_thesis
                })
            else:
                claims_to_check.append({
                    "claim": str(fact),
                    "context": {},
                    "primary_thesis": primary_thesis
                })
        
        if not claims_to_check:
            logger.info("No claims to fact-check.")
            return FactCheckResult(
                claim_results=[],
                examined_sources=[],
                search_queries_used=[],
                summary=FactCheckSummary(
                    total_sources_found=0,
                    credible_sources=[],
                    supporting_evidence=0,
                    contradicting_evidence=0,
                )
            )

        logger.info(f"Claims to check: {claims_to_check}")

        primary_topic = context.primary_topic.lower() if context.primary_topic else "general"
        checker_class = self.FACT_CHECKER_REGISTRY.get(primary_topic, GeneralFactChecker)
        fact_checker = checker_class(self.search_tool)

        logger.info(f"Using '{primary_topic}' fact-checker.")

        all_claim_results = []
        all_examined_sources = set()
        all_search_queries = set()
        
        num_claims = len(claims_to_check)
        for claim_info in claims_to_check:

            claim_text = claim_info["claim"]
            claim_context = claim_info["context"]
            claim_primary_thesis = claim_info["primary_thesis"]
            
            # Generate contextual search queries using both the claim and its context
            search_queries = await self._generate_contextual_search_queries(
                claim_text,
                claim_context,
                claim_primary_thesis,
                context.temporal_analysis
            )
            all_search_queries.update(search_queries)

            claim_result_str = await fact_checker.check(claim_text, search_queries, context.temporal_analysis)
            try:
                claim_result = json.loads(claim_result_str)
                # Add context information to the result
                claim_result["context"] = claim_context
                claim_result["primary_thesis"] = claim_primary_thesis
            except json.JSONDecodeError:
                logger.error(f"Failed to decode fact-checker response for claim: '{claim_text}'")
                claim_result = {
                    "error": "Failed to get a valid response from fact-checker.",
                    "context": claim_context,
                    "primary_thesis": claim_primary_thesis
                }
            all_claim_results.append(claim_result)
            all_examined_sources.update(claim_result.get("examined_sources", []))

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
        temporal_analysis: Optional[TemporalAnalysisResult] = None
    ) -> List[str]:
        """Generate enhanced search queries using hierarchical context."""
        try:
            parser = PydanticOutputParser(pydantic_object=QueryGenerationOutput)
            
            # Create an enhanced claim description for search query generation
            enhanced_claim = claim
            if claim_context:
                context_str = ", ".join([f"{k}: {v}" for k, v in claim_context.items() if v])
                if context_str:
                    enhanced_claim = f"{claim} (Context: {context_str})"
            
            if primary_thesis:
                enhanced_claim = f"{enhanced_claim} (Supporting: {primary_thesis})"
            
            temporal_context = "No specific temporal context provided."
            if temporal_analysis:
                temporal_context = (
                f"Estimated Publication Date: {temporal_analysis.post_creation_date}. "
                f"Key Dates Mentioned: {temporal_analysis.mentioned_dates}. "
                f"Relationship: {temporal_analysis.relationship_description}."
            )

            prompt_template = prompt_manager.get_prompt_template("query_generation")
            prompt = prompt_template.partial(
                claim=enhanced_claim,
                temporal_context=temporal_context
            )
            
            formatted_prompt = prompt.format(format_instructions=parser.get_format_instructions())
            logger.info(f"Formatted prompt for QUERY GENERATION: {formatted_prompt}")
            
            response = await self.llm_manager.invoke_text_only(formatted_prompt)
            logger.info(f"Response from QUERY GENERATION: {response}")
            parsed_response = parser.parse(response)
            
            # Extract query strings from the parsed Pydantic objects
            queries = [query.query for query in parsed_response.queries]
            
            logger.info(f"Generated {len(queries)} contextual search queries for claim: '{claim}'")
            return queries
        except Exception as e:
            logger.error(f"Failed to generate contextual search queries: {e}", exc_info=True)
            return []

    def _compile_summary(self, all_claim_results: List[Dict[str, Any]]) -> FactCheckSummary:
        """Compile a summary from individual claim check results."""
        total_sources_found = 0
        credible_sources = 0
        supporting_evidence = 0
        contradicting_evidence = 0

        for result in all_claim_results:
            total_sources_found += result.get("total_sources_found", 0)
            credible_sources += len(result.get("credible_sources", []))
            supporting_evidence += result.get("supporting_evidence", 0)
            contradicting_evidence += result.get("contradicting_evidence", 0)

        return FactCheckSummary(
            total_sources_found=total_sources_found,
            credible_sources=credible_sources,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
        )
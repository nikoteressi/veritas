"""
Service for handling the fact-checking process.
"""
import logging
import json
from typing import Dict, Any, List, Type, Optional

from langchain_core.output_parsers import JsonOutputParser

from agent.llm import llm_manager
from agent.prompts import QUERY_GENERATION_PROMPT, DOMAIN_SPECIFIC_PROMPTS
from agent.fact_checkers.base import BaseFactChecker
from agent.fact_checkers.general_checker import GeneralFactChecker
from agent.fact_checkers.financial_checker import FinancialFactChecker
from app.exceptions import AgentError
from app.schemas import FactCheckResult, FactCheckSummary
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

    async def check(
        self,
        extracted_info: Dict[str, Any],
        user_prompt: str,
        progress_callback: Optional[callable] = None
    ) -> FactCheckResult:
        """
        Fact-check all identified claims from the extracted information.

        Args:
            extracted_info: The structured information from the image analysis.
            user_prompt: The original user prompt.
            progress_callback: Optional callback for progress updates.

        Returns:
            A FactCheckResult object containing the results.
        """
        claims = extracted_info.get("claims", [])
        if not claims:
            logger.info("No claims to fact-check.")
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

        primary_topic = extracted_info.get("primary_topic", "general").lower()
        checker_class = self.FACT_CHECKER_REGISTRY.get(primary_topic, GeneralFactChecker)
        fact_checker = checker_class(self.search_tool)

        logger.info(f"Using '{primary_topic}' fact-checker.")

        all_claim_results = []
        all_examined_sources = set()
        all_search_queries = set()
        
        num_claims = len(claims)
        for i, claim in enumerate(claims):
            if progress_callback:
                await progress_callback(
                    "Fact-checking claims...", 50 + int((i / num_claims) * 30)
                )

            search_queries = await self._generate_search_queries(
                claim,
                DOMAIN_SPECIFIC_PROMPTS.get(primary_topic, "general fact-checking"),
                extracted_info.get("temporal_analysis", {})
            )
            all_search_queries.update(search_queries)

            claim_result_str = await fact_checker.check(claim, search_queries, extracted_info.get("temporal_analysis", {}))
            try:
                claim_result = json.loads(claim_result_str)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode fact-checker response for claim: '{claim}'")
                claim_result = {"error": "Failed to get a valid response from fact-checker."}
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

    async def _generate_search_queries(
        self, claim: str, role_description: str, temporal_context: Dict[str, Any]
    ) -> List[str]:
        """Generate search queries for a given claim using the LLM."""
        try:
            parser = JsonOutputParser()
            prompt = QUERY_GENERATION_PROMPT.partial(
                claim=claim,
                role_description=role_description,
                temporal_context=json.dumps(temporal_context),
                format_instructions=parser.get_format_instructions(),
            )
            
            response = await self.llm_manager.invoke_text_only(prompt.format())
            parsed_response = parser.parse(response)
            
            queries = parsed_response.get("search_queries", [])
            logger.info(f"Generated {len(queries)} search queries for claim: '{claim}'")
            return queries
        except Exception as e:
            logger.error(f"Failed to generate search queries: {e}", exc_info=True)
            return [claim]  # Fallback to using the claim itself as a query

    def _compile_summary(self, claim_results: List[Dict[str, Any]]) -> FactCheckSummary:
        """Compile a summary from individual claim check results."""
        total_sources = 0
        credible_sources = 0
        supporting_evidence = 0
        contradicting_evidence = 0

        for result in claim_results:
            total_sources += result.get("total_sources_found", 0)
            credible_sources += result.get("credible_sources", 0)
            supporting_evidence += result.get("supporting_evidence", 0)
            contradicting_evidence += result.get("contradicting_evidence", 0)

        return FactCheckSummary(
            total_sources_found=total_sources,
            credible_sources=credible_sources,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
        ) 
import json
import logging
from typing import Any, Dict, List, Optional

from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import JsonOutputParser

from agent.models import FactCheckerResponse
from agent.analyzers.temporal_analyzer import TemporalAnalysisResult
from agent.fact_checkers.base import BaseFactChecker
from agent.llm import llm_manager
from agent.services.web_scraper import WebScraper

logger = logging.getLogger(__name__)


class FinancialFactChecker(BaseFactChecker):
    """
    A fact-checker specialized for financial claims.
    """
    role_description = "A financial fact-checker responsible for verifying claims about stocks, companies, and market activities."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.web_scraper = WebScraper()

    async def _select_credible_sources(self, claim: str, search_results: List[Dict[str, Any]]) -> List[str]:
        """
        Uses an LLM to select the most credible and relevant sources from search results.
        
        Raises:
            ValueError: If the LLM response is invalid or contains no URLs.
        """
        logger.info("Selecting credible sources using LLM.")
        parser = JsonOutputParser()
        
        try:
            prompt_template = self.prompt_manager.get_prompt_template("source_selection")
            prompt = await prompt_template.aformat(
                claim=claim,
                search_results=json.dumps(search_results),
                format_instructions=parser.get_format_instructions(),
            )
            response = await llm_manager.invoke_text_only(prompt)
            parsed_response = parser.parse(response)

            if not isinstance(parsed_response, dict) or "credible_urls" not in parsed_response or not parsed_response["credible_urls"]:
                raise ValueError("LLM response for source selection is invalid or empty.")

            logger.info(f"LLM selected {len(parsed_response['credible_urls'])} credible sources.")
            return parsed_response["credible_urls"]
        except Exception as e:
            logger.error(f"Failed to select credible sources for claim '{claim}': {e}")
            raise

    async def analyze_search_results(self, claim: str, search_results: List[Dict[str, Any]], temporal_context: Optional[TemporalAnalysisResult]) -> str:
        """
        Analyzes search results by first selecting credible sources, scraping them, 
        and then performing a final analysis.
        """
        parser = PydanticOutputParser(pydantic_object=FactCheckerResponse)
        
        try:
            credible_urls = await self._select_credible_sources(claim, search_results)
            scraped_results = await self.web_scraper.scrape_urls(credible_urls)
            
            successful_scrapes = [
                result for result in scraped_results if result["status"] == "success" and result["content"]
            ]

            if not successful_scrapes:
                raise ValueError("Failed to scrape any of the selected credible sources.")

            scraped_content_str = json.dumps(successful_scrapes)
            prompt_template = self.prompt_manager.get_prompt_template("claim_analysis_with_scraped_content")
            prompt = await prompt_template.aformat(
                role_description=self.role_description,
                claim=claim,
                scraped_content=scraped_content_str,
                temporal_context=temporal_context.model_dump_json() if temporal_context else "{}",
                format_instructions=parser.get_format_instructions(),
            )
            response = await llm_manager.invoke_text_only(prompt)
            parsed_response = parser.parse(response)
            
            return parsed_response.model_dump_json()
        except Exception as e:
            logger.error(f"Error analyzing financial search results for claim '{claim}': {e}")
            # Return a valid JSON object matching the schema on error
            error_response = FactCheckerResponse(
                assessment="error",
                summary=f"An error occurred during analysis: {e}",
                confidence=0.0,
                supporting_evidence=0,
                contradicting_evidence=0,
                credible_sources=[],
            )
            return error_response.model_dump_json()
import json
import logging
from typing import Any, Dict, List, Optional

from langchain.output_parsers import PydanticOutputParser

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

    def __init__(self, *args, **kwargs):
        # Remove domain from kwargs if present to avoid conflicts
        kwargs.pop('domain', None)
        super().__init__(*args, domain="financial", **kwargs)

    async def analyze_search_results(self, claim: str, search_results: List[Dict[str, Any]], temporal_context: Optional[TemporalAnalysisResult]) -> FactCheckerResponse:
        """
        Analyzes search results by first selecting credible sources, scraping them, 
        and then performing a final analysis.
        """
        parser = PydanticOutputParser(pydantic_object=FactCheckerResponse)
        
        try:
            credible_urls = await self._select_credible_sources(claim, search_results)
            
            # Use WebScraper as async context manager for proper resource cleanup
            async with WebScraper() as web_scraper:
                scraped_results = await web_scraper.scrape_urls(credible_urls)
            
            successful_scrapes = [
                result for result in scraped_results if result["status"] == "success" and result["content"]
            ]

            if not successful_scrapes:
                # Log detailed error information for debugging
                failed_scrapes = [result for result in scraped_results if result["status"] != "success"]
                error_details = []
                for failed in failed_scrapes:
                    error_msg = failed.get("error_message", "Unknown error")
                    error_details.append(f"URL: {failed['url']}, Error: {error_msg}")
                
                logger.error(f"Failed to scrape any credible sources. Details: {'; '.join(error_details)}")
                raise ValueError("Failed to scrape any of the selected credible sources.")

            logger.info(f"Successfully scraped {len(successful_scrapes)} out of {len(credible_urls)} sources")
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
            
            return parsed_response
        except Exception as e:
            logger.error(f"Error analyzing financial search results for claim '{claim}': {e}")
            # Return a valid FactCheckerResponse object on error
            return FactCheckerResponse(
                assessment="error",
                summary=f"An error occurred during analysis: {e}",
                confidence=0.0,
                supporting_evidence=0,
                contradicting_evidence=0,
                credible_sources=[],
            )
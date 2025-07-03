import json
import logging
from typing import Any, Dict, List

from langchain.output_parsers import PydanticOutputParser

from .base import BaseFactChecker
from agent.llm import llm_manager
from agent.prompts import CLAIM_ANALYSIS_PROMPT
from app.schemas import FactCheckerResponse

logger = logging.getLogger(__name__)


class FinancialFactChecker(BaseFactChecker):
    """
    Fact-checker specializing in financial claims, using a structured
    Pydantic output for robustness.
    """
    role_description = "A financial fact-checker responsible for verifying claims about stocks, companies, and market activities."

    async def analyze_search_results(self, claim: str, search_results: List[Dict[str, Any]], temporal_context: Dict[str, Any]) -> str:
        """Analyze financial search results asynchronously."""
        parser = PydanticOutputParser(pydantic_object=FactCheckerResponse)
        
        try:
            prompt = CLAIM_ANALYSIS_PROMPT.format(
                role_description=self.role_description,
                claim=claim,
                search_results=json.dumps(search_results),
                temporal_context=json.dumps(temporal_context),
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
                credible_sources=0,
            )
            return error_response.model_dump_json() 
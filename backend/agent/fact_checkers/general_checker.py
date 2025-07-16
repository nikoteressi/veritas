import json
import logging
from typing import Any, Dict, List, Optional

from langchain.output_parsers import PydanticOutputParser

from agent.models import FactCheckerResponse
from agent.analyzers.temporal_analyzer import TemporalAnalysisResult
from agent.fact_checkers.base import BaseFactChecker
from agent.llm import llm_manager

logger = logging.getLogger(__name__)


class GeneralFactChecker(BaseFactChecker):
    """
    A fact-checker for general-purpose claims.
    """
    role_description: str = (
        "A versatile fact-checker that verifies general claims by cross-referencing "
        "reputable news sources and established fact-checking organizations."
    )

    async def analyze_search_results(self, claim: str, search_results: List[Dict[str, Any]], temporal_context: Optional[TemporalAnalysisResult]) -> str:
        """Analyze general search results asynchronously."""
        parser = PydanticOutputParser(pydantic_object=FactCheckerResponse)
        
        try:
            prompt_template = self.prompt_manager.get_prompt_template("claim_analysis")
            prompt = await prompt_template.aformat(
                role_description=self.role_description,
                claim=claim,
                search_results=json.dumps(search_results),
                temporal_context=temporal_context.model_dump_json() if temporal_context else "{}",
                format_instructions=parser.get_format_instructions(),
            )
            response = await llm_manager.invoke_text_only(prompt)
            parsed_response = parser.parse(response)
            
            return parsed_response.model_dump_json()
        except Exception as e:
            logger.error(f"Error analyzing general search results for claim '{claim}': {e}")
            error_response = FactCheckerResponse(
                assessment="error",
                summary=f"An error occurred during analysis: {e}",
                confidence=0.0,
                supporting_evidence=0,
                contradicting_evidence=0,
                credible_sources=0,
            )
            return error_response.model_dump_json()
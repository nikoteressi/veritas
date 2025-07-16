"""
Refactored temporal analysis module for detecting temporal mismatches in social media posts.
"""
from ast import parse
import logging
import re
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser

from agent.analyzers.base_analyzer import BaseAnalyzer
from agent.models.verification_context import VerificationContext

from app.exceptions import TemporalAnalysisError

logger = logging.getLogger(__name__)


class TemporalAnalysisResult(BaseModel):
    post_creation_date: Optional[str] = Field(None, description="The identified creation date of the post.")
    mentioned_dates: List[Dict[str, str]] = Field(default_factory=list, description="A list of other dates mentioned in the text and their context.")
    relationship_description: str = Field(..., description="A description of the relationship between the post date and mentioned dates.")


class TemporalAnalyzer(BaseAnalyzer):
    """Analyzes temporal context of social media posts using an LLM."""

    def __init__(self, llm_manager, prompt_manager):
        super().__init__("temporal")
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager

    async def analyze(self, context: VerificationContext) -> TemporalAnalysisResult:
        """Analyzes temporal context using the reasoning LLM."""
        if not context.screenshot_data:
            raise TemporalAnalysisError("Screenshot data not found in context for temporal analysis.")

        output_parser = PydanticOutputParser(pydantic_object=TemporalAnalysisResult)

        try:
            prompt_template = self.prompt_manager.get_prompt_template("temporal_analysis_llm")
            prompt = prompt_template.partial(current_date=datetime.now().strftime("%Y-%m-%d"),
                                             post_text=context.screenshot_data.post_content.text_body,
                                             post_timestamp=context.screenshot_data.post_content.timestamp,
                                             format_instructions=output_parser.get_format_instructions())

            formatted_prompt = prompt.format()
            response = await self.llm_manager.invoke_text_only(formatted_prompt)
            # Remove the <think> block from the response
            clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            # Remove json markdown formatting
            clean_response = re.sub(r"```json\n|```", "", clean_response).strip()
            parsed_response = output_parser.parse(clean_response)
            logger.info(f"LLM parsed response: {parsed_response}")

            if not isinstance(parsed_response, TemporalAnalysisResult):
                raise TemporalAnalysisError("Failed to parse temporal analysis from LLM response.")

            logger.info("Successfully performed LLM-based temporal analysis.")
            return parsed_response

        except Exception as e:
            logger.error(f"Error during LLM-based temporal analysis: {e}", exc_info=True)
            raise TemporalAnalysisError(f"An unexpected error occurred during temporal analysis: {e}") from e
            
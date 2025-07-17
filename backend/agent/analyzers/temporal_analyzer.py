"""
Refactored temporal analysis module for detecting temporal mismatches in social media posts.
"""
import logging
from datetime import datetime

from langchain_core.output_parsers import PydanticOutputParser

from agent.analyzers.base_analyzer import BaseAnalyzer
from agent.models.verification_context import VerificationContext
from agent.models.temporal_analysis import TemporalAnalysisResult

from app.exceptions import TemporalAnalysisError

logger = logging.getLogger(__name__)


class TemporalAnalyzer(BaseAnalyzer):
    """Analyzes temporal context of social media posts using an LLM."""

    def __init__(self, llm_manager, prompt_manager):
        super().__init__("temporal")
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager

    async def analyze(self, context: VerificationContext) -> TemporalAnalysisResult:
        """Analyzes temporal context using the reasoning LLM."""
        if not context.screenshot_data:
            raise TemporalAnalysisError(
                "Screenshot data not found in context for temporal analysis.")

        output_parser = PydanticOutputParser(
            pydantic_object=TemporalAnalysisResult)

        try:
            prompt_template = self.prompt_manager.get_prompt_template(
                "temporal_analysis")
            prompt = await prompt_template.aformat(
                current_date=datetime.now(),
                post_content=context.screenshot_data,
                format_instructions=output_parser.get_format_instructions())

            logger.info("TEMPORAL ANALYSIS PROMPT: %s", prompt)

            response = await self.llm_manager.invoke_text_only(prompt)
            parsed_response = output_parser.parse(response)
            logger.info("TEMPORAL ANALYSIS PARSED RESULT: %s", parsed_response)

            if not isinstance(parsed_response, TemporalAnalysisResult):
                raise TemporalAnalysisError(
                    "Failed to parse temporal analysis from LLM response.")

            logger.info("Successfully performed LLM-based temporal analysis.")
            return parsed_response

        except Exception as e:
            logger.error(
                "Error during LLM-based temporal analysis: %s", e, exc_info=True)
            raise TemporalAnalysisError(
                f"An unexpected error occurred during temporal analysis: {e}") from e

"""
Refactored temporal analysis module for detecting temporal mismatches in social media posts.
"""
from __future__ import annotations

import logging
from datetime import datetime

from langchain_core.output_parsers import PydanticOutputParser

from agent.analyzers.base_analyzer import BaseAnalyzer
from agent.llm import OllamaLLMManager
from agent.models.temporal_analysis import TemporalAnalysisResult
from agent.models.verification_context import VerificationContext
from agent.prompts import PromptManager
from app.exceptions import TemporalAnalysisError
from app.models.progress_callback import NoOpProgressCallback, ProgressCallback

logger = logging.getLogger(__name__)


class TemporalAnalyzer(BaseAnalyzer):
    """Analyzes temporal context of social media posts using an LLM."""

    def __init__(self, llm_manager: OllamaLLMManager, prompt_manager: PromptManager):
        super().__init__("temporal")
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager
        self.progress_callback: ProgressCallback = NoOpProgressCallback()

    def set_progress_callback(self, callback: ProgressCallback | None) -> None:
        """Set the progress callback for detailed progress reporting."""
        self.progress_callback = callback or NoOpProgressCallback()

    async def analyze(self, context: VerificationContext) -> TemporalAnalysisResult:
        """Analyzes temporal context using the reasoning LLM."""
        # Initial progress
        await self.progress_callback.update_progress(0, 100, "Starting temporal analysis...")

        if not context.screenshot_data:
            raise TemporalAnalysisError(
                "Screenshot data not found in context for temporal analysis.")

        # Progress: Preparing analysis
        await self.progress_callback.update_progress(20, 100, "Preparing temporal analysis components...")

        output_parser = PydanticOutputParser(
            pydantic_object=TemporalAnalysisResult)

        try:
            # Progress: Building prompt
            await self.progress_callback.update_progress(40, 100, "Building temporal analysis prompt...")

            prompt_template = self.prompt_manager.get_prompt_template(
                "temporal_analysis")
            prompt = await prompt_template.aformat(
                current_date=datetime.now(),
                post_content=context.screenshot_data,
                format_instructions=output_parser.get_format_instructions(),
            )

            logger.info("TEMPORAL ANALYSIS PROMPT: %s", prompt)

            # Progress: Invoking LLM
            await self.progress_callback.update_progress(60, 100, "Analyzing temporal context with LLM...")

            response = await self.llm_manager.invoke_text_only(prompt)

            # Progress: Parsing response
            await self.progress_callback.update_progress(80, 100, "Parsing temporal analysis results...")

            parsed_response = output_parser.parse(response)
            logger.info("TEMPORAL ANALYSIS PARSED RESULT: %s", parsed_response)

            if not isinstance(parsed_response, TemporalAnalysisResult):
                raise TemporalAnalysisError(
                    "Failed to parse temporal analysis from LLM response.")

            # Final progress
            await self.progress_callback.update_progress(100, 100, "Temporal analysis completed")

            logger.info("Successfully performed LLM-based temporal analysis.")
            return parsed_response

        except Exception as e:
            logger.error(
                "Error during LLM-based temporal analysis: %s", e, exc_info=True)
            raise TemporalAnalysisError(
                f"An unexpected error occurred during temporal analysis: {e}") from e

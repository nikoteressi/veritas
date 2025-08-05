"""
Post analyzer service module for analyzing social media posts.
Provides functionality to analyze screenshots and temporal data using LLM.
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_core.output_parsers import PydanticOutputParser

from agent.llm.manager import OllamaLLMManager
from agent.prompts.manager import PromptManager
from agent.models.post_analysis_result import PostAnalysisResult
from agent.models.verification_context import VerificationContext
from app.models.progress_callback import ProgressCallback, NoOpProgressCallback

logger = logging.getLogger(__name__)


class PostAnalyzerService:
    """
    Service class for analyzing social media posts.

    Analyzes screenshots and temporal data using LLM to extract relevant information
    and generate analysis results. Works with VerificationContext to process post data
    and returns structured PostAnalysisResult.

    Attributes:
        llm_manager: Manager for LLM interactions
        prompt_manager: Manager for handling analysis prompts
    """

    def __init__(self, llm_manager: OllamaLLMManager, prompt_manager: PromptManager):
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager
        self.progress_callback: ProgressCallback = NoOpProgressCallback()

    def set_progress_callback(self, callback: Optional[ProgressCallback]) -> None:
        """Set the progress callback for detailed progress reporting."""
        self.progress_callback = callback or NoOpProgressCallback()

    async def analyze(self, context: VerificationContext) -> PostAnalysisResult:
        """
        Analyzes a social media post using the provided verification context.

        Processes screenshot data and temporal information to generate structured analysis results.
        Uses LLM to extract relevant information from the post content.

        Args:
            context (VerificationContext): Context containing screenshot data and temporal info

        Returns:
            PostAnalysisResult: Structured analysis results containing extracted information

        Raises:
            ValueError: If screenshot data is missing from the context
        """
        # Initial progress
        await self.progress_callback.update_progress(0, 100, "Starting post analysis...")

        if not context.screenshot_data:
            raise ValueError("Screenshot data is required for post analysis.")

        # Progress: Preparing analysis
        await self.progress_callback.update_progress(20, 100, "Preparing analysis components...")

        output_parser = PydanticOutputParser(
            pydantic_object=PostAnalysisResult)

        # Get temporal analysis using the new typed method
        temporal_analysis = context.get_temporal_analysis()

        # Ensure temporal_analysis is not None for formatting
        safe_temporal_analysis = (
            temporal_analysis if temporal_analysis is not None else "No temporal analysis available"
        )

        # Progress: Building prompt
        await self.progress_callback.update_progress(40, 100, "Building analysis prompt...")

        prompt_template = self.prompt_manager.get_prompt_template(
            "post_analysis")
        prompt = await prompt_template.aformat(
            screenshot_data=context.screenshot_data,
            temporal_analysis=safe_temporal_analysis,
            format_instructions=output_parser.get_format_instructions(),
        )

        logger.info("POST ANALYSIS PROMPT: %s", prompt)

        # Progress: Invoking LLM
        await self.progress_callback.update_progress(60, 100, "Analyzing post content with LLM...")

        response = await self.llm_manager.invoke_text_only(prompt)

        # Progress: Parsing response
        await self.progress_callback.update_progress(80, 100, "Parsing analysis results...")

        parsed_response = output_parser.parse(response)
        logger.info("POST ANALYSIS PARSED RESULT: %s", parsed_response)

        # Final progress
        await self.progress_callback.update_progress(100, 100, "Post analysis completed")

        return parsed_response

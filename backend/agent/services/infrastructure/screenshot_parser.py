"""
Service for parsing screenshots using a vision model.
"""
from __future__ import annotations

import logging

from langchain_core.output_parsers import PydanticOutputParser

from agent.llm import llm_manager
from agent.models.screenshot_data import ScreenshotData
from agent.prompts import prompt_manager
from app.exceptions import ScreenshotParsingError
from app.models.progress_callback import NoOpProgressCallback, ProgressCallback

logger = logging.getLogger(__name__)


class ScreenshotParserService:
    """Service to parse screenshots into structured data."""

    def __init__(self):
        self.progress_callback: ProgressCallback = NoOpProgressCallback()

    def set_progress_callback(self, callback: ProgressCallback | None) -> None:
        """Set the progress callback for detailed progress reporting."""
        self.progress_callback = callback or NoOpProgressCallback()

    async def parse(self, image_bytes: bytes) -> ScreenshotData:
        """Parses the screenshot and returns structured data."""
        await self.progress_callback.update_progress(10, 100, "Initializing screenshot parser...")

        output_parser = PydanticOutputParser(pydantic_object=ScreenshotData)

        try:
            await self.progress_callback.update_progress(30, 100, "Building parsing prompt...")
            prompt_template = prompt_manager.get_prompt_template("screenshot_parsing")
            prompt = prompt_template.partial(format_instructions=output_parser.get_format_instructions())
            prompt_value = prompt.format_prompt()

            await self.progress_callback.update_progress(50, 100, "Processing image with vision model...")
            response = await llm_manager.invoke_multimodal(text=prompt_value.to_string(), image_bytes=image_bytes)

            try:
                await self.progress_callback.update_progress(80, 100, "Parsing response data...")
                # Parse the cleaned JSON string into a ScreenshotData object
                parsed_data = output_parser.parse(response)
                logger.info("PARSED IMAGE DATA: %s", parsed_data)

                await self.progress_callback.update_progress(100, 100, "Screenshot parsing completed")
                logger.info("Successfully parsed screenshot data.")
                return parsed_data
            except Exception as parse_error:
                logger.error("Failed to parse JSON response: %s", parse_error, exc_info=True)
                logger.error("Invalid JSON string: %s", response)
                raise ScreenshotParsingError(f"Failed to parse screenshot due to: {parse_error}") from parse_error

        except Exception as e:
            logger.error("Error parsing screenshot: %s", e, exc_info=True)
            raise ScreenshotParsingError(f"An unexpected error occurred during screenshot parsing: {e}") from e


# Singleton instance
screenshot_parser_service = ScreenshotParserService()

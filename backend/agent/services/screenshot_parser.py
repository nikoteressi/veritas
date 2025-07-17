"""
Service for parsing screenshots using a vision model.
"""
import logging

from langchain_core.output_parsers import PydanticOutputParser

from agent.llm import llm_manager
from agent.models.screenshot_data import ScreenshotData
from agent.prompt_manager import prompt_manager
from app.exceptions import ScreenshotParsingError

logger = logging.getLogger(__name__)


class ScreenshotParserService:
    """Service to parse screenshots into structured data."""

    async def parse(self, image_bytes: bytes) -> ScreenshotData:
        """Parses the screenshot and returns structured data."""
        output_parser = PydanticOutputParser(pydantic_object=ScreenshotData)

        try:
            prompt_template = prompt_manager.get_prompt_template("screenshot_parsing")
            prompt = prompt_template.partial(format_instructions=output_parser.get_format_instructions())
            prompt_value = prompt.format_prompt()
            response = await llm_manager.invoke_multimodal(
                text=prompt_value.to_string(),
                image_bytes=image_bytes
            )

            try:
                # Parse the cleaned JSON string into a ScreenshotData object
                parsed_data = output_parser.parse(response)
                logger.info(f"PARSED IMAGE DATA: {parsed_data}")

                logger.info("Successfully parsed screenshot data.")
                return parsed_data
            except Exception as parse_error:
                logger.error(f"Failed to parse JSON response: {parse_error}", exc_info=True)
                logger.error(f"Invalid JSON string: {clean_response}")
                raise ScreenshotParsingError(f"Failed to parse screenshot due to: {parse_error}") from parse_error

        except Exception as e:
            logger.error(f"Error parsing screenshot: {e}", exc_info=True)
            raise ScreenshotParsingError(f"An unexpected error occurred during screenshot parsing: {e}") from e


# Singleton instance
screenshot_parser_service = ScreenshotParserService()
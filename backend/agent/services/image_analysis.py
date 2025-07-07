"""
Service for handling image analysis using a multimodal LLM.
"""
import logging
from datetime import datetime
from typing import Dict, Any

from langchain_core.output_parsers import JsonOutputParser

from agent.llm import llm_manager
from agent.prompts import MULTIMODAL_ANALYSIS_PROMPT
from app.exceptions import LLMError
from app.schemas import ImageAnalysisResult

logger = logging.getLogger(__name__)


class ImageAnalysisService:
    """Service to analyze images with a multimodal LLM."""

    def __init__(self):
        self.llm_manager = llm_manager

    async def analyze(self, image_bytes: bytes, user_prompt: str) -> ImageAnalysisResult:
        """
        Analyze the image using a multimodal LLM and return a structured result.

        Args:
            image_bytes: Raw image bytes.
            user_prompt: The user's question or prompt.

        Returns:
            An ImageAnalysisResult object.
            
        Raises:
            LLMError: If the analysis or parsing fails.
        """
        output_parser = JsonOutputParser(pydantic_object=ImageAnalysisResult)

        try:
            current_date = datetime.now().strftime("%Y-%m-%d")

            prompt = MULTIMODAL_ANALYSIS_PROMPT.partial(
                format_instructions=output_parser.get_format_instructions(),
                current_date=current_date
            )

            # Format the final prompt with user input
            final_prompt = await prompt.aformat(user_prompt=user_prompt)

            llm_output = await self.llm_manager.invoke_multimodal(final_prompt, image_bytes)

            parsed_output = await output_parser.aparse(llm_output)

            # Ensure we have a Pydantic model instance
            if isinstance(parsed_output, dict):
                analysis_result = ImageAnalysisResult(**parsed_output)
            else:
                analysis_result = parsed_output

            # Inject additional context that wasn't part of the initial model analysis
            analysis_result.contextual_information = {
                "current_date": current_date,
                "user_prompt": user_prompt
            }
            
            logger.info("Image analysis completed and JSON extracted.")
            return analysis_result

        except Exception as e:
            logger.error(f"Image analysis failed: {e}", exc_info=True)
            raise LLMError(
                f"Image analysis failed during processing or parsing: {e}",
                error_code="IMAGE_ANALYSIS_FAILED"
            )

# Singleton instance
image_analysis_service = ImageAnalysisService() 
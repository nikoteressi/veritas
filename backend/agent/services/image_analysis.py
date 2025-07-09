"""
Service for handling image analysis using a multimodal LLM.
"""
import logging
from datetime import datetime
from typing import Dict, Any

from langchain_core.output_parsers import JsonOutputParser

from agent.llm import llm_manager
from agent.prompt_manager import prompt_manager
from app.exceptions import LLMError
from agent.models import ImageAnalysisResult

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

            prompt_template = prompt_manager.get_prompt_template("multimodal_analysis")

            prompt = prompt_template.partial(
                format_instructions=output_parser.get_format_instructions(),
                current_date=current_date,
            )

            llm_output = await self.llm_manager.invoke_multimodal(prompt, image_bytes)

            # Clean the output to fix common JSON issues
            cleaned_output = self._clean_json_output(llm_output)
            
            parsed_output = await output_parser.aparse(cleaned_output)

            # Ensure we have a Pydantic model instance
            if isinstance(parsed_output, dict):
                analysis_result = ImageAnalysisResult(**parsed_output)
            else:
                analysis_result = parsed_output
            
            logger.info("Image analysis completed and JSON extracted.")
            return analysis_result

        except Exception as e:
            logger.error(f"Image analysis failed: {e}", exc_info=True)
            raise LLMError(
                f"Image analysis failed during processing or parsing: {e}",
                error_code="IMAGE_ANALYSIS_FAILED"
            )
    
    def _clean_json_output(self, output: str) -> str:
        """
        Clean JSON output to fix common formatting issues.
        
        Args:
            output: Raw JSON output from LLM
            
        Returns:
            Cleaned JSON string
        """
        import re
        
        # Remove underscores from numbers (e.g., 4_970 -> 4970)
        # This regex finds numbers with underscores and removes the underscores
        output = re.sub(r'(\d+)_(\d+)', r'\1\2', output)
        
        # Remove any additional underscores that might be in larger numbers
        while '_' in output and re.search(r'\d+_\d+', output):
            output = re.sub(r'(\d+)_(\d+)', r'\1\2', output)
        
        return output

# Singleton instance
image_analysis_service = ImageAnalysisService() 
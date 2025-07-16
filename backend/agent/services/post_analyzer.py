import logging
import re
from typing import Dict, Any

from ..models.post_analysis_result import PostAnalysisResult
from ..models.verification_context import VerificationContext

from langchain_core.output_parsers import PydanticOutputParser

logger = logging.getLogger(__name__)

class PostAnalyzerService:
    def __init__(self, llm_manager, prompt_manager):
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager

    async def analyze(self, context: VerificationContext) -> PostAnalysisResult:
        if not context.screenshot_data:
            raise ValueError("Screenshot data is required for post analysis.")

        output_parser = PydanticOutputParser(pydantic_object=PostAnalysisResult)

        prompt_template = self.prompt_manager.get_prompt_template("post_analysis")
        prompt = prompt_template.partial(
            screenshot_data=context.screenshot_data,
            temporal_analysis=context.temporal_analysis,
            format_instructions=output_parser.get_format_instructions())
        
        formatted_prompt = prompt.format()
        logger.info(f"Post analysis prompt: {formatted_prompt}")

        response = await self.llm_manager.invoke_text_only(formatted_prompt)
        clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        clean_response = re.sub(r"```json\n|```", "", clean_response).strip()
        formatted_response = output_parser.parse(clean_response)
        logger.info(f"Post analysis response: {formatted_response}")

        return formatted_response
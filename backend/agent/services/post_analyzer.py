import logging

from langchain_core.output_parsers import PydanticOutputParser

from ..models.post_analysis_result import PostAnalysisResult
from ..models.verification_context import VerificationContext

logger = logging.getLogger(__name__)


class PostAnalyzerService:
    def __init__(self, llm_manager, prompt_manager):
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager

    async def analyze(self, context: VerificationContext) -> PostAnalysisResult:
        if not context.screenshot_data:
            raise ValueError("Screenshot data is required for post analysis.")

        output_parser = PydanticOutputParser(
            pydantic_object=PostAnalysisResult)

        # Get temporal analysis using the new typed method
        temporal_analysis = context.get_temporal_analysis()

        prompt_template = self.prompt_manager.get_prompt_template(
            "post_analysis")
        prompt = prompt_template.partial(
            screenshot_data=context.screenshot_data,
            temporal_analysis=temporal_analysis,
            format_instructions=output_parser.get_format_instructions())

        formatted_prompt = prompt.format()
        logger.info("POST ANALYSIS PROMPT: %s", formatted_prompt)

        response = await self.llm_manager.invoke_text_only(formatted_prompt)
        parsed_response = output_parser.parse(response)
        logger.info("POST ANALYSIS PARSED RESULT: %s", parsed_response)

        return parsed_response

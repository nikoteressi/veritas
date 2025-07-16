import logging

from ..llm import OllamaLLMManager
from ..models.verification_context import VerificationContext
from ..prompt_manager import PromptManager

logger = logging.getLogger(__name__)

class SummarizerService:
    def __init__(self, llm_manager: OllamaLLMManager, prompt_manager: PromptManager):
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager

    async def summarize(self, context: VerificationContext) -> str:
        if not context.fact_check_result:
            raise ValueError("Fact check result is required for summarization.")

        prompt_template = self.prompt_manager.get_prompt_template("summarization")
        prompt = await prompt_template.aformat(
            user_prompt=context.user_prompt,
            temporal_analysis=context.temporal_analysis.model_dump_json(indent=2) if context.temporal_analysis else "{}",
            research_results=context.fact_check_result.model_dump_json(indent=2),
        )

        response = await self.llm_manager.invoke_text_only(prompt)

        return response
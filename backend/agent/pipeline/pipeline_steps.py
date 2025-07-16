"""
Pipeline steps for the verification workflow.
"""
import logging
from typing import Dict, Any, Optional, Type, Callable
from abc import ABC, abstractmethod

from agent.models.verification_context import VerificationContext
from agent.analyzers.temporal_analyzer import TemporalAnalyzer
from agent.analyzers.motives_analyzer import MotivesAnalyzer
from agent.llm import llm_manager
from agent.prompt_manager import prompt_manager
from agent.services.validation_service import validation_service
from agent.services.image_analysis import image_analysis_service
from agent.services.fact_checking import FactCheckingService
from agent.services.verdict import verdict_service
from agent.services.reputation import reputation_service
from agent.services.storage import storage_service
from agent.services.screenshot_parser import screenshot_parser_service
from agent.services.post_analyzer import PostAnalyzerService
from agent.services.summarizer import SummarizerService

from agent.services.result_compiler import ResultCompiler
from agent.tools import searxng_tool

logger = logging.getLogger(__name__)


class BasePipelineStep(ABC):
    """Base class for verification pipeline steps."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    async def execute(self, context: VerificationContext) -> VerificationContext:
        """
        Execute the verification step.
        
        Args:
            context: Verification context containing all necessary data
            
        Returns:
            Updated verification context
        """
        pass
    
    async def safe_execute(self, context: VerificationContext) -> VerificationContext:
        """
        Safely execute the step with error handling and logging.
        
        Args:
            context: Verification context containing all necessary data
            
        Returns:
            Updated verification context
        """
        self.logger.info(f"Starting {self.name} step")
        try:
            result_context = await self.execute(context)
            self.logger.info(f"Completed {self.name} step successfully")
            return result_context
        except Exception as e:
            self.logger.error(f"Failed {self.name} step: {e}", exc_info=True)
            raise


class ValidationStep(BasePipelineStep):
    """Step for validating the verification request."""
    
    def __init__(self):
        super().__init__("Validation")
    
    async def execute(self, context: VerificationContext) -> VerificationContext:
        """Validate the verification request."""
        validated_data = validation_service.validate_verification_request(
            file_data=context.image_bytes,
            prompt=context.user_prompt,
            filename=context.filename,
            session_id=context.session_id
        )
        
        context.validated_data = validated_data
        return context


class SummarizationStep(BasePipelineStep):
    """Step for generating the final summary."""

    def __init__(self):
        super().__init__("Summarization")
        self.summarizer_service = SummarizerService(llm_manager, prompt_manager)

    async def execute(self, context: VerificationContext) -> VerificationContext:
        """Generate the final summary."""
        if context.event_service:
            await context.event_service.emit_summarization_started()

        summary = await self.summarizer_service.summarize(context)
        context.summary = summary

        if context.event_service:
            await context.event_service.emit_summarization_completed()

        return context

class ScreenshotParsingStep(BasePipelineStep):
    """Step for parsing the screenshot into structured data."""
    
    def __init__(self):
        super().__init__("Screenshot Parsing")
    
    async def execute(self, context: VerificationContext) -> VerificationContext:
        """Parse the screenshot and save the result to the context."""
        if context.event_service:
            await context.event_service.emit_screenshot_parsing_started()
        
        screenshot_data = await screenshot_parser_service.parse(context.image_bytes)
        context.screenshot_data = screenshot_data
        
        if context.event_service:
            await context.event_service.emit_screenshot_parsing_completed(screenshot_data.model_dump())
            
        return context





class ReputationRetrievalStep(BasePipelineStep):
    """Step for retrieving user reputation."""
    
    def __init__(self):
        super().__init__("Reputation Retrieval")
    
    async def execute(self, context: VerificationContext) -> VerificationContext:
        """Get or create user reputation."""
        # Emit start event
        if context.event_service:
            await context.event_service.emit_reputation_retrieval_started()
        
        analysis_result = context.analysis_result
        username = getattr(analysis_result, 'username', None) or "unknown"
        
        user_reputation = await reputation_service.get_or_create(
            context.db, 
            username
        )
        
        context.user_reputation = user_reputation
        
        # Emit completion event
        if context.event_service:
            await context.event_service.emit_reputation_retrieval_completed(username)
        
        return context


class TemporalAnalysisStep(BasePipelineStep):
    """Step for performing LLM-based temporal analysis."""

    def __init__(self):
        super().__init__("Temporal Analysis")
        self.analyzer = TemporalAnalyzer(llm_manager, prompt_manager)

    async def execute(self, context: VerificationContext) -> VerificationContext:
        """Executes the temporal analysis and updates the context."""
        if context.event_service:
            await context.event_service.emit_temporal_analysis_started()

        temporal_analysis_result = await self.analyzer.analyze(context)
        context.temporal_analysis = temporal_analysis_result

        if context.event_service:
            await context.event_service.emit_temporal_analysis_completed()

        return context


class PostAnalysisStep(BasePipelineStep):
    """Step for analyzing the parsed post data to extract thesis and facts."""

    def __init__(self):
        super().__init__("Post Analysis")
        self.analyzer = PostAnalyzerService(llm_manager, prompt_manager)

    async def execute(self, context: VerificationContext) -> VerificationContext:
        """Executes the post analysis and updates the context."""
        if context.event_service:
            await context.event_service.emit_post_analysis_started()

        analysis_result = await self.analyzer.analyze(context)
        context.fact_hierarchy = analysis_result.fact_hierarchy
        context.primary_topic = analysis_result.primary_topic
        if context.event_service:
            await context.event_service.emit_post_analysis_completed()

        return context


class MotivesAnalysisStep(BasePipelineStep):
    """Step for performing LLM-based motives analysis."""

    def __init__(self):
        super().__init__("Motives Analysis")
        self.analyzer = MotivesAnalyzer(llm_manager, prompt_manager)

    async def execute(self, context: VerificationContext) -> VerificationContext:
        """Executes the motives analysis and updates the context."""
        if context.event_service:
            await context.event_service.emit_motives_analysis_started()

        motives_analysis_result = await self.analyzer.analyze(context)
        context.motives_analysis = motives_analysis_result

        if context.event_service:
            await context.event_service.emit_motives_analysis_completed()

        return context


class FactCheckingStep(BasePipelineStep):
    """Step for performing fact-checking."""
    
    def __init__(self):
        super().__init__("Fact Checking")
        self.fact_checking_service = FactCheckingService(searxng_tool)
    
    async def execute(self, context: VerificationContext) -> VerificationContext:
        """Perform fact-checking."""
        # Extract claims to determine total count
        supporting_facts = context.fact_hierarchy.supporting_facts
        total_claims = len(supporting_facts)
        
        # Emit start event
        if context.event_service:
            await context.event_service.emit_fact_checking_started(total_claims)
        
        fact_check_result = await self.fact_checking_service.check(context)
        
        context.fact_check_result = fact_check_result
        
        # Emit completion event
        if context.event_service:
            await context.event_service.emit_fact_checking_completed()
        
        return context


class VerdictGenerationStep(BasePipelineStep):
    """Step for generating the final verdict."""
    
    def __init__(self):
        super().__init__("Verdict Generation")
    
    async def execute(self, context: VerificationContext) -> VerificationContext:
        """Generate final verdict."""
        # Emit start event
        if context.event_service:
            await context.event_service.emit_verdict_generation_started()
        
        extracted_info = context.get_extracted_info()
        verdict_result = await verdict_service.generate(
            context.fact_check_result,
            context.user_prompt,
            extracted_info.get("temporal_analysis", {}),
            {}  # motives_analysis not available yet - runs after verdict generation
        )
        
        context.verdict_result = verdict_result
        
        # Emit completion event
        if context.event_service:
            await context.event_service.emit_verdict_generation_completed(verdict_result.verdict)
        
        return context


class ReputationUpdateStep(BasePipelineStep):
    """Step for updating user reputation."""
    
    def __init__(self):
        super().__init__("Reputation Update")
    
    async def execute(self, context: VerificationContext) -> VerificationContext:
        """Update user reputation."""
        # Emit start event
        if context.event_service:
            await context.event_service.emit_reputation_update_started()
        
        extracted_info = context.get_extracted_info()
        username = extracted_info.get("username", "unknown")
        
        updated_reputation = await reputation_service.update(
            context.db,
            username,
            context.verdict_result.verdict
        )
        
        context.updated_reputation = updated_reputation
        
        # Emit completion event
        if context.event_service:
            await context.event_service.emit_reputation_update_completed()
        
        return context


class ResultStorageStep(BasePipelineStep):
    """Step for storing verification results."""
    
    def __init__(self):
        super().__init__("Result Storage")
    
    async def execute(self, context: VerificationContext) -> VerificationContext:
        """Save verification result to database."""
        # Emit start event
        if context.event_service:
            await context.event_service.emit_result_storage_started()
        
        reputation_data = self._compile_reputation_data(context.updated_reputation)
        
        verification_record = await storage_service.save_verification_result(
            db=context.db,
            user_nickname=context.updated_reputation.nickname,
            image_bytes=context.image_bytes,
            user_prompt=context.user_prompt,
            extracted_info=context.get_extracted_info(),
            verdict_result=context.verdict_result.model_dump(),
            reputation_data=reputation_data
        )
        
        context.verification_record = verification_record
        context.verification_id = str(verification_record.id)
        
        # Emit completion event
        if context.event_service:
            await context.event_service.emit_result_storage_completed()
        
        return context
    
    def _compile_reputation_data(self, reputation) -> Dict[str, Any]:
        """Compile reputation data into a dictionary."""
        return {
            "nickname": reputation.nickname,
            "true_count": reputation.true_count,
            "partially_true_count": reputation.partially_true_count,
            "false_count": reputation.false_count,
            "ironic_count": reputation.ironic_count,
            "total_posts_checked": reputation.total_posts_checked,
            "warning_issued": reputation.warning_issued,
            "notification_issued": reputation.notification_issued,
            "created_at": reputation.created_at,
            "last_checked_date": reputation.last_checked_date
        }


class PipelineStepRegistry:
    """Registry for managing pipeline steps."""
    
    def __init__(self):
        self._steps: Dict[str, Type[BasePipelineStep]] = {
            "validation": ValidationStep,
            "screenshot_parsing": ScreenshotParsingStep,
            "temporal_analysis": TemporalAnalysisStep,
            "post_analysis": PostAnalysisStep,
            "reputation_retrieval": ReputationRetrievalStep,
            "fact_checking": FactCheckingStep,
            "summarization": SummarizationStep,
            "verdict_generation": VerdictGenerationStep,
            "motives_analysis": MotivesAnalysisStep,
            "reputation_update": ReputationUpdateStep,
            "result_storage": ResultStorageStep,
        }
    
    def get_step_class(self, step_name: str) -> Type[BasePipelineStep]:
        """Get step class by name."""
        if step_name not in self._steps:
            raise ValueError(f"Unknown pipeline step: {step_name}")
        return self._steps[step_name]
    
    def create_step(self, step_name: str) -> BasePipelineStep:
        """Create step instance by name."""
        step_class = self.get_step_class(step_name)
        return step_class()
    
    def register_step(self, step_name: str, step_class: Type[BasePipelineStep]) -> None:
        """Register a new step class."""
        self._steps[step_name] = step_class
    
    def get_available_steps(self) -> list[str]:
        """Get list of available step names."""
        return list(self._steps.keys())
    
    def create_pipeline_steps(self, step_names: list[str]) -> list[BasePipelineStep]:
        """Create a list of pipeline steps from step names."""
        steps = []
        for step_name in step_names:
            try:
                step = self.create_step(step_name)
                steps.append(step)
            except ValueError as e:
                logger.error(f"Failed to create step '{step_name}': {e}")
                raise
        return steps


# Global step registry instance
step_registry = PipelineStepRegistry()
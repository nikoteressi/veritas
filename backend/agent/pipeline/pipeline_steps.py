"""
Pipeline steps for the verification workflow.
"""
from __future__ import annotations

import logging
from typing import Any

from agent.analyzers.motives_analyzer import MotivesAnalyzer
from agent.analyzers.temporal_analyzer import TemporalAnalyzer
from agent.llm import llm_manager
from agent.models.verification_context import VerificationContext
from agent.pipeline.base_step import BasePipelineStep
from agent.pipeline.graph_fact_checking_step import GraphFactCheckingStep
from agent.prompts import prompt_manager
from app.models.progress_callback import PipelineProgressCallback
from app.config import VerificationSteps

from ..services.analysis.post_analyzer import PostAnalyzerService
from ..services.infrastructure.screenshot_parser import screenshot_parser_service
from ..services.infrastructure.storage import storage_service
from ..services.output.summarizer import SummarizerService
from ..services.output.verdict import verdict_service
from ..services.processing.validation_service import validation_service
from ..services.reputation.reputation import reputation_service

logger = logging.getLogger(__name__)


class ValidationStep(BasePipelineStep):
    """Step for validating the verification request."""

    def __init__(self):
        super().__init__("Validation")

    async def execute(self, context: VerificationContext) -> VerificationContext:
        """Validate the verification request."""
        await self.update_progress(0.1, "Starting validation...")
        
        await self.update_progress(0.3, "Validating file data...")
        
        await self.update_progress(0.6, "Validating request parameters...")
        
        validated_data = validation_service.validate_verification_request(
            file_data=context.image_bytes,
            prompt=context.user_prompt,
            filename=context.filename,
            session_id=context.session_id,
        )

        await self.update_progress(0.9, "Validation completed...")
        context.validated_data = validated_data
        
        await self.update_progress(1.0, "Validation finished")
        return context


class SummarizationStep(BasePipelineStep):
    """Step for generating the final summary."""

    def __init__(self):
        super().__init__("Summarization")
        self.summarizer_service = SummarizerService(
            llm_manager, prompt_manager)

    async def execute(self, context: VerificationContext) -> VerificationContext:
        """Generate the final summary."""
        # Setup progress callback
        if context.progress_manager and context.session_id:
            callback = PipelineProgressCallback(
                context.progress_manager,
                context.session_id,
                VerificationSteps.SUMMARIZATION.value
            )
            self.summarizer_service.set_progress_callback(callback)

        if context.event_service:
            await context.event_service.emit_summarization_started()

        # Get typed summarization result
        summarization_result = await self.summarizer_service.summarize(context)

        # Store typed result using new method
        context.set_summarization_result(summarization_result)

        # Keep backward compatibility with summary field
        context.summary = summarization_result.summary

        if context.event_service:
            await context.event_service.emit_summarization_completed()

        return context


class ScreenshotParsingStep(BasePipelineStep):
    """Step for parsing the screenshot into structured data."""

    def __init__(self):
        super().__init__("Screenshot Parsing")

    async def execute(self, context: VerificationContext) -> VerificationContext:
        """Parse the screenshot and save the result to the context."""
        # Setup progress callback for the service
        if context.progress_manager and context.session_id:
            callback = PipelineProgressCallback(
                context.progress_manager,
                context.session_id,
                VerificationSteps.SCREENSHOT_PARSING.value
            )
            screenshot_parser_service.set_progress_callback(callback)
        else:
            logger.warning(f"ScreenshotParsingStep: No progress manager or session_id - progress_manager: {context.progress_manager}, session_id: {context.session_id}")
        
        if context.event_service:
            await context.event_service.emit_screenshot_parsing_started()

        # The service now handles its own progress updates
        screenshot_data = await screenshot_parser_service.parse(context.image_bytes)
        
        context.screenshot_data = screenshot_data

        if context.event_service:
            model_data = screenshot_data.model_dump()
            await context.event_service.emit_screenshot_parsing_completed(model_data)

        return context


class ReputationRetrievalStep(BasePipelineStep):
    """Step for retrieving user reputation."""

    def __init__(self):
        super().__init__("Reputation Retrieval")

    async def execute(self, context: VerificationContext) -> VerificationContext:
        """Get or create user reputation."""
        await self.update_progress(0.1, "Starting reputation retrieval...")
        
        # Emit start event
        if context.event_service:
            await context.event_service.emit_reputation_retrieval_started()

        await self.update_progress(0.3, "Extracting username...")
        username = context.screenshot_data.post_content.author
        logger.info("RETRIEVING REPUTATION FOR USER: %s", username)

        await self.update_progress(0.6, "Querying reputation database...")
        user_reputation = await reputation_service.get_or_create(context.db, username)
        logger.info("RETRIEVED REPUTATION FOR USER: %s", username)

        await self.update_progress(0.8, "Processing reputation data...")
        context.user_reputation = user_reputation

        # Emit completion event
        if context.event_service:
            await context.event_service.emit_reputation_retrieval_completed(username)

        await self.update_progress(1.0, "Reputation retrieval completed")
        return context


class TemporalAnalysisStep(BasePipelineStep):
    """Step for performing LLM-based temporal analysis."""

    def __init__(self):
        super().__init__("Temporal Analysis")
        self.analyzer = TemporalAnalyzer(llm_manager, prompt_manager)

    async def execute(self, context: VerificationContext) -> VerificationContext:
        """Executes the temporal analysis and updates the context."""
        # Setup progress callback
        if context.progress_manager and context.session_id:
            callback = PipelineProgressCallback(
                context.progress_manager,
                context.session_id,
                VerificationSteps.TEMPORAL_ANALYSIS.value
            )
            self.analyzer.set_progress_callback(callback)

        if context.event_service:
            await context.event_service.emit_temporal_analysis_started()

        # Get typed result from analyzer
        temporal_analysis_result = await self.analyzer.analyze(context)

        # Use new typed method
        context.set_temporal_analysis(temporal_analysis_result)

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
        # Setup progress callback
        if context.progress_manager and context.session_id:
            callback = PipelineProgressCallback(
                context.progress_manager,
                context.session_id,
                VerificationSteps.POST_ANALYSIS.value
            )
            self.analyzer.set_progress_callback(callback)

        if context.event_service:
            await context.event_service.emit_post_analysis_started()

        analysis_result = await self.analyzer.analyze(context)

        # Set fact hierarchy and sync claims
        if analysis_result.fact_hierarchy:
            context.set_fact_hierarchy(analysis_result.fact_hierarchy)

        context.primary_topic = analysis_result.primary_topic
        context.post_analysis_result = analysis_result

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
        # Setup progress callback
        if context.progress_manager and context.session_id:
            callback = PipelineProgressCallback(
                context.progress_manager,
                context.session_id,
                VerificationSteps.MOTIVES_ANALYSIS.value
            )
            self.analyzer.set_progress_callback(callback)

        if context.event_service:
            await context.event_service.emit_motives_analysis_started()

        # Get summarization result (now available since motives analysis runs after summarization)
        summarization_result = context.get_summarization_result()

        # Log the enhanced context
        if summarization_result:
            logger.info(
                "Motives analysis using summarization: %s...",
                summarization_result.summary[:100],
            )
        else:
            logger.warning(
                "No summarization result available for motives analysis")

        # Get typed result from analyzer
        motives_analysis_result = await self.analyzer.analyze(context)

        # Debug logging
        logger.info(
            f"Motives analysis result type: {type(motives_analysis_result)}")
        logger.info(
            f"Motives analysis result primary_motive: {motives_analysis_result.primary_motive if motives_analysis_result else 'None'}"
        )

        # Use new typed method
        context.set_motives_analysis(motives_analysis_result)

        # Verify it was set correctly
        verification = context.get_motives_analysis()
        logger.info(
            f"Verification after setting: {verification.primary_motive if verification else 'None'}")

        if context.event_service:
            await context.event_service.emit_motives_analysis_completed()

        return context


# FactCheckingStep has been replaced with GraphFactCheckingStep
# The old implementation is removed to avoid confusion


class VerdictGenerationStep(BasePipelineStep):
    """Step for generating the final verdict."""

    def __init__(self):
        super().__init__("Verdict Generation")

    async def execute(self, context: VerificationContext) -> VerificationContext:
        """Generate final verdict using summarization and motives analysis results."""
        await self.update_progress(0.1, "Starting verdict generation...")
        
        # Setup progress callback
        if context.progress_manager and context.session_id:
            callback = PipelineProgressCallback(
                context.progress_manager,
                context.session_id,
                VerificationSteps.VERDICT_GENERATION.value
            )
            verdict_service.set_progress_callback(callback)

        # Emit start event
        if context.event_service:
            await context.event_service.emit_verdict_generation_started()

        await self.update_progress(0.2, "Gathering analysis results...")
        # Use new typed methods - now including motives analysis
        temporal_analysis = context.get_temporal_analysis()
        summarization_result = context.get_summarization_result()
        motives_analysis_result = context.get_motives_analysis()

        await self.update_progress(0.4, "Processing analysis data...")
        # Debug logging for motives analysis
        logger.info(
            f"Retrieved motives analysis type: {type(motives_analysis_result)}")
        logger.info(
            f"Retrieved motives analysis primary_motive: {motives_analysis_result.primary_motive if motives_analysis_result else 'None'}"
        )
        logger.info(
            f"Context motives_analysis_result field: {context.motives_analysis_result}")

        # Use summarization result if available, otherwise fallback to context.summary
        summary_text = summarization_result.summary if summarization_result else context.summary

        await self.update_progress(0.6, "Preparing verdict generation...")
        # Log the enhanced context for verdict generation
        if motives_analysis_result:
            logger.info(
                "Verdict generation using motives analysis: %s",
                motives_analysis_result.primary_motive,
            )
        else:
            logger.warning(
                "No motives analysis result available for verdict generation")

        await self.update_progress(0.8, "Generating final verdict...")
        verdict_result = await verdict_service.generate(
            context.fact_check_result,
            context.user_prompt,
            temporal_analysis,  # Pass the object directly, not the dict
            # Now available - motives analysis runs before verdict generation
            motives_analysis_result,
            summary_text,  # Pass the summary from summarization service
        )

        await self.update_progress(0.9, "Finalizing verdict...")
        context.verdict_result = verdict_result

        # Emit completion event
        if context.event_service:
            await context.event_service.emit_verdict_generation_completed(verdict_result.verdict)

        await self.update_progress(1.0, "Verdict generation completed")
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

        # Get username from typed extracted info
        username = "unknown"
        if context.extracted_info_typed and context.extracted_info_typed.username:
            username = context.extracted_info_typed.username

        updated_reputation = await reputation_service.update(context.db, username, context.verdict_result.verdict)

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

        reputation_data = self._compile_reputation_data(
            context.updated_reputation)

        # Use typed extracted info and add claims from context
        extracted_info_dict = {}
        if context.extracted_info_typed:
            extracted_info_dict = context.extracted_info_typed.model_dump()

        # Debug logging for claims
        logger.info(f"Context claims: {context.claims}")
        logger.info(f"Context claims type: {type(context.claims)}")
        logger.info(
            f"Context claims length: {len(context.claims) if context.claims else 'None'}")

        # Add claims from context.claims to the extracted_info_dict
        extracted_info_dict["claims"] = context.claims if context.claims else []

        logger.info(
            f"Final extracted_info_dict claims: {extracted_info_dict.get('claims', [])}")

        verification_record = await storage_service.save_verification_result(
            db=context.db,
            user_nickname=context.updated_reputation.nickname,
            image_bytes=context.image_bytes,
            user_prompt=context.user_prompt,
            extracted_info=extracted_info_dict,
            verdict_result=context.verdict_result.model_dump(),
            reputation_data=reputation_data,
        )

        context.verification_record = verification_record
        context.verification_id = str(verification_record.id)

        # Emit completion event
        if context.event_service:
            await context.event_service.emit_result_storage_completed()

        return context

    def _compile_reputation_data(self, reputation) -> dict[str, Any]:
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
            "last_checked_date": reputation.last_checked_date,
        }


class PipelineStepRegistry:
    """Registry for managing pipeline steps."""

    def __init__(self):
        self._steps: dict[str, type[BasePipelineStep]] = {
            "validation": ValidationStep,
            "screenshot_parsing": ScreenshotParsingStep,
            "temporal_analysis": TemporalAnalysisStep,
            "post_analysis": PostAnalysisStep,
            "reputation_retrieval": ReputationRetrievalStep,
            # Replaced with graph-based implementation
            "fact_checking": GraphFactCheckingStep,
            "summarization": SummarizationStep,
            "verdict_generation": VerdictGenerationStep,
            "motives_analysis": MotivesAnalysisStep,
            "reputation_update": ReputationUpdateStep,
            "result_storage": ResultStorageStep,
        }

    def get_step_class(self, step_name: str) -> type[BasePipelineStep]:
        """Get step class by name."""
        if step_name not in self._steps:
            raise ValueError(f"Unknown pipeline step: {step_name}")
        return self._steps[step_name]

    def create_step(self, step_name: str) -> BasePipelineStep:
        """Create step instance by name."""
        step_class = self.get_step_class(step_name)
        return step_class()

    def register_step(self, step_name: str, step_class: type[BasePipelineStep]) -> None:
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
                logger.error("Failed to create step '%s': %s", step_name, e)
                raise
        return steps


# Global step registry instance
step_registry = PipelineStepRegistry()

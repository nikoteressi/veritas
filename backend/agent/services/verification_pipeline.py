"""
Verification pipeline service for orchestrating individual verification steps.
"""
import logging
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions import AgentError
from app.schemas import ImageAnalysisResult, FactCheckResult, VerdictResult
from agent.services.validation_service import validation_service
from agent.services.image_analysis import image_analysis_service
from agent.services.fact_checking import FactCheckingService
from agent.services.verdict import verdict_service
from agent.services.reputation import reputation_service
from agent.services.storage import storage_service
from agent.services.progress_tracking import ProgressTrackingService
from agent.services.result_compiler import ResultCompiler
from agent.temporal_analysis import temporal_analyzer
from agent.motives_analyzer import motives_analyzer
from agent.tools import searxng_tool

logger = logging.getLogger(__name__)


class VerificationPipelineStep:
    """Base class for verification pipeline steps."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the verification step.
        
        Args:
            context: Verification context containing all necessary data
            
        Returns:
            Updated context with step results
        """
        self.logger.info(f"Starting {self.name} step")
        try:
            result = await self._execute_step(context)
            self.logger.info(f"Completed {self.name} step successfully")
            return result
        except Exception as e:
            self.logger.error(f"Failed {self.name} step: {e}", exc_info=True)
            raise
    
    async def _execute_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Override this method in subclasses."""
        raise NotImplementedError


class ValidationStep(VerificationPipelineStep):
    """Step for validating the verification request."""
    
    def __init__(self):
        super().__init__("Validation")
    
    async def _execute_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the verification request."""
        validated_data = validation_service.validate_verification_request(
            file_data=context['image_bytes'],
            prompt=context['user_prompt'],
            filename=context.get('filename'),
            session_id=context.get('session_id')
        )
        
        context.update(validated_data)
        return context


class ImageAnalysisStep(VerificationPipelineStep):
    """Step for analyzing the uploaded image."""
    
    def __init__(self):
        super().__init__("Image Analysis")
    
    async def _execute_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform image analysis."""
        analysis_result = await image_analysis_service.analyze(
            context['image_bytes'], 
            context['prompt']
        )
        
        context['analysis_result'] = analysis_result
        return context


class ReputationRetrievalStep(VerificationPipelineStep):
    """Step for retrieving user reputation."""
    
    def __init__(self):
        super().__init__("Reputation Retrieval")
    
    async def _execute_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get or create user reputation."""
        analysis_result = context['analysis_result']
        username = getattr(analysis_result, 'username', None) or "unknown"
        
        user_reputation = await reputation_service.get_or_create(
            context['db'], 
            username
        )
        
        context['user_reputation'] = user_reputation
        return context


class TemporalAnalysisStep(VerificationPipelineStep):
    """Step for performing temporal analysis."""
    
    def __init__(self):
        super().__init__("Temporal Analysis")
    
    async def _execute_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform temporal analysis."""
        analysis_result = context['analysis_result']
        extracted_info = analysis_result.dict()
        
        temporal_analysis = temporal_analyzer.analyze_temporal_context(extracted_info)
        extracted_info["temporal_analysis"] = temporal_analysis
        
        context['extracted_info'] = extracted_info
        return context


class MotivesAnalysisStep(VerificationPipelineStep):
    """Step for performing motives analysis."""
    
    def __init__(self):
        super().__init__("Motives Analysis")
    
    async def _execute_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform motives analysis."""
        extracted_info = context['extracted_info']
        motives_analysis = await motives_analyzer.analyze_motives(extracted_info)
        extracted_info["motives_analysis"] = motives_analysis
        
        context['extracted_info'] = extracted_info
        return context


class FactCheckingStep(VerificationPipelineStep):
    """Step for performing fact-checking."""
    
    def __init__(self):
        super().__init__("Fact Checking")
        self.fact_checking_service = FactCheckingService(searxng_tool)
    
    async def _execute_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform fact-checking."""
        progress_service = context.get('progress_service')
        progress_callback = None
        if progress_service:
            progress_callback = progress_service.update_fact_checking_progress
        
        fact_check_result = await self.fact_checking_service.check(
            context['extracted_info'],
            context['user_prompt'],
            progress_callback
        )
        
        context['fact_check_result'] = fact_check_result
        return context


class VerdictGenerationStep(VerificationPipelineStep):
    """Step for generating the final verdict."""
    
    def __init__(self):
        super().__init__("Verdict Generation")
    
    async def _execute_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final verdict."""
        extracted_info = context['extracted_info']
        verdict_result = await verdict_service.generate(
            context['fact_check_result'],
            context['user_prompt'],
            extracted_info.get("temporal_analysis", {}),
            extracted_info.get("motives_analysis", {})
        )
        
        context['verdict_result'] = verdict_result
        return context


class ReputationUpdateStep(VerificationPipelineStep):
    """Step for updating user reputation."""
    
    def __init__(self):
        super().__init__("Reputation Update")
    
    async def _execute_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Update user reputation."""
        extracted_info = context['extracted_info']
        username = extracted_info.get("username", "unknown")
        
        updated_reputation = await reputation_service.update(
            context['db'],
            username,
            context['verdict_result'].verdict
        )
        
        context['updated_reputation'] = updated_reputation
        return context


class ResultStorageStep(VerificationPipelineStep):
    """Step for storing verification results."""
    
    def __init__(self):
        super().__init__("Result Storage")
    
    async def _execute_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Save verification result to database."""
        reputation_data = self._compile_reputation_data(context['updated_reputation'])
        
        verification_record = await storage_service.save_verification_result(
            db=context['db'],
            user_nickname=context['updated_reputation'].nickname,
            image_bytes=context['image_bytes'],
            user_prompt=context['user_prompt'],
            extracted_info=context['extracted_info'],
            verdict_result=context['verdict_result'].dict(),
            reputation_data=reputation_data
        )
        
        context['verification_record'] = verification_record
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


class VerificationPipeline:
    """
    Pipeline for executing verification steps in sequence.
    
    This class manages the execution of individual verification steps,
    providing better separation of concerns and error handling.
    """
    
    def __init__(self):
        self.steps = [
            ValidationStep(),
            ImageAnalysisStep(),
            ReputationRetrievalStep(),
            TemporalAnalysisStep(),
            MotivesAnalysisStep(),
            FactCheckingStep(),
            VerdictGenerationStep(),
            ReputationUpdateStep(),
            ResultStorageStep()
        ]
    
    async def execute(
        self,
        image_bytes: bytes,
        user_prompt: str,
        db: AsyncSession,
        session_id: str,
        progress_callback: Optional[callable] = None,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete verification pipeline.
        
        Args:
            image_bytes: Image content to analyze
            user_prompt: User's question/prompt
            db: Database session
            session_id: Session identifier
            progress_callback: Optional progress callback
            filename: Optional filename for display
            
        Returns:
            Complete verification result
        """
        # Initialize context with input data
        context = {
            'image_bytes': image_bytes,
            'user_prompt': user_prompt,
            'db': db,
            'session_id': session_id,
            'filename': filename,
            'progress_service': ProgressTrackingService(progress_callback) if progress_callback else None,
            'result_compiler': ResultCompiler()
        }
        
        # Start timing
        if context['result_compiler']:
            context['result_compiler'].start_timing()
        
        # Execute progress tracking
        if context['progress_service']:
            await context['progress_service'].start_verification()
        
        try:
            # Execute each step in sequence
            for i, step in enumerate(self.steps):
                # Update progress
                if context['progress_service']:
                    await self._update_progress_for_step(context['progress_service'], i)
                
                # Execute step
                context = await step.execute(context)
            
            # Compile final result
            final_result = await self._compile_final_result(context)
            
            # Store in vector database
            await storage_service.store_in_vector_db(final_result)
            
            # Complete progress tracking
            if context['progress_service']:
                await context['progress_service'].complete_verification()
            
            processing_time = context['result_compiler'].get_processing_time()
            logger.info(f"Verification completed in {processing_time}s: {context['verdict_result'].verdict}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise AgentError(f"Verification pipeline failed: {e}") from e
    
    async def _update_progress_for_step(self, progress_service: ProgressTrackingService, step_index: int):
        """Update progress based on current step."""
        step_names = [
            'start_verification',
            'start_analysis', 
            'start_analysis',
            'start_temporal_analysis',
            'start_motives_analysis',
            'start_fact_checking',
            'start_verdict_generation',
            'start_reputation_update',
            'start_saving_results'
        ]
        
        if step_index < len(step_names):
            method_name = step_names[step_index]
            method = getattr(progress_service, method_name, None)
            if method:
                await method()
    
    async def _compile_final_result(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compile the final verification result."""
        reputation_data = {
            "nickname": context['updated_reputation'].nickname,
            "true_count": context['updated_reputation'].true_count,
            "partially_true_count": context['updated_reputation'].partially_true_count,
            "false_count": context['updated_reputation'].false_count,
            "ironic_count": context['updated_reputation'].ironic_count,
            "total_posts_checked": context['updated_reputation'].total_posts_checked,
            "warning_issued": context['updated_reputation'].warning_issued,
            "notification_issued": context['updated_reputation'].notification_issued,
            "created_at": context['updated_reputation'].created_at,
            "last_checked_date": context['updated_reputation'].last_checked_date
        }
        
        warnings = reputation_service.generate_warnings(context['updated_reputation'])
        
        final_result = context['result_compiler'].compile_result(
            verification_id=str(context['verification_record'].id),
            analysis_result=context['analysis_result'],
            fact_check_result=context['fact_check_result'],
            verdict_result=context['verdict_result'],
            extracted_info=context['extracted_info'],
            reputation_data=reputation_data,
            warnings=warnings,
            user_prompt=context['user_prompt'],
            image_bytes=context['image_bytes'],
            filename=context.get('filename')
        )
        
        return final_result


# Singleton instance
verification_pipeline = VerificationPipeline() 
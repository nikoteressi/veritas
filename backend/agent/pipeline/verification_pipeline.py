"""
Configurable verification pipeline service for orchestrating verification steps.
"""
import logging
from typing import Dict, Any, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions import AgentError
from app.config import settings
from agent.models.verification_context import VerificationContext
from agent.pipeline.pipeline_steps import step_registry, BasePipelineStep
from agent.services.progress_tracking import ProgressTrackingService
from agent.services.result_compiler import ResultCompiler
from agent.services.storage import storage_service
from agent.services.reputation import reputation_service

logger = logging.getLogger(__name__)


class VerificationPipeline:
    """
    Configurable pipeline for executing verification steps in sequence.
    
    This class manages the execution of individual verification steps using the
    VerificationContext model, providing better separation of concerns, type safety,
    and configurability.
    """
    
    def __init__(self, step_names: Optional[List[str]] = None):
        """
        Initialize the verification pipeline.
        
        Args:
            step_names: Optional list of step names to use. If None, uses default configuration.
        """
        if step_names is None:
            step_names = settings.get_pipeline_steps()
        
        try:
            self.steps = step_registry.create_pipeline_steps(step_names)
            self.step_names = step_names
            logger.info(f"Initialized verification pipeline with steps: {step_names}")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline with steps {step_names}: {e}")
            raise AgentError(f"Pipeline initialization failed: {e}") from e
    
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
        # Initialize verification context
        try:
            context = VerificationContext(
                image_bytes=image_bytes,
                user_prompt=user_prompt,
                session_id=session_id,
                filename=filename,
                db=db,
                progress_service=ProgressTrackingService(progress_callback) if progress_callback else None,
                result_compiler=ResultCompiler()
            )
        except Exception as e:
            logger.error(f"Failed to create verification context: {e}")
            raise AgentError(f"Context initialization failed: {e}") from e
        
        # Start timing
        if context.result_compiler:
            context.result_compiler.start_timing()
        
        # Execute progress tracking
        if context.progress_service:
            await context.progress_service.start_verification()
        
        try:
            # Execute each step in sequence
            for i, step in enumerate(self.steps):
                # Update progress
                if context.progress_service:
                    await self._update_progress_for_step(context.progress_service, i)
                
                # Execute step safely
                context = await step.safe_execute(context)
            
            # Compile final result
            final_result = await self._compile_final_result(context)
            
            # Store in vector database
            await storage_service.store_in_vector_db(final_result)
            
            # Complete progress tracking
            if context.progress_service:
                await context.progress_service.complete_verification()
            
            processing_time = context.result_compiler.get_processing_time()
            logger.info(f"Verification completed in {processing_time}s: {context.verdict_result.verdict}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise AgentError(f"Verification pipeline failed: {e}") from e
    
    async def _update_progress_for_step(self, progress_service: ProgressTrackingService, step_index: int):
        """Update progress based on current step."""
        step_progress_methods = {
            'validation': 'start_verification',
            'image_analysis': 'start_analysis',
            'reputation_retrieval': 'start_analysis',
            'temporal_analysis': 'start_temporal_analysis',
            'motives_analysis': 'start_motives_analysis',
            'fact_checking': 'start_fact_checking',
            'verdict_generation': 'start_verdict_generation',
            'reputation_update': 'start_reputation_update',
            'result_storage': 'start_saving_results'
        }
        
        if step_index < len(self.step_names):
            step_name = self.step_names[step_index]
            method_name = step_progress_methods.get(step_name)
            
            if method_name:
                method = getattr(progress_service, method_name, None)
                if method:
                    await method()
    
    async def _compile_final_result(self, context: VerificationContext) -> Dict[str, Any]:
        """Compile the final verification result."""
        reputation_data = {
            "nickname": context.updated_reputation.nickname,
            "true_count": context.updated_reputation.true_count,
            "partially_true_count": context.updated_reputation.partially_true_count,
            "false_count": context.updated_reputation.false_count,
            "ironic_count": context.updated_reputation.ironic_count,
            "total_posts_checked": context.updated_reputation.total_posts_checked,
            "warning_issued": context.updated_reputation.warning_issued,
            "notification_issued": context.updated_reputation.notification_issued,
            "created_at": context.updated_reputation.created_at,
            "last_checked_date": context.updated_reputation.last_checked_date
        }
        
        warnings = reputation_service.generate_warnings(context.updated_reputation)
        
        final_result = context.result_compiler.compile_result(
            verification_id=str(context.verification_record.id),
            analysis_result=context.analysis_result,
            fact_check_result=context.fact_check_result,
            verdict_result=context.verdict_result,
            extracted_info=context.get_extracted_info(),
            reputation_data=reputation_data,
            warnings=warnings,
            user_prompt=context.user_prompt,
            image_bytes=context.image_bytes,
            filename=context.filename
        )
        
        return final_result
    
    def get_step_names(self) -> List[str]:
        """Get the list of step names in this pipeline."""
        return self.step_names.copy()
    
    def get_steps(self) -> List[BasePipelineStep]:
        """Get the list of step instances in this pipeline."""
        return self.steps.copy()
    
    def add_step(self, step_name: str, position: Optional[int] = None) -> None:
        """
        Add a step to the pipeline.
        
        Args:
            step_name: Name of the step to add
            position: Position to insert the step. If None, appends to the end.
        """
        try:
            step = step_registry.create_step(step_name)
            
            if position is None:
                self.steps.append(step)
                self.step_names.append(step_name)
            else:
                self.steps.insert(position, step)
                self.step_names.insert(position, step_name)
            
            logger.info(f"Added step '{step_name}' to pipeline at position {position or len(self.steps)-1}")
        except Exception as e:
            logger.error(f"Failed to add step '{step_name}': {e}")
            raise AgentError(f"Failed to add step to pipeline: {e}") from e
    
    def remove_step(self, step_name: str) -> bool:
        """
        Remove a step from the pipeline.
        
        Args:
            step_name: Name of the step to remove
            
        Returns:
            True if the step was removed, False if it wasn't found
        """
        try:
            if step_name in self.step_names:
                index = self.step_names.index(step_name)
                self.steps.pop(index)
                self.step_names.remove(step_name)
                logger.info(f"Removed step '{step_name}' from pipeline")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove step '{step_name}': {e}")
            raise AgentError(f"Failed to remove step from pipeline: {e}") from e
    
    def reorder_steps(self, new_step_names: List[str]) -> None:
        """
        Reorder pipeline steps.
        
        Args:
            new_step_names: New order of step names
        """
        try:
            # Validate that all step names are available
            for step_name in new_step_names:
                if step_name not in step_registry.get_available_steps():
                    raise ValueError(f"Unknown step: {step_name}")
            
            # Create new step instances
            new_steps = step_registry.create_pipeline_steps(new_step_names)
            
            self.steps = new_steps
            self.step_names = new_step_names
            
            logger.info(f"Reordered pipeline steps to: {new_step_names}")
        except Exception as e:
            logger.error(f"Failed to reorder steps: {e}")
            raise AgentError(f"Failed to reorder pipeline steps: {e}") from e


# Create default pipeline instance
def create_default_pipeline() -> VerificationPipeline:
    """Create a pipeline with default configuration."""
    return VerificationPipeline()


# Create customizable pipeline instance
def create_custom_pipeline(step_names: List[str]) -> VerificationPipeline:
    """Create a pipeline with custom step configuration."""
    return VerificationPipeline(step_names)


# Global default pipeline instance
verification_pipeline = create_default_pipeline() 
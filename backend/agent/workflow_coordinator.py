"""
Lightweight coordinator for the verification workflow.
"""
import logging
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions import AgentError
from agent.pipeline.verification_pipeline import verification_pipeline

logger = logging.getLogger(__name__)


class WorkflowCoordinator:
    """
    Lightweight coordinator that orchestrates the verification workflow.
    
    This class now delegates to a modular pipeline for better separation of concerns.
    """
    
    def __init__(self):
        self.pipeline = verification_pipeline
    
    async def execute_verification(
        self,
        image_bytes: bytes,
        user_prompt: str,
        db: AsyncSession,
        session_id: str,
        progress_callback: Optional[callable] = None,
        event_callback: Optional[callable] = None,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete verification workflow.
        
        This method now delegates to the modular verification pipeline
        for better separation of concerns and maintainability.
        
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
        try:
            return await self.pipeline.execute(
                image_bytes=image_bytes,
                user_prompt=user_prompt,
                db=db,
                session_id=session_id,
                progress_callback=progress_callback,
                event_callback=event_callback,
                filename=filename
            )
        except Exception as e:
            logger.error(f"Workflow coordination failed: {e}", exc_info=True)
            raise AgentError(f"Verification workflow failed: {e}") from e
    



# Singleton instance
workflow_coordinator = WorkflowCoordinator() 
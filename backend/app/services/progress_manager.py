"""
Progress Management Service

This service provides centralized management of progress tracking,
step definitions, and smooth progress calculations.
"""

from typing import Dict, List, Optional
from datetime import datetime
import asyncio
import logging

from app.models.progress import (
    ProgressStep,
    PipelineConfiguration,
    SmartProgressCalculator,
    StepStatus,
    StepsDefinitionData,
    ProgressUpdateData,
    StepUpdateData
)
from app.config import VerificationSteps, settings
from app.schemas.websocket import (
    StepsDefinitionMessage,
    ProgressUpdateMessage,
    StepUpdateMessage
)
from app.websocket_manager import connection_manager

logger = logging.getLogger(__name__)


class StepDefinitionManager:
    """Manages step definitions and creates pipeline configurations."""

    # Step weights for progress calculation (total should be ~100)
    STEP_WEIGHTS = {
        VerificationSteps.VALIDATION: 5.0,
        VerificationSteps.SCREENSHOT_PARSING: 10.0,
        VerificationSteps.TEMPORAL_ANALYSIS: 8.0,
        VerificationSteps.POST_ANALYSIS: 12.0,
        VerificationSteps.REPUTATION_RETRIEVAL: 8.0,
        VerificationSteps.FACT_CHECKING: 35.0,  # Heaviest step
        VerificationSteps.SUMMARIZATION: 7.0,
        VerificationSteps.VERDICT_GENERATION: 8.0,
        VerificationSteps.MOTIVES_ANALYSIS: 5.0,
        VerificationSteps.REPUTATION_UPDATE: 2.0,
    }

    # Estimated durations in seconds
    STEP_DURATIONS = {
        VerificationSteps.VALIDATION: 2,
        VerificationSteps.SCREENSHOT_PARSING: 5,
        VerificationSteps.TEMPORAL_ANALYSIS: 4,
        VerificationSteps.POST_ANALYSIS: 6,
        VerificationSteps.REPUTATION_RETRIEVAL: 3,
        VerificationSteps.FACT_CHECKING: 15,  # Longest step
        VerificationSteps.SUMMARIZATION: 4,
        VerificationSteps.VERDICT_GENERATION: 5,
        VerificationSteps.MOTIVES_ANALYSIS: 3,
        VerificationSteps.REPUTATION_UPDATE: 2,
    }

    @classmethod
    def create_pipeline_configuration(cls, session_id: str) -> PipelineConfiguration:
        """Create a complete pipeline configuration with all steps."""
        steps = []
        total_duration = 0

        for verification_step in VerificationSteps:
            step = ProgressStep(
                id=verification_step.value,
                name=settings.get_step_display_name(
                    verification_step),
                description=settings.get_progress_message(verification_step),
                estimated_duration=cls.STEP_DURATIONS.get(
                    verification_step, 5),
                weight=cls.STEP_WEIGHTS.get(verification_step, 5.0),
                status=StepStatus.PENDING,
                metadata={
                    "verification_step": verification_step.value,
                    "order": list(VerificationSteps).index(verification_step)
                }
            )
            steps.append(step)
            total_duration += step.estimated_duration

        return PipelineConfiguration(
            steps=steps,
            session_id=session_id
        )

    @classmethod
    def create_custom_pipeline(cls, session_id: str, step_configs: List[Dict]) -> PipelineConfiguration:
        """Create a custom pipeline configuration from step configs."""
        steps = []
        total_duration = 0

        for config in step_configs:
            step = ProgressStep(
                id=config["id"],
                name=config["name"],
                description=config.get("description", ""),
                estimated_duration=config.get("estimated_duration", 5),
                weight=config.get("weight", 5.0),
                status=StepStatus.PENDING,
                metadata=config.get("metadata", {})
            )
            steps.append(step)
            total_duration += step.estimated_duration

        return PipelineConfiguration(
            steps=steps,
            session_id=session_id
        )


class ProgressManager:
    """Manages progress tracking for verification sessions."""

    def __init__(self):
        self.active_sessions: Dict[str, SmartProgressCalculator] = {}
        self.websocket_manager = None  # Will be injected

    def set_websocket_manager(self, websocket_manager):
        """Inject WebSocket manager dependency."""
        self.websocket_manager = websocket_manager

    async def start_session(self, session_id: str, custom_steps: Optional[List[Dict]] = None) -> PipelineConfiguration:
        """Start a new progress tracking session."""
        try:
            # Create pipeline configuration
            if custom_steps:
                pipeline_config = StepDefinitionManager.create_custom_pipeline(
                    session_id, custom_steps)
            else:
                pipeline_config = StepDefinitionManager.create_pipeline_configuration(
                    session_id)

            # Create progress calculator
            calculator = SmartProgressCalculator(
                pipeline_config=pipeline_config,
                start_time=datetime.now()
            )

            # Store session
            self.active_sessions[session_id] = calculator

            # Send steps definition to frontend
            await self._send_steps_definition(session_id, pipeline_config)

            logger.info(
                "Started progress session %s with %d steps", session_id, len(pipeline_config.steps))
            return pipeline_config

        except Exception as e:
            logger.error("Failed to start progress session %s: %s",
                         session_id, e)
            raise

    async def update_step_status(self, session_id: str, step_id: str, status: StepStatus,
                                 progress: float = 0.0, message: str = "") -> bool:
        """Update step status and send progress update."""
        try:
            calculator = self.active_sessions.get(session_id)
            if not calculator:
                logger.warning("No active session found for %s", session_id)
                return False

            # Update step status
            success = calculator.update_step_status(step_id, status, progress)
            if not success:
                logger.warning(
                    "Failed to update step %s in session %s", step_id, session_id)
                return False

            # Calculate overall progress
            overall_progress = calculator.calculate_weighted_progress(progress)

            # Send step update (progress should be 0-100 for frontend)
            await self._send_step_update(session_id, step_id, status, progress * 100, message)

            # Send progress update
            await self._send_progress_update(
                session_id=session_id,
                current_progress=overall_progress,
                target_progress=overall_progress,
                current_step_id=step_id,
                message=message
            )

            logger.debug(
                "Updated step %s to %s with progress %.2f", step_id, status, progress)
            return True

        except Exception as e:
            logger.error("Failed to update step status: %s", e)
            return False

    async def update_smooth_progress(self, session_id: str, step_id: str,
                                     start_progress: float, target_progress: float,
                                     duration_ms: int = 300, message: str = "") -> bool:
        """Send smooth progress updates with animation."""
        try:
            calculator = self.active_sessions.get(session_id)
            if not calculator:
                logger.warning(
                    "No calculator found for session %s", session_id)
                return False

            # Generate smooth progress updates
            updates = calculator.get_smooth_progress_updates(
                start_progress, target_progress, duration_ms
            )

            # Send each update with appropriate delay
            for i, (progress, delay) in enumerate(updates):
                await asyncio.sleep(delay / 1000.0)  # Convert to seconds

                await self._send_progress_update(
                    session_id=session_id,
                    current_progress=progress * 100,  # Convert to 0-100 range
                    target_progress=target_progress * 100,  # Convert to 0-100 range
                    current_step_id=step_id,
                    message=message,
                    animation_duration=200  # Longer duration for smoother updates
                )

            # If we reached 100%, update step status to COMPLETED
            if target_progress >= 1.0:
                status = StepStatus.COMPLETED
                success = calculator.update_step_status(
                    step_id, status, target_progress)
                if success:
                    await self._send_step_update(session_id, step_id, status, target_progress * 100, message or "Completed")

            return True

        except Exception as e:
            logger.error("Failed to send smooth progress updates: %s", e)
            return False

    async def complete_session(self, session_id: str) -> bool:
        """Complete a progress tracking session."""
        try:
            calculator = self.active_sessions.get(session_id)
            if not calculator:
                return False

            # Mark all remaining steps as completed
            for step in calculator.pipeline_config.steps:
                if step.status not in [StepStatus.COMPLETED, StepStatus.FAILED]:
                    step.status = StepStatus.COMPLETED
                    step.completed_at = datetime.now()

            # Send final progress update
            await self._send_progress_update(
                session_id=session_id,
                current_progress=100.0,
                target_progress=100.0,
                current_step_id="completion",
                message="Verification completed successfully"
            )

            # Clean up session
            del self.active_sessions[session_id]

            logger.info("Completed progress session %s", session_id)
            return True

        except Exception as e:
            logger.error("Failed to complete session %s: %s", session_id, e)
            return False

    def get_session_status(self, session_id: str) -> Optional[Dict]:
        """Get current status of a progress session."""
        calculator = self.active_sessions.get(session_id)
        if not calculator:
            return None

        completed_steps = [
            step.id for step in calculator.pipeline_config.steps
            if step.status == StepStatus.COMPLETED
        ]
        current_step = calculator.get_current_step()
        overall_progress = calculator.calculate_weighted_progress()

        return {
            "session_id": session_id,
            "overall_progress": overall_progress,
            "current_step": current_step.id if current_step else None,
            "completed_steps": completed_steps,
            "total_steps": len(calculator.pipeline_config.steps)
        }

    def get_step_progress(self, session_id: str, step_id: str) -> float:
        """Get current progress of a specific step (0.0-1.0)."""
        calculator = self.active_sessions.get(session_id)
        if not calculator:
            logger.warning("No calculator found for session %s", session_id)
            return 0.0

        step = calculator.pipeline_config.get_step_by_id(step_id)
        if not step:
            logger.warning("Step %s not found in session %s",
                           step_id, session_id)
            return 0.0

        # Return progress as 0.0-1.0 (step.progress_percentage is 0-100)
        return step.progress_percentage / 100.0

    # Private methods for WebSocket communication
    async def _send_steps_definition(self, session_id: str, pipeline_config: PipelineConfiguration):
        """Send steps definition to frontend."""
        if not self.websocket_manager:
            logger.warning(
                "No websocket_manager available for session %s", session_id)
            return

        data = StepsDefinitionData(
            steps=pipeline_config.steps,
            session_id=session_id
        )

        message = StepsDefinitionMessage(data=data)
        logger.info(
            "Sending steps_definition to session %s: %d steps", session_id,
            len(pipeline_config.steps))
        logger.debug("Steps definition message: %s", message.model_dump())
        await self.websocket_manager.send_message_to_session(session_id, message.model_dump())

    async def _send_progress_update(self, session_id: str, current_progress: float,
                                    target_progress: float, current_step_id: str,
                                    message: str, animation_duration: int = 300):
        """Send progress update to frontend."""
        if not self.websocket_manager:
            logger.warning(
                "No websocket_manager available for progress update to session %s", session_id)
            return

        data = ProgressUpdateData(
            current_progress=current_progress,
            target_progress=target_progress,
            animation_duration=animation_duration,
            current_step_id=current_step_id,
            message=message
        )

        message_obj = ProgressUpdateMessage(data=data)
        logger.debug("Progress update message: %s", message_obj.model_dump())
        await self.websocket_manager.send_message_to_session(session_id, message_obj.model_dump())

    async def _send_step_update(self, session_id: str, step_id: str, status: StepStatus,
                                progress: float, message: str):
        """Send step update to frontend."""
        if not self.websocket_manager:
            logger.warning(
                "No websocket_manager available for step update to session %s", session_id)
            return

        data = StepUpdateData(
            step_id=step_id,
            status=status,
            progress=progress,
            message=message
        )

        message_obj = StepUpdateMessage(data=data)
        await self.websocket_manager.send_message_to_session(session_id, message_obj.model_dump())


# Global progress manager instance
progress_manager = ProgressManager()

# Initialize progress manager with websocket manager


def initialize_progress_manager():
    """Initialize progress manager with websocket manager dependency."""
    progress_manager.set_websocket_manager(connection_manager)

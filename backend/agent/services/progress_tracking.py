"""
Service for tracking and reporting verification progress.
"""
import logging
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger(__name__)


class ProgressTrackingService:
    """Manages progress tracking and reporting for verification workflows."""
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        self.progress_callback = progress_callback
        self.current_step = 0
        self.total_steps = 10  # Default total steps
        self.step_descriptions = {
            10: "Analyzing image content...",
            25: "Extracting claims and user information...",
            40: "Performing temporal analysis...",
            45: "Analyzing potential motives...",
            50: "Fact-checking with external sources...",
            80: "Generating final verdict...",
            90: "Updating user reputation...",
            95: "Saving results...",
            100: "Verification complete!"
        }
    
    async def update_progress(self, message: str, progress: int, details: Optional[str] = None):
        """Update progress with a custom message and percentage."""
        if self.progress_callback:
            try:
                await self.progress_callback(message, progress, details)
                logger.debug(f"Progress updated: {progress}% - {message}")
            except Exception as e:
                logger.warning(f"Failed to send progress update: {e}")
        else:
            logger.warning("No progress callback set!")
    
    async def update_step(self, step_key: int):
        """Update progress using predefined steps."""
        if step_key in self.step_descriptions:
            await self.update_progress(self.step_descriptions[step_key], step_key, None)
    
    async def start_verification(self):
        """Signal the start of verification."""
        await self.update_step(10)
    
    async def start_analysis(self):
        """Signal the start of analysis phase."""
        await self.update_step(25)
    
    async def start_temporal_analysis(self):
        """Signal the start of temporal analysis."""
        await self.update_step(40)
    
    async def start_motives_analysis(self):
        """Signal the start of motives analysis."""
        await self.update_step(45)
    
    async def start_fact_checking(self):
        """Signal the start of fact-checking."""
        await self.update_step(50)
    
    async def update_fact_checking_progress(self, claim_index: int, total_claims: int):
        """Update fact-checking progress based on claim processing."""
        if total_claims > 0:
            progress = 50 + int((claim_index / total_claims) * 30)
            details = f"Processing claim {claim_index + 1} of {total_claims}"
            await self.update_progress("Fact-checking with external sources...", progress, details)
    
    async def start_verdict_generation(self):
        """Signal the start of verdict generation."""
        await self.update_step(80)
    
    async def start_reputation_update(self):
        """Signal the start of reputation update."""
        await self.update_step(90)
    
    async def start_saving_results(self):
        """Signal the start of saving results."""
        await self.update_step(95)
    
    async def complete_verification(self):
        """Signal the completion of verification."""
        await self.update_step(100)
    
    def set_callback(self, callback: Optional[Callable]):
        """Set or update the progress callback."""
        self.progress_callback = callback
    
    def has_callback(self) -> bool:
        """Check if a progress callback is set."""
        return self.progress_callback is not None 
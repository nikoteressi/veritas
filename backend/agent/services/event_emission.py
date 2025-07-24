"""
from __future__ import annotations

Event emission service for the verification pipeline.
"""

import logging
from collections.abc import Callable
from typing import Any, Optional

from app.schemas import ProgressEvent

logger = logging.getLogger(__name__)


class EventEmissionService:
    """Manages event emission for verification workflows."""

    def __init__(self, event_callback: Optional[Callable] = None):
        self.event_callback = event_callback

    async def emit_event(self, event_name: str, payload: dict[str, Any] = None):
        """Emit an event with optional payload."""
        if payload is None:
            payload = {}

        event = ProgressEvent(event_name=event_name, payload=payload)

        if self.event_callback:
            try:
                await self.event_callback(event)
                logger.debug(f"Event emitted: {event_name}")
            except Exception as e:
                logger.warning(f"Failed to emit event {event_name}: {e}")
        else:
            logger.warning(f"No event callback set for event: {event_name}")

    async def emit_verification_started(self):
        """Signal the start of verification."""
        await self.emit_event("VERIFICATION_STARTED")

    async def emit_screenshot_parsing_started(self):
        """Signal the start of screenshot parsing."""
        await self.emit_event("SCREENSHOT_PARSING_STARTED")

    async def emit_screenshot_parsing_completed(self, extracted_info: dict[str, Any]):
        """Signal the completion of screenshot parsing."""
        await self.emit_event(
            "SCREENSHOT_PARSING_COMPLETED", {"extracted_info": extracted_info}
        )

    async def emit_post_analysis_started(self):
        """Signal the start of post analysis."""
        await self.emit_event("POST_ANALYSIS_STARTED")

    async def emit_post_analysis_completed(self):
        """Signal the completion of post analysis."""
        await self.emit_event("POST_ANALYSIS_COMPLETED")

    async def emit_reputation_retrieval_started(self):
        """Signal the start of reputation retrieval."""
        await self.emit_event("REPUTATION_RETRIEVAL_STARTED")

    async def emit_reputation_retrieval_completed(self, username: str):
        """Signal the completion of reputation retrieval."""
        await self.emit_event("REPUTATION_RETRIEVAL_COMPLETED", {"username": username})

    async def emit_temporal_analysis_started(self):
        """Signal the start of temporal analysis."""
        await self.emit_event("TEMPORAL_ANALYSIS_STARTED")

    async def emit_temporal_analysis_completed(self):
        """Signal the completion of temporal analysis."""
        await self.emit_event("TEMPORAL_ANALYSIS_COMPLETED")

    async def emit_motives_analysis_started(self):
        """Signal the start of motives analysis."""
        await self.emit_event("MOTIVES_ANALYSIS_STARTED")

    async def emit_motives_analysis_completed(self):
        """Signal the completion of motives analysis."""
        await self.emit_event("MOTIVES_ANALYSIS_COMPLETED")

    async def emit_fact_checking_started(self, total_claims: int):
        """Signal the start of fact-checking."""
        await self.emit_event("FACT_CHECKING_STARTED", {"total_claims": total_claims})

    async def emit_fact_checking_item_completed(
        self, checked: int, total: int, claim_text: str
    ):
        """Signal the completion of a fact-checking item."""
        await self.emit_event(
            "FACT_CHECKING_ITEM_COMPLETED",
            {"checked": checked, "total": total, "claim_text": claim_text},
        )

    async def emit_fact_checking_completed(self):
        """Signal the completion of fact-checking."""
        await self.emit_event("FACT_CHECKING_COMPLETED")

    async def emit_summarization_started(self):
        """Signal the start of summarization."""
        await self.emit_event("SUMMARIZATION_STARTED")

    async def emit_summarization_completed(self):
        """Signal the completion of summarization."""
        await self.emit_event("SUMMARIZATION_COMPLETED")

    async def emit_verdict_generation_started(self):
        """Signal the start of verdict generation."""
        await self.emit_event("VERDICT_GENERATION_STARTED")

    async def emit_verdict_generation_completed(self, verdict: str):
        """Signal the completion of verdict generation."""
        await self.emit_event("VERDICT_GENERATION_COMPLETED", {"verdict": verdict})

    async def emit_reputation_update_started(self):
        """Signal the start of reputation update."""
        await self.emit_event("REPUTATION_UPDATE_STARTED")

    async def emit_reputation_update_completed(self):
        """Signal the completion of reputation update."""
        await self.emit_event("REPUTATION_UPDATE_COMPLETED")

    async def emit_result_storage_started(self):
        """Signal the start of result storage."""
        await self.emit_event("RESULT_STORAGE_STARTED")

    async def emit_result_storage_completed(self):
        """Signal the completion of result storage."""
        await self.emit_event("RESULT_STORAGE_COMPLETED")

    async def emit_verification_completed(self):
        """Signal the completion of verification."""
        await self.emit_event("VERIFICATION_COMPLETED")

    def set_callback(self, callback: Optional[Callable]):
        """Set or update the event callback."""
        self.event_callback = callback

    def has_callback(self) -> bool:
        """Check if an event callback is set."""
        return self.event_callback is not None

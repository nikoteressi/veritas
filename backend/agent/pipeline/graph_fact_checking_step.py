"""
Graph-based fact checking pipeline step.
"""
from __future__ import annotations

import logging

from agent.models.internal import FactCheckResult
from agent.models.verification_context import VerificationContext
from agent.pipeline.base_step import BasePipelineStep
from agent.tools import searxng_tool
from app.config import VerificationSteps
from app.exceptions import AgentError
from app.models.progress_callback import PipelineProgressCallback

from ..services.graph.graph_fact_checking import GraphFactCheckingService

logger = logging.getLogger(__name__)


class GraphFactCheckingStep(BasePipelineStep):
    """
    Graph-based fact checking pipeline step.

    This step replaces the traditional FactCheckingStep with a graph-based approach
    that processes related facts together for more efficient and accurate verification.
    """

    def __init__(self):
        super().__init__("Graph Fact Checking")

        self.graph_fact_checking_service = GraphFactCheckingService(
            search_tool=searxng_tool)

    async def execute(self, context: VerificationContext) -> VerificationContext:
        """Execute graph-based fact checking."""
        # Setup progress callback - let the service handle all progress updates
        if context.progress_manager and context.session_id:
            callback = PipelineProgressCallback(
                context.progress_manager,
                context.session_id,
                VerificationSteps.FACT_CHECKING.value
            )
            self.graph_fact_checking_service.set_progress_callback(callback)

        # Emit start event
        if context.event_service:
            total_claims = len(context.claims) if context.claims else 0
            await context.event_service.emit_fact_checking_started(total_claims)

        # Validate required data
        if not context.claims:
            logger.warning("No claims found for fact checking")

            context.fact_check_result = FactCheckResult(
                overall_verdict="INSUFFICIENT_DATA",
                confidence_score=0.0,
                claim_results=[],
                reasoning="No claims found to fact-check"
            )
            # Update progress for early return case
            await self.update_progress(1.0, "No claims to fact-check")
            return context

        # Log claims for debugging
        logger.info(
            f"Processing {len(context.claims)} claims for fact checking")
        for i, claim in enumerate(context.claims):
            logger.info(f"Claim {i+1}: {claim}")

        # Execute graph-based fact checking - service handles all progress updates
        fact_check_result = await self.graph_fact_checking_service.verify_facts(context)

        # Store result in context
        context.fact_check_result = fact_check_result

        # Emit completion event
        if context.event_service:
            await context.event_service.emit_fact_checking_completed()

        return context

    def get_service(self) -> GraphFactCheckingService:
        """
        Get the underlying graph fact checking service.

        Returns:
            The GraphFactCheckingService instance
        """
        return self.graph_fact_checking_service

    async def close(self):
        """Close all resources and cleanup."""
        try:
            if hasattr(self, "graph_fact_checking_service") and self.graph_fact_checking_service:
                await self.graph_fact_checking_service.close()
                logger.info("GraphFactCheckingStep closed successfully")
        except Exception as e:
            raise AgentError(
                f"Failed to close GraphFactCheckingStep: {e}") from e

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

"""
Graph-based fact checking pipeline step.
"""
from __future__ import annotations

import logging

from agent.models.verification_context import VerificationContext
from agent.pipeline.base_step import BasePipelineStep
from agent.tools import searxng_tool
from app.exceptions import AgentError
from app.config import VerificationSteps
from app.models.progress import StepStatus

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
        """
        Perform graph-based fact-checking.

        Args:
            context: Verification context containing fact hierarchy and claims

        Returns:
            Updated verification context with fact check results
        """
        # Validate that we have the necessary data
        if not context.fact_hierarchy:
            logger.warning(
                "No fact hierarchy available for graph-based fact checking")
            return context

        if not context.claims:
            logger.warning("No claims available for graph-based fact checking")
            return context

        # Extract claims to determine total count for progress tracking
        total_claims = len(context.claims)

        # Send initial progress update
        if context.progress_manager and context.session_id:
            await context.progress_manager.update_step_status(
                session_id=context.session_id,
                step_id=VerificationSteps.FACT_CHECKING.value,
                status=StepStatus.IN_PROGRESS,
                progress=0.1,
                message=f"Starting graph-based fact checking for {total_claims} claims..."
            )

        # Emit start event
        if context.event_service:
            await context.event_service.emit_fact_checking_started(total_claims)

        try:
            # Send progress update for graph building
            if context.progress_manager and context.session_id:
                await context.progress_manager.update_step_status(
                    session_id=context.session_id,
                    step_id=VerificationSteps.FACT_CHECKING.value,
                    status=StepStatus.IN_PROGRESS,
                    progress=0.3,
                    message="Building fact graph and analyzing relationships..."
                )

            # Perform graph-based fact checking
            fact_check_result = await self.graph_fact_checking_service.verify_facts(context)

            # Send progress update for verification completion
            if context.progress_manager and context.session_id:
                await context.progress_manager.update_step_status(
                    session_id=context.session_id,
                    step_id=VerificationSteps.FACT_CHECKING.value,
                    status=StepStatus.IN_PROGRESS,
                    progress=0.9,
                    message="Finalizing fact verification results..."
                )

            # Store the result in context
            context.fact_check_result = fact_check_result

            # Log summary of results
            logger.info(
                "Graph fact checking completed: %d facts verified",
                len(fact_check_result.claim_results),
            )

            # Emit completion event
            if context.event_service:
                await context.event_service.emit_fact_checking_completed()

        except (ValueError, RuntimeError, AttributeError, TypeError) as e:
            logger.error("Graph fact checking failed: %s", e, exc_info=True)

            # Send error progress update
            if context.progress_manager and context.session_id:
                await context.progress_manager.update_step_status(
                    session_id=context.session_id,
                    step_id=VerificationSteps.FACT_CHECKING.value,
                    status=StepStatus.FAILED,
                    progress=0.0,
                    message=f"Fact checking failed: {str(e)}"
                )

            # Emit error event if available
            if context.event_service and hasattr(context.event_service, "emit_fact_checking_error"):
                await context.event_service.emit_fact_checking_error(str(e))

            # Re-raise as AgentError to be handled by the pipeline
            raise AgentError(f"Graph fact checking failed: {e}") from e

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

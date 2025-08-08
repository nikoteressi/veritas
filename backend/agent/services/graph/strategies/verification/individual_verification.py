"""
Individual verification strategy implementation.

This module implements verification of individual fact nodes one by one.
"""

import asyncio
import logging
from typing import Any

from agent.models.graph import FactCluster, FactNode, VerificationStatus
from agent.models.graph_verification_models import VerificationResponse
from agent.services.graph.interfaces.verification_strategy import VerificationStrategy

logger = logging.getLogger(__name__)


class IndividualVerificationStrategy(VerificationStrategy):
    """
    Verification strategy for individual fact nodes.

    Verifies each fact node independently without considering
    relationships to other nodes.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize individual verification strategy."""
        self._config = self._get_default_config()
        if config:
            self._config.update(config)

        self._validate_config(self._config)
        self._llm_service = None  # Will be injected
        self._search_service = None  # Will be injected

        logger.info(f"Initialized {self.get_strategy_name()} with config: {self._config}")

    async def verify_node(self, node: FactNode, context: dict[str, Any] | None = None) -> VerificationResponse:
        """
        Verify a single fact node.

        Args:
            node: Fact node to verify
            context: Optional verification context

        Returns:
            Verification response with results
        """
        if not self._llm_service:
            raise RuntimeError("LLM service not configured")

        logger.info(f"Starting individual verification for node: {node.node_id}")

        try:
            # Update node status
            node.verification_status = VerificationStatus.IN_PROGRESS

            # Prepare verification context
            verification_context = self.prepare_verification_context(node, context)

            # Search for evidence if search service is available
            evidence = []
            if self._search_service and self._config["use_search"]:
                evidence = await self._search_for_evidence(node)

            # Perform LLM-based verification
            verification_result = await self._verify_with_llm(node, evidence, verification_context)

            # Update node status based on result
            if verification_result.is_verified:
                node.verification_status = VerificationStatus.VERIFIED
                node.confidence = verification_result.confidence_score
            else:
                node.verification_status = VerificationStatus.FAILED
                node.confidence = max(0.0, 1.0 - verification_result.confidence_score)

            # Update metadata with verification details
            if not node.metadata:
                node.metadata = {}

            node.metadata.update(
                {
                    "verification_method": "individual",
                    "verification_timestamp": verification_context.get("timestamp"),
                    "evidence_count": len(evidence),
                    "verification_details": verification_result.explanation,
                }
            )

            logger.info(f"Completed verification for node {node.node_id}: {verification_result.is_verified}")
            return verification_result

        except Exception as e:
            logger.error(f"Error verifying node {node.node_id}: {str(e)}")
            node.verification_status = VerificationStatus.FAILED

            # Return error response
            return VerificationResponse(
                is_verified=False,
                confidence_score=0.0,
                explanation=f"Verification failed due to error: {str(e)}",
                evidence_summary="No evidence collected due to error",
                sources_used=[],
            )

    async def verify_cluster(
        self, cluster: FactCluster, context: dict[str, Any] | None = None
    ) -> list[VerificationResponse]:
        """
        Verify all nodes in a cluster individually.

        Args:
            cluster: Fact cluster to verify
            context: Optional verification context

        Returns:
            List of verification responses for each node
        """
        logger.info(f"Starting individual verification for cluster: {cluster.cluster_id}")

        results = []

        # Process nodes based on configuration
        if self._config["parallel_processing"]:
            # Verify nodes in parallel with concurrency limit
            semaphore = asyncio.Semaphore(self._config["max_concurrent"])

            async def verify_with_semaphore(node):
                async with semaphore:
                    return await self.verify_node(node, context)

            tasks = [verify_with_semaphore(node) for node in cluster.nodes]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error in parallel verification: {str(result)}")
                    results[i] = VerificationResponse(
                        is_verified=False,
                        confidence_score=0.0,
                        explanation=f"Parallel verification failed: {str(result)}",
                        evidence_summary="No evidence due to error",
                        sources_used=[],
                    )
        else:
            # Verify nodes sequentially
            for node in cluster.nodes:
                result = await self.verify_node(node, context)
                results.append(result)

                # Add delay between requests if configured
                if self._config["request_delay"] > 0:
                    await asyncio.sleep(self._config["request_delay"])

        logger.info(f"Completed cluster verification: {len(results)} nodes processed")
        return results

    async def cross_verify(
        self, nodes: list[FactNode], context: dict[str, Any] | None = None
    ) -> list[VerificationResponse]:
        """
        Cross-verify nodes (fallback to individual verification).

        Args:
            nodes: List of fact nodes to cross-verify
            context: Optional verification context

        Returns:
            List of verification responses
        """
        logger.info("Cross-verification requested, falling back to individual verification")

        # Create temporary cluster for processing
        temp_cluster = FactCluster(
            id="temp_cross_verify", nodes=nodes, cluster_type=None, verification_strategy="individual"
        )

        return await self.verify_cluster(temp_cluster, context)

    async def _search_for_evidence(self, node: FactNode) -> list[dict[str, Any]]:
        """Search for evidence supporting the fact."""
        try:
            # Use search service to find relevant evidence
            search_results = await self._search_service.search(
                query=node.claim, max_results=self._config["max_evidence_sources"]
            )

            return search_results
        except Exception as e:
            logger.warning(f"Evidence search failed for node {node.node_id}: {str(e)}")
            return []

    async def _verify_with_llm(
        self, node: FactNode, evidence: list[dict[str, Any]], context: dict[str, Any]
    ) -> VerificationResponse:
        """Verify fact using LLM."""
        # Prepare prompt for LLM
        prompt = self._build_verification_prompt(node, evidence, context)

        # Call LLM service
        response = await self._llm_service.verify_fact(
            claim=node.claim, evidence=evidence, context=context, prompt=prompt
        )

        return response

    def _build_verification_prompt(
        self, node: FactNode, evidence: list[dict[str, Any]], context: dict[str, Any]
    ) -> str:
        """Build verification prompt for LLM."""
        prompt_parts = [
            f"Please verify the following claim: {node.claim}",
            f"Domain: {node.domain or 'General'}",
        ]

        if evidence:
            prompt_parts.append("Available evidence:")
            for i, ev in enumerate(evidence[: self._config["max_evidence_sources"]], 1):
                prompt_parts.append(f"{i}. {ev.get('content', 'No content')}")

        if context and context.get("additional_context"):
            prompt_parts.append(f"Additional context: {context['additional_context']}")

        prompt_parts.append("Provide a detailed verification response with confidence score.")

        return "\n\n".join(prompt_parts)

    def prepare_verification_context(self, node: FactNode, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Prepare verification context for a node.

        Args:
            node: Fact node being verified
            context: Optional additional context

        Returns:
            Prepared verification context
        """
        from datetime import datetime

        verification_context = {
            "node_id": node.node_id,
            "claim": node.claim,
            "domain": node.domain,
            "timestamp": datetime.now().isoformat(),
            "strategy": self.get_strategy_name(),
            "config": self._config.copy(),
        }

        if context:
            verification_context.update(context)

        return verification_context

    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "individual_verification"

    def validate_config(self, config: dict[str, Any]) -> bool:
        """
        Validate configuration parameters.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        return self._validate_config(config)

    def _validate_config(self, config: dict[str, Any]) -> bool:
        """Internal config validation."""
        if "parallel_processing" in config:
            if not isinstance(config["parallel_processing"], bool):
                raise ValueError("parallel_processing must be a boolean")

        if "max_concurrent" in config:
            if not isinstance(config["max_concurrent"], int) or config["max_concurrent"] < 1:
                raise ValueError("max_concurrent must be a positive integer")

        if "request_delay" in config:
            if not isinstance(config["request_delay"], int | float) or config["request_delay"] < 0:
                raise ValueError("request_delay must be a non-negative number")

        if "use_search" in config:
            if not isinstance(config["use_search"], bool):
                raise ValueError("use_search must be a boolean")

        if "max_evidence_sources" in config:
            if not isinstance(config["max_evidence_sources"], int) or config["max_evidence_sources"] < 1:
                raise ValueError("max_evidence_sources must be a positive integer")

        return True

    def get_config(self) -> dict[str, Any]:
        """Get current configuration."""
        return self._config.copy()

    def update_config(self, config: dict[str, Any]) -> None:
        """
        Update configuration.

        Args:
            config: New configuration parameters
        """
        new_config = self._config.copy()
        new_config.update(config)
        self._validate_config(new_config)
        self._config = new_config
        logger.info(f"Updated config for {self.get_strategy_name()}: {self._config}")

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration."""
        return {
            "parallel_processing": True,
            "max_concurrent": 3,
            "request_delay": 0.5,  # Seconds between requests
            "use_search": True,
            "max_evidence_sources": 5,
        }

    def set_llm_service(self, llm_service):
        """Set LLM service dependency."""
        self._llm_service = llm_service

    def set_search_service(self, search_service):
        """Set search service dependency."""
        self._search_service = search_service

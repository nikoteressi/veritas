"""
Batch verification strategy implementation.

This module implements batch verification of fact nodes,
processing multiple facts together for efficiency.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any

from agent.models import FactCluster, FactNode, VerificationResult
from agent.services.graph.interfaces import VerificationStrategy

logger = logging.getLogger(__name__)


class BatchVerificationStrategy(VerificationStrategy):
    """
    Batch verification strategy.

    This strategy processes multiple fact nodes together in batches,
    which can be more efficient for certain types of verification
    and allows for cross-referencing within the batch.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize batch verification strategy.

        Args:
            config: Strategy configuration
        """
        self._config = config or {}
        self._batch_size = self._config.get("batch_size", 5)
        self._parallel_batches = self._config.get("parallel_batches", True)
        self._max_concurrent_batches = self._config.get(
            "max_concurrent_batches", 2)
        self._use_search = self._config.get("use_search", True)
        self._max_evidence_sources = self._config.get(
            "max_evidence_sources", 3)

        # Services (would be injected in real implementation)
        self._llm_service = None  # Placeholder for LLM service
        self._search_service = None  # Placeholder for search service

        logger.info("BatchVerificationStrategy initialized")

    async def verify_node(self, node: FactNode, context_nodes: list[FactNode] | None = None) -> VerificationResult:
        """
        Verify a single node (delegates to batch verification).

        Args:
            node: Fact node to verify
            context_nodes: Optional context nodes

        Returns:
            Verification result
        """
        # For single node verification, create a batch of one
        batch_result = await self.verify_cluster(
            FactCluster(
                id=str(uuid.uuid4()),
                node_ids=[node.node_id],
                cluster_type="single_node",
                verification_strategy="batch_verification",
                metadata={},
                created_at=datetime.now(),
            ),
            [node] + (context_nodes or []),
        )

        return batch_result

    async def verify_cluster(self, cluster: FactCluster, all_nodes: list[FactNode]) -> VerificationResult:
        """
        Verify a cluster of fact nodes using batch processing.

        Args:
            cluster: Fact cluster to verify
            all_nodes: All available nodes for context

        Returns:
            Verification result for the cluster
        """
        try:
            # Get nodes in cluster
            cluster_nodes = [
                node for node in all_nodes if node.node_id in cluster.node_ids]

            if not cluster_nodes:
                return self._create_error_result(cluster.cluster_id, "No nodes found in cluster")

            # Split into batches
            batches = self._create_batches(cluster_nodes)

            # Process batches
            if self._parallel_batches and len(batches) > 1:
                batch_results = await self._process_batches_parallel(batches, all_nodes)
            else:
                batch_results = await self._process_batches_sequential(batches, all_nodes)

            # Combine batch results
            combined_result = self._combine_batch_results(
                cluster.cluster_id, batch_results)

            logger.info(
                f"Batch verification completed for cluster {cluster.cluster_id}")
            return combined_result

        except Exception as e:
            logger.error(
                f"Batch verification failed for cluster {cluster.cluster_id}: {str(e)}")
            return self._create_error_result(cluster.cluster_id, str(e))

    async def cross_verify(
        self, nodes: list[FactNode], verification_results: list[VerificationResult]
    ) -> list[VerificationResult]:
        """
        Perform cross-verification between nodes and their results.

        Args:
            nodes: Nodes to cross-verify
            verification_results: Existing verification results

        Returns:
            Updated verification results
        """
        try:
            # Group nodes into batches for cross-verification
            batches = self._create_batches(nodes)
            updated_results = []

            for batch in batches:
                # Find relevant verification results for this batch
                batch_node_ids = {node.node_id for node in batch}
                batch_results = [
                    result for result in verification_results if result.node_id in batch_node_ids]

                # Perform cross-verification within batch
                cross_verified = await self._cross_verify_batch(batch, batch_results)
                updated_results.extend(cross_verified)

            logger.info(f"Cross-verification completed for {len(nodes)} nodes")
            return updated_results

        except Exception as e:
            logger.error(f"Cross-verification failed: {str(e)}")
            return verification_results  # Return original results on error

    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "batch_verification"

    def validate_config(self, config: dict[str, Any]) -> bool:
        """
        Validate strategy configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if configuration is valid
        """
        try:
            batch_size = config.get("batch_size", 5)
            if not isinstance(batch_size, int) or batch_size <= 0:
                return False

            max_concurrent_batches = config.get("max_concurrent_batches", 2)
            if not isinstance(max_concurrent_batches, int) or max_concurrent_batches <= 0:
                return False

            max_evidence_sources = config.get("max_evidence_sources", 3)
            if not isinstance(max_evidence_sources, int) or max_evidence_sources <= 0:
                return False

            return True

        except Exception:
            return False

    def get_config(self) -> dict[str, Any]:
        """Get current configuration."""
        return self._config.copy()

    def update_config(self, config: dict[str, Any]) -> None:
        """
        Update strategy configuration.

        Args:
            config: New configuration
        """
        if self.validate_config(config):
            self._config.update(config)
            self._batch_size = self._config.get("batch_size", 5)
            self._parallel_batches = self._config.get("parallel_batches", True)
            self._max_concurrent_batches = self._config.get(
                "max_concurrent_batches", 2)
            self._use_search = self._config.get("use_search", True)
            self._max_evidence_sources = self._config.get(
                "max_evidence_sources", 3)
            logger.info("Batch verification strategy configuration updated")
        else:
            raise ValueError("Invalid configuration")

    async def prepare_verification_context(self, nodes: list[FactNode]) -> dict[str, Any]:
        """
        Prepare verification context for batch processing.

        Args:
            nodes: Nodes to prepare context for

        Returns:
            Verification context
        """
        context = {
            "total_nodes": len(nodes),
            "batch_size": self._batch_size,
            "estimated_batches": (len(nodes) + self._batch_size - 1) // self._batch_size,
            "parallel_processing": self._parallel_batches,
            "use_search": self._use_search,
            "domains": list(set(node.metadata.get("domain", "general") for node in nodes)),
            "sources": list(set(node.source for node in nodes if node.source)),
            "timestamp_range": self._get_timestamp_range(nodes),
        }

        return context

    def _create_batches(self, nodes: list[FactNode]) -> list[list[FactNode]]:
        """Create batches from nodes."""
        batches = []
        for i in range(0, len(nodes), self._batch_size):
            batch = nodes[i: i + self._batch_size]
            batches.append(batch)
        return batches

    async def _process_batches_parallel(
        self, batches: list[list[FactNode]], all_nodes: list[FactNode]
    ) -> list[VerificationResult]:
        """Process batches in parallel."""
        semaphore = asyncio.Semaphore(self._max_concurrent_batches)

        async def process_batch_with_semaphore(batch):
            async with semaphore:
                return await self._process_single_batch(batch, all_nodes)

        tasks = [process_batch_with_semaphore(batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and flatten results
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed: {str(result)}")
            else:
                valid_results.extend(result)

        return valid_results

    async def _process_batches_sequential(
        self, batches: list[list[FactNode]], all_nodes: list[FactNode]
    ) -> list[VerificationResult]:
        """Process batches sequentially."""
        all_results = []
        for batch in batches:
            batch_results = await self._process_single_batch(batch, all_nodes)
            all_results.extend(batch_results)
        return all_results

    async def _process_single_batch(self, batch: list[FactNode], all_nodes: list[FactNode]) -> list[VerificationResult]:
        """Process a single batch of nodes."""
        results = []

        # Collect evidence for the entire batch
        batch_evidence = await self._collect_batch_evidence(batch)

        # Verify each node in the batch with batch context
        for node in batch:
            try:
                result = await self._verify_node_in_batch(node, batch, batch_evidence, all_nodes)
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Failed to verify node {node.node_id} in batch: {str(e)}")
                error_result = self._create_error_result(
                    None, str(e), node.node_id)
                results.append(error_result)

        return results

    async def _collect_batch_evidence(self, batch: list[FactNode]) -> dict[str, Any]:
        """Collect evidence for a batch of nodes."""
        if not self._use_search or not self._search_service:
            return {"evidence": [], "sources": []}

        # Combine claims for batch search
        batch_claims = [node.claim for node in batch]

        try:
            # This would use the actual search service
            # For now, return placeholder evidence
            evidence = {
                "evidence": [f"Evidence for batch of {len(batch)} claims", "Cross-referenced information available"],
                "sources": ["batch_search_result_1", "batch_search_result_2"],
                "search_queries": batch_claims[: self._max_evidence_sources],
            }

            return evidence

        except Exception as e:
            logger.error(f"Failed to collect batch evidence: {str(e)}")
            return {"evidence": [], "sources": [], "error": str(e)}

    async def _verify_node_in_batch(
        self, node: FactNode, batch: list[FactNode], batch_evidence: dict[str, Any], all_nodes: list[FactNode]
    ) -> VerificationResult:
        """Verify a single node within batch context."""
        # Create verification prompt with batch context
        batch_context = [
            other_node.claim for other_node in batch if other_node.node_id != node.node_id]

        # This would use the actual LLM service
        # For now, create a mock verification result
        confidence_score = 0.7  # Placeholder
        verification_status = "verified" if confidence_score > 0.5 else "disputed"

        reasoning = f"Verified in batch context with {len(batch_context)} related claims. "
        reasoning += f"Evidence sources: {len(batch_evidence.get('sources', []))}"

        result = VerificationResult(
            result_id=str(uuid.uuid4()),
            node_id=node.node_id,
            cluster_id=None,
            verification_status=verification_status,
            confidence_score=confidence_score,
            evidence=batch_evidence.get("evidence", []),
            reasoning=reasoning,
            metadata={
                "batch_size": len(batch),
                "batch_context": batch_context[:3],  # Limit context size
                "verification_method": "batch_processing",
                "evidence_sources": len(batch_evidence.get("sources", [])),
            },
            verified_at=datetime.now(),
        )

        return result

    async def _cross_verify_batch(
        self, batch: list[FactNode], batch_results: list[VerificationResult]
    ) -> list[VerificationResult]:
        """Perform cross-verification within a batch."""
        updated_results = []

        for result in batch_results:
            # Find conflicting results within batch
            conflicts = [
                other_result
                for other_result in batch_results
                if (
                    other_result.result_id != result.result_id
                    and other_result.verification_status != result.verification_status
                )
            ]

            if conflicts:
                # Adjust confidence based on conflicts
                conflict_penalty = min(0.2 * len(conflicts), 0.5)
                adjusted_confidence = max(
                    0.1, result.confidence_score - conflict_penalty)

                # Create updated result
                updated_result = VerificationResult(
                    result_id=result.result_id,
                    node_id=result.node_id,
                    cluster_id=result.cluster_id,
                    verification_status=result.verification_status,
                    confidence_score=adjusted_confidence,
                    evidence=result.evidence,
                    reasoning=result.reasoning +
                    f" (Adjusted for {len(conflicts)} conflicts)",
                    metadata={**result.metadata, "cross_verification": True,
                              "conflicts_detected": len(conflicts)},
                    verified_at=datetime.now(),
                )
                updated_results.append(updated_result)
            else:
                updated_results.append(result)

        return updated_results

    def _combine_batch_results(self, cluster_id: str, batch_results: list[VerificationResult]) -> VerificationResult:
        """Combine results from multiple batches into cluster result."""
        if not batch_results:
            return self._create_error_result(cluster_id, "No batch results to combine")

        # Calculate overall statistics
        total_nodes = len(batch_results)
        verified_count = sum(
            1 for result in batch_results if result.verification_status == "verified")
        disputed_count = sum(
            1 for result in batch_results if result.verification_status == "disputed")

        avg_confidence = sum(
            result.confidence_score for result in batch_results) / total_nodes

        # Determine overall status
        if verified_count > disputed_count:
            overall_status = "verified"
        elif disputed_count > verified_count:
            overall_status = "disputed"
        else:
            overall_status = "inconclusive"

        # Combine evidence
        all_evidence = []
        for result in batch_results:
            all_evidence.extend(result.evidence)

        # Create combined result
        combined_result = VerificationResult(
            result_id=str(uuid.uuid4()),
            node_id=None,
            cluster_id=cluster_id,
            verification_status=overall_status,
            confidence_score=avg_confidence,
            evidence=list(set(all_evidence)),  # Remove duplicates
            reasoning=f"Batch verification of {total_nodes} nodes: "
            f"{verified_count} verified, {disputed_count} disputed",
            metadata={
                "verification_method": "batch_processing",
                "total_nodes": total_nodes,
                "verified_count": verified_count,
                "disputed_count": disputed_count,
                "batch_results": [result.result_id for result in batch_results],
            },
            verified_at=datetime.now(),
        )

        return combined_result

    def _create_error_result(
        self, cluster_id: str | None, error_message: str, node_id: str | None = None
    ) -> VerificationResult:
        """Create an error verification result."""
        return VerificationResult(
            result_id=str(uuid.uuid4()),
            node_id=node_id,
            cluster_id=cluster_id,
            verification_status="error",
            confidence_score=0.0,
            evidence=[],
            reasoning=f"Verification failed: {error_message}",
            metadata={"error": error_message},
            verified_at=datetime.now(),
        )

    def _get_timestamp_range(self, nodes: list[FactNode]) -> dict[str, Any]:
        """Get timestamp range for nodes."""
        timestamps = [node.created_at for node in nodes if node.created_at]

        if not timestamps:
            return {"min": None, "max": None}

        return {
            "min": min(timestamps).isoformat(),
            "max": max(timestamps).isoformat(),
            "span_hours": (max(timestamps) - min(timestamps)).total_seconds() / 3600,
        }

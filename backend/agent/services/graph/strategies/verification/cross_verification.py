"""
Cross-verification strategy implementation.

This module implements cross-verification of fact nodes,
comparing facts against each other to identify conflicts and consensus.
"""

import logging
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any

from agent.models import FactCluster, FactNode, VerificationResult
from agent.services.graph.interfaces import VerificationStrategy

logger = logging.getLogger(__name__)


class CrossVerificationStrategy(VerificationStrategy):
    """
    Cross-verification strategy.

    This strategy compares facts against each other to identify
    conflicts, consensus, and supporting evidence through
    cross-referencing and consistency checking.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize cross-verification strategy.

        Args:
            config: Strategy configuration
        """
        self._config = config or {}
        self._comparison_threshold = self._config.get("comparison_threshold", 0.7)
        self._consensus_threshold = self._config.get("consensus_threshold", 0.6)
        self._max_cross_checks = self._config.get("max_cross_checks", 10)
        self._use_search = self._config.get("use_search", True)

        # Services (would be injected in real implementation)
        self._llm_service = None  # Placeholder for LLM service
        self._search_service = None  # Placeholder for search service
        self._similarity_service = None  # Placeholder for similarity service

        logger.info("CrossVerificationStrategy initialized")

    async def verify_node(self, node: FactNode, context_nodes: list[FactNode] | None = None) -> VerificationResult:
        """
        Verify a single node through cross-verification.

        Args:
            node: Fact node to verify
            context_nodes: Optional context nodes for cross-verification

        Returns:
            Verification result
        """
        if not context_nodes:
            # Without context, perform basic verification
            return await self._verify_node_standalone(node)

        # Perform cross-verification with context
        return await self._cross_verify_single_node(node, context_nodes)

    async def verify_cluster(self, cluster: FactCluster, all_nodes: list[FactNode]) -> VerificationResult:
        """
        Verify a cluster through cross-verification analysis.

        Args:
            cluster: Fact cluster to verify
            all_nodes: All available nodes for context

        Returns:
            Verification result for the cluster
        """
        try:
            # Get nodes in cluster
            cluster_nodes = [node for node in all_nodes if node.node_id in cluster.node_ids]

            if not cluster_nodes:
                return self._create_error_result(cluster.cluster_id, "No nodes found in cluster")

            # Get context nodes (nodes not in cluster)
            context_nodes = [node for node in all_nodes if node.node_id not in cluster.node_ids]

            # Perform cross-verification analysis
            cross_verification_results = await self._perform_cluster_cross_verification(cluster_nodes, context_nodes)

            # Analyze consensus and conflicts
            consensus_analysis = self._analyze_consensus(cross_verification_results)

            # Create cluster verification result
            cluster_result = self._create_cluster_result(
                cluster.cluster_id, cross_verification_results, consensus_analysis
            )

            logger.info(f"Cross-verification completed for cluster {cluster.cluster_id}")
            return cluster_result

        except Exception as e:
            logger.error(f"Cross-verification failed for cluster {cluster.cluster_id}: {str(e)}")
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
            Updated verification results with cross-verification insights
        """
        try:
            # Create node lookup
            node_lookup = {node.node_id: node for node in nodes}

            # Group results by verification status
            status_groups = defaultdict(list)
            for result in verification_results:
                status_groups[result.verification_status].append(result)

            # Perform cross-verification analysis
            updated_results = []

            for result in verification_results:
                if result.node_id not in node_lookup:
                    updated_results.append(result)
                    continue

                node = node_lookup[result.node_id]

                # Find conflicting and supporting results
                conflicts, supports = self._find_conflicts_and_supports(result, verification_results, nodes)

                # Update result with cross-verification insights
                updated_result = await self._update_result_with_cross_verification(
                    result, node, conflicts, supports, nodes
                )

                updated_results.append(updated_result)

            logger.info(f"Cross-verification analysis completed for {len(nodes)} nodes")
            return updated_results

        except Exception as e:
            logger.error(f"Cross-verification analysis failed: {str(e)}")
            return verification_results  # Return original results on error

    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "cross_verification"

    def validate_config(self, config: dict[str, Any]) -> bool:
        """
        Validate strategy configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if configuration is valid
        """
        try:
            comparison_threshold = config.get("comparison_threshold", 0.7)
            if not isinstance(comparison_threshold, int | float) or not 0 <= comparison_threshold <= 1:
                return False

            consensus_threshold = config.get("consensus_threshold", 0.6)
            if not isinstance(consensus_threshold, int | float) or not 0 <= consensus_threshold <= 1:
                return False

            max_cross_checks = config.get("max_cross_checks", 10)
            if not isinstance(max_cross_checks, int) or max_cross_checks <= 0:
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
            self._comparison_threshold = self._config.get("comparison_threshold", 0.7)
            self._consensus_threshold = self._config.get("consensus_threshold", 0.6)
            self._max_cross_checks = self._config.get("max_cross_checks", 10)
            self._use_search = self._config.get("use_search", True)
            logger.info("Cross-verification strategy configuration updated")
        else:
            raise ValueError("Invalid configuration")

    async def prepare_verification_context(self, nodes: list[FactNode]) -> dict[str, Any]:
        """
        Prepare verification context for cross-verification.

        Args:
            nodes: Nodes to prepare context for

        Returns:
            Verification context
        """
        # Analyze node relationships and similarities
        similarity_matrix = await self._compute_similarity_matrix(nodes)

        context = {
            "total_nodes": len(nodes),
            "similarity_matrix": similarity_matrix,
            "potential_conflicts": self._identify_potential_conflicts(nodes, similarity_matrix),
            "consensus_groups": self._identify_consensus_groups(nodes, similarity_matrix),
            "domains": list(set(node.metadata.get("domain", "general") for node in nodes)),
            "sources": list(set(node.source for node in nodes if node.source)),
            "cross_verification_pairs": min(len(nodes) * (len(nodes) - 1) // 2, self._max_cross_checks),
        }

        return context

    async def _verify_node_standalone(self, node: FactNode) -> VerificationResult:
        """Verify a single node without cross-verification context."""
        # Basic verification without cross-referencing
        confidence_score = 0.5  # Neutral confidence without context

        result = VerificationResult(
            result_id=str(uuid.uuid4()),
            node_id=node.node_id,
            cluster_id=None,
            verification_status="inconclusive",
            confidence_score=confidence_score,
            evidence=[],
            reasoning="No context available for cross-verification",
            metadata={"verification_method": "standalone", "cross_verification": False},
            verified_at=datetime.now(),
        )

        return result

    async def _cross_verify_single_node(self, node: FactNode, context_nodes: list[FactNode]) -> VerificationResult:
        """Cross-verify a single node against context nodes."""
        # Find similar and conflicting nodes
        similar_nodes = await self._find_similar_nodes(node, context_nodes)
        conflicting_nodes = await self._find_conflicting_nodes(node, context_nodes)

        # Calculate consensus score
        consensus_score = self._calculate_consensus_score(node, similar_nodes, conflicting_nodes)

        # Determine verification status
        if consensus_score >= self._consensus_threshold:
            status = "verified"
            confidence = min(0.9, 0.5 + consensus_score * 0.4)
        elif consensus_score <= -self._consensus_threshold:
            status = "disputed"
            confidence = min(0.9, 0.5 + abs(consensus_score) * 0.4)
        else:
            status = "inconclusive"
            confidence = 0.5

        # Collect evidence
        evidence = self._collect_cross_verification_evidence(node, similar_nodes, conflicting_nodes)

        reasoning = self._generate_cross_verification_reasoning(node, similar_nodes, conflicting_nodes, consensus_score)

        result = VerificationResult(
            result_id=str(uuid.uuid4()),
            node_id=node.node_id,
            cluster_id=None,
            verification_status=status,
            confidence_score=confidence,
            evidence=evidence,
            reasoning=reasoning,
            metadata={
                "verification_method": "cross_verification",
                "consensus_score": consensus_score,
                "similar_nodes": len(similar_nodes),
                "conflicting_nodes": len(conflicting_nodes),
                "context_nodes": len(context_nodes),
            },
            verified_at=datetime.now(),
        )

        return result

    async def _perform_cluster_cross_verification(
        self, cluster_nodes: list[FactNode], context_nodes: list[FactNode]
    ) -> list[VerificationResult]:
        """Perform cross-verification for all nodes in a cluster."""
        results = []

        for node in cluster_nodes:
            # Create context that includes other cluster nodes and external context
            node_context = [
                other_node for other_node in cluster_nodes if other_node.node_id != node.node_id
            ] + context_nodes

            result = await self._cross_verify_single_node(node, node_context)
            results.append(result)

        return results

    async def _find_similar_nodes(self, target_node: FactNode, candidate_nodes: list[FactNode]) -> list[FactNode]:
        """Find nodes similar to the target node."""
        similar_nodes = []

        for candidate in candidate_nodes:
            similarity = await self._calculate_node_similarity(target_node, candidate)
            if similarity >= self._comparison_threshold:
                similar_nodes.append(candidate)

        return similar_nodes

    async def _find_conflicting_nodes(self, target_node: FactNode, candidate_nodes: list[FactNode]) -> list[FactNode]:
        """Find nodes that conflict with the target node."""
        conflicting_nodes = []

        for candidate in candidate_nodes:
            conflict_score = await self._calculate_conflict_score(target_node, candidate)
            if conflict_score >= self._comparison_threshold:
                conflicting_nodes.append(candidate)

        return conflicting_nodes

    async def _calculate_node_similarity(self, node1: FactNode, node2: FactNode) -> float:
        """Calculate similarity between two nodes."""
        # This would use actual similarity service
        # For now, use simple text-based similarity

        # Check if claims are similar
        claim_similarity = self._simple_text_similarity(node1.claim, node2.claim)

        # Check domain similarity
        domain1 = node1.metadata.get("domain", "general")
        domain2 = node2.metadata.get("domain", "general")
        domain_similarity = 1.0 if domain1 == domain2 else 0.0

        # Check source similarity
        source_similarity = 0.0
        if node1.source and node2.source:
            source_similarity = 1.0 if node1.source == node2.source else 0.0

        # Weighted average
        similarity = claim_similarity * 0.7 + domain_similarity * 0.2 + source_similarity * 0.1

        return similarity

    async def _calculate_conflict_score(self, node1: FactNode, node2: FactNode) -> float:
        """Calculate conflict score between two nodes."""
        # This would use actual conflict detection
        # For now, use simple heuristics

        # Check for explicit contradictions
        claim1_lower = node1.claim.lower()
        claim2_lower = node2.claim.lower()

        # Simple contradiction detection
        contradiction_keywords = [
            ("true", "false"),
            ("yes", "no"),
            ("exists", "does not exist"),
            ("is", "is not"),
            ("will", "will not"),
            ("can", "cannot"),
        ]

        conflict_score = 0.0
        for pos_word, neg_word in contradiction_keywords:
            if (pos_word in claim1_lower and neg_word in claim2_lower) or (
                neg_word in claim1_lower and pos_word in claim2_lower
            ):
                conflict_score = max(conflict_score, 0.8)

        # Check domain conflicts
        domain1 = node1.metadata.get("domain", "general")
        domain2 = node2.metadata.get("domain", "general")
        if domain1 == domain2:
            # Same domain increases potential for conflict
            conflict_score = max(conflict_score, 0.3)

        return conflict_score

    def _calculate_consensus_score(
        self, node: FactNode, similar_nodes: list[FactNode], conflicting_nodes: list[FactNode]
    ) -> float:
        """Calculate consensus score for a node."""
        support_weight = len(similar_nodes)
        conflict_weight = len(conflicting_nodes)

        if support_weight + conflict_weight == 0:
            return 0.0

        # Normalize to [-1, 1] range
        consensus_score = (support_weight - conflict_weight) / (support_weight + conflict_weight)

        return consensus_score

    def _collect_cross_verification_evidence(
        self, node: FactNode, similar_nodes: list[FactNode], conflicting_nodes: list[FactNode]
    ) -> list[str]:
        """Collect evidence from cross-verification."""
        evidence = []

        if similar_nodes:
            evidence.append(f"Supported by {len(similar_nodes)} similar claims")
            for similar_node in similar_nodes[:3]:  # Limit to top 3
                evidence.append(f"Supporting claim: {similar_node.claim[:100]}...")

        if conflicting_nodes:
            evidence.append(f"Conflicts with {len(conflicting_nodes)} opposing claims")
            for conflicting_node in conflicting_nodes[:3]:  # Limit to top 3
                evidence.append(f"Conflicting claim: {conflicting_node.claim[:100]}...")

        return evidence

    def _generate_cross_verification_reasoning(
        self, node: FactNode, similar_nodes: list[FactNode], conflicting_nodes: list[FactNode], consensus_score: float
    ) -> str:
        """Generate reasoning for cross-verification result."""
        reasoning_parts = []

        reasoning_parts.append(f"Cross-verification analysis of claim: '{node.claim[:100]}...'")

        if similar_nodes:
            reasoning_parts.append(f"Found {len(similar_nodes)} supporting claims")

        if conflicting_nodes:
            reasoning_parts.append(f"Found {len(conflicting_nodes)} conflicting claims")

        reasoning_parts.append(f"Consensus score: {consensus_score:.2f}")

        if consensus_score >= self._consensus_threshold:
            reasoning_parts.append("Strong consensus supports this claim")
        elif consensus_score <= -self._consensus_threshold:
            reasoning_parts.append("Strong consensus disputes this claim")
        else:
            reasoning_parts.append("No clear consensus found")

        return ". ".join(reasoning_parts)

    def _find_conflicts_and_supports(
        self, result: VerificationResult, all_results: list[VerificationResult], nodes: list[FactNode]
    ) -> tuple[list[VerificationResult], list[VerificationResult]]:
        """Find conflicting and supporting results."""
        conflicts = []
        supports = []

        for other_result in all_results:
            if other_result.result_id == result.result_id:
                continue

            # Check for status conflicts
            if result.verification_status == "verified" and other_result.verification_status == "disputed":
                conflicts.append(other_result)
            elif result.verification_status == "disputed" and other_result.verification_status == "verified":
                conflicts.append(other_result)
            elif result.verification_status == other_result.verification_status:
                supports.append(other_result)

        return conflicts, supports

    async def _update_result_with_cross_verification(
        self,
        result: VerificationResult,
        node: FactNode,
        conflicts: list[VerificationResult],
        supports: list[VerificationResult],
        all_nodes: list[FactNode],
    ) -> VerificationResult:
        """Update verification result with cross-verification insights."""
        # Calculate cross-verification adjustment
        support_factor = len(supports) * 0.1
        conflict_factor = len(conflicts) * 0.1

        # Adjust confidence
        adjusted_confidence = result.confidence_score
        adjusted_confidence += support_factor
        adjusted_confidence -= conflict_factor
        adjusted_confidence = max(0.1, min(0.9, adjusted_confidence))

        # Update reasoning
        cross_verification_info = []
        if supports:
            cross_verification_info.append(f"{len(supports)} supporting results")
        if conflicts:
            cross_verification_info.append(f"{len(conflicts)} conflicting results")

        updated_reasoning = result.reasoning
        if cross_verification_info:
            updated_reasoning += f" Cross-verification: {', '.join(cross_verification_info)}"

        # Create updated result
        updated_result = VerificationResult(
            result_id=result.result_id,
            node_id=result.node_id,
            cluster_id=result.cluster_id,
            verification_status=result.verification_status,
            confidence_score=adjusted_confidence,
            evidence=result.evidence,
            reasoning=updated_reasoning,
            metadata={
                **result.metadata,
                "cross_verification_applied": True,
                "supporting_results": len(supports),
                "conflicting_results": len(conflicts),
                "confidence_adjustment": adjusted_confidence - result.confidence_score,
            },
            verified_at=datetime.now(),
        )

        return updated_result

    async def _compute_similarity_matrix(self, nodes: list[FactNode]) -> dict[str, dict[str, float]]:
        """Compute similarity matrix for nodes."""
        matrix = {}

        for i, node1 in enumerate(nodes):
            matrix[node1.node_id] = {}
            for j, node2 in enumerate(nodes):
                if i == j:
                    matrix[node1.node_id][node2.node_id] = 1.0
                else:
                    similarity = await self._calculate_node_similarity(node1, node2)
                    matrix[node1.node_id][node2.node_id] = similarity

        return matrix

    def _identify_potential_conflicts(
        self, nodes: list[FactNode], similarity_matrix: dict[str, dict[str, float]]
    ) -> list[tuple[str, str]]:
        """Identify potential conflicts between nodes."""
        conflicts = []

        for i, node1 in enumerate(nodes):
            for _j, node2 in enumerate(nodes[i + 1 :], i + 1):
                # High similarity might indicate potential conflict
                # if they have different verification outcomes
                similarity = similarity_matrix[node1.node_id][node2.node_id]
                if similarity > 0.7:  # High similarity threshold
                    conflicts.append((node1.node_id, node2.node_id))

        return conflicts[: self._max_cross_checks]

    def _identify_consensus_groups(
        self, nodes: list[FactNode], similarity_matrix: dict[str, dict[str, float]]
    ) -> list[list[str]]:
        """Identify groups of nodes with consensus."""
        groups = []
        visited = set()

        for node in nodes:
            if node.node_id in visited:
                continue

            # Find all nodes similar to this one
            group = [node.node_id]
            visited.add(node.node_id)

            for other_node in nodes:
                if (
                    other_node.node_id not in visited
                    and similarity_matrix[node.node_id][other_node.node_id] >= self._consensus_threshold
                ):
                    group.append(other_node.node_id)
                    visited.add(other_node.node_id)

            if len(group) > 1:
                groups.append(group)

        return groups

    def _analyze_consensus(self, verification_results: list[VerificationResult]) -> dict[str, Any]:
        """Analyze consensus from verification results."""
        status_counts = defaultdict(int)
        confidence_scores = []

        for result in verification_results:
            status_counts[result.verification_status] += 1
            confidence_scores.append(result.confidence_score)

        total_results = len(verification_results)
        avg_confidence = sum(confidence_scores) / total_results if confidence_scores else 0.0

        # Determine consensus
        max_status = max(status_counts.items(), key=lambda x: x[1]) if status_counts else ("inconclusive", 0)
        consensus_strength = max_status[1] / total_results if total_results > 0 else 0.0

        analysis = {
            "total_results": total_results,
            "status_distribution": dict(status_counts),
            "consensus_status": max_status[0],
            "consensus_strength": consensus_strength,
            "average_confidence": avg_confidence,
            "has_strong_consensus": consensus_strength >= self._consensus_threshold,
        }

        return analysis

    def _create_cluster_result(
        self, cluster_id: str, verification_results: list[VerificationResult], consensus_analysis: dict[str, Any]
    ) -> VerificationResult:
        """Create cluster verification result from individual results."""
        # Use consensus analysis to determine cluster result
        cluster_status = consensus_analysis["consensus_status"]
        cluster_confidence = consensus_analysis["average_confidence"]

        # Adjust confidence based on consensus strength
        if consensus_analysis["has_strong_consensus"]:
            cluster_confidence = min(0.9, cluster_confidence * 1.2)
        else:
            cluster_confidence = max(0.1, cluster_confidence * 0.8)

        # Collect all evidence
        all_evidence = []
        for result in verification_results:
            all_evidence.extend(result.evidence)

        reasoning = (
            f"Cross-verification analysis of {consensus_analysis['total_results']} nodes. "
            f"Consensus: {consensus_analysis['consensus_status']} "
            f"(strength: {consensus_analysis['consensus_strength']:.2f})"
        )

        cluster_result = VerificationResult(
            result_id=str(uuid.uuid4()),
            node_id=None,
            cluster_id=cluster_id,
            verification_status=cluster_status,
            confidence_score=cluster_confidence,
            evidence=list(set(all_evidence)),  # Remove duplicates
            reasoning=reasoning,
            metadata={
                "verification_method": "cross_verification",
                "consensus_analysis": consensus_analysis,
                "individual_results": [result.result_id for result in verification_results],
            },
            verified_at=datetime.now(),
        )

        return cluster_result

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
            reasoning=f"Cross-verification failed: {error_message}",
            metadata={"error": error_message},
            verified_at=datetime.now(),
        )

    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation."""
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

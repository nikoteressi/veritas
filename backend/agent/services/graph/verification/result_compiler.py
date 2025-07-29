"""
Result compiler for graph verification results.

Handles compilation and formatting of verification results.
"""

import logging
from datetime import datetime
from typing import Any

from agent.models.graph import FactGraph, VerificationStatus

from ..graph_config import ClusterVerificationResult

logger = logging.getLogger(__name__)


class ResultCompiler:
    """Compiles and formats verification results."""

    def compile_overall_results(
        self,
        successful_results: list[ClusterVerificationResult],
        individual_results: list[dict[str, Any]],
        failed_clusters: list[str],
        verification_time: float,
    ) -> dict[str, Any]:
        """Compile overall verification results."""
        # Count total facts processed
        total_facts = sum(len(result.individual_results) for result in successful_results)
        total_facts += len(individual_results)

        # Count verdicts
        verdict_counts = {"TRUE": 0, "FALSE": 0, "UNKNOWN": 0, "ERROR": 0}

        for result in successful_results:
            for fact_result in result.individual_results.values():
                verdict = fact_result.get("verdict", "UNKNOWN")
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        for result in individual_results:
            if isinstance(result, dict):
                verdict = result.get("verdict", "UNKNOWN")
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        # Calculate overall confidence
        all_confidences = []
        for result in successful_results:
            all_confidences.append(result.confidence)
            for fact_result in result.individual_results.values():
                confidence = fact_result.get("confidence", 0.0)
                if confidence > 0:
                    all_confidences.append(confidence)

        for result in individual_results:
            if isinstance(result, dict):
                confidence = result.get("confidence", 0.0)
                if confidence > 0:
                    all_confidences.append(confidence)

        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

        # Count contradictions
        total_contradictions = sum(len(result.contradictions_found) for result in successful_results)

        return {
            "verification_summary": {
                "total_facts": total_facts,
                "total_clusters": len(successful_results),
                "failed_clusters": len(failed_clusters),
                "verdict_distribution": verdict_counts,
                "average_confidence": round(avg_confidence, 3),
                "contradictions_found": total_contradictions,
                "verification_time": round(verification_time, 2),
            },
            "cluster_results": [result.to_dict() for result in successful_results],
            "individual_results": individual_results,
            "failed_clusters": failed_clusters,
            "metadata": {
                "verification_timestamp": datetime.now().isoformat(),
                "engine_version": "modular_v1.0",
            },
        }

    async def update_graph_with_results(
        self,
        graph: FactGraph,
        cluster_results: list[ClusterVerificationResult],
        individual_results: list[dict[str, Any]],
    ):
        """Update graph nodes with verification results."""
        # Update nodes from cluster results
        for cluster_result in cluster_results:
            cluster = graph.clusters.get(cluster_result.cluster_id)
            if not cluster:
                continue

            for node in cluster.nodes:
                fact_result = cluster_result.individual_results.get(node.id)
                if fact_result:
                    self._update_node_with_result(node, fact_result)

        # Update individual nodes
        for result in individual_results:
            if isinstance(result, dict) and "node_id" in result:
                node = graph.get_node(result["node_id"])
                if node:
                    self._update_node_with_result(node, result)

    def _update_node_with_result(self, node, result: dict[str, Any]):
        """Update a single node with verification result."""
        verdict = result.get("verdict", "UNKNOWN")
        confidence = result.get("confidence", 0.0)

        # Map verdict to VerificationStatus
        status_mapping = {
            "TRUE": VerificationStatus.VERIFIED,
            "FALSE": VerificationStatus.FAILED,
            "UNKNOWN": VerificationStatus.PENDING,
            "ERROR": VerificationStatus.FAILED,
        }

        node.verification_status = status_mapping.get(verdict, VerificationStatus.PENDING)
        node.confidence_score = confidence

        # Update metadata
        if not node.metadata:
            node.metadata = {}

        node.metadata.update(
            {
                "verification_verdict": verdict,
                "verification_reasoning": result.get("reasoning", ""),
                "evidence_count": len(result.get("evidence_used", [])),
                "last_verified": datetime.now().isoformat(),
            }
        )

    def format_cluster_summary(self, result: ClusterVerificationResult) -> dict[str, Any]:
        """Format a cluster result into a summary."""
        return {
            "cluster_id": result.cluster_id,
            "overall_verdict": result.overall_verdict,
            "confidence": round(result.confidence, 3),
            "facts_count": len(result.individual_results),
            "contradictions": len(result.contradictions_found),
            "verification_time": round(result.verification_time, 2),
            "cluster_type": result.metadata.get("cluster_type", "unknown"),
            "evidence_sources": len(result.supporting_evidence),
        }

    def create_detailed_report(self, overall_results: dict[str, Any], include_evidence: bool = False) -> dict[str, Any]:
        """Create a detailed verification report."""
        summary = overall_results.get("verification_summary", {})

        report = {
            "executive_summary": {
                "total_facts_verified": summary.get("total_facts", 0),
                "overall_confidence": summary.get("average_confidence", 0.0),
                "verification_time": summary.get("verification_time", 0.0),
                "quality_score": self._calculate_quality_score(summary),
            },
            "verdict_breakdown": summary.get("verdict_distribution", {}),
            "cluster_summaries": [],
            "issues_detected": {
                "contradictions": summary.get("contradictions_found", 0),
                "failed_verifications": summary.get("failed_clusters", 0),
                "low_confidence_facts": self._count_low_confidence_facts(overall_results),
            },
        }

        # Add cluster summaries
        for cluster_result in overall_results.get("cluster_results", []):
            if isinstance(cluster_result, dict):
                cluster_summary = {
                    "cluster_id": cluster_result.get("cluster_id"),
                    "verdict": cluster_result.get("overall_verdict"),
                    "confidence": cluster_result.get("confidence"),
                    "facts_count": len(cluster_result.get("individual_results", {})),
                }
                report["cluster_summaries"].append(cluster_summary)

        # Optionally include evidence
        if include_evidence:
            report["evidence_summary"] = self._create_evidence_summary(overall_results)

        return report

    def _calculate_quality_score(self, summary: dict[str, Any]) -> float:
        """Calculate overall quality score for verification."""
        confidence = summary.get("average_confidence", 0.0)
        total_facts = summary.get("total_facts", 1)
        contradictions = summary.get("contradictions_found", 0)
        failed_clusters = summary.get("failed_clusters", 0)

        # Base score from confidence
        quality_score = confidence

        # Penalize contradictions
        if total_facts > 0:
            contradiction_penalty = min(0.3, contradictions / total_facts)
            quality_score -= contradiction_penalty

        # Penalize failed verifications
        failure_penalty = min(0.2, failed_clusters / max(1, total_facts / 5))
        quality_score -= failure_penalty

        return max(0.0, min(1.0, quality_score))

    def _count_low_confidence_facts(self, overall_results: dict[str, Any]) -> int:
        """Count facts with low confidence scores."""
        low_confidence_count = 0
        threshold = 0.3

        # Check cluster results
        for cluster_result in overall_results.get("cluster_results", []):
            if isinstance(cluster_result, dict):
                for fact_result in cluster_result.get("individual_results", {}).values():
                    if fact_result.get("confidence", 0.0) < threshold:
                        low_confidence_count += 1

        # Check individual results
        for result in overall_results.get("individual_results", []):
            if isinstance(result, dict) and result.get("confidence", 0.0) < threshold:
                low_confidence_count += 1

        return low_confidence_count

    def _create_evidence_summary(self, overall_results: dict[str, Any]) -> dict[str, Any]:
        """Create summary of evidence used in verification."""
        all_sources = set()
        evidence_count = 0

        for cluster_result in overall_results.get("cluster_results", []):
            if isinstance(cluster_result, dict):
                for evidence in cluster_result.get("supporting_evidence", []):
                    if isinstance(evidence, dict) and "url" in evidence:
                        all_sources.add(evidence["url"])
                        evidence_count += 1

        return {
            "unique_sources": len(all_sources),
            "total_evidence_pieces": evidence_count,
            "sources_list": list(all_sources)[:10],  # Limit to first 10
        }

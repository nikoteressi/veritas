"""Result compilation service for graph-based fact checking.

This service handles conversion of graph verification results to standard formats
and compilation of fact check summaries.
"""

import logging
from typing import Any

from agent.models import ClaimResult, FactCheckResult, FactCheckSummary
from agent.services.graph.uncertainty_analyzer import UncertaintyAnalyzer

logger = logging.getLogger(__name__)


class VerificationResultCompiler:
    """Compiles graph verification results into standard formats.

    Handles conversion of graph verification results to FactCheckResult,
    ClaimResult, and FactCheckSummary formats.
    """

    def __init__(self, uncertainty_analyzer: UncertaintyAnalyzer):
        """Initialize the result compiler.

        Args:
            uncertainty_analyzer: Uncertainty analyzer for confidence calculations
        """
        self.uncertainty_analyzer = uncertainty_analyzer
        self.logger = logging.getLogger(__name__)

    async def convert_to_fact_check_result(
        self,
        verification_results: dict[str, Any],
        graph_nodes: dict[str, Any],
        claim: str,
        sources: list[str] | None = None,
        queries: list[str] | None = None,
    ) -> FactCheckResult:
        """Convert graph verification results to FactCheckResult format.

        Args:
            verification_results: Results from graph verification
            graph_nodes: Graph nodes with metadata
            claim: Original claim being verified
            sources: List of sources used
            queries: List of queries used

        Returns:
            FactCheckResult: Standardized fact check result
        """
        try:
            # Extract cluster results
            cluster_results = verification_results.get("cluster_results", [])
            individual_results = verification_results.get("individual_results", [])

            # Collect all evidence and confidence scores
            all_evidence = []
            confidence_scores = []

            # Process cluster results
            for cluster_result in cluster_results:
                _cluster_verdict = cluster_result.get("verdict", "INSUFFICIENT_EVIDENCE")
                cluster_confidence = cluster_result.get("confidence", 0.0)
                cluster_evidence = cluster_result.get("evidence", [])

                all_evidence.extend(cluster_evidence)
                confidence_scores.append(cluster_confidence)

                # Process individual results within cluster
                for individual_result in cluster_result.get("individual_results", {}).values():
                    individual_confidence = individual_result.get("confidence", 0.0)
                    individual_evidence = individual_result.get("evidence", [])

                    all_evidence.extend(individual_evidence)
                    confidence_scores.append(individual_confidence)

            # Process standalone individual results
            for individual_result in individual_results:
                individual_confidence = individual_result.get("confidence", 0.0)
                individual_evidence = individual_result.get("evidence", [])

                all_evidence.extend(individual_evidence)
                confidence_scores.append(individual_confidence)

            # Calculate overall confidence
            overall_confidence = (
                await self.uncertainty_analyzer.calculate_confidence_score(confidence_scores)
                if confidence_scores
                else 0.0
            )

            # Determine overall verdict based on confidence and evidence
            if overall_confidence >= 0.8:
                verdict = "TRUE"
            elif overall_confidence >= 0.6:
                verdict = "PARTIALLY_TRUE"
            elif overall_confidence >= 0.4:
                verdict = "MIXED"
            elif overall_confidence >= 0.2:
                verdict = "PARTIALLY_FALSE"
            elif overall_confidence > 0.0:
                verdict = "FALSE"
            else:
                verdict = "INSUFFICIENT_EVIDENCE"

            # Collect sources from graph nodes and parameters
            all_sources = set()
            if sources:
                all_sources.update(sources)

            for node in graph_nodes.values():
                node_sources = node.get("sources", [])
                if isinstance(node_sources, list):
                    all_sources.update(node_sources)
                elif isinstance(node_sources, str):
                    all_sources.add(node_sources)

            # Collect queries
            all_queries = queries or []

            # Create FactCheckResult
            result = FactCheckResult(
                claim=claim,
                verdict=verdict,
                confidence=overall_confidence,
                evidence=all_evidence,
                sources=list(all_sources),
                queries=all_queries,
                metadata={
                    "graph_verification": True,
                    "cluster_count": len(cluster_results),
                    "individual_count": len(individual_results),
                    "node_count": len(graph_nodes),
                    "uncertainty_level": await self.uncertainty_analyzer.assess_uncertainty_level(verification_results),
                },
            )

            self.logger.info(
                "Converted graph verification to FactCheckResult: verdict=%s, confidence=%.3f",
                verdict,
                overall_confidence,
            )

            return result

        except Exception as e:
            self.logger.error("Error converting to FactCheckResult: %s", e)
            # Return minimal result on error
            return FactCheckResult(
                claim=claim,
                verdict="INSUFFICIENT_EVIDENCE",
                confidence=0.0,
                evidence=[],
                sources=sources or [],
                queries=queries or [],
                metadata={"error": str(e)},
            )

    async def convert_to_claim_result(
        self,
        verification_results: dict[str, Any],
        graph_nodes: dict[str, Any],
        claim: str,
    ) -> ClaimResult:
        """Convert graph verification results to ClaimResult format.

        Args:
            verification_results: Results from graph verification
            graph_nodes: Graph nodes with metadata
            claim: Original claim being verified

        Returns:
            ClaimResult: Standardized claim result
        """
        try:
            # First convert to FactCheckResult
            fact_check_result = await self.convert_to_fact_check_result(verification_results, graph_nodes, claim)

            # Map verdict to ClaimResult format
            verdict_mapping = {
                "TRUE": "supported",
                "PARTIALLY_TRUE": "partially_supported",
                "MIXED": "mixed",
                "PARTIALLY_FALSE": "partially_refuted",
                "FALSE": "refuted",
                "INSUFFICIENT_EVIDENCE": "insufficient_evidence",
            }

            verdict = verdict_mapping.get(fact_check_result.verdict, "insufficient_evidence")

            # Extract supporting and contradicting evidence
            supporting_evidence = []
            contradicting_evidence = []

            for evidence in fact_check_result.evidence:
                # Simple heuristic: evidence with high confidence is supporting,
                # evidence with low confidence might be contradicting
                evidence_confidence = evidence.get("confidence", 0.5) if isinstance(evidence, dict) else 0.5

                if evidence_confidence >= 0.6:
                    supporting_evidence.append(evidence)
                elif evidence_confidence <= 0.4:
                    contradicting_evidence.append(evidence)
                else:
                    supporting_evidence.append(evidence)  # Default to supporting

            # Create ClaimResult
            result = ClaimResult(
                claim=claim,
                verdict=verdict,
                confidence=fact_check_result.confidence,
                supporting_evidence=supporting_evidence,
                contradicting_evidence=contradicting_evidence,
                sources=fact_check_result.sources,
                metadata=fact_check_result.metadata,
            )

            self.logger.info(
                "Converted graph verification to ClaimResult: verdict=%s, confidence=%.3f",
                verdict,
                fact_check_result.confidence,
            )

            return result

        except Exception as e:
            self.logger.error("Error converting to ClaimResult: %s", e)
            # Return minimal result on error
            return ClaimResult(
                claim=claim,
                verdict="insufficient_evidence",
                confidence=0.0,
                supporting_evidence=[],
                contradicting_evidence=[],
                sources=[],
                metadata={"error": str(e)},
            )

    async def compile_fact_check_summary(
        self,
        verification_results: dict[str, Any],
        graph_nodes: dict[str, Any],
        claims: list[str],
    ) -> FactCheckSummary:
        """Compile a FactCheckSummary from graph verification results.

        Args:
            verification_results: Results from graph verification
            graph_nodes: Graph nodes with metadata
            claims: List of claims that were verified

        Returns:
            FactCheckSummary: Compiled summary of fact checking results
        """
        try:
            # Convert each claim to ClaimResult
            claim_results = []
            for claim in claims:
                claim_result = await self.convert_to_claim_result(verification_results, graph_nodes, claim)
                claim_results.append(claim_result)

            # Calculate overall statistics
            total_claims = len(claim_results)
            if total_claims == 0:
                overall_confidence = 0.0
                overall_verdict = "insufficient_evidence"
            else:
                # Calculate average confidence
                overall_confidence = sum(result.confidence for result in claim_results) / total_claims

                # Determine overall verdict based on individual verdicts
                verdict_counts = {}
                for result in claim_results:
                    verdict = result.verdict
                    verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

                # Most common verdict becomes overall verdict
                overall_verdict = max(verdict_counts, key=verdict_counts.get)

            # Collect all sources
            all_sources = set()
            for result in claim_results:
                all_sources.update(result.sources)

            # Create summary metadata
            summary_metadata = {
                "graph_verification": True,
                "total_claims": total_claims,
                "node_count": len(graph_nodes),
                "cluster_count": len(verification_results.get("cluster_results", [])),
                "individual_count": len(verification_results.get("individual_results", [])),
                "uncertainty_level": await self.uncertainty_analyzer.assess_uncertainty_level(verification_results),
                "verdict_distribution": {
                    verdict: count
                    for verdict, count in {
                        result.verdict: sum(1 for r in claim_results if r.verdict == result.verdict)
                        for result in claim_results
                    }.items()
                }
                if claim_results
                else {},
            }

            # Create FactCheckSummary
            summary = FactCheckSummary(
                claims=claim_results,
                overall_verdict=overall_verdict,
                overall_confidence=overall_confidence,
                sources=list(all_sources),
                metadata=summary_metadata,
            )

            self.logger.info(
                "Compiled FactCheckSummary: %d claims, overall_verdict=%s, confidence=%.3f",
                total_claims,
                overall_verdict,
                overall_confidence,
            )

            return summary

        except Exception as e:
            self.logger.error("Error compiling FactCheckSummary: %s", e)
            # Return minimal summary on error
            return FactCheckSummary(
                claims=[],
                overall_verdict="insufficient_evidence",
                overall_confidence=0.0,
                sources=[],
                metadata={"error": str(e)},
            )

    async def add_extended_metadata(
        self,
        result: FactCheckResult,
        verification_results: dict[str, Any],
        graph_nodes: dict[str, Any],
    ) -> FactCheckResult:
        """Add extended metadata to a FactCheckResult.

        Args:
            result: Original FactCheckResult
            verification_results: Graph verification results
            graph_nodes: Graph nodes with metadata

        Returns:
            FactCheckResult: Result with extended metadata
        """
        try:
            # Calculate additional metrics
            node_confidences = [node.get("confidence", 0.0) for node in graph_nodes.values()]

            extended_metadata = {
                **result.metadata,
                "graph_metrics": {
                    "avg_node_confidence": (sum(node_confidences) / len(node_confidences) if node_confidences else 0.0),
                    "min_node_confidence": min(node_confidences) if node_confidences else 0.0,
                    "max_node_confidence": max(node_confidences) if node_confidences else 0.0,
                    "confidence_variance": (
                        sum((c - sum(node_confidences) / len(node_confidences)) ** 2 for c in node_confidences)
                        / len(node_confidences)
                        if len(node_confidences) > 1
                        else 0.0
                    ),
                },
                "verification_details": {
                    "cluster_verdicts": [
                        cluster.get("verdict", "UNKNOWN") for cluster in verification_results.get("cluster_results", [])
                    ],
                    "individual_verdicts": [
                        individual.get("verdict", "UNKNOWN")
                        for individual in verification_results.get("individual_results", [])
                    ],
                },
                "uncertainty_analysis": await self.uncertainty_analyzer.analyze_verification_uncertainty(
                    verification_results, graph_nodes
                ),
            }

            # Create new result with extended metadata
            extended_result = FactCheckResult(
                claim=result.claim,
                verdict=result.verdict,
                confidence=result.confidence,
                evidence=result.evidence,
                sources=result.sources,
                queries=result.queries,
                metadata=extended_metadata,
            )

            self.logger.debug("Added extended metadata to FactCheckResult")
            return extended_result

        except Exception as e:
            self.logger.warning("Error adding extended metadata: %s", e)
            return result  # Return original result on error

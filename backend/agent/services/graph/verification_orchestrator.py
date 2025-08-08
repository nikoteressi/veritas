"""Fact verification orchestrator for graph-based fact checking.

This service coordinates the entire graph-based fact verification process,
managing dependencies and orchestrating the verification workflow.
"""

import logging
from typing import Any

from agent.models import ClaimResult, FactCheckResult, FactCheckSummary
from agent.services.graph.cache_manager import GraphCacheManager
from agent.services.graph.graph_builder import GraphBuilder
from agent.services.graph.graph_verifier import GraphVerifier
from agent.services.graph.result_compiler import VerificationResultCompiler
from agent.services.graph.uncertainty_analyzer import UncertaintyAnalyzer

logger = logging.getLogger(__name__)


class FactVerificationOrchestrator:
    """Orchestrates graph-based fact verification process.

    Coordinates graph building, verification, result compilation,
    and caching for fact checking operations.
    """

    def __init__(
        self,
        graph_builder: GraphBuilder,
        graph_verifier: GraphVerifier,
        cache_manager: GraphCacheManager | None = None,
        uncertainty_analyzer: UncertaintyAnalyzer | None = None,
        result_compiler: VerificationResultCompiler | None = None,
    ):
        """Initialize the verification orchestrator.

        Args:
            graph_builder: Service for building knowledge graphs
            graph_verifier: Service for verifying facts using graphs
            cache_manager: Optional cache manager for results
            uncertainty_analyzer: Optional uncertainty analyzer
            result_compiler: Optional result compiler
        """
        self.graph_builder = graph_builder
        self.graph_verifier = graph_verifier
        self.cache_manager = cache_manager or GraphCacheManager()
        self.uncertainty_analyzer = uncertainty_analyzer or UncertaintyAnalyzer()
        self.result_compiler = result_compiler or VerificationResultCompiler(
            self.uncertainty_analyzer)
        self.logger = logging.getLogger(__name__)

    async def verify_claim(
        self,
        claim: str,
        evidence_texts: list[str],
        sources: list[str] | None = None,
        queries: list[str] | None = None,
        use_cache: bool = True,
        progress_callback: callable | None = None,
    ) -> FactCheckResult:
        """Verify a single claim using graph-based fact checking.

        Args:
            claim: The claim to verify
            evidence_texts: List of evidence texts
            sources: Optional list of sources
            queries: Optional list of queries used
            use_cache: Whether to use caching
            progress_callback: Optional progress callback function

        Returns:
            FactCheckResult: Verification result
        """
        try:
            self.logger.info(
                "Starting graph-based verification for claim: %s", claim[:100])

            # Check cache first
            if use_cache:
                cached_result = await self.cache_manager.get_cached_result(claim, evidence_texts)
                if cached_result:
                    self.logger.info("Retrieved cached result for claim")
                    return cached_result

            # Report progress
            if progress_callback:
                progress_callback("Building knowledge graph...")

            # Build knowledge graph
            self.logger.info("Building knowledge graph from evidence")
            graph_nodes, graph_edges = await self.graph_builder.build_graph(evidence_texts, claim)

            if progress_callback:
                progress_callback("Verifying claim against graph...")

            # Verify claim using graph
            self.logger.info("Verifying claim using knowledge graph")
            verification_results = await self.graph_verifier.verify_claim(claim, graph_nodes, graph_edges)

            if progress_callback:
                progress_callback("Compiling verification results...")

            # Compile results
            self.logger.info("Compiling verification results")
            fact_check_result = await self.result_compiler.convert_to_fact_check_result(
                verification_results, graph_nodes, claim, sources, queries
            )

            # Add extended metadata
            fact_check_result = await self.result_compiler.add_extended_metadata(
                fact_check_result, verification_results, graph_nodes
            )

            # Cache result
            if use_cache:
                await self.cache_manager.cache_result(claim, evidence_texts, fact_check_result)

            if progress_callback:
                progress_callback("Verification complete")

            self.logger.info(
                "Graph-based verification completed: verdict=%s, confidence=%.3f",
                fact_check_result.verdict,
                fact_check_result.confidence,
            )

            return fact_check_result

        except Exception as e:
            self.logger.error("Error in graph-based claim verification: %s", e)
            # Return error result
            return FactCheckResult(
                claim=claim,
                verdict="INSUFFICIENT_EVIDENCE",
                confidence=0.0,
                evidence=[],
                sources=sources or [],
                queries=queries or [],
                metadata={"error": str(e), "verification_method": "graph"},
            )

    async def verify_multiple_claims(
        self,
        claims: list[str],
        evidence_texts: list[str],
        sources: list[str] | None = None,
        queries: list[str] | None = None,
        use_cache: bool = True,
        progress_callback: callable | None = None,
    ) -> FactCheckSummary:
        """Verify multiple claims using graph-based fact checking.

        Args:
            claims: List of claims to verify
            evidence_texts: List of evidence texts
            sources: Optional list of sources
            queries: Optional list of queries used
            use_cache: Whether to use caching
            progress_callback: Optional progress callback function

        Returns:
            FactCheckSummary: Summary of verification results
        """
        try:
            self.logger.info(
                "Starting graph-based verification for %d claims", len(claims))

            if progress_callback:
                progress_callback("Building shared knowledge graph...")

            # Build shared knowledge graph for all claims
            self.logger.info("Building shared knowledge graph from evidence")
            graph_nodes, graph_edges = await self.graph_builder.build_graph(evidence_texts, " ".join(claims))

            # Verify each claim
            claim_results = []
            for i, claim in enumerate(claims):
                try:
                    if progress_callback:
                        progress_callback(
                            f"Verifying claim {i + 1}/{len(claims)}...")

                    # Check cache first
                    cached_result = None
                    if use_cache:
                        cached_result = await self.cache_manager.get_cached_result(claim, evidence_texts)

                    if cached_result:
                        # Convert cached FactCheckResult to ClaimResult
                        claim_result = await self.result_compiler.convert_to_claim_result(
                            {
                                "individual_results": [
                                    {"verdict": cached_result.verdict,
                                        "confidence": cached_result.confidence}
                                ]
                            },
                            graph_nodes,
                            claim,
                        )
                    else:
                        # Verify claim using shared graph
                        verification_results = await self.graph_verifier.verify_claim(claim, graph_nodes, graph_edges)

                        # Convert to ClaimResult
                        claim_result = await self.result_compiler.convert_to_claim_result(
                            verification_results, graph_nodes, claim
                        )

                        # Cache the result as FactCheckResult
                        if use_cache:
                            fact_check_result = await self.result_compiler.convert_to_fact_check_result(
                                verification_results, graph_nodes, claim, sources, queries
                            )
                            await self.cache_manager.cache_result(claim, evidence_texts, fact_check_result)

                    claim_results.append(claim_result)

                except Exception as e:
                    self.logger.error(
                        "Error verifying claim '%s': %s", claim, e)
                    # Add error result for this claim
                    error_result = ClaimResult(
                        claim=claim,
                        verdict="insufficient_evidence",
                        confidence=0.0,
                        supporting_evidence=[],
                        contradicting_evidence=[],
                        sources=sources or [],
                        metadata={"error": str(e)},
                    )
                    claim_results.append(error_result)

            if progress_callback:
                progress_callback("Compiling summary...")

            # Compile summary using shared graph data
            verification_results = {
                "cluster_results": [],
                "individual_results": [
                    {
                        "verdict": result.verdict,
                        "confidence": result.confidence,
                        "evidence": result.supporting_evidence + result.contradicting_evidence,
                    }
                    for result in claim_results
                ],
            }

            summary = await self.result_compiler.compile_fact_check_summary(verification_results, graph_nodes, claims)

            if progress_callback:
                progress_callback("Verification complete")

            self.logger.info(
                "Graph-based multi-claim verification completed: %d claims, overall_verdict=%s",
                len(claims),
                summary.overall_verdict,
            )

            return summary

        except Exception as e:
            self.logger.error(
                "Error in graph-based multi-claim verification: %s", e)
            # Return error summary
            return FactCheckSummary(
                claims=[],
                overall_verdict="insufficient_evidence",
                overall_confidence=0.0,
                sources=sources or [],
                metadata={"error": str(e), "verification_method": "graph"},
            )

    async def verify_with_custom_graph(
        self,
        claim: str,
        graph_nodes: dict[str, Any],
        graph_edges: list[dict[str, Any]],
        sources: list[str] | None = None,
        queries: list[str] | None = None,
    ) -> FactCheckResult:
        """Verify a claim using a pre-built custom graph.

        Args:
            claim: The claim to verify
            graph_nodes: Pre-built graph nodes
            graph_edges: Pre-built graph edges
            sources: Optional list of sources
            queries: Optional list of queries used

        Returns:
            FactCheckResult: Verification result
        """
        try:
            self.logger.info(
                "Verifying claim with custom graph: %s", claim[:100])

            # Verify claim using provided graph
            verification_results = await self.graph_verifier.verify_claim(claim, graph_nodes, graph_edges)

            # Compile results
            fact_check_result = await self.result_compiler.convert_to_fact_check_result(
                verification_results, graph_nodes, claim, sources, queries
            )

            # Add extended metadata
            fact_check_result = await self.result_compiler.add_extended_metadata(
                fact_check_result, verification_results, graph_nodes
            )

            self.logger.info(
                "Custom graph verification completed: verdict=%s, confidence=%.3f",
                fact_check_result.verdict,
                fact_check_result.confidence,
            )

            return fact_check_result

        except Exception as e:
            self.logger.error("Error in custom graph verification: %s", e)
            return FactCheckResult(
                claim=claim,
                verdict="INSUFFICIENT_EVIDENCE",
                confidence=0.0,
                evidence=[],
                sources=sources or [],
                queries=queries or [],
                metadata={"error": str(
                    e), "verification_method": "custom_graph"},
            )

    async def invalidate_cache(self, claim: str, evidence_texts: list[str]) -> None:
        """Invalidate cached results for a specific claim and evidence.

        Args:
            claim: The claim to invalidate
            evidence_texts: Evidence texts to invalidate
        """
        await self.cache_manager.invalidate_cache(claim, evidence_texts)
        self.logger.info("Invalidated cache for claim: %s", claim[:100])

    async def clear_all_cache(self) -> None:
        """Clear all cached results."""
        await self.cache_manager.clear_cache()
        self.logger.info("Cleared all cached results")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict[str, Any]: Cache statistics
        """
        return self.cache_manager.get_cache_stats()

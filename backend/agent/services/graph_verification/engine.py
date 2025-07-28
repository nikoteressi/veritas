"""
Enhanced modular graph verification engine.

Main orchestrator that coordinates all verification modules with intelligent caching,
adaptive thresholds, and performance monitoring.
"""

import asyncio
import gc
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from agent.models.graph import FactCluster, FactGraph, FactNode
from agent.models.verification_context import VerificationContext
from agent.ollama_embeddings import OllamaEmbeddingFunction
from agent.prompt_manager import PromptManager
from agent.services.graph_config import ClusterVerificationResult, VerificationConfig
from app.config import settings

from .cluster_analyzer import ClusterAnalyzer
from .evidence_gatherer import EnhancedEvidenceGatherer
from .response_parser import ResponseParser
from .result_compiler import ResultCompiler
from .source_manager import EnhancedSourceManager
from .utils import CacheManager
from .verification_processor import EnhancedVerificationProcessor

if TYPE_CHECKING:
    from agent.tools import SearxNGSearchTool

logger = logging.getLogger(__name__)


class EnhancedGraphVerificationEngine:
    """
    Enhanced modular engine for verifying fact clusters using graph-based approaches.

    This is the main orchestrator that coordinates all verification modules
    to process clusters of related facts together, leveraging their relationships
    for more efficient and accurate verification with intelligent caching,
    adaptive thresholds, and performance monitoring.
    """

    def __init__(self, search_tool: "SearxNGSearchTool", config: VerificationConfig):
        self._search_tool = search_tool
        self.config = config or VerificationConfig()

        # Initialize embeddings and prompt manager
        self.embeddings = OllamaEmbeddingFunction(
            ollama_url=settings.ollama_base_url,
            model_name=settings.embedding_model_name,
        )
        self.prompt_manager = PromptManager()
        self.logger = logging.getLogger(__name__)

        # Initialize enhanced verification modules
        self.evidence_gatherer = EnhancedEvidenceGatherer(search_tool, self.config)
        self.source_manager = EnhancedSourceManager(
            max_concurrent_scrapes=self.config.max_concurrent_scrapes
        )
        self.verification_processor = EnhancedVerificationProcessor()
        self.cluster_analyzer = ClusterAnalyzer()
        self.result_compiler = ResultCompiler()
        self.response_parser = ResponseParser()

        # Initialize cache manager
        self.cache_manager = CacheManager(
            max_size=(
                self.config.cache_size if hasattr(self.config, "cache_size") else 1000
            ),
            ttl_seconds=3600,
        )

        # Performance tracking
        self.performance_metrics = {
            "total_verifications": 0,
            "total_verification_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "error_count": 0,
        }

        self.logger.info("GraphVerificationEngine initialized successfully")

        # Memory optimization: trigger garbage collection after initialization
        gc.collect()
        self.logger.debug(
            "Memory cleanup completed after GraphVerificationEngine initialization"
        )

    async def verify_graph(
        self, graph: FactGraph, context: VerificationContext
    ) -> dict[str, Any]:
        """
        Verify all clusters in a fact graph with enhanced performance monitoring.

        Args:
            graph: The fact graph to verify
            context: Verification context with additional data

        Returns:
            Dict containing overall verification results with performance metrics
        """
        start_time = datetime.now()
        self.logger.info(
            "Starting enhanced graph verification with %d clusters", len(graph.clusters)
        )

        # Update performance metrics
        self.performance_metrics["total_verifications"] += 1

        # Verify clusters in parallel with concurrency limit
        semaphore = asyncio.Semaphore(self.config.max_concurrent_verifications)

        async def verify_cluster_with_semaphore(
            cluster: FactCluster,
        ) -> ClusterVerificationResult:
            async with semaphore:
                return await self.verify_cluster(cluster, graph, context)

        cluster_tasks = [
            verify_cluster_with_semaphore(cluster)
            for cluster in graph.clusters.values()
        ]

        cluster_results = await asyncio.gather(*cluster_tasks, return_exceptions=True)

        # Process results and handle exceptions
        successful_results = []
        failed_clusters = []

        for i, result in enumerate(cluster_results):
            if isinstance(result, Exception):
                cluster_id = list(graph.clusters.keys())[i]
                self.logger.error("Failed to verify cluster %s: %s", cluster_id, result)
                failed_clusters.append(cluster_id)
                self.performance_metrics["error_count"] += 1
            else:
                successful_results.append(result)

        # Verify individual nodes not in clusters
        individual_nodes = self._get_unclustered_nodes(graph)
        individual_results = []

        if individual_nodes:
            individual_tasks = [
                self.verification_processor.verify_individual_node(node, context)
                for node in individual_nodes
            ]
            individual_results = await asyncio.gather(
                *individual_tasks, return_exceptions=True
            )
            # Filter out exceptions and count errors
            filtered_results = []
            for result in individual_results:
                if isinstance(result, Exception):
                    self.performance_metrics["error_count"] += 1
                else:
                    filtered_results.append(result)
            individual_results = filtered_results

        # Compile overall results
        verification_time = (datetime.now() - start_time).total_seconds()
        self.performance_metrics["total_verification_time"] += verification_time

        overall_result = self.result_compiler.compile_overall_results(
            successful_results, individual_results, failed_clusters, verification_time
        )

        # Add performance metrics to result
        overall_result["performance_metrics"] = await self.get_performance_metrics()

        # Update graph with verification results
        await self.result_compiler.update_graph_with_results(
            graph, successful_results, individual_results
        )

        self.logger.info(
            "Enhanced graph verification completed in %.2fs", verification_time
        )
        return overall_result

    async def verify_cluster(
        self, cluster: FactCluster, graph: FactGraph, context: VerificationContext
    ) -> ClusterVerificationResult:
        """
        Verify a single cluster of facts using modular approach.

        Args:
            cluster: The cluster to verify
            graph: The complete graph for context
            context: Verification context

        Returns:
            ClusterVerificationResult with detailed results
        """
        start_time = datetime.now()
        self.logger.info(
            "Verifying cluster %s with %d facts using modular approach",
            cluster.id,
            len(cluster.nodes),
        )

        try:
            # Step 1: Gather evidence for the cluster
            evidence = await self.evidence_gatherer.gather_cluster_evidence(
                cluster, context
            )

            # Step 1.5: Extract and scrape credible sources from evidence
            credible_urls = set()
            for evidence_item in evidence:
                if "credible_sources" in evidence_item:
                    credible_urls.update(evidence_item["credible_sources"])

            credible_urls_list = list(credible_urls)

            if credible_urls_list:
                self.logger.info(
                    "Scraping %d credible sources for cluster %s",
                    len(credible_urls_list),
                    cluster.id,
                )

                # Scrape the credible sources
                scraped_content = await self.source_manager.scrape_sources_batch(
                    credible_urls_list
                )

                # Log summary instead of full content to prevent BlockingIOError
                content_summary = {
                    url: (
                        f"Content length: {len(content)} chars, Preview: {content[:100]}..."
                        if len(content) > 100
                        else content
                    )
                    for url, content in scraped_content.items()
                }
                self.logger.info("Scraped content summary: %s", content_summary)

                self.logger.info(
                    "Successfully scraped %d sources for cluster %s",
                    len(scraped_content),
                    cluster.id,
                )

                # Memory optimization: trigger garbage collection after scraping
                gc.collect()
                self.logger.debug("Memory cleanup completed after scraping")

                # Filter scraped content to create enriched evidence
                cluster_queries = []
                for evidence_item in evidence:
                    if "query" in evidence_item:
                        cluster_queries.append(evidence_item["query"])

                enriched_evidence = (
                    await self.source_manager.filter_evidence_for_cluster(
                        cluster_queries, scraped_content
                    )
                )

                # Log summary of enriched evidence to prevent BlockingIOError
                evidence_summary = f"Enriched evidence: {len(enriched_evidence)} items"
                if enriched_evidence:
                    first_keys = (
                        list(enriched_evidence[0].keys())
                        if enriched_evidence[0]
                        else "empty"
                    )
                    evidence_summary += f", first item keys: {first_keys}"
                self.logger.info(evidence_summary)

                # Combine original evidence with scraped content
                evidence.extend(enriched_evidence)
            else:
                self.logger.warning(
                    "No credible sources found for cluster %s", cluster.id
                )

            # Step 2: Verify individual facts within cluster context
            individual_results = await self.verification_processor.verify_cluster_facts(
                cluster, evidence, context
            )

            # Log summary of individual results to prevent BlockingIOError
            results_summary = f"Individual results: {len(individual_results)} items"
            if individual_results:
                first_key = next(iter(individual_results))
                result_keys = (
                    list(individual_results[first_key].keys())
                    if isinstance(individual_results[first_key], dict)
                    else "non-dict"
                )
                results_summary += f", sample result keys: {result_keys}"
            self.logger.info(results_summary)

            # Step 3: Cross-verify facts against each other
            cross_verification_results = []
            if self.config.enable_cross_verification:
                cross_verification_results = (
                    await self.cluster_analyzer.cross_verify_cluster_facts(
                        cluster, individual_results, evidence
                    )
                )

            # Log summary of cross-verification results to prevent BlockingIOError
            cross_summary = (
                f"Cross-verification results: {len(cross_verification_results)} items"
            )
            self.logger.info(cross_summary)

            # Step 4: Detect contradictions
            contradictions = []
            if self.config.enable_contradiction_detection:
                contradictions = (
                    await self.cluster_analyzer.detect_cluster_contradictions(
                        cluster, individual_results, graph
                    )
                )

            # Log summary of contradictions to prevent BlockingIOError
            contradictions_summary = f"Contradictions: {len(contradictions)} items"
            self.logger.info(contradictions_summary)

            # Step 5: Compile cluster verdict
            overall_verdict, confidence = self.cluster_analyzer.compile_cluster_verdict(
                individual_results, cross_verification_results, contradictions
            )

            verification_time = (datetime.now() - start_time).total_seconds()

            result = ClusterVerificationResult(
                cluster_id=cluster.id,
                overall_verdict=overall_verdict,
                confidence=confidence,
                individual_results=individual_results,
                cross_verification_results=cross_verification_results,
                contradictions_found=contradictions,
                supporting_evidence=evidence,
                verification_time=verification_time,
                metadata={
                    "cluster_type": cluster.cluster_type.value,
                    "verification_strategy": cluster.verification_strategy,
                    "node_count": len(cluster.nodes),
                    "engine_version": "modular_v1.0",
                },
            )
            # Log summary of cluster verification result to prevent BlockingIOError
            result_summary = (
                f"Cluster {cluster.id} verification result: "
                f"verdict={result.overall_verdict}, "
                f"confidence={result.confidence:.2f}, "
                f"individual_results={len(result.individual_results)}, "
                f"cross_verification={len(result.cross_verification_results)}, "
                f"contradictions={len(result.contradictions_found)}"
            )
            self.logger.info(result_summary)

            self.logger.info(
                "Cluster %s verified: %s (confidence: %.2f) in %.2f s",
                cluster.id,
                overall_verdict,
                confidence,
                verification_time,
            )

            return result

        except (TimeoutError, OSError, ValueError, KeyError, RuntimeError) as e:
            self.logger.error("Failed to verify cluster %s: %s", cluster.id, e)
            # Return error result
            return ClusterVerificationResult(
                cluster_id=cluster.id,
                overall_verdict="ERROR",
                confidence=0.0,
                individual_results={},
                cross_verification_results=[],
                contradictions_found=[],
                supporting_evidence=[],
                verification_time=(datetime.now() - start_time).total_seconds(),
                metadata={"error": str(e), "engine_version": "modular_v1.0"},
            )

    async def verify_clusters(
        self, clusters: list[FactCluster], context: VerificationContext
    ) -> dict[str, dict[str, Any]]:
        """
        Verify multiple clusters using efficient batch approach.

        This method provides backward compatibility with the original interface
        while using the new modular architecture.
        """
        try:
            self.logger.info(
                "Starting modular batch verification for %d clusters", len(clusters)
            )

            # Step 1: Generate search queries for all clusters
            all_search_queries = []
            cluster_queries_map = {}

            for cluster in clusters:
                search_queries = (
                    await self.evidence_gatherer.create_cluster_search_queries(cluster)
                )
                cluster_queries_map[cluster.id] = search_queries
                all_search_queries.extend(search_queries)

            self.logger.info(
                "Generated %s total search queries for all clusters",
                len(all_search_queries),
            )

            # Step 2: Execute batch search
            search_results = await self.evidence_gatherer.execute_searches_batch(
                all_search_queries
            )

            # Step 3: Select credible sources
            all_claims = []
            for cluster in clusters:
                all_claims.extend([node.claim for node in cluster.nodes])

            combined_claim = (
                f"Multiple claims verification: {'; '.join(all_claims[:5])}"
            )
            credible_urls = await self.evidence_gatherer.select_credible_sources_batch(
                combined_claim, search_results
            )

            self.logger.info(
                "Selected %s credible sources for batch scraping", len(credible_urls)
            )

            # Step 4: Scrape sources
            scraped_content = await self.source_manager.scrape_sources_batch(
                credible_urls
            )

            self.logger.info("Successfully scraped %d sources", len(scraped_content))

            # Memory optimization: trigger garbage collection after batch scraping
            gc.collect()
            self.logger.debug("Memory cleanup completed after batch scraping")

            # Step 5: Verify each cluster
            all_results = {}

            for cluster in clusters:
                try:
                    # Filter evidence for this cluster
                    cluster_queries = cluster_queries_map.get(cluster.id, [])
                    cluster_evidence = (
                        await self.source_manager.filter_evidence_for_cluster(
                            cluster_queries, scraped_content
                        )
                    )

                    # Verify cluster facts
                    cluster_results = (
                        await self.verification_processor.verify_cluster_facts(
                            cluster, cluster_evidence, context
                        )
                    )

                    all_results.update(cluster_results)

                    self.logger.info(
                        "Completed verification for cluster %s: %d facts verified",
                        cluster.id,
                        len(cluster_results),
                    )

                except (OSError, ValueError, KeyError, RuntimeError) as e:
                    self.logger.error("Failed to verify cluster %s: %s", cluster.id, e)
                    # Add error results for all nodes in this cluster
                    for node in cluster.nodes:
                        all_results[node.id] = {
                            "node_id": node.id,
                            "claim": node.claim,
                            "verdict": "ERROR",
                            "confidence": 0.0,
                            "reasoning": f"Cluster verification failed: {str(e)}",
                            "evidence_used": [],
                        }

            return all_results

        except (TimeoutError, OSError, ValueError, KeyError, RuntimeError) as e:
            self.logger.error("Modular batch verification failed: %s", e)
            # Return error results for all nodes
            all_results = {}
            for cluster in clusters:
                for node in cluster.nodes:
                    all_results[node.id] = {
                        "node_id": node.id,
                        "claim": node.claim,
                        "verdict": "ERROR",
                        "confidence": 0.0,
                        "reasoning": f"Batch verification failed: {str(e)}",
                        "evidence_used": [],
                    }
            return all_results

    def _get_unclustered_nodes(self, graph: FactGraph) -> list[FactNode]:
        """Get nodes that are not part of any cluster."""
        clustered_node_ids = set()
        for cluster in graph.clusters.values():
            clustered_node_ids.update(node.id for node in cluster.nodes)

        unclustered_nodes = []
        for node in graph.nodes.values():
            if node.id not in clustered_node_ids:
                unclustered_nodes.append(node)

        return unclustered_nodes

    # Legacy methods for backward compatibility
    async def _gather_cluster_evidence(
        self, cluster: FactCluster, context: VerificationContext
    ) -> list[dict[str, Any]]:
        """Legacy method - delegates to evidence gatherer."""
        return await self.evidence_gatherer.gather_cluster_evidence(cluster, context)

    async def _verify_cluster_facts(
        self,
        cluster: FactCluster,
        evidence: list[dict[str, Any]],
        context: VerificationContext,
    ) -> dict[str, dict[str, Any]]:
        """Legacy method - delegates to verification processor."""
        return await self.verification_processor.verify_cluster_facts(
            cluster, evidence, context
        )

    async def _cross_verify_cluster_facts(
        self,
        cluster: FactCluster,
        individual_results: dict[str, dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Legacy method - delegates to cluster analyzer."""
        return await self.cluster_analyzer.cross_verify_cluster_facts(
            cluster, individual_results, evidence
        )

    async def _detect_cluster_contradictions(
        self,
        cluster: FactCluster,
        individual_results: dict[str, dict[str, Any]],
        graph: FactGraph,
    ) -> list[dict[str, Any]]:
        """Legacy method - delegates to cluster analyzer."""
        return await self.cluster_analyzer.detect_cluster_contradictions(
            cluster, individual_results, graph
        )

    def _compile_cluster_verdict(
        self,
        individual_results: dict[str, dict[str, Any]],
        cross_verification_results: list[dict[str, Any]],
        contradictions: list[dict[str, Any]],
    ) -> tuple[str, float]:
        """Legacy method - delegates to cluster analyzer."""
        return self.cluster_analyzer.compile_cluster_verdict(
            individual_results, cross_verification_results, contradictions
        )

    async def _verify_individual_node(
        self, node: FactNode, graph: FactGraph, context: VerificationContext
    ) -> dict[str, Any]:
        """Legacy method - delegates to verification processor."""
        return await self.verification_processor.verify_individual_node(node, context)

    # Additional utility methods
    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics from all components."""
        # Get metrics from individual components
        evidence_metrics = await self.evidence_gatherer.get_cache_stats()
        source_metrics = await self.source_manager.get_cache_stats()
        verification_metrics = (
            await self.verification_processor.get_performance_metrics()
        )

        # Calculate engine-level metrics
        avg_verification_time = self.performance_metrics[
            "total_verification_time"
        ] / max(1, self.performance_metrics["total_verifications"])

        error_rate = self.performance_metrics["error_count"] / max(
            1, self.performance_metrics["total_verifications"]
        )

        return {
            "engine_metrics": {
                "total_verifications": self.performance_metrics["total_verifications"],
                "total_verification_time": self.performance_metrics[
                    "total_verification_time"
                ],
                "average_verification_time": avg_verification_time,
                "error_count": self.performance_metrics["error_count"],
                "error_rate": error_rate,
                "cache_size": len(self.cache_manager.cache),
                "max_cache_size": self.cache_manager.max_size,
            },
            "component_metrics": {
                "evidence_gatherer": evidence_metrics,
                "source_manager": source_metrics,
                "verification_processor": verification_metrics,
            },
        }

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache_manager.cache),
            "max_size": self.cache_manager.max_size,
            "ttl_seconds": self.cache_manager.ttl_seconds,
        }

    async def clear_cache(self):
        """Clear all caches."""
        self.cache_manager.clear()
        await self.evidence_gatherer.clear_cache()
        await self.source_manager.clear_cache()
        await self.verification_processor.clear_cache()
        self.logger.info("All caches cleared")

    async def optimize_performance(self):
        """Optimize performance across all components."""
        # Optimize individual components
        evidence_recommendations = await self.evidence_gatherer.optimize_performance()
        # source_manager doesn't return recommendations
        await self.source_manager.optimize_cache()
        verification_recommendations = (
            await self.verification_processor.optimize_performance()
        )

        # Engine-level optimizations
        engine_recommendations = []

        # Check if cache needs resizing
        cache_usage = len(self.cache_manager.cache) / self.cache_manager.max_size
        if cache_usage > 0.9:
            engine_recommendations.append("Consider increasing cache size")
        elif cache_usage < 0.1:
            engine_recommendations.append("Consider decreasing cache size")

        # Check error rate
        if self.performance_metrics["total_verifications"] > 0:
            error_rate = (
                self.performance_metrics["error_count"]
                / self.performance_metrics["total_verifications"]
            )
            if error_rate > 0.1:
                engine_recommendations.append(
                    "High error rate detected - review configuration"
                )

        all_recommendations = {
            "engine": engine_recommendations,
            "evidence_gatherer": evidence_recommendations,
            # source_manager doesn't return specific recommendations
            "source_manager": ["Cache optimization completed"],
            "verification_processor": verification_recommendations,
        }

        self.logger.info(
            "Performance optimization completed. Recommendations: %s",
            all_recommendations,
        )
        return all_recommendations

    def get_module_info(self) -> dict[str, str]:
        """Get information about loaded modules."""
        return {
            "evidence_gatherer": type(self.evidence_gatherer).__name__,
            "source_manager": type(self.source_manager).__name__,
            "verification_processor": type(self.verification_processor).__name__,
            "cluster_analyzer": type(self.cluster_analyzer).__name__,
            "result_compiler": type(self.result_compiler).__name__,
            "response_parser": type(self.response_parser).__name__,
            "engine_version": "enhanced_v2.0",
        }

    async def close(self):
        """Close all modules and clean up resources."""
        # Close enhanced components
        if hasattr(self, "verification_processor") and self.verification_processor:
            await self.verification_processor.close()
            logger.info("EnhancedGraphVerificationEngine: VerificationProcessor closed")

        if hasattr(self, "source_manager") and self.source_manager:
            await self.source_manager.close()
            logger.info("EnhancedGraphVerificationEngine: SourceManager closed")

        if hasattr(self, "evidence_gatherer") and self.evidence_gatherer:
            await self.evidence_gatherer.close()
            logger.info("EnhancedGraphVerificationEngine: EvidenceGatherer closed")

        # Clear all caches
        if hasattr(self, "cache_manager") and self.cache_manager:
            self.cache_manager.clear()

        logger.info("EnhancedGraphVerificationEngine: All resources cleaned up")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Backward compatibility alias
GraphVerificationEngine = EnhancedGraphVerificationEngine

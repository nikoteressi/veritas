"""
Relevance orchestrator for Veritas system.

Maintains the existing public interface of RelevanceIntegrationManager while delegating
responsibilities to specialized components.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.exceptions import ValidationError

from ..cache.cache_monitor import CacheMonitor
from .relevance_embeddings_coordinator import RelevanceEmbeddingsCoordinator

logger = logging.getLogger(__name__)


class RelevanceOrchestrator:
    """
    Orchestrates relevance operations while maintaining the existing public interface.

    This class serves as the main entry point for relevance operations, delegating
    responsibilities to specialized components while preserving backward compatibility.
    """

    def __init__(self):
        """Initialize the relevance orchestrator."""
        self.cache_monitor: CacheMonitor | None = None
        self.embeddings_coordinator: RelevanceEmbeddingsCoordinator | None = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if the orchestrator has been initialized."""
        return self._initialized

    async def initialize(self) -> bool:
        """Initialize all orchestrator components."""
        try:
            logger.info("Initializing RelevanceOrchestrator...")

            # Initialize cache monitor
            self.cache_monitor = CacheMonitor()

            # Initialize embeddings coordinator
            self.embeddings_coordinator = RelevanceEmbeddingsCoordinator()
            await self.embeddings_coordinator.initialize()

            self._initialized = True
            logger.info("RelevanceOrchestrator initialized successfully")
            return True

        except Exception as e:
            raise ValidationError(
                f"Failed to initialize RelevanceOrchestrator: {e}") from e

    async def calculate_comprehensive_relevance(
        self, query: str, document: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Calculate comprehensive relevance score using hybrid scoring and temporal analysis.

        This method maintains the existing interface while delegating to specialized components.
        """
        if not self._initialized:
            logger.error("RelevanceOrchestrator not initialized")
            return {"error": "Orchestrator not initialized"}

        try:
            # Get hybrid relevance score
            hybrid_result = await self.embeddings_coordinator.calculate_hybrid_relevance(
                query, document, metadata)

            # Get temporal analysis
            temporal_result = await self.embeddings_coordinator.analyze_temporal_relevance(
                query, document, metadata)

            # Get explainable score for transparency
            explainable_result = await self.embeddings_coordinator.get_explainable_score(
                query, document, metadata)

            # Get adaptive threshold - use factual for fact verification queries
            threshold_context = {
                "query": query,
                "document_preview": document[:100],
                "query_type": metadata.get("query_type", "factual") if metadata else "factual",
                "source_type": metadata.get("source_type", "web") if metadata else "web"
            }
            threshold = await self.embeddings_coordinator.get_adaptive_threshold(threshold_context)

            # Combine results
            comprehensive_score = {
                "hybrid_score": hybrid_result.get("score", 0.0),
                "temporal_score": temporal_result.get("adjusted_relevance", 0.0),
                "explainable_score": explainable_result.get("final_score", 0.0),
                "adaptive_threshold": threshold,
                "metadata": {
                    "hybrid_details": hybrid_result,
                    "temporal_details": temporal_result,
                    "explanation": explainable_result.get("explanation", ""),
                    "confidence": explainable_result.get("confidence", 0.0),
                },
            }

            # Calculate final weighted score
            final_score = (
                comprehensive_score["hybrid_score"] * 0.5
                + comprehensive_score["temporal_score"] * 0.3
                + comprehensive_score["explainable_score"] * 0.2
            )

            comprehensive_score["final_score"] = final_score
            comprehensive_score["is_relevant"] = final_score >= threshold

            return comprehensive_score

        except Exception as e:
            raise ValidationError(
                f"Failed to calculate comprehensive relevance: {e}") from e

    async def batch_analyze_relevance(
        self,
        queries: list[str],
        documents: list[str],
        metadata_list: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Analyze relevance for multiple query-document pairs in batch.

        Maintains the existing interface while leveraging specialized components.
        """
        if not self._initialized:
            logger.error("RelevanceOrchestrator not initialized")
            return [{"error": "Orchestrator not initialized"}] * len(queries)

        try:
            # Prepare metadata list
            if metadata_list is None:
                metadata_list = [None] * len(queries)

            # Process in parallel
            tasks = [
                self.calculate_comprehensive_relevance(query, doc, metadata)
                for query, doc, metadata in zip(queries, documents, metadata_list, strict=False)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions in results
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    processed_results.append({"error": str(result)})
                else:
                    processed_results.append(result)

            return processed_results

        except Exception as e:
            raise ValidationError(
                f"Failed to process batch relevance analysis: {e}") from e

    async def get_performance_report(self) -> str:
        """
        Generate comprehensive performance report for all relevance components.

        Maintains the existing interface while leveraging specialized monitoring.
        """
        if not self._initialized:
            return "Error: RelevanceOrchestrator not initialized"

        try:
            report = "=== VERITAS RELEVANCE SYSTEM PERFORMANCE REPORT ===\n\n"

            # Cache performance report
            if self.cache_monitor:
                cache_report = await self.cache_monitor.generate_performance_report()
                report += "--- CACHE PERFORMANCE ---\n"
                report += cache_report + "\n\n"

            # Embeddings performance metrics
            if self.embeddings_coordinator:
                embeddings_metrics = await self.embeddings_coordinator.get_performance_metrics()
                report += "--- EMBEDDINGS PERFORMANCE ---\n"
                for component, metrics in embeddings_metrics.items():
                    report += f"  {component.upper()}:\n"
                    for key, value in metrics.items():
                        report += f"    {key}: {value}\n"
                report += "\n"

            return report

        except Exception as e:
            raise ValidationError(
                f"Failed to generate performance report: {e}") from e

    async def optimize_performance(self) -> dict[str, Any]:
        """
        Optimize performance of all relevance components.

        Maintains the existing interface while delegating to specialized optimizers.
        """
        if not self._initialized:
            return {"error": "Orchestrator not initialized"}

        optimization_results = {}

        try:
            # Optimize embeddings performance
            if self.embeddings_coordinator:
                embeddings_optimization = await self.embeddings_coordinator\
                    .optimize_embeddings_performance()
                optimization_results["embeddings"] = embeddings_optimization

            # Optimize cache settings
            if self.cache_monitor:
                cache_optimization = await self.cache_monitor.optimize_cache_settings(
                    "embedding_cache")
                optimization_results["cache"] = cache_optimization

            logger.info("Performance optimization completed")
            return optimization_results

        except Exception as e:
            raise ValidationError(
                f"Failed to optimize performance: {e}") from e

    async def check_health(self) -> dict[str, Any]:
        """
        Check health of all relevance system components.

        Provides basic health checking for orchestrator components.
        """
        if not self._initialized:
            return {"status": "error", "message": "Orchestrator not initialized"}

        try:
            health_status = {"status": "healthy", "components": {}}

            # Check embeddings coordinator health
            if self.embeddings_coordinator:
                health_status["components"]["embeddings_coordinator"] = "healthy"
            else:
                health_status["components"]["embeddings_coordinator"] = "not_initialized"

            # Check cache monitor health
            if self.cache_monitor:
                health_status["components"]["cache_monitor"] = "healthy"
            else:
                health_status["components"]["cache_monitor"] = "not_initialized"

            return health_status

        except Exception as e:
            raise ValidationError(
                f"Failed to check system health: {e}") from e

    async def get_cache_metrics(self) -> dict[str, Any]:
        """
        Get cache performance metrics.

        Delegates to the CacheMonitor for detailed cache metrics.
        """
        if not self._initialized or not self.cache_monitor:
            return {"error": "Cache monitor not available"}

        try:
            return await self.cache_monitor.collect_cache_metrics()
        except Exception as e:
            raise ValidationError(
                f"Failed to get cache metrics: {e}") from e

    async def close(self):
        """Clean up all orchestrator resources."""
        try:
            logger.info("Closing RelevanceOrchestrator...")

            # Close embeddings coordinator
            if self.embeddings_coordinator:
                await self.embeddings_coordinator.close()

            # Stop cache monitoring
            if self.cache_monitor and hasattr(self.cache_monitor, "stop_monitoring"):
                await self.cache_monitor.stop_monitoring()

            self._initialized = False
            logger.info("RelevanceOrchestrator closed successfully")

        except Exception as e:
            raise ValidationError(
                f"Failed to close RelevanceOrchestrator: {e}") from e


# Global instance management (maintaining existing interface)
_relevance_orchestrator: RelevanceOrchestrator | None = None


def get_relevance_manager() -> RelevanceOrchestrator:
    """
    Get global relevance manager instance.

    Maintains the existing function name for backward compatibility.
    """
    global _relevance_orchestrator
    if _relevance_orchestrator is None:
        _relevance_orchestrator = RelevanceOrchestrator()
    return _relevance_orchestrator


async def close_relevance_manager():
    """
    Close global relevance manager instance.

    Maintains the existing function name for backward compatibility.
    """
    global _relevance_orchestrator
    if _relevance_orchestrator is not None:
        await _relevance_orchestrator.close()
        _relevance_orchestrator = None

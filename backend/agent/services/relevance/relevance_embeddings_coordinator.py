"""
Relevance-specific embeddings coordination for Veritas system.

Handles embeddings generation, caching, and coordination for relevance scoring.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.exceptions import ValidationError
from app.cache import CacheType, get_cache, get_general_cache

from ..analysis.adaptive_thresholds import AdaptiveThresholds
from ..infrastructure.enhanced_ollama_embeddings import EnhancedOllamaEmbeddings
from .cached_hybrid_relevance_scorer import CachedHybridRelevanceScorer
from .explainable_relevance_scorer import ExplainableRelevanceScorer

logger = logging.getLogger(__name__)


class RelevanceEmbeddingsCoordinator:
    """Coordinates embeddings generation and relevance scoring for the Veritas system."""

    def __init__(self):
        """Initialize the relevance embeddings coordinator."""
        self.embeddings: EnhancedOllamaEmbeddings | None = None
        self.hybrid_scorer: CachedHybridRelevanceScorer | None = None
        self.temporal_cache = None  # Will be initialized from cache_factory
        self.explainable_scorer: ExplainableRelevanceScorer | None = None
        self.adaptive_thresholds: AdaptiveThresholds | None = None
        self.cache = None  # Will be initialized from cache_factory
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize all relevance-specific components."""
        try:
            logger.info("Initializing RelevanceEmbeddingsCoordinator...")

            # Initialize embeddings
            self.embeddings = EnhancedOllamaEmbeddings()

            # Initialize temporal analysis cache
            self.temporal_cache = await get_cache(CacheType.TEMPORAL)

            # Initialize unified cache for general caching needs
            self.cache = await get_general_cache()

            # Initialize adaptive thresholds
            self.adaptive_thresholds = AdaptiveThresholds()

            # Initialize hybrid scorer with shared embeddings
            self.hybrid_scorer = CachedHybridRelevanceScorer(
                shared_embeddings=self.embeddings, cache_size=200)

            # Initialize explainable scorer
            self.explainable_scorer = ExplainableRelevanceScorer(
                hybrid_scorer=self.hybrid_scorer, cache_size=200)

            self._initialized = True
            logger.info(
                "RelevanceEmbeddingsCoordinator initialized successfully")
            return True

        except Exception as e:
            raise ValidationError(
                f"Failed to initialize RelevanceEmbeddingsCoordinator: {e}") from e

    async def generate_embeddings(self, text: str) -> list[float] | None:
        """Generate embeddings for the given text."""
        if not self._initialized or not self.embeddings:
            logger.error("RelevanceEmbeddingsCoordinator not initialized")
            return None

        try:
            return await self.embeddings.embed_text(text)

        except Exception as e:
            raise ValidationError(f"Failed to generate embeddings: {e}") from e

    async def calculate_hybrid_relevance(
        self, query: str, document: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Calculate hybrid relevance score using multiple scoring methods."""
        if not self._initialized or not self.hybrid_scorer:
            logger.error("RelevanceEmbeddingsCoordinator not initialized")
            return {"error": "Coordinator not initialized"}

        try:
            # Call hybrid scorer with explain=True to get detailed results
            result = await self.hybrid_scorer.calculate_hybrid_score(query, document, use_cache=True, explain=True)

            # Handle the tuple return (score, details)
            if isinstance(result, tuple) and len(result) == 2:
                score, details = result
                return {
                    "score": score,
                    "details": details,
                    "method": "hybrid_scoring"
                }
            # Handle float return (fallback for compatibility)
            elif isinstance(result, (int, float)):
                return {
                    "score": float(result),
                    "details": {},
                    "method": "hybrid_scoring"
                }
            # Handle dict return (if already in correct format)
            elif isinstance(result, dict):
                if "score" not in result:
                    # If it's a dict but missing score key, try to extract it
                    result["score"] = result.get("hybrid_score", 0.0)
                return result
            else:
                logger.warning(
                    f"Unexpected result type from hybrid scorer: {type(result)}")
                return {
                    "score": 0.0,
                    "details": {},
                    "method": "hybrid_scoring",
                    "error": f"Unexpected result type: {type(result)}"
                }
        except Exception as e:
            raise ValidationError(
                f"Failed to calculate hybrid relevance: {e}") from e

    async def get_explainable_score(
        self, query: str, document: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Get explainable relevance score with reasoning."""
        if not self._initialized or not self.explainable_scorer:
            logger.error("RelevanceEmbeddingsCoordinator not initialized")
            return {"error": "Coordinator not initialized"}

        try:
            # Extract content_date from metadata if available
            content_date = metadata.get("date") if metadata else None
            return await self.explainable_scorer.explain_relevance(query, document, content_date)
        except Exception as e:
            raise ValidationError(
                f"Failed to get explainable score: {e}") from e

    async def analyze_temporal_relevance(
        self, query: str, document: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Analyze temporal aspects of relevance."""
        if not self._initialized or not self.temporal_cache:
            logger.error("RelevanceEmbeddingsCoordinator not initialized")
            return {"error": "Coordinator not initialized"}

        try:
            # Extract content_date from metadata or use current time
            content_date = metadata.get("date") if metadata else None
            # Calculate base relevance score (simplified for now)
            base_relevance = 0.5

            logger.debug(
                f"Calling temporal_cache.analyze_temporal_relevance with query={query[:50]}..., content_date={content_date}, base_relevance={base_relevance}")

            result = await self.temporal_cache.analyze_temporal_relevance(
                query=query,
                content=document,
                content_date=content_date,
                base_relevance=base_relevance,
                time_window="default",
                use_cache=True,
            )

            logger.debug(
                f"Temporal analysis result type: {type(result)}, value: {result}")

            return result

        except Exception as e:
            logger.error(f"Failed to analyze temporal relevance: {e}")
            raise ValidationError(
                f"Failed to analyze temporal relevance: {e}") from e

    async def get_adaptive_threshold(self, context: dict[str, Any] = None) -> float:
        """Get adaptive threshold for the given context."""
        if not self._initialized or not self.adaptive_thresholds:
            logger.error("RelevanceEmbeddingsCoordinator not initialized")
            return 0.5  # Default threshold

        try:
            # Extract query_type and source_type from context or use defaults
            query_type = context.get(
                "query_type", "general") if context else "general"
            source_type = context.get(
                "source_type", "web") if context else "web"
            return await self.adaptive_thresholds.get_adaptive_threshold(
                query_type=query_type, source_type=source_type, context=context
            )

        except Exception as e:
            raise ValidationError(
                f"Failed to get adaptive threshold: {e}") from e

    async def batch_process_embeddings(self, texts: list[str]) -> list[list[float] | None]:
        """Process multiple texts for embeddings in batch."""
        if not self._initialized or not self.embeddings:
            logger.error("RelevanceEmbeddingsCoordinator not initialized")
            return [None] * len(texts)

        try:
            tasks = [self.generate_embeddings(text) for text in texts]
            return await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            raise ValidationError(
                f"Failed to process batch embeddings: {e}") from e

    async def optimize_embeddings_performance(self) -> dict[str, Any]:
        """Optimize embeddings performance based on usage patterns."""
        if not self._initialized:
            return {"error": "Coordinator not initialized"}

        optimization_results = {}

        try:
            # Optimize embeddings cache
            if self.embeddings:
                embeddings_optimization = await self.embeddings.optimize_cache()
                optimization_results["embeddings"] = embeddings_optimization

            # Note: AdaptiveThresholds doesn't have an optimize method
            # It automatically adjusts based on performance data

            logger.info("Embeddings performance optimization completed")
            return optimization_results

        except Exception as e:
            raise ValidationError(
                f"Failed to optimize embeddings performance: {e}") from e

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics for all embeddings components."""
        if not self._initialized:
            return {"error": "Coordinator not initialized"}

        metrics = {}

        try:
            # Embeddings metrics
            if self.embeddings:
                metrics["embeddings"] = await self.embeddings.get_performance_metrics()

            # Note: Temporal cache from unified system doesn't have get_stats method
            # Note: AdaptiveThresholds doesn't have get_stats method
            # These components track their own internal metrics

            return metrics

        except Exception as e:
            raise ValidationError(
                f"Failed to get performance metrics: {e}") from e

    async def close(self):
        """Clean up resources (cache is managed by factory)."""
        try:
            logger.info("Closing RelevanceEmbeddingsCoordinator...")

            if self.embeddings:
                await self.embeddings.close()

            # Cache is managed by CacheFactory, no need to close here
            # Note: AdaptiveThresholds doesn't have a close method
            # It doesn't require explicit cleanup

            self._initialized = False
            logger.info("RelevanceEmbeddingsCoordinator closed successfully")

        except Exception as e:
            raise ValidationError(
                f"Failed to close RelevanceEmbeddingsCoordinator: {e}") from e


# Global instance management
_relevance_embeddings_coordinator: RelevanceEmbeddingsCoordinator | None = None


def get_relevance_embeddings_coordinator() -> RelevanceEmbeddingsCoordinator:
    """Get global relevance embeddings coordinator instance."""
    global _relevance_embeddings_coordinator
    if _relevance_embeddings_coordinator is None:
        _relevance_embeddings_coordinator = RelevanceEmbeddingsCoordinator()
    return _relevance_embeddings_coordinator


async def close_relevance_embeddings_coordinator():
    """Close global relevance embeddings coordinator instance."""
    global _relevance_embeddings_coordinator
    if _relevance_embeddings_coordinator is not None:
        await _relevance_embeddings_coordinator.close()
        _relevance_embeddings_coordinator = None

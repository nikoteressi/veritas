"""
Relevance-specific embeddings coordination for Veritas system.

Handles embeddings generation, caching, and coordination for relevance scoring.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ..infrastructure.enhanced_ollama_embeddings import EnhancedOllamaEmbeddings
from .cached_hybrid_relevance_scorer import CachedHybridRelevanceScorer
from .explainable_relevance_scorer import ExplainableRelevanceScorer
from ..cache.temporal_analysis_cache import TemporalAnalysisCache
from ..cache.intelligent_cache import IntelligentCache
from ..analysis.adaptive_thresholds import AdaptiveThresholds

logger = logging.getLogger(__name__)


class RelevanceEmbeddingsCoordinator:
    """Coordinates embeddings generation and relevance scoring for the Veritas system."""

    def __init__(self):
        """Initialize the relevance embeddings coordinator."""
        self.embeddings: Optional[EnhancedOllamaEmbeddings] = None
        self.hybrid_scorer: Optional[CachedHybridRelevanceScorer] = None
        self.temporal_cache: Optional[TemporalAnalysisCache] = None
        self.explainable_scorer: Optional[ExplainableRelevanceScorer] = None
        self.intelligent_cache: Optional[IntelligentCache] = None
        self.adaptive_thresholds: Optional[AdaptiveThresholds] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize all relevance-specific components."""
        try:
            logger.info("Initializing RelevanceEmbeddingsCoordinator...")

            # Initialize embeddings
            self.embeddings = EnhancedOllamaEmbeddings()

            # Initialize intelligent cache
            self.intelligent_cache = IntelligentCache()
            await self.intelligent_cache.initialize()

            # Initialize temporal analysis cache
            self.temporal_cache = TemporalAnalysisCache()

            # Initialize adaptive thresholds
            self.adaptive_thresholds = AdaptiveThresholds()

            # Initialize hybrid scorer with shared embeddings
            self.hybrid_scorer = CachedHybridRelevanceScorer(
                shared_embeddings=self.embeddings,
                cache_size=200
            )

            # Initialize explainable scorer
            self.explainable_scorer = ExplainableRelevanceScorer(
                hybrid_scorer=self.hybrid_scorer,
                cache_size=200
            )

            self._initialized = True
            logger.info(
                "RelevanceEmbeddingsCoordinator initialized successfully")
            return True

        except Exception as e:
            logger.error(
                f"Failed to initialize RelevanceEmbeddingsCoordinator: {e}")
            return False

    async def generate_embeddings(self, text: str) -> Optional[List[float]]:
        """Generate embeddings for the given text."""
        if not self._initialized or not self.embeddings:
            logger.error("RelevanceEmbeddingsCoordinator not initialized")
            return None

        try:
            return await self.embeddings.embed_text(text)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return None

    async def calculate_hybrid_relevance(
        self,
        query: str,
        document: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Calculate hybrid relevance score using multiple scoring methods."""
        if not self._initialized or not self.hybrid_scorer:
            logger.error("RelevanceEmbeddingsCoordinator not initialized")
            return {"error": "Coordinator not initialized"}

        try:
            return await self.hybrid_scorer.calculate_hybrid_score(query, document, metadata)
        except Exception as e:
            logger.error(f"Error calculating hybrid relevance: {e}")
            return {"error": str(e)}

    async def get_explainable_score(
        self,
        query: str,
        document: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get explainable relevance score with reasoning."""
        if not self._initialized or not self.explainable_scorer:
            logger.error("RelevanceEmbeddingsCoordinator not initialized")
            return {"error": "Coordinator not initialized"}

        try:
            return await self.explainable_scorer.explain_relevance(query, document, metadata)
        except Exception as e:
            logger.error(f"Error getting explainable score: {e}")
            return {"error": str(e)}

    async def analyze_temporal_relevance(
        self,
        query: str,
        document: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze temporal aspects of relevance."""
        if not self._initialized or not self.temporal_cache:
            logger.error("RelevanceEmbeddingsCoordinator not initialized")
            return {"error": "Coordinator not initialized"}

        try:
            # Extract content_date from metadata or use current time
            content_date = metadata.get('date') if metadata else None
            # Calculate base relevance score (simplified for now)
            base_relevance = 0.5
            return await self.temporal_cache.analyze_temporal_relevance(
                query=query, 
                content=document, 
                content_date=content_date, 
                base_relevance=base_relevance,
                time_window=None,
                use_cache=True
            )
        except Exception as e:
            logger.error(f"Error analyzing temporal relevance: {e}")
            return {"error": str(e)}

    async def get_adaptive_threshold(self, context: Dict[str, Any] = None) -> float:
        """Get adaptive threshold for the given context."""
        if not self._initialized or not self.adaptive_thresholds:
            logger.error("RelevanceEmbeddingsCoordinator not initialized")
            return 0.5  # Default threshold

        try:
            # Extract query_type and source_type from context or use defaults
            query_type = context.get('query_type', 'general') if context else 'general'
            source_type = context.get('source_type', 'web') if context else 'web'
            return await self.adaptive_thresholds.get_adaptive_threshold(
                query_type=query_type,
                source_type=source_type,
                context=context
            )
        except Exception as e:
            logger.error(f"Error getting adaptive threshold: {e}")
            return 0.5  # Default threshold

    async def batch_process_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Process multiple texts for embeddings in batch."""
        if not self._initialized or not self.embeddings:
            logger.error("RelevanceEmbeddingsCoordinator not initialized")
            return [None] * len(texts)

        try:
            tasks = [self.generate_embeddings(text) for text in texts]
            return await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in batch processing embeddings: {e}")
            return [None] * len(texts)

    async def optimize_embeddings_performance(self) -> Dict[str, Any]:
        """Optimize embeddings performance based on usage patterns."""
        if not self._initialized:
            return {"error": "Coordinator not initialized"}

        optimization_results = {}

        try:
            # Optimize embeddings cache
            if self.embeddings:
                embeddings_optimization = await self.embeddings.optimize_cache()
                optimization_results["embeddings"] = embeddings_optimization

            # Optimize intelligent cache
            if self.intelligent_cache:
                await self.intelligent_cache.optimize()
                cache_stats = await self.intelligent_cache.get_stats()
                optimization_results["cache"] = cache_stats

            # Note: AdaptiveThresholds doesn't have an optimize method
            # It automatically adjusts based on performance data

            logger.info("Embeddings performance optimization completed")
            return optimization_results

        except Exception as e:
            logger.error(f"Error optimizing embeddings performance: {e}")
            return {"error": str(e)}

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all embeddings components."""
        if not self._initialized:
            return {"error": "Coordinator not initialized"}

        metrics = {}

        try:
            # Embeddings metrics
            if self.embeddings:
                metrics["embeddings"] = await self.embeddings.get_performance_metrics()

            # Cache metrics
            if self.intelligent_cache:
                metrics["cache"] = await self.intelligent_cache.get_stats()

            # Note: TemporalAnalysisCache doesn't have get_stats method
            # Note: AdaptiveThresholds doesn't have get_stats method
            # These components track their own internal metrics

            return metrics

        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e)}

    async def close(self):
        """Clean up resources."""
        try:
            logger.info("Closing RelevanceEmbeddingsCoordinator...")

            if self.embeddings:
                await self.embeddings.close()

            if self.intelligent_cache:
                await self.intelligent_cache.close()

            if self.temporal_cache:
                await self.temporal_cache.close()

            # Note: AdaptiveThresholds doesn't have a close method
            # It doesn't require explicit cleanup

            self._initialized = False
            logger.info("RelevanceEmbeddingsCoordinator closed successfully")

        except Exception as e:
            logger.error(f"Error closing RelevanceEmbeddingsCoordinator: {e}")


# Global instance management
_relevance_embeddings_coordinator: Optional[RelevanceEmbeddingsCoordinator] = None


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

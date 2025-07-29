"""
Extended cache monitor for relevance-specific cache monitoring.

Extends the base CacheMonitor to include monitoring for relevance system caches
while maintaining compatibility with the existing infrastructure.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from .cache_monitor import CacheMonitor
from ..relevance.relevance_embeddings_coordinator import RelevanceEmbeddingsCoordinator

logger = logging.getLogger(__name__)


class RelevanceCacheMonitor(CacheMonitor):
    """
    Extended cache monitor that includes relevance-specific cache monitoring.
    
    This class extends the base CacheMonitor to include monitoring for
    caches used by the relevance system components.
    """

    def __init__(self, embeddings_coordinator: Optional[RelevanceEmbeddingsCoordinator] = None):
        """
        Initialize the relevance cache monitor.
        
        Args:
            embeddings_coordinator: RelevanceEmbeddingsCoordinator instance for monitoring
        """
        super().__init__()
        self.embeddings_coordinator = embeddings_coordinator
        
        # Add relevance-specific monitoring data
        self._monitoring_data.update({
            "temporal_cache": [],
            "adaptive_thresholds_cache": [],
            "explainable_scorer_cache": []
        })

    async def collect_cache_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Collect comprehensive metrics from all caches including relevance caches.
        
        Returns:
            dict: Comprehensive cache metrics
        """
        try:
            # Get base cache metrics
            metrics = await super().collect_cache_metrics()

            # Add relevance-specific cache metrics
            if self.embeddings_coordinator and self.embeddings_coordinator._initialized:
                relevance_metrics = await self._collect_relevance_cache_metrics()
                metrics.update(relevance_metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error collecting cache metrics: {e}")
            return {}

    async def _collect_relevance_cache_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Collect metrics from relevance-specific caches.
        
        Returns:
            dict: Relevance cache metrics
        """
        relevance_metrics = {}

        try:
            # Temporal cache metrics
            if self.embeddings_coordinator.temporal_cache:
                temporal_stats = await self._get_temporal_cache_stats()
                relevance_metrics["temporal_cache"] = {
                    **temporal_stats,
                    "timestamp": datetime.now().isoformat(),
                    "cache_type": "temporal"
                }

            # Adaptive thresholds cache metrics
            if self.embeddings_coordinator.adaptive_thresholds:
                thresholds_stats = await self._get_adaptive_thresholds_stats()
                relevance_metrics["adaptive_thresholds_cache"] = {
                    **thresholds_stats,
                    "timestamp": datetime.now().isoformat(),
                    "cache_type": "adaptive_thresholds"
                }

            # Explainable scorer cache metrics
            if self.embeddings_coordinator.explainable_scorer:
                explainable_stats = await self._get_explainable_scorer_stats()
                relevance_metrics["explainable_scorer_cache"] = {
                    **explainable_stats,
                    "timestamp": datetime.now().isoformat(),
                    "cache_type": "explainable_scorer"
                }

            # Store metrics for trend analysis
            for cache_name, cache_metrics in relevance_metrics.items():
                if cache_name in self._monitoring_data:
                    self._monitoring_data[cache_name].append(cache_metrics)
                    
                    # Keep only last 100 measurements
                    if len(self._monitoring_data[cache_name]) > 100:
                        self._monitoring_data[cache_name] = self._monitoring_data[cache_name][-100:]

            return relevance_metrics

        except Exception as e:
            logger.error(f"Error collecting relevance cache metrics: {e}")
            return {}

    async def _get_temporal_cache_stats(self) -> Dict[str, Any]:
        """Get statistics from temporal analysis cache."""
        try:
            if hasattr(self.embeddings_coordinator.temporal_cache, 'get_stats'):
                return await self.embeddings_coordinator.temporal_cache.get_stats()
            else:
                # Fallback basic stats
                return {
                    "hit_rate": 0.0,
                    "miss_rate": 1.0,
                    "total_requests": 0,
                    "cache_size": 0,
                    "memory_usage": 0.0
                }
        except Exception as e:
            logger.error(f"Error getting temporal cache stats: {e}")
            return {"error": str(e)}

    async def _get_adaptive_thresholds_stats(self) -> Dict[str, Any]:
        """Get statistics from adaptive thresholds cache."""
        try:
            if hasattr(self.embeddings_coordinator.adaptive_thresholds, 'get_stats'):
                return await self.embeddings_coordinator.adaptive_thresholds.get_stats()
            else:
                # Fallback basic stats
                return {
                    "hit_rate": 0.0,
                    "miss_rate": 1.0,
                    "total_requests": 0,
                    "cache_size": 0,
                    "memory_usage": 0.0
                }
        except Exception as e:
            logger.error(f"Error getting adaptive thresholds stats: {e}")
            return {"error": str(e)}

    async def _get_explainable_scorer_stats(self) -> Dict[str, Any]:
        """Get statistics from explainable scorer cache."""
        try:
            if hasattr(self.embeddings_coordinator.explainable_scorer, 'get_cache_stats'):
                return await self.embeddings_coordinator.explainable_scorer.get_cache_stats()
            else:
                # Fallback basic stats
                return {
                    "hit_rate": 0.0,
                    "miss_rate": 1.0,
                    "total_requests": 0,
                    "cache_size": 0,
                    "memory_usage": 0.0
                }
        except Exception as e:
            logger.error(f"Error getting explainable scorer stats: {e}")
            return {"error": str(e)}

    async def generate_performance_report(self) -> str:
        """
        Generate comprehensive performance report including relevance caches.
        
        Returns:
            str: Detailed performance report
        """
        try:
            # Get base performance report
            base_report = await super().generate_performance_report()

            # Add relevance-specific cache report
            relevance_report = await self._generate_relevance_cache_report()

            return base_report + relevance_report

        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return f"Error generating performance report: {e}"

    async def _generate_relevance_cache_report(self) -> str:
        """Generate performance report for relevance-specific caches."""
        try:
            relevance_metrics = await self._collect_relevance_cache_metrics()
            
            if not relevance_metrics:
                return "\n=== RELEVANCE CACHE PERFORMANCE ===\nNo relevance cache data available\n"

            report = "\n=== RELEVANCE CACHE PERFORMANCE ===\n"
            report += f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

            for cache_name, cache_stats in relevance_metrics.items():
                if "error" in cache_stats:
                    report += f"--- {cache_name.upper()} ---\n"
                    report += f"  Error: {cache_stats['error']}\n\n"
                    continue

                report += f"--- {cache_name.upper()} ---\n"

                # Basic stats
                hit_rate = cache_stats.get("hit_rate", 0) * 100
                miss_rate = cache_stats.get("miss_rate", 0) * 100

                report += f"  Hit Rate: {hit_rate:.2f}%\n"
                report += f"  Miss Rate: {miss_rate:.2f}%\n"
                report += f"  Total Requests: {cache_stats.get('total_requests', 0)}\n"
                report += f"  Cache Size: {cache_stats.get('cache_size', 0)} entries\n"

                if "memory_usage" in cache_stats:
                    report += f"  Memory Usage: {cache_stats['memory_usage']:.2f} MB\n"

                # Performance analysis
                performance_grade = self._calculate_performance_grade(cache_stats)
                report += f"  Performance Grade: {performance_grade}\n"

                # Relevance-specific recommendations
                recommendations = self._generate_relevance_cache_recommendations(cache_name, cache_stats)
                if recommendations:
                    report += "  Recommendations:\n"
                    for rec in recommendations:
                        report += f"    - {rec}\n"

                report += "\n"

            return report

        except Exception as e:
            logger.error(f"Error generating relevance cache report: {e}")
            return f"\n=== RELEVANCE CACHE PERFORMANCE ===\nError: {e}\n"

    def _generate_relevance_cache_recommendations(
        self, cache_name: str, cache_stats: Dict[str, Any]
    ) -> list[str]:
        """Generate optimization recommendations for relevance-specific caches."""
        recommendations = []

        hit_rate = cache_stats.get("hit_rate", 0)
        cache_size = cache_stats.get("cache_size", 0)
        total_requests = cache_stats.get("total_requests", 0)

        # Cache-specific recommendations
        if cache_name == "temporal_cache":
            if hit_rate < 0.6:
                recommendations.append(
                    "Temporal cache hit rate is low. Consider increasing TTL for temporal analysis results."
                )
            if cache_size > 5000:
                recommendations.append(
                    "Large temporal cache. Consider implementing time-based eviction policies."
                )

        elif cache_name == "adaptive_thresholds_cache":
            if hit_rate < 0.8:
                recommendations.append(
                    "Adaptive thresholds should have high reuse. Review threshold calculation frequency."
                )
            if total_requests > 1000 and cache_size < 50:
                recommendations.append(
                    "Consider increasing adaptive thresholds cache size for better performance."
                )

        elif cache_name == "explainable_scorer_cache":
            if hit_rate < 0.5:
                recommendations.append(
                    "Explainable scorer cache hit rate is low. Consider caching explanation components."
                )

        # General recommendations
        if hit_rate < 0.4:
            recommendations.append(
                f"Critical hit rate for {cache_name}. Review caching strategy and key patterns."
            )

        return recommendations

    async def optimize_relevance_cache_settings(self, cache_name: str) -> Dict[str, Any]:
        """
        Suggest optimal cache settings for relevance-specific caches.
        
        Args:
            cache_name: Name of the relevance cache to optimize
            
        Returns:
            dict: Optimization recommendations
        """
        try:
            # Check if it's a relevance cache
            relevance_caches = ["temporal_cache", "adaptive_thresholds_cache", "explainable_scorer_cache"]
            
            if cache_name in relevance_caches:
                trends = await self.get_cache_trends(cache_name)
                
                if not trends:
                    return {"error": "Insufficient data for optimization"}

                recommendations = {}

                # Relevance-specific optimization logic
                if cache_name == "temporal_cache":
                    if trends.get("avg_hit_rate", 0) < 0.6:
                        recommendations["ttl"] = "Increase TTL for temporal analysis results (suggest 1-2 hours)"
                        recommendations["strategy"] = "Consider implementing sliding window caching"

                elif cache_name == "adaptive_thresholds_cache":
                    if trends.get("avg_hit_rate", 0) < 0.8:
                        recommendations["size"] = "Increase cache size for better threshold reuse"
                        recommendations["strategy"] = "Implement context-based caching keys"

                elif cache_name == "explainable_scorer_cache":
                    if trends.get("avg_hit_rate", 0) < 0.5:
                        recommendations["strategy"] = "Cache explanation components separately"
                        recommendations["ttl"] = "Consider longer TTL for stable explanations"

                return recommendations

            else:
                # Fall back to base optimization for non-relevance caches
                return await super().optimize_cache_settings(cache_name)

        except Exception as e:
            logger.error(f"Error optimizing relevance cache settings: {e}")
            return {"error": str(e)}


# Global instance management
_relevance_cache_monitor: Optional[RelevanceCacheMonitor] = None


def get_relevance_cache_monitor(embeddings_coordinator: RelevanceEmbeddingsCoordinator = None) -> RelevanceCacheMonitor:
    """
    Get global relevance cache monitor instance.
    
    Args:
        embeddings_coordinator: RelevanceEmbeddingsCoordinator instance
        
    Returns:
        RelevanceCacheMonitor: Global instance
    """
    global _relevance_cache_monitor
    if _relevance_cache_monitor is None:
        _relevance_cache_monitor = RelevanceCacheMonitor(embeddings_coordinator)
    return _relevance_cache_monitor


def close_relevance_cache_monitor():
    """Close global relevance cache monitor instance."""
    global _relevance_cache_monitor
    if _relevance_cache_monitor is not None:
        _relevance_cache_monitor = None
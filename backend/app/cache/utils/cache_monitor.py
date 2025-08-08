"""
Base cache monitor for the unified cache system.

Provides monitoring and performance reporting capabilities for the unified cache system.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from app.exceptions import CacheError
from .factory import cache_factory, get_general_cache

logger = logging.getLogger(__name__)


class CacheMonitor:
    """
    Base cache monitor for the unified cache system.

    Provides monitoring and performance reporting capabilities for all cache components
    in the unified cache system.
    """

    def __init__(self):
        """Initialize the cache monitor."""
        self.cache_factory = cache_factory

    async def generate_performance_report(self) -> str:
        """
        Generate comprehensive performance report for the unified cache system.

        Returns:
            str: Detailed performance report
        """
        try:
            report_lines = [
                "=== UNIFIED CACHE SYSTEM PERFORMANCE REPORT ===",
                f"Generated at: {datetime.now().isoformat()}",
                ""
            ]

            # Check if cache factory is initialized
            if not self.cache_factory.is_initialized():
                report_lines.extend([
                    "Cache system not initialized",
                    ""
                ])
                return "\n".join(report_lines)

            # Get unified cache metrics
            unified_metrics = await self._collect_unified_cache_metrics()

            if not unified_metrics:
                report_lines.extend([
                    "No cache metrics available",
                    ""
                ])
                return "\n".join(report_lines)

            # Generate report for each cache component
            for cache_name, cache_stats in unified_metrics.items():
                if "error" in cache_stats:
                    report_lines.extend([
                        f"--- {cache_name.upper().replace('_', ' ')} ---",
                        f"  Error: {cache_stats['error']}",
                        ""
                    ])
                    continue

                cache_display_name = cache_name.replace("_", " ").title()
                report_lines.extend([
                    f"--- {cache_display_name} ---",
                    f"  Hit Rate: {cache_stats.get('hit_rate', 0):.2%}",
                    f"  Miss Rate: {cache_stats.get('miss_rate', 0):.2%}",
                    f"  Total Requests: {cache_stats.get('total_requests', 0):,}",
                    f"  Cache Size: {cache_stats.get('cache_size', 0):,} items",
                    f"  Memory Usage: {cache_stats.get('memory_usage', 0):.2f} MB",
                    ""
                ])

                # Add performance grade
                performance_grade = self._calculate_performance_grade(
                    cache_stats)
                report_lines.append(
                    f"  Performance Grade: {performance_grade}")

                # Add recommendations
                recommendations = self._generate_cache_recommendations(
                    cache_name, cache_stats)
                if recommendations:
                    report_lines.append("  Recommendations:")
                    for rec in recommendations:
                        report_lines.append(f"    - {rec}")

                report_lines.append("")

            return "\n".join(report_lines)

        except CacheError as e:
            logger.error("Cache error generating performance report: %s", e)
            return f"Cache error generating performance report: {e}"
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Error generating performance report: %s", e)
            return f"Error generating performance report: {e}"

    async def _collect_unified_cache_metrics(self) -> dict[str, dict[str, Any]]:
        """
        Collect metrics from the unified cache system.

        Returns:
            dict: Unified cache metrics
        """
        try:
            unified_metrics = {}

            # Get cache manager metrics
            if hasattr(self.cache_factory, '_cache_manager'):
                cache_manager = await get_general_cache()
                if hasattr(cache_manager, 'get_stats'):
                    manager_stats = cache_manager.get_stats()
                    unified_metrics["cache_manager"] = {
                        **manager_stats,
                        "timestamp": datetime.now().isoformat(),
                        "cache_type": "unified_manager",
                    }

            # Get specialized cache metrics
            for cache_name in ["embedding_cache", "verification_cache", "temporal_cache"]:
                try:
                    cache_instance = getattr(
                        self.cache_factory, f"_{cache_name}", None)
                    if cache_instance:
                        # Try different methods to get stats
                        cache_stats = None
                        if hasattr(cache_instance, 'get_stats'):
                            cache_stats = await cache_instance.get_stats() if hasattr(cache_instance.get_stats, '__call__') else cache_instance.get_stats()
                        elif hasattr(cache_instance, 'get_metrics'):
                            cache_stats = await cache_instance.get_metrics() if hasattr(cache_instance.get_metrics, '__call__') else cache_instance.get_metrics()

                        if cache_stats:
                            unified_metrics[cache_name] = {
                                **cache_stats,
                                "timestamp": datetime.now().isoformat(),
                                "cache_type": cache_name,
                            }
                except CacheError as e:
                    logger.error(
                        "Cache error getting metrics for %s: %s", cache_name, e)
                    unified_metrics[cache_name] = {
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                        "cache_type": cache_name,
                    }
                except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                    logger.error(
                        "Could not get metrics for %s: %s", cache_name, e)
                    unified_metrics[cache_name] = {
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                        "cache_type": cache_name,
                    }

            return unified_metrics

        except CacheError as e:
            logger.error("Cache error collecting unified cache metrics: %s", e)
            return {}
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Error collecting unified cache metrics: %s", e)
            return {}

    def _calculate_performance_grade(self, cache_stats: dict[str, Any]) -> str:
        """
        Calculate performance grade based on cache statistics.

        Args:
            cache_stats: Cache statistics

        Returns:
            str: Performance grade (A, B, C, D, F)
        """
        try:
            hit_rate = cache_stats.get("hit_rate", 0)

            if hit_rate >= 0.9:
                return "A (Excellent)"
            elif hit_rate >= 0.8:
                return "B (Good)"
            elif hit_rate >= 0.7:
                return "C (Fair)"
            elif hit_rate >= 0.5:
                return "D (Poor)"
            else:
                return "F (Critical)"
        except (ValueError, TypeError, AttributeError) as e:
            logger.error("Error calculating performance grade: %s", e)
            return "Unknown"

    def _generate_cache_recommendations(self, cache_name: str, cache_stats: dict[str, Any]) -> list[str]:
        """
        Generate optimization recommendations for cache.

        Args:
            cache_name: Name of the cache
            cache_stats: Cache statistics

        Returns:
            list: List of recommendations
        """
        recommendations = []

        try:
            hit_rate = cache_stats.get("hit_rate", 0)
            cache_size = cache_stats.get("cache_size", 0)
            memory_usage = cache_stats.get("memory_usage", 0)

            # Hit rate recommendations
            if hit_rate < 0.5:
                recommendations.append(
                    "Hit rate is critically low. Consider increasing cache size or reviewing cache strategy.")
            elif hit_rate < 0.7:
                recommendations.append(
                    "Hit rate could be improved. Consider optimizing cache keys or increasing TTL.")

            # Cache size recommendations
            if cache_size == 0:
                recommendations.append(
                    "Cache appears to be empty. Verify cache is being used properly.")
            elif cache_size > 10000:
                recommendations.append(
                    "Cache size is large. Consider implementing cache eviction policies.")

            # Memory usage recommendations
            if memory_usage > 100:  # MB
                recommendations.append(
                    "High memory usage detected. Consider reducing cache size or implementing compression.")

            # Cache-specific recommendations
            if cache_name == "embedding_cache" and hit_rate < 0.8:
                recommendations.append(
                    "Embedding cache hit rate is low. Consider caching embeddings for longer periods.")
            elif cache_name == "verification_cache" and hit_rate < 0.7:
                recommendations.append(
                    "Verification cache hit rate is low. Consider caching verification results more aggressively.")
            elif cache_name == "temporal_cache" and hit_rate < 0.6:
                recommendations.append(
                    "Temporal cache hit rate is low. Consider adjusting temporal analysis caching strategy.")

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(
                "Error generating recommendations for %s: %s", cache_name, e)

        return recommendations

    async def get_cache_metrics(self) -> dict[str, Any]:
        """
        Get comprehensive cache metrics.

        Returns:
            dict: Cache metrics
        """
        try:
            return await self._collect_unified_cache_metrics()
        except CacheError as e:
            logger.error("Cache error getting cache metrics: %s", e)
            return {"error": str(e)}
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Error getting cache metrics: %s", e)
            return {"error": str(e)}

    async def optimize_cache_settings(self) -> dict[str, Any]:
        """
        Optimize cache settings based on current performance.

        Returns:
            dict: Optimization results
        """
        try:
            metrics = await self._collect_unified_cache_metrics()
            optimizations = {}

            for cache_name, cache_stats in metrics.items():
                if "error" in cache_stats:
                    continue

                hit_rate = cache_stats.get("hit_rate", 0)
                cache_size = cache_stats.get("cache_size", 0)

                # Suggest optimizations
                if hit_rate < 0.7 and cache_size < 1000:
                    optimizations[cache_name] = {
                        "action": "increase_size",
                        "current_size": cache_size,
                        "suggested_size": min(cache_size * 2, 5000),
                        "reason": "Low hit rate with small cache size"
                    }
                elif hit_rate > 0.95 and cache_size > 5000:
                    optimizations[cache_name] = {
                        "action": "decrease_size",
                        "current_size": cache_size,
                        "suggested_size": max(cache_size // 2, 1000),
                        "reason": "High hit rate with large cache size"
                    }

            return {
                "status": "completed",
                "optimizations": optimizations,
                "timestamp": datetime.now().isoformat()
            }

        except CacheError as e:
            logger.error("Cache error optimizing cache settings: %s", e)
            return {"error": str(e)}
        except (ValueError, TypeError, AttributeError, RuntimeError, KeyError) as e:
            logger.error("Error optimizing cache settings: %s", e)
            return {"error": str(e)}

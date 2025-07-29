"""
Cache monitoring and performance analysis for Veritas system.

Provides comprehensive monitoring of cache performance, hit rates, and optimization recommendations.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any

from .intelligent_cache import get_embedding_cache, get_verification_cache

logger = logging.getLogger(__name__)


class CacheMonitor:
    """Monitors cache performance and provides optimization insights."""

    def __init__(self):
        self.embedding_cache = get_embedding_cache()
        self.verification_cache = get_verification_cache()
        self._monitoring_data: dict[str, list[dict[str, Any]]] = {
            "embedding_cache": [],
            "verification_cache": [],
        }
        self._start_time = time.time()
        self._monitoring_active = False

    async def collect_cache_metrics(self) -> dict[str, dict[str, Any]]:
        """Collect comprehensive metrics from all caches."""
        metrics = {}

        # Collect embedding cache metrics
        embedding_stats = await self.embedding_cache.get_stats()
        metrics["embedding_cache"] = {
            **embedding_stats,
            "timestamp": datetime.now().isoformat(),
            "cache_type": "embedding",
        }

        # Collect verification cache metrics
        verification_stats = await self.verification_cache.get_stats()
        metrics["verification_cache"] = {
            **verification_stats,
            "timestamp": datetime.now().isoformat(),
            "cache_type": "verification",
        }

        # Store metrics for trend analysis
        for cache_name, cache_metrics in metrics.items():
            self._monitoring_data[cache_name].append(cache_metrics)

            # Keep only last 100 measurements to prevent memory bloat
            if len(self._monitoring_data[cache_name]) > 100:
                self._monitoring_data[cache_name] = self._monitoring_data[cache_name][
                    -100:
                ]

        return metrics

    async def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        metrics = await self.collect_cache_metrics()

        report = "=== VERITAS CACHE PERFORMANCE REPORT ===\n"
        report += f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Monitoring duration: {(time.time() - self._start_time) / 3600:.2f} hours\n\n"

        for cache_name, cache_stats in metrics.items():
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

            # Recommendations
            recommendations = self._generate_recommendations(cache_name, cache_stats)
            if recommendations:
                report += "  Recommendations:\n"
                for rec in recommendations:
                    report += f"    - {rec}\n"

            report += "\n"

        # Overall system recommendations
        system_recommendations = self._generate_system_recommendations(metrics)
        if system_recommendations:
            report += "--- SYSTEM-WIDE RECOMMENDATIONS ---\n"
            for rec in system_recommendations:
                report += f"  - {rec}\n"

        return report

    def _calculate_performance_grade(self, cache_stats: dict[str, Any]) -> str:
        """Calculate performance grade based on cache metrics."""
        hit_rate = cache_stats.get("hit_rate", 0)

        if hit_rate >= 0.9:
            return "A+ (Excellent)"
        elif hit_rate >= 0.8:
            return "A (Very Good)"
        elif hit_rate >= 0.7:
            return "B (Good)"
        elif hit_rate >= 0.6:
            return "C (Fair)"
        elif hit_rate >= 0.5:
            return "D (Poor)"
        else:
            return "F (Critical)"

    def _generate_recommendations(
        self, cache_name: str, cache_stats: dict[str, Any]
    ) -> list[str]:
        """Generate optimization recommendations for specific cache."""
        recommendations = []

        hit_rate = cache_stats.get("hit_rate", 0)
        cache_size = cache_stats.get("cache_size", 0)
        total_requests = cache_stats.get("total_requests", 0)

        # Hit rate recommendations
        if hit_rate < 0.6:
            recommendations.append(
                "Low hit rate detected. Consider increasing cache size or TTL values."
            )

        if hit_rate < 0.4:
            recommendations.append(
                "Critical hit rate. Review caching strategy and key patterns."
            )

        # Cache size recommendations
        if cache_size > 10000:
            recommendations.append(
                "Large cache size. Consider implementing more aggressive eviction policies."
            )

        if cache_size < 100 and total_requests > 1000:
            recommendations.append(
                "Cache size may be too small for request volume. Consider increasing capacity."
            )

        # Cache-specific recommendations
        if cache_name == "embedding_cache":
            if hit_rate < 0.7:
                recommendations.append(
                    "Consider implementing semantic similarity search for embeddings."
                )

        elif cache_name == "verification_cache":
            if hit_rate < 0.8:
                recommendations.append(
                    "Verification results should have high reuse. Review dependency tracking."
                )

        return recommendations

    def _generate_system_recommendations(
        self, metrics: dict[str, dict[str, Any]]
    ) -> list[str]:
        """Generate system-wide optimization recommendations."""
        recommendations = []

        # Calculate average hit rate across all caches
        total_hit_rate = 0
        cache_count = 0

        for cache_stats in metrics.values():
            if "hit_rate" in cache_stats:
                total_hit_rate += cache_stats["hit_rate"]
                cache_count += 1

        avg_hit_rate = total_hit_rate / cache_count if cache_count > 0 else 0

        if avg_hit_rate < 0.7:
            recommendations.append(
                "Overall cache performance is below optimal. Consider cache warming strategies."
            )

        if avg_hit_rate < 0.5:
            recommendations.append(
                "Critical: System-wide cache performance is poor. Review caching architecture."
            )

        # Memory usage recommendations
        total_memory = sum(
            cache_stats.get("memory_usage", 0) for cache_stats in metrics.values()
        )

        if total_memory > 500:  # MB
            recommendations.append(
                "High memory usage detected. Consider implementing cache compression."
            )

        return recommendations

    async def get_cache_trends(
        self, cache_name: str, hours: int = 24
    ) -> dict[str, Any]:
        """Get performance trends for a specific cache."""
        if cache_name not in self._monitoring_data:
            return {}

        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_data = [
            data
            for data in self._monitoring_data[cache_name]
            if datetime.fromisoformat(data["timestamp"]) > cutoff_time
        ]

        if not recent_data:
            return {}

        # Calculate trends
        hit_rates = [data.get("hit_rate", 0) for data in recent_data]
        cache_sizes = [data.get("cache_size", 0) for data in recent_data]

        return {
            "data_points": len(recent_data),
            "avg_hit_rate": sum(hit_rates) / len(hit_rates) if hit_rates else 0,
            "hit_rate_trend": (
                "improving"
                if hit_rates[-1] > hit_rates[0]
                else "declining" if hit_rates else "stable"
            ),
            "avg_cache_size": sum(cache_sizes) / len(cache_sizes) if cache_sizes else 0,
            "size_trend": (
                "growing"
                if cache_sizes[-1] > cache_sizes[0]
                else "shrinking" if cache_sizes else "stable"
            ),
        }

    async def optimize_cache_settings(self, cache_name: str) -> dict[str, Any]:
        """Suggest optimal cache settings based on usage patterns."""
        trends = await self.get_cache_trends(cache_name)

        if not trends:
            return {"error": "Insufficient data for optimization"}

        recommendations = {}

        # TTL recommendations
        if trends.get("hit_rate_trend") == "declining":
            recommendations["ttl"] = (
                "Consider increasing TTL values to improve hit rates"
            )
        elif trends.get("avg_hit_rate", 0) > 0.9:
            recommendations["ttl"] = (
                "Hit rate is excellent, current TTL settings are optimal"
            )

        # Size recommendations
        if (
            trends.get("size_trend") == "growing"
            and trends.get("avg_hit_rate", 0) < 0.7
        ):
            recommendations["size"] = (
                "Cache is growing but hit rate is low, review eviction policy"
            )

        # Strategy recommendations
        avg_hit_rate = trends.get("avg_hit_rate", 0)
        if avg_hit_rate < 0.6:
            recommendations["strategy"] = (
                "Consider switching to SIMILARITY strategy for better performance"
            )

        return recommendations

    async def start_monitoring(self, interval_seconds: int = 300):
        """Start continuous monitoring with specified interval."""
        logger.info(f"Starting cache monitoring with {interval_seconds}s interval")
        self._monitoring_active = True

        while self._monitoring_active:
            try:
                await self.collect_cache_metrics()
                logger.debug("Cache metrics collected successfully")
            except Exception as e:
                logger.error(f"Error collecting cache metrics: {e}")

            await asyncio.sleep(interval_seconds)

    async def stop_monitoring(self):
        """Stop continuous monitoring."""
        logger.info("Stopping cache monitoring")
        self._monitoring_active = False


# Global cache monitor instance
_cache_monitor: CacheMonitor | None = None


def get_cache_monitor() -> CacheMonitor:
    """Get global cache monitor instance."""
    global _cache_monitor
    if _cache_monitor is None:
        _cache_monitor = CacheMonitor()
    return _cache_monitor
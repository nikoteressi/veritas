"""
Adaptive thresholds for dynamic relevance scoring calibration.

Automatically adjusts relevance thresholds based on query patterns, source types, and performance metrics.
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Any

from ..cache.intelligent_cache import IntelligentCache

logger = logging.getLogger(__name__)


class AdaptiveThresholds:
    """Manages dynamic threshold calibration for relevance scoring."""

    def __init__(self):
        self.cache = IntelligentCache(max_memory_size=500)

        # Default thresholds
        self.default_thresholds = {
            "relevance_threshold": 0.05,
            "high_confidence_threshold": 0.7,
            "low_confidence_threshold": 0.3,
            "source_quality_threshold": 0.6,
        }

        # Performance tracking
        self._performance_history: list[dict[str, Any]] = []
        self._calibration_data: dict[str, list[float]] = {
            "precision_scores": [],
            "recall_scores": [],
            "f1_scores": [],
            "source_retention_rates": [],
        }

    async def get_adaptive_threshold(
        self,
        query_type: str = "general",
        source_type: str = "web",
        context: dict[str, Any] | None = None,
    ) -> float:
        """Get dynamically calibrated relevance threshold."""

        cache_key = f"threshold:{query_type}:{source_type}"

        # Try to get cached threshold
        cached_threshold = await self.cache.get(cache_key)
        if cached_threshold is not None:
            return cached_threshold

        # Calculate adaptive threshold
        base_threshold = self.default_thresholds["relevance_threshold"]

        # Adjust based on query type
        query_adjustment = self._get_query_type_adjustment(query_type)

        # Adjust based on source type
        source_adjustment = self._get_source_type_adjustment(source_type)

        # Adjust based on recent performance
        performance_adjustment = await self._get_performance_adjustment()

        # Adjust based on context
        context_adjustment = self._get_context_adjustment(context)

        # Calculate final threshold
        adaptive_threshold = (
            base_threshold + query_adjustment + source_adjustment + performance_adjustment + context_adjustment
        )

        # Ensure threshold is within reasonable bounds
        adaptive_threshold = max(0.01, min(0.9, adaptive_threshold))

        # Cache the threshold
        await self.cache.set(
            cache_key,
            adaptive_threshold,
            ttl_seconds=3600,  # Cache for 1 hour
            dependencies=["performance_data", f"query_type:{query_type}"],
        )

        logger.debug(f"Adaptive threshold for {query_type}/{source_type}: {adaptive_threshold:.4f}")
        return adaptive_threshold

    def _get_query_type_adjustment(self, query_type: str) -> float:
        """Get threshold adjustment based on query type."""
        adjustments = {
            # Lower threshold for factual queries (more inclusive)
            "factual": -0.02,
            # Higher threshold for opinion queries (more selective)
            "opinion": 0.05,
            "temporal": -0.01,  # Slightly lower for temporal queries
            "causal": 0.02,  # Slightly higher for causal queries
            "domain_specific": -0.03,  # Lower for domain-specific queries
            "general": 0.0,  # No adjustment for general queries
        }
        return adjustments.get(query_type, 0.0)

    def _get_source_type_adjustment(self, source_type: str) -> float:
        """Get threshold adjustment based on source type."""
        adjustments = {
            # Lower threshold for academic sources (more inclusive)
            "academic": -0.03,
            "news": 0.01,  # Slightly higher for news sources
            "legal": -0.02,  # Lower for legal documents
            "government": -0.02,  # Lower for government sources
            # Much higher for social media (more selective)
            "social_media": 0.08,
            "blog": 0.05,  # Higher for blog posts
            "wiki": 0.02,  # Slightly higher for wiki sources
            "web": 0.0,  # No adjustment for general web sources
        }
        return adjustments.get(source_type, 0.0)

    async def _get_performance_adjustment(self) -> float:
        """Get threshold adjustment based on recent performance metrics."""
        if not self._calibration_data["precision_scores"]:
            return 0.0

        # Get recent performance data (last 50 measurements)
        recent_precision = self._calibration_data["precision_scores"][-50:]
        recent_recall = self._calibration_data["recall_scores"][-50:]
        recent_retention = self._calibration_data["source_retention_rates"][-50:]

        if not recent_precision:
            return 0.0

        avg_precision = statistics.mean(recent_precision)
        avg_recall = statistics.mean(recent_recall)
        avg_retention = statistics.mean(recent_retention) if recent_retention else 0.5

        # If precision is low, lower threshold (be more inclusive)
        if avg_precision < 0.5:
            precision_adjustment = -0.03
        elif avg_precision > 0.8:
            precision_adjustment = 0.02
        else:
            precision_adjustment = 0.0

        # If recall is low, lower threshold (be more inclusive)
        if avg_recall < 0.4:
            recall_adjustment = -0.04
        elif avg_recall > 0.7:
            recall_adjustment = 0.01
        else:
            recall_adjustment = 0.0

        # If source retention is too low, lower threshold
        if avg_retention < 0.3:
            retention_adjustment = -0.05
        elif avg_retention > 0.8:
            retention_adjustment = 0.02
        else:
            retention_adjustment = 0.0

        total_adjustment = precision_adjustment + recall_adjustment + retention_adjustment

        logger.debug(
            f"Performance adjustment: {total_adjustment:.4f} (P:{avg_precision:.3f}, R:{avg_recall:.3f}, Ret:{avg_retention:.3f})"
        )
        return total_adjustment

    def _get_context_adjustment(self, context: dict[str, Any] | None) -> float:
        """Get threshold adjustment based on context information."""
        if not context:
            return 0.0

        adjustment = 0.0

        # Adjust based on query complexity
        query_length = context.get("query_length", 0)
        if query_length > 100:  # Long, complex queries
            adjustment -= 0.02  # Lower threshold for complex queries
        elif query_length < 20:  # Short queries
            adjustment += 0.01  # Higher threshold for short queries

        # Adjust based on urgency
        urgency = context.get("urgency", "normal")
        if urgency == "high":
            adjustment -= 0.03  # Lower threshold for urgent queries
        elif urgency == "low":
            adjustment += 0.02  # Higher threshold for non-urgent queries

        # Adjust based on domain expertise required
        expertise_level = context.get("expertise_level", "general")
        if expertise_level == "expert":
            adjustment -= 0.02  # Lower threshold for expert-level queries
        elif expertise_level == "basic":
            adjustment += 0.01  # Higher threshold for basic queries

        return adjustment

    async def record_performance_metrics(
        self,
        precision: float,
        recall: float,
        f1_score: float,
        source_retention_rate: float,
        query_type: str = "general",
        source_type: str = "web",
    ):
        """Record performance metrics for threshold calibration."""

        # Store metrics
        self._calibration_data["precision_scores"].append(precision)
        self._calibration_data["recall_scores"].append(recall)
        self._calibration_data["f1_scores"].append(f1_score)
        self._calibration_data["source_retention_rates"].append(source_retention_rate)

        # Keep only last 1000 measurements to prevent memory bloat
        for metric_list in self._calibration_data.values():
            if len(metric_list) > 1000:
                metric_list[:] = metric_list[-1000:]

        # Store performance record
        performance_record = {
            "timestamp": datetime.now().isoformat(),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "source_retention_rate": source_retention_rate,
            "query_type": query_type,
            "source_type": source_type,
        }

        self._performance_history.append(performance_record)

        # Keep only last 500 records
        if len(self._performance_history) > 500:
            self._performance_history = self._performance_history[-500:]

        # Invalidate cached thresholds to force recalibration
        await self.cache.invalidate_dependencies("performance_data")

        logger.debug(
            f"Recorded performance: P={precision:.3f}, R={recall:.3f}, F1={f1_score:.3f}, Retention={source_retention_rate:.3f}"
        )

    async def get_threshold_recommendations(self) -> dict[str, Any]:
        """Get recommendations for threshold optimization."""
        if not self._calibration_data["precision_scores"]:
            return {"error": "Insufficient performance data"}

        recent_data = {key: values[-50:] for key, values in self._calibration_data.items()}

        recommendations = {}

        # Analyze precision
        avg_precision = statistics.mean(recent_data["precision_scores"])
        if avg_precision < 0.5:
            recommendations["precision"] = (
                "Low precision detected. Consider raising thresholds or improving scoring algorithm."
            )
        elif avg_precision > 0.9:
            recommendations["precision"] = "Excellent precision. Current thresholds are working well."

        # Analyze recall
        avg_recall = statistics.mean(recent_data["recall_scores"])
        if avg_recall < 0.4:
            recommendations["recall"] = (
                "Low recall detected. Consider lowering thresholds to capture more relevant sources."
            )
        elif avg_recall > 0.8:
            recommendations["recall"] = "Excellent recall. Good balance between inclusivity and selectivity."

        # Analyze source retention
        avg_retention = statistics.mean(recent_data["source_retention_rates"])
        if avg_retention < 0.3:
            recommendations["retention"] = (
                "Critical: Too many sources being filtered out. Significantly lower thresholds needed."
            )
        elif avg_retention > 0.8:
            recommendations["retention"] = (
                "High source retention. Consider slightly raising thresholds for better quality."
            )

        # Overall recommendation
        f1_scores = recent_data["f1_scores"]
        if f1_scores:
            avg_f1 = statistics.mean(f1_scores)
            if avg_f1 < 0.5:
                recommendations["overall"] = "Poor overall performance. Review entire relevance scoring system."
            elif avg_f1 > 0.7:
                recommendations["overall"] = "Good overall performance. Fine-tune for specific use cases."

        return recommendations

    async def calibrate_thresholds_batch(
        self, test_queries: list[dict[str, Any]], ground_truth: list[list[str]]
    ) -> dict[str, float]:
        """Calibrate thresholds using batch testing with ground truth data."""

        best_thresholds = {}
        best_f1_score = 0.0

        # Test different threshold values
        threshold_range = [i * 0.01 for i in range(1, 91)]  # 0.01 to 0.90

        for threshold in threshold_range:
            total_precision = 0.0
            total_recall = 0.0
            valid_tests = 0

            for query_data, expected_sources in zip(test_queries, ground_truth, strict=False):
                # Simulate relevance scoring with this threshold
                # This would integrate with actual scoring system
                predicted_sources = await self._simulate_scoring(query_data, threshold)

                if predicted_sources and expected_sources:
                    precision = len(set(predicted_sources) & set(expected_sources)) / len(predicted_sources)
                    recall = len(set(predicted_sources) & set(expected_sources)) / len(expected_sources)

                    total_precision += precision
                    total_recall += recall
                    valid_tests += 1

            if valid_tests > 0:
                avg_precision = total_precision / valid_tests
                avg_recall = total_recall / valid_tests
                f1_score = (
                    2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
                    if (avg_precision + avg_recall) > 0
                    else 0
                )

                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    best_thresholds = {
                        "relevance_threshold": threshold,
                        "precision": avg_precision,
                        "recall": avg_recall,
                        "f1_score": f1_score,
                    }

        logger.info(
            f"Calibration complete. Best threshold: {best_thresholds.get('relevance_threshold', 0.05):.3f} (F1: {best_f1_score:.3f})"
        )
        return best_thresholds

    async def _simulate_scoring(self, query_data: dict[str, Any], threshold: float) -> list[str]:
        """Simulate relevance scoring for calibration purposes."""
        # This is a placeholder - in real implementation, this would use the actual scoring system
        # For now, return empty list
        return []

    async def get_performance_summary(self, days: int = 7) -> dict[str, Any]:
        """Get performance summary for the specified number of days."""
        cutoff_date = datetime.now() - timedelta(days=days)

        recent_records = [
            record for record in self._performance_history if datetime.fromisoformat(record["timestamp"]) > cutoff_date
        ]

        if not recent_records:
            return {"error": "No performance data available for the specified period"}

        # Calculate averages
        avg_precision = statistics.mean([r["precision"] for r in recent_records])
        avg_recall = statistics.mean([r["recall"] for r in recent_records])
        avg_f1 = statistics.mean([r["f1_score"] for r in recent_records])
        avg_retention = statistics.mean([r["source_retention_rate"] for r in recent_records])

        # Calculate trends
        precision_trend = self._calculate_trend([r["precision"] for r in recent_records])
        recall_trend = self._calculate_trend([r["recall"] for r in recent_records])

        return {
            "period_days": days,
            "total_measurements": len(recent_records),
            "averages": {
                "precision": avg_precision,
                "recall": avg_recall,
                "f1_score": avg_f1,
                "source_retention_rate": avg_retention,
            },
            "trends": {"precision": precision_trend, "recall": recall_trend},
            "performance_grade": self._calculate_performance_grade(avg_f1),
        }

    def _calculate_trend(self, values: list[float]) -> str:
        """Calculate trend direction for a list of values."""
        if len(values) < 2:
            return "insufficient_data"

        first_half = values[: len(values) // 2]
        second_half = values[len(values) // 2 :]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        if second_avg > first_avg * 1.05:
            return "improving"
        elif second_avg < first_avg * 0.95:
            return "declining"
        else:
            return "stable"

    def _calculate_performance_grade(self, f1_score: float) -> str:
        """Calculate performance grade based on F1 score."""
        if f1_score >= 0.8:
            return "A"
        elif f1_score >= 0.7:
            return "B"
        elif f1_score >= 0.6:
            return "C"
        elif f1_score >= 0.5:
            return "D"
        else:
            return "F"


# Global adaptive thresholds instance
_adaptive_thresholds: AdaptiveThresholds | None = None


def get_adaptive_thresholds() -> AdaptiveThresholds:
    """Get global adaptive thresholds instance."""
    global _adaptive_thresholds
    if _adaptive_thresholds is None:
        _adaptive_thresholds = AdaptiveThresholds()
    return _adaptive_thresholds

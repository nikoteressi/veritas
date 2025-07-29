"""
Temporal analysis cache for time-aware relevance scoring.

Provides temporal relevance analysis considering recency, trends, and time-sensitive content
with intelligent caching for improved performance.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from dateutil import parser as date_parser

from .intelligent_cache import CacheStrategy, IntelligentCache

logger = logging.getLogger(__name__)


class TemporalAnalysisCache:
    """Temporal analysis with intelligent caching for time-aware relevance."""

    def __init__(
        self,
        cache_size: int = 500,
        default_decay_rate: float = 0.1,
        trend_window_days: int = 30,
        recency_weight: float = 0.3,
        trend_weight: float = 0.4,
        content_weight: float = 0.3,
    ):
        """
        Initialize temporal analysis cache.

        Args:
            cache_size: Maximum cache size
            default_decay_rate: Default temporal decay rate (0-1)
            trend_window_days: Window for trend analysis in days
            recency_weight: Weight for recency factor (0-1)
            trend_weight: Weight for trend factor (0-1)
            content_weight: Weight for content relevance (0-1)
        """
        self.cache = IntelligentCache(max_memory_size=cache_size)
        self.default_decay_rate = default_decay_rate
        self.trend_window_days = trend_window_days

        # Normalize weights
        total_weight = recency_weight + trend_weight + content_weight
        if total_weight > 0:
            self.recency_weight = recency_weight / total_weight
            self.trend_weight = trend_weight / total_weight
            self.content_weight = content_weight / total_weight
        else:
            self.recency_weight = 0.33
            self.trend_weight = 0.33
            self.content_weight = 0.34

        # Performance metrics
        self.metrics = {
            "total_analyses": 0,
            "cache_hits": 0,
            "recency_calculations": 0,
            "trend_calculations": 0,
            "temporal_adjustments": 0,
            "processing_time": 0.0,
        }

        logger.info("Temporal analysis cache initialized")

    def _generate_cache_key(self, query: str, time_window: str, analysis_type: str = "temporal") -> str:
        """Generate cache key for temporal analysis."""
        combined = f"{query}|{time_window}|{analysis_type}"
        text_hash = hashlib.md5(combined.encode("utf-8")).hexdigest()
        return f"temporal:{analysis_type}:{text_hash}"

    def _parse_date(self, date_str: str) -> datetime | None:
        """Parse date string to datetime object."""
        try:
            if isinstance(date_str, datetime):
                return date_str

            # Try common date formats
            return date_parser.parse(date_str)

        except Exception as e:
            logger.debug(f"Failed to parse date '{date_str}': {e}")
            return None

    def calculate_recency_score(
        self,
        content_date: str | datetime,
        reference_date: datetime | None = None,
        decay_rate: float | None = None,
    ) -> float:
        """
        Calculate recency score based on content age.

        Args:
            content_date: Date of the content
            reference_date: Reference date (default: now)
            decay_rate: Temporal decay rate (default: class default)

        Returns:
            Recency score (0-1)
        """
        try:
            if reference_date is None:
                reference_date = datetime.now()

            if decay_rate is None:
                decay_rate = self.default_decay_rate

            # Parse content date
            parsed_date = self._parse_date(content_date)
            if parsed_date is None:
                return 0.5  # Neutral score for unparseable dates

            # Calculate age in days
            age_delta = reference_date - parsed_date
            age_days = age_delta.total_seconds() / (24 * 3600)

            # Handle future dates
            if age_days < 0:
                return 1.0  # Future content gets maximum recency

            # Exponential decay
            recency_score = np.exp(-decay_rate * age_days)
            return max(min(recency_score, 1.0), 0.0)

        except Exception as e:
            logger.error(f"Failed to calculate recency score: {e}")
            return 0.5

    def calculate_trend_score(
        self,
        query: str,
        content_dates: list[str | datetime],
        reference_date: datetime | None = None,
    ) -> float:
        """
        Calculate trend score based on content frequency over time.

        Args:
            query: Search query
            content_dates: List of content dates
            reference_date: Reference date (default: now)

        Returns:
            Trend score (0-1)
        """
        try:
            if reference_date is None:
                reference_date = datetime.now()

            if not content_dates:
                return 0.0

            # Parse dates
            parsed_dates = []
            for date_str in content_dates:
                parsed_date = self._parse_date(date_str)
                if parsed_date:
                    parsed_dates.append(parsed_date)

            if not parsed_dates:
                return 0.0

            # Filter dates within trend window
            window_start = reference_date - timedelta(days=self.trend_window_days)
            recent_dates = [d for d in parsed_dates if d >= window_start]

            if not recent_dates:
                return 0.0

            # Calculate trend metrics
            total_content = len(recent_dates)
            days_with_content = len(set(d.date() for d in recent_dates))

            # Frequency score
            # Normalize to 10 items
            frequency_score = min(total_content / 10.0, 1.0)

            # Consistency score
            consistency_score = days_with_content / min(self.trend_window_days, 30)

            # Recent activity boost
            recent_week = reference_date - timedelta(days=7)
            recent_activity = len([d for d in recent_dates if d >= recent_week])
            activity_boost = min(recent_activity / 5.0, 0.3)  # Max 30% boost

            # Combine scores
            trend_score = frequency_score * 0.4 + consistency_score * 0.4 + activity_boost * 0.2
            return max(min(trend_score, 1.0), 0.0)

        except Exception as e:
            logger.error(f"Failed to calculate trend score: {e}")
            return 0.0

    async def analyze_temporal_relevance(
        self,
        query: str,
        content: str,
        content_date: str | datetime,
        base_relevance: float,
        time_window: str = "default",
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """
        Analyze temporal relevance of content.

        Args:
            query: Search query
            content: Content text
            content_date: Date of the content
            base_relevance: Base relevance score
            time_window: Time window identifier
            use_cache: Whether to use caching

        Returns:
            Temporal analysis results
        """
        start_time = time.time()
        self.metrics["total_analyses"] += 1

        cache_key = self._generate_cache_key(query, time_window, "relevance")

        # Try cache first
        if use_cache:
            cached_result = await self.cache.get(cache_key)
            if cached_result is not None:
                self.metrics["cache_hits"] += 1
                return cached_result

        # Calculate temporal factors
        recency_score = self.calculate_recency_score(content_date)
        self.metrics["recency_calculations"] += 1

        # For trend analysis, we need historical data
        # In a real implementation, this would query a database
        trend_score = 0.5  # Placeholder - would need historical content dates
        self.metrics["trend_calculations"] += 1

        # Calculate temporal adjustment
        temporal_factor = (
            self.recency_weight * recency_score
            + self.trend_weight * trend_score
            + self.content_weight * 1.0  # Content relevance is already factored in base_relevance
        )

        # Apply temporal adjustment
        adjusted_relevance = base_relevance * temporal_factor
        self.metrics["temporal_adjustments"] += 1

        # Create analysis result
        analysis = {
            "query": query,
            "content_date": str(content_date),
            "base_relevance": base_relevance,
            "recency_score": recency_score,
            "trend_score": trend_score,
            "temporal_factor": temporal_factor,
            "adjusted_relevance": adjusted_relevance,
            "weights": {
                "recency": self.recency_weight,
                "trend": self.trend_weight,
                "content": self.content_weight,
            },
            "timestamp": time.time(),
        }

        # Cache result
        if use_cache:
            dependencies = [
                f"query:{hashlib.md5(query.encode()).hexdigest()[:8]}",
                f"time_window:{time_window}",
            ]

            await self.cache.set(
                cache_key,
                analysis,
                ttl_seconds=1800,  # 30 minutes
                level=CacheStrategy.LRU,
                dependencies=dependencies,
            )

        # Update metrics
        self.metrics["processing_time"] += time.time() - start_time

        return analysis

    async def batch_temporal_analysis(
        self,
        query: str,
        content_items: list[dict[str, Any]],
        time_window: str = "default",
    ) -> list[dict[str, Any]]:
        """
        Perform temporal analysis on multiple content items.

        Args:
            query: Search query
            content_items: List of content items with 'content', 'date', 'relevance' keys
            time_window: Time window identifier

        Returns:
            List of temporal analysis results
        """
        tasks = []
        for item in content_items:
            task = self.analyze_temporal_relevance(
                query=query,
                content=item.get("content", ""),
                content_date=item.get("date", datetime.now()),
                base_relevance=item.get("relevance", 0.0),
                time_window=time_window,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

    def calculate_time_decay(
        self,
        content_date: str | datetime,
        half_life_days: float = 30.0,
        reference_date: datetime | None = None,
    ) -> float:
        """
        Calculate time decay using half-life model.

        Args:
            content_date: Date of the content
            half_life_days: Half-life in days
            reference_date: Reference date (default: now)

        Returns:
            Decay factor (0-1)
        """
        try:
            if reference_date is None:
                reference_date = datetime.now()

            parsed_date = self._parse_date(content_date)
            if parsed_date is None:
                return 0.5

            # Calculate age in days
            age_delta = reference_date - parsed_date
            age_days = age_delta.total_seconds() / (24 * 3600)

            if age_days < 0:
                return 1.0  # Future content

            # Half-life decay
            if half_life_days <= 0:
                return 0.5  # Default value for invalid half-life

            decay_factor = 0.5 ** (age_days / half_life_days)
            return max(min(decay_factor, 1.0), 0.0)

        except Exception as e:
            logger.error(f"Failed to calculate time decay: {e}")
            return 0.5

    def update_weights(self, recency_weight: float, trend_weight: float, content_weight: float):
        """Update temporal analysis weights."""
        total_weight = recency_weight + trend_weight + content_weight
        if total_weight > 0:
            self.recency_weight = recency_weight / total_weight
            self.trend_weight = trend_weight / total_weight
            self.content_weight = content_weight / total_weight

            # Clear cache as weights changed
            self.cache.clear()
            logger.info(
                f"Updated temporal weights: Recency={self.recency_weight:.2f}, "
                f"Trend={self.trend_weight:.2f}, Content={self.content_weight:.2f}"
            )

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics."""
        total_analyses = self.metrics["total_analyses"]
        if total_analyses == 0:
            return self.metrics.copy()

        cache_hit_rate = self.metrics["cache_hits"] / total_analyses
        avg_processing_time = self.metrics["processing_time"] / total_analyses

        return {
            **self.metrics,
            "cache_hit_rate": cache_hit_rate,
            "avg_processing_time": avg_processing_time,
        }

    def clear_cache(self):
        """Clear temporal analysis cache."""
        self.cache.clear()
        logger.info("Temporal analysis cache cleared")

    def optimize_cache(self):
        """Optimize cache performance."""
        self.cache.optimize()
        logger.info("Temporal analysis cache optimized")

    async def close(self):
        """Close and cleanup resources."""
        await self.cache.close()
        logger.info("Temporal analysis cache closed")


# Factory function for easy initialization
def create_temporal_analysis_cache(
    cache_size: int = 500,
    recency_weight: float = 0.3,
    trend_weight: float = 0.4,
    content_weight: float = 0.3,
) -> TemporalAnalysisCache:
    """
    Factory function to create temporal analysis cache.

    Args:
        cache_size: Cache size
        recency_weight: Weight for recency factor
        trend_weight: Weight for trend factor
        content_weight: Weight for content relevance

    Returns:
        TemporalAnalysisCache instance
    """
    return TemporalAnalysisCache(
        cache_size=cache_size,
        recency_weight=recency_weight,
        trend_weight=trend_weight,
        content_weight=content_weight,
    )

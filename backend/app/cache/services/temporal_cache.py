"""
Temporal analysis cache service.

Specialized cache for temporal analysis with time-aware relevance scoring,
trend analysis, and time-based invalidation.
"""
import logging
import time
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from app.exceptions import CacheError
from app.cache.core import CacheManager
from .base_cache_service import CacheServiceInterface

logger = logging.getLogger(__name__)


class TemporalCache(CacheServiceInterface):
    """
    Specialized cache for temporal analysis.

    Features:
    - Time-aware relevance scoring
    - Trend analysis caching
    - Time-based invalidation
    - Recency scoring
    - Temporal pattern recognition
    """

    def __init__(self, cache_manager: CacheManager):
        """
        Initialize temporal cache.

        Args:
            cache_manager: Unified cache manager instance
        """
        self.cache_manager = cache_manager
        self._cache_type = 'temporal'

        # Temporal analysis parameters
        self.recency_decay_factor = 0.1  # How quickly relevance decays over time
        self.trend_window_hours = 24     # Window for trend analysis
        self.max_age_days = 30           # Maximum age for cached temporal data

    @property
    def cache_type(self) -> str:
        """Return the cache type."""
        return self._cache_type

    async def get_temporal_relevance(
        self,
        content: str,
        query_time: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get temporal relevance score for content.

        Args:
            content: Content to analyze
            query_time: Time of query (uses current time if None)

        Returns:
            Temporal relevance data or None if not cached
        """
        effective_query_time = query_time or datetime.now()
        key = self._generate_temporal_key(content, 'relevance')

        cached_data = await self.cache_manager.get(key, self._cache_type)

        if cached_data:
            # Calculate current relevance based on time decay
            current_relevance = self._calculate_time_adjusted_relevance(
                cached_data, effective_query_time
            )

            return {
                **cached_data,
                'current_relevance': current_relevance,
                'query_time': effective_query_time.isoformat()
            }

        return None

    async def set_temporal_relevance(
        self,
        content: str,
        base_relevance: float,
        content_timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store temporal relevance data for content.

        Args:
            content: Content to store relevance for
            base_relevance: Base relevance score
            content_timestamp: When the content was created
            metadata: Additional metadata
            ttl: Time to live in seconds

        Returns:
            True if successful, False otherwise
        """
        key = self._generate_temporal_key(content, 'relevance')

        temporal_data = {
            'content': content,
            'base_relevance': base_relevance,
            'content_timestamp': content_timestamp.isoformat(),
            'cached_at': datetime.now().isoformat(),
            'metadata': metadata or {},
            'access_count': 0,
            'last_accessed': time.time()
        }

        return await self.cache_manager.set(key, temporal_data, ttl, self._cache_type)

    async def get_trend_analysis(
        self,
        topic: str,
        time_window: Optional[timedelta] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get trend analysis for a topic.

        Args:
            topic: Topic to analyze trends for
            time_window: Time window for analysis

        Returns:
            Trend analysis data or None if not cached
        """
        effective_window = time_window or timedelta(
            hours=self.trend_window_hours)
        key = self._generate_temporal_key(topic, 'trend')

        cached_trend = await self.cache_manager.get(key, self._cache_type)

        if cached_trend:
            # Check if trend data is still valid for the requested window
            cached_window = timedelta(
                seconds=cached_trend.get('window_seconds', 0))

            if cached_window >= effective_window:
                return cached_trend

        return None

    async def set_trend_analysis(
        self,
        topic: str,
        trend_data: Dict[str, Any],
        time_window: timedelta,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store trend analysis data.

        Args:
            topic: Topic the trend analysis is for
            trend_data: Trend analysis results
            time_window: Time window used for analysis
            ttl: Time to live in seconds

        Returns:
            True if successful, False otherwise
        """
        key = self._generate_temporal_key(topic, 'trend')

        cached_trend = {
            'topic': topic,
            'trend_data': trend_data,
            'window_seconds': time_window.total_seconds(),
            'analyzed_at': datetime.now().isoformat(),
            'valid_until': (datetime.now() + time_window).isoformat()
        }

        return await self.cache_manager.set(key, cached_trend, ttl, self._cache_type)

    async def get_recency_scores(
        self,
        content_list: List[str],
        reference_time: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Get recency scores for multiple content items.

        Args:
            content_list: List of content to score
            reference_time: Reference time for scoring

        Returns:
            Dictionary mapping content to recency scores
        """
        effective_ref_time = reference_time or datetime.now()

        # Generate keys for batch retrieval
        keys = [self._generate_temporal_key(
            content, 'relevance') for content in content_list]
        cached_data = await self.cache_manager.get_many(keys, self._cache_type)

        recency_scores = {}

        for content, key in zip(content_list, keys):
            if key in cached_data and cached_data[key]:
                data = cached_data[key]
                score = self._calculate_recency_score(data, effective_ref_time)
                recency_scores[content] = score
            else:
                recency_scores[content] = 0.0

        return recency_scores

    async def update_access_patterns(
        self,
        content: str,
        access_time: Optional[datetime] = None
    ) -> bool:
        """
        Update access patterns for content.

        Args:
            content: Content that was accessed
            access_time: Time of access

        Returns:
            True if successful, False otherwise
        """
        key = self._generate_temporal_key(content, 'relevance')
        cached_data = await self.cache_manager.get(key, self._cache_type)

        if cached_data:
            # Update access information
            cached_data['access_count'] = cached_data.get(
                'access_count', 0) + 1
            cached_data['last_accessed'] = time.time()

            if access_time:
                cached_data['last_access_time'] = access_time.isoformat()

            return await self.cache_manager.set(key, cached_data, cache_type=self._cache_type)

        return False

    async def cleanup(self) -> int:
        """Clean up expired cache entries (implements CacheServiceInterface)."""
        return await self.cleanup_expired_temporal_data()

    async def cleanup_expired_temporal_data(self) -> int:
        """
        Clean up temporal data that has exceeded maximum age.

        Returns:
            Number of cleaned entries
        """
        cutoff_time = datetime.now() - timedelta(days=self.max_age_days)

        # This is a simplified cleanup - in a real implementation,
        # you might want to scan through keys more efficiently
        pattern = "*"

        # For now, we'll rely on TTL-based expiration
        # In a more sophisticated implementation, you could:
        # 1. Maintain an index of temporal data by timestamp
        # 2. Use Redis SCAN to iterate through keys
        # 3. Check timestamps and delete expired entries

        logger.info("Temporal cache cleanup completed (TTL-based)")
        return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache service statistics (implements CacheServiceInterface)."""
        return await self.get_temporal_stats()

    async def get_temporal_stats(self) -> Dict[str, Any]:
        """Get temporal cache statistics."""
        base_stats = self.cache_manager.get_stats()

        temporal_stats = {
            'recency_decay_factor': self.recency_decay_factor,
            'trend_window_hours': self.trend_window_hours,
            'max_age_days': self.max_age_days
        }

        return {
            **base_stats,
            'temporal_stats': temporal_stats
        }

    def _generate_temporal_key(self, content: str, analysis_type: str) -> str:
        """
        Generate cache key for temporal analysis.

        Args:
            content: Content or topic
            analysis_type: Type of temporal analysis

        Returns:
            Cache key
        """
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
        return f"{content_hash}:{analysis_type}"

    def _calculate_time_adjusted_relevance(
        self,
        cached_data: Dict[str, Any],
        query_time: datetime
    ) -> float:
        """
        Calculate relevance adjusted for time decay.

        Args:
            cached_data: Cached temporal data
            query_time: Current query time

        Returns:
            Time-adjusted relevance score
        """
        try:
            base_relevance = cached_data.get('base_relevance', 0.0)
            content_timestamp = datetime.fromisoformat(
                cached_data['content_timestamp'])

            # Calculate time difference in hours
            time_diff = (query_time - content_timestamp).total_seconds() / 3600

            # Apply exponential decay
            decay_factor = max(
                0.0, 1.0 - (time_diff * self.recency_decay_factor / 24))

            return base_relevance * decay_factor

        except CacheError as e:
            logger.error(
                "Cache error calculating time-adjusted relevance: %s", e)
            return 0.0
        except Exception as e:
            logger.error("Error calculating time-adjusted relevance: %s", e)
            return 0.0

    def _calculate_recency_score(
        self,
        cached_data: Dict[str, Any],
        reference_time: datetime
    ) -> float:
        """
        Calculate recency score for content.

        Args:
            cached_data: Cached temporal data
            reference_time: Reference time for scoring

        Returns:
            Recency score (0-1)
        """
        try:
            content_timestamp = datetime.fromisoformat(
                cached_data['content_timestamp'])

            # Calculate hours since content creation
            hours_old = (reference_time -
                         content_timestamp).total_seconds() / 3600

            # Calculate recency score with exponential decay
            # More recent content gets higher scores
            recency_score = max(
                0.0, 1.0 - (hours_old * self.recency_decay_factor / 24))

            # Boost score based on access patterns
            access_count = cached_data.get('access_count', 0)
            access_boost = min(0.2, access_count * 0.01)  # Max 20% boost

            return min(1.0, recency_score + access_boost)

        except CacheError as e:
            logger.error("Cache error calculating recency score: %s", e)
            return 0.0
        except Exception as e:
            logger.error("Error calculating recency score: %s", e)
            return 0.0

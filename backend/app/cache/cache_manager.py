"""
Unified cache manager.

Central cache management system that replaces all existing cache implementations
with a unified, high-performance Redis-based solution.
"""
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set

from redis.exceptions import RedisError

from app.exceptions import CacheError

from .config import CacheConfig, TTL_PRESETS, KEY_PREFIXES, cache_config
from .connection_manager import ConnectionManager
from .serialization_manager import SerializationManager

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Unified cache manager with Redis backend.

    Features:
    - Unified Redis-based caching
    - Intelligent serialization
    - Batch operations
    - Pipeline support
    - Memory cache layer
    - Performance monitoring
    - Health checks
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize cache manager.

        Args:
            config: Cache configuration (uses default if None)
        """
        self.config = config or cache_config
        self.config.validate()

        # Core components
        self.connection_manager = ConnectionManager()
        self.serialization_manager = SerializationManager(
            compression_threshold=self.config.compression_threshold,
            compression_level=self.config.compression_level
        )

        # Performance metrics
        self._metrics = {
            'hits': 0,
            'misses': 0,
            'memory_hits': 0,
            'redis_hits': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0,
            'total_operations': 0
        }

        # Health status
        self._is_healthy = True
        self._last_health_check = 0

        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()

    async def initialize(self) -> None:
        """Initialize cache manager and start background tasks."""
        try:
            # Initialize connection manager
            await self.connection_manager.initialize()

            # Start background tasks
            if self.config.enable_metrics:
                task = asyncio.create_task(self._metrics_collector())
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

            task = asyncio.create_task(self._health_checker())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

            logger.info("Cache manager initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize cache manager: %s", e)
            raise

    async def close(self) -> None:
        """Close cache manager and cleanup resources."""
        try:
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()

            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)

            # Close connection manager
            await self.connection_manager.close()

            logger.info("Cache manager closed successfully")

        except Exception as e:
            raise CacheError(f"Error closing cache manager: {e}") from e

    async def get(self, key: str, cache_type: str = 'default') -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key
            cache_type: Cache type for key prefixing

        Returns:
            Cached value or None if not found
        """
        full_key = self._build_key(key, cache_type)

        try:
            self._metrics['total_operations'] += 1

            # Check Redis cache
            redis_result = await self._get_from_redis(full_key)
            if redis_result is not None:
                self._metrics['hits'] += 1
                self._metrics['redis_hits'] += 1

                return redis_result

            self._metrics['misses'] += 1
            return None

        except (ConnectionError, TimeoutError, ValueError, TypeError) as e:
            self._metrics['errors'] += 1
            logger.error("Error getting cache key %s: %s", full_key, e)
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        cache_type: str = 'default'
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
            cache_type: Cache type for key prefixing and TTL selection

        Returns:
            True if successful, False otherwise
        """
        full_key = self._build_key(key, cache_type)
        effective_ttl = ttl or self._get_ttl_for_type(cache_type)

        try:
            self._metrics['total_operations'] += 1
            self._metrics['sets'] += 1

            # Store in Redis
            success = await self._set_in_redis(full_key, value, effective_ttl)

            if success:
                return True

            return False

        except (ConnectionError, TimeoutError, ValueError, TypeError, OSError) as e:
            self._metrics['errors'] += 1
            logger.error("Error setting cache key %s: %s", full_key, e)
            return False

    async def delete(self, key: str, cache_type: str = 'default') -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key
            cache_type: Cache type for key prefixing

        Returns:
            True if successful, False otherwise
        """
        full_key = self._build_key(key, cache_type)

        try:
            self._metrics['total_operations'] += 1
            self._metrics['deletes'] += 1

            # Remove from Redis
            return await self._delete_from_redis(full_key)

        except (ConnectionError, TimeoutError, ValueError, TypeError, OSError) as e:
            self._metrics['errors'] += 1
            logger.error("Error deleting cache key %s: %s", full_key, e)
            return False

    async def exists(self, key: str, cache_type: str = 'default') -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key
            cache_type: Cache type for key prefixing

        Returns:
            True if key exists, False otherwise
        """
        full_key = self._build_key(key, cache_type)
        # Check Redis
        try:
            redis = await self.connection_manager.get_client()
            return await redis.exists(full_key) > 0
        except (ConnectionError, TimeoutError, ValueError, TypeError, OSError) as e:
            logger.error("Error checking key existence %s: %s", full_key, e)
            return False

    async def clear_by_pattern(self, pattern: str, cache_type: str = 'default') -> int:
        """
        Clear cache entries matching pattern.

        Args:
            pattern: Key pattern (supports wildcards)
            cache_type: Cache type for key prefixing

        Returns:
            Number of deleted keys
        """
        full_pattern = self._build_key(pattern, cache_type)

        try:
            redis = await self.connection_manager.get_client()

            # Find matching keys
            keys = await redis.keys(full_pattern)

            if not keys:
                return 0

            # Remove from Redis
            deleted = await redis.delete(*keys)

            self._metrics['deletes'] += deleted
            return deleted

        except (ConnectionError, TimeoutError, ValueError, TypeError, OSError, RedisError) as e:
            self._metrics['errors'] += 1
            logger.error("Error clearing cache pattern %s: %s",
                         full_pattern, e)
            return 0

    async def get_many(
        self,
        keys: List[str],
        cache_type: str = 'default'
    ) -> Dict[str, Any]:
        """
        Get multiple values from cache.

        Args:
            keys: List of cache keys
            cache_type: Cache type for key prefixing

        Returns:
            Dictionary of key-value pairs
        """
        if not keys:
            return {}

        full_keys = [self._build_key(key, cache_type) for key in keys]
        result = {}

        try:
            # Check memory cache first
            memory_results = {}
            redis_keys = []

            for i, full_key in enumerate(full_keys):
                redis_keys.append((keys[i], full_key))

            # Get remaining keys from Redis
            if redis_keys:
                redis_results = await self._get_many_from_redis(
                    [full_key for _, full_key in redis_keys]
                )

                for (original_key, full_key), value in zip(redis_keys, redis_results):
                    if value is not None:
                        result[original_key] = value
                        self._metrics['redis_hits'] += 1
                    else:
                        self._metrics['misses'] += 1

            # Combine results
            result.update(memory_results)
            self._metrics['hits'] += len(result)
            self._metrics['total_operations'] += len(keys)

            return result

        except (ConnectionError, TimeoutError, ValueError, TypeError, OSError, RedisError) as e:
            self._metrics['errors'] += 1
            logger.error("Error getting multiple cache keys: %s", e)
            return {}

    async def set_many(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None,
        cache_type: str = 'default'
    ) -> bool:
        """
        Set multiple values in cache.

        Args:
            mapping: Dictionary of key-value pairs
            ttl: Time to live in seconds
            cache_type: Cache type for key prefixing

        Returns:
            True if all successful, False otherwise
        """
        if not mapping:
            return True

        effective_ttl = ttl or self._get_ttl_for_type(cache_type)

        try:
            # Prepare data for Redis
            redis_mapping = {}
            for key, value in mapping.items():
                full_key = self._build_key(key, cache_type)
                redis_mapping[full_key] = value

            # Set in Redis
            success = await self._set_many_in_redis(redis_mapping, effective_ttl)

            self._metrics['sets'] += len(mapping)
            self._metrics['total_operations'] += len(mapping)

            return success

        except (ConnectionError, TimeoutError, ValueError, TypeError, OSError, RedisError) as e:
            self._metrics['errors'] += 1
            logger.error("Error setting multiple cache keys: %s", e)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_ops = self._metrics['total_operations']
        hit_rate = (self._metrics['hits'] /
                    total_ops * 100) if total_ops > 0 else 0

        return {
            'metrics': self._metrics.copy(),
            'hit_rate': hit_rate,
            'is_healthy': self._is_healthy,
            'connection_stats': self.connection_manager.get_pool_stats()
        }

    def _build_key(self, key: str, cache_type: str) -> str:
        """Build full cache key with prefix."""
        prefix = KEY_PREFIXES.get(cache_type, '')
        return f"{prefix}{key}"

    def _get_ttl_for_type(self, cache_type: str) -> int:
        """Get TTL for cache type."""
        return TTL_PRESETS.get(cache_type, self.config.default_ttl)

    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        try:
            redis = await self.connection_manager.get_client()
            data = await redis.get(key)

            if data is None:
                return None

            return self.serialization_manager.deserialize(data)

        except (ConnectionError, TimeoutError, ValueError, TypeError, RedisError) as e:
            logger.error("Error getting from Redis key %s: %s", key, e)
            return None

    async def _set_in_redis(self, key: str, value: Any, ttl: int) -> bool:
        """Set value in Redis."""
        try:
            redis = await self.connection_manager.get_client()
            serialized_data = self.serialization_manager.serialize(value)

            await redis.setex(key, ttl, serialized_data)
            return True

        except (ConnectionError, TimeoutError, ValueError, TypeError, OSError, RedisError) as e:
            logger.error("Error setting in Redis key %s: %s", key, e)
            return False

    async def _delete_from_redis(self, key: str) -> bool:
        """Delete value from Redis."""
        try:
            redis = await self.connection_manager.get_client()
            result = await redis.delete(key)
            return result > 0

        except (ConnectionError, TimeoutError, ValueError, TypeError, OSError, RedisError) as e:
            logger.error("Error deleting from Redis key %s: %s", key, e)
            return False

    async def _get_many_from_redis(self, keys: List[str]) -> List[Optional[Any]]:
        """Get multiple values from Redis."""
        try:
            redis = await self.connection_manager.get_client()
            data_list = await redis.mget(keys)

            results = []
            for data in data_list:
                if data is None:
                    results.append(None)
                else:
                    try:
                        results.append(
                            self.serialization_manager.deserialize(data))
                    except (ConnectionError, TimeoutError, ValueError, TypeError, RedisError) as e:
                        logger.error("Error deserializing data: %s", e)
                        results.append(None)

            return results

        except (ConnectionError, TimeoutError, ValueError, TypeError, OSError, RedisError) as e:
            logger.error("Error getting multiple from Redis: %s", e)
            return [None] * len(keys)

    async def _set_many_in_redis(self, mapping: Dict[str, Any], ttl: int) -> bool:
        """Set multiple values in Redis."""
        try:
            redis = await self.connection_manager.get_client()

            # Use pipeline for efficiency
            pipe = redis.pipeline()

            for key, value in mapping.items():
                serialized_data = self.serialization_manager.serialize(value)
                pipe.setex(key, ttl, serialized_data)

            await pipe.execute()
            return True

        except (ConnectionError, TimeoutError, ValueError, TypeError, OSError, RedisError) as e:
            logger.error("Error setting multiple in Redis: %s", e)
            return False

    async def _metrics_collector(self) -> None:
        """Background task to collect and log metrics."""
        while True:
            try:
                await asyncio.sleep(self.config.metrics_interval)

                stats = self.get_stats()
                logger.info("Cache metrics: %s", stats)

            except asyncio.CancelledError:
                break
            except (ValueError, OSError, RuntimeError) as e:
                logger.error("Error in metrics collector: %s", e)

    async def _health_checker(self) -> None:
        """Background task to check cache health."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                # Check Redis connectivity
                is_healthy = await self.connection_manager.health_check()

                if is_healthy != self._is_healthy:
                    self._is_healthy = is_healthy
                    if is_healthy:
                        logger.info("Cache health restored")
                    else:
                        logger.warning("Cache health degraded")

                self._last_health_check = time.time()

            except asyncio.CancelledError:
                break
            except (ValueError, OSError, RuntimeError, ConnectionError, TimeoutError) as e:
                logger.error("Error in health checker: %s", e)
                self._is_healthy = False


# Global cache manager instance
cache_manager = CacheManager()

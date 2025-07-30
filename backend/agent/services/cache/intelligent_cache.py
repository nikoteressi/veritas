"""
from __future__ import annotations

Advanced caching system for fact verification with intelligent cache management.

This module provides multi-level caching for embeddings, verification results,
and graph data with automatic cache invalidation and optimization.
"""

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import redis.asyncio as redis
from app.config import Settings
from app.exceptions import CacheError
from app.redis_client import redis_manager

settings = Settings()


class CacheLevel(Enum):
    """Cache levels for different types of data."""

    MEMORY = "memory"  # In-memory cache (fastest)
    REDIS = "redis"  # Redis cache (fast, persistent)
    DISK = "disk"  # Disk cache (slower, very persistent)


class CacheStrategy(Enum):
    """Cache invalidation strategies."""

    TTL = "ttl"  # Time-to-live
    LRU = "lru"  # Least recently used
    SIMILARITY = "similarity"  # Semantic similarity-based
    DEPENDENCY = "dependency"  # Dependency-based invalidation


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds

    def touch(self):
        """Update access information."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class IntelligentCache:
    """
    Advanced multi-level caching system with intelligent cache management.

    Features:
    - Multi-level caching (memory, Redis, disk)
    - Intelligent cache invalidation
    - Semantic similarity-based cache lookup
    - Automatic cache optimization
    - Dependency tracking
    """

    def __init__(self, redis_client: redis.Redis = None, max_memory_size: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.max_memory_size = max_memory_size
        self.similarity_threshold = 0.7  # Threshold for similarity search

        # Memory cache
        self.memory_cache: dict[str, CacheEntry] = {}

        # Redis cache - use redis_manager for better connection management
        self.redis_client = redis_client

        # Cache statistics
        self.stats = {"hits": 0, "misses": 0,
                      "evictions": 0, "memory_usage": 0}

        # Dependency tracking
        self.dependency_map: dict[str, list[str]] = {}

        # Initialize cache system
        # Note: Redis initialization will be done lazily when first accessed

    async def initialize(self) -> bool:
        """Initialize the cache system."""
        try:
            # Try to initialize Redis connection
            await self._get_redis_client()
            self.logger.info("Cache system initialized successfully")
            return True
        except (redis.ConnectionError, redis.TimeoutError, redis.RedisError) as e:
            self.logger.warning("Cache initialization warning: %s", e)
            # Cache can still work with memory only
            return True
        except Exception as e:
            raise CacheError(
                f"Unexpected error during cache initialization: {e}") from e

    async def _get_redis_client(self) -> redis.Redis | None:
        """Get Redis client, initializing if needed."""
        if self.redis_client is None:
            try:
                # Check if redis_manager is available and initialized
                if hasattr(redis_manager, "_async_pool_binary") and redis_manager._async_pool_binary is not None:
                    self.redis_client = redis_manager.get_async_client_binary()
                else:
                    # Try to initialize redis_manager
                    await redis_manager.init_redis()
                    if redis_manager._async_pool_binary is not None:
                        self.redis_client = redis_manager.get_async_client_binary()
                    else:
                        self.logger.warning(
                            "Redis is not available, using memory cache only")
                        return None
            except (redis.ConnectionError, redis.TimeoutError, redis.RedisError) as e:
                self.logger.warning("Failed to get Redis client: %s", e)
                return None
            except Exception as e:
                raise CacheError(
                    f"Unexpected error getting Redis client: {e}") from e
        return self.redis_client

    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data."""
        if isinstance(data, str):
            content = data
        elif isinstance(data, dict):
            content = json.dumps(data, sort_keys=True)
        elif isinstance(data, (list, tuple)):
            content = json.dumps(list(data), sort_keys=True)
        else:
            content = str(data)

        hash_obj = hashlib.sha256(content.encode("utf-8"))
        return f"{prefix}:{hash_obj.hexdigest()[:16]}"

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        return pickle.dumps(value)

    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        return pickle.loads(data)

    async def get(self, key: str, default: Any = None, similarity_search: bool = False) -> Any:
        """
        Get value from cache with optional similarity search.

        Args:
            key: Cache key
            default: Default value if not found
            similarity_search: Enable semantic similarity search

        Returns:
            Cached value or default
        """
        # Try memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if not entry.is_expired():
                entry.touch()
                self.stats["hits"] += 1
                return entry.value
            else:
                # Remove expired entry
                del self.memory_cache[key]

        # Try Redis cache
        redis_client = await self._get_redis_client()
        if redis_client is not None:
            try:
                data = await redis_client.get(key)
                if data:
                    value = self._deserialize_value(data)
                    # Store in memory cache for faster access
                    self._store_memory(key, value)
                    self.stats["hits"] += 1
                    return value
            except (redis.ConnectionError, redis.TimeoutError, redis.RedisError) as e:
                self.logger.warning("Redis get error: %s", e)
            except Exception as e:
                raise CacheError(
                    f"Unexpected error during Redis get operation: {e}") from e

        # Try similarity search if enabled
        if similarity_search:
            similar_value = self._similarity_search(key)
            if similar_value is not None:
                self.stats["hits"] += 1
                return similar_value

        self.stats["misses"] += 1
        return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
        level: CacheLevel = CacheLevel.MEMORY,
        dependencies: list[str] = None,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            level: Cache level to use
            dependencies: List of dependency keys

        Returns:
            True if successful
        """
        try:
            # Store in memory cache
            if level in [CacheLevel.MEMORY, CacheLevel.REDIS]:
                self._store_memory(key, value, ttl_seconds, dependencies)

            # Store in Redis cache
            if level == CacheLevel.REDIS:
                await self._store_redis(key, value, ttl_seconds)

            return True
        except Exception as e:
            raise CacheError(
                f"Failed to set cache entry for key '{key}': {e}") from e

    def _store_memory(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
        dependencies: list[str] = None,
    ):
        """Store value in memory cache."""
        # Check if we need to evict entries
        if len(self.memory_cache) >= self.max_memory_size:
            self._evict_memory_entries()

        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            ttl_seconds=ttl_seconds,
            dependencies=dependencies or [],
        )

        self.memory_cache[key] = entry
        self.stats["memory_usage"] = len(self.memory_cache)

    async def _store_redis(self, key: str, value: Any, ttl_seconds: int | None = None):
        """Store value in Redis cache."""
        try:
            redis_client = await self._get_redis_client()
            if redis_client is not None:
                serialized = self._serialize_value(value)
                if ttl_seconds:
                    await redis_client.setex(key, ttl_seconds, serialized)
                else:
                    await redis_client.set(key, serialized)
        except (redis.ConnectionError, redis.TimeoutError, redis.RedisError) as e:
            self.logger.warning("Redis set error: %s", e)
        except Exception as e:
            raise CacheError(
                f"Unexpected error during Redis set operation: {e}") from e

    def _evict_memory_entries(self, count: int = None):
        """Evict entries from memory cache using LRU strategy."""
        if not count:
            count = max(1, len(self.memory_cache) // 10)  # Evict 10%

        # Sort by last accessed time (LRU)
        sorted_entries = sorted(self.memory_cache.items(),
                                key=lambda x: x[1].last_accessed)

        for i in range(min(count, len(sorted_entries))):
            key = sorted_entries[i][0]
            del self.memory_cache[key]
            self.stats["evictions"] += 1

        self.stats["memory_usage"] = len(self.memory_cache)

    def _similarity_search(self, target_key: str) -> Any:
        """Search for similar cached entries using semantic similarity."""
        # This is a simplified version - in practice, you'd use embeddings
        # For now, we'll use string similarity

        best_similarity = 0.0
        best_value = None

        for cached_key, entry in self.memory_cache.items():
            if entry.is_expired():
                continue

            # Simple string similarity (can be replaced with embedding similarity)
            similarity = self._calculate_string_similarity(
                target_key, cached_key)

            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_value = entry.value
                entry.touch()

        return best_value

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity (simplified)."""
        # Simple Jaccard similarity
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())

        if not set1 and not set2:
            return 1.0

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    async def delete(self, key: str) -> bool:
        """Delete entry from all cache levels."""
        deleted = False

        # Delete from memory
        if key in self.memory_cache:
            del self.memory_cache[key]
            deleted = True

        # Delete from Redis (graceful degradation)
        redis_client = await self._get_redis_client()
        if redis_client is not None:
            try:
                result = await redis_client.delete(key)
                if result > 0:
                    deleted = True
            except Exception as e:
                raise CacheError(
                    f"Failed to delete entry from Redis for key '{key}': {e}") from e

        return deleted

    async def invalidate_dependencies(self, dependency_key: str):
        """Invalidate all cache entries that depend on a specific key."""
        to_delete = []

        for key, entry in self.memory_cache.items():
            if dependency_key in entry.dependencies:
                to_delete.append(key)

        for key in to_delete:
            await self.delete(key)

        self.logger.debug(
            "Invalidated %d entries dependent on %s", len(to_delete), dependency_key)

    async def clear(self, pattern: str = None):
        """Clear cache entries matching pattern."""
        if pattern:
            # Clear specific pattern
            to_delete = [key for key in self.memory_cache.keys()
                         if pattern in key]
            for key in to_delete:
                await self.delete(key)
        else:
            # Clear all
            self.memory_cache.clear()
            redis_client = await self._get_redis_client()
            if redis_client is not None:
                try:
                    await redis_client.flushdb()
                except Exception as e:
                    self.logger.warning(f"Redis clear error: {e}")

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / \
            total_requests if total_requests > 0 else 0.0

        stats = {
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "evictions": self.stats["evictions"],
            "memory_entries": len(self.memory_cache),
            "memory_usage": self.stats["memory_usage"],
            "redis_available": self.redis_client is not None,
        }

        # Add Redis-specific stats if available (graceful degradation)
        redis_client = await self._get_redis_client()
        if redis_client is not None:
            try:
                redis_info = await redis_client.info()
                stats["redis_stats"] = {
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "used_memory": redis_info.get("used_memory_human", "N/A"),
                    "keyspace_hits": redis_info.get("keyspace_hits", 0),
                    "keyspace_misses": redis_info.get("keyspace_misses", 0),
                    "total_commands_processed": redis_info.get("total_commands_processed", 0),
                }
            except Exception as e:
                stats["redis_error"] = str(e)
                raise CacheError(
                    f"Failed to get Redis stats: {e}") from e

        return stats

    async def touch(self, key: str) -> bool:
        """Update last accessed time for cache entry."""
        if key in self.memory_cache:
            self.memory_cache[key].last_accessed = datetime.now()
            return True

        # Check if exists in Redis and update access time (graceful degradation)
        redis_client = await self._get_redis_client()
        if redis_client is not None:
            try:
                exists = await redis_client.exists(key)
                if exists:
                    # Get current TTL and reset it to update access time
                    ttl = await redis_client.ttl(key)
                    if ttl > 0:
                        await redis_client.expire(key, ttl)
                    return True
            except Exception as e:
                raise CacheError(
                    f"Failed to touch entry in Redis for key '{key}': {e}") from e

        return False

    async def optimize(self):
        """Optimize cache by removing expired entries and managing memory."""
        try:
            # Remove expired entries from memory
            expired_keys = []
            for key, entry in self.memory_cache.items():
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                del self.memory_cache[key]

            # Clean up Redis expired entries (Redis handles this automatically)
            redis_client = await self._get_redis_client()
            if redis_client is not None:
                try:
                    # Get memory usage info
                    info = await redis_client.info("memory")
                    self.logger.debug(
                        "Redis memory usage: %s", info.get('used_memory_human', 'unknown'))
                except Exception as e:
                    raise CacheError(
                        f"Failed to get Redis memory info: {e}") from e

            self.logger.info(
                f"Removed {len(expired_keys)} expired entries during optimization")

            # Update stats
            self.stats["memory_usage"] = len(self.memory_cache)

        except Exception as e:
            raise CacheError(f"Failed to optimize cache: {e}") from e

    async def close(self):
        """Close cache connections and cleanup resources."""
        try:
            # Clear memory cache
            self.memory_cache.clear()

            # Close Redis connection if exists (graceful degradation)
            if self.redis_client is not None:
                try:
                    await self.redis_client.close()
                    self.redis_client = None
                except Exception as e:
                    raise CacheError(
                        f"Error closing Redis connection: {e}") from e

            self.logger.info("Cache closed successfully")
        except Exception as e:
            raise CacheError(f"Failed to close cache properly: {e}") from e

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class EmbeddingCache(IntelligentCache):
    """
    Specialized cache for embeddings with semantic similarity search.
    """

    def __init__(self, redis_client: redis.Redis = None, max_memory_size: int = 5000):
        super().__init__(redis_client, max_memory_size)
        self.embedding_index: dict[str, np.ndarray] = {}
        self.similarity_threshold = 0.90  # Higher threshold for embeddings

    async def get_embedding(self, text: str, model_name: str = "default") -> np.ndarray | None:
        """Get embedding from cache with semantic similarity search."""
        key = self._generate_key(f"embedding:{model_name}", text)

        # Try exact match first
        embedding = await self.get(key)
        if embedding is not None:
            return embedding

        # Try semantic similarity search
        similar_embedding = self._find_similar_embedding(text, model_name)
        if similar_embedding is not None:
            # Cache the result for future use
            await self.set_embedding(text, similar_embedding, model_name)
            return similar_embedding

        return None

    async def set_embedding(
        self,
        text: str,
        embedding: np.ndarray,
        model_name: str = "default",
        ttl_seconds: int = 86400,
    ):
        """Set embedding in cache."""
        key = self._generate_key(f"embedding:{model_name}", text)

        # Store embedding
        await self.set(key, embedding, ttl_seconds, CacheLevel.REDIS)

        # Store in embedding index for similarity search
        self.embedding_index[key] = embedding

        # Limit index size
        if len(self.embedding_index) > self.max_memory_size:
            # Remove oldest entries
            keys_to_remove = list(self.embedding_index.keys())[:100]
            for k in keys_to_remove:
                del self.embedding_index[k]

    def _find_similar_embedding(self, text: str, model_name: str) -> np.ndarray | None:
        """Find similar embedding using cosine similarity."""
        if not self.embedding_index:
            return None

        # Get embeddings for comparison (this is simplified - in practice,
        # you'd need to generate embedding for the input text)
        target_key = self._generate_key(f"embedding:{model_name}", text)

        best_similarity = 0.0
        best_embedding = None

        for cached_key, cached_embedding in self.embedding_index.items():
            if not cached_key.startswith(f"embedding:{model_name}"):
                continue

            # For demonstration, we'll use a simple similarity check
            # In practice, you'd compute cosine similarity between embeddings
            key_similarity = self._calculate_string_similarity(
                target_key, cached_key)

            if key_similarity > self.similarity_threshold and key_similarity > best_similarity:
                best_similarity = key_similarity
                best_embedding = cached_embedding

        return best_embedding


class VerificationCache(IntelligentCache):
    """
    Specialized cache for verification results with dependency tracking.
    """

    def __init__(self, redis_client: redis.Redis = None):
        super().__init__(redis_client, max_memory_size=2000)

    async def get_verification_result(self, claim: str, sources: list[str] = None) -> dict[str, Any] | None:
        """Get verification result from cache."""
        # Create key from claim and sources
        cache_data = {"claim": claim, "sources": sorted(
            sources) if sources else []}
        key = self._generate_key("verification", cache_data)

        return await self.get(key, similarity_search=True)

    async def set_verification_result(
        self,
        claim: str,
        result: dict[str, Any],
        sources: list[str] = None,
        ttl_seconds: int = 3600,
    ):
        """Set verification result in cache."""
        cache_data = {"claim": claim, "sources": sorted(
            sources) if sources else []}
        key = self._generate_key("verification", cache_data)

        # Add dependencies on sources
        dependencies = [self._generate_key(
            "source", source) for source in (sources or [])]

        await self.set(key, result, ttl_seconds, CacheLevel.REDIS, dependencies)

    async def invalidate_source_results(self, source: str):
        """Invalidate all verification results that depend on a specific source."""
        source_key = self._generate_key("source", source)
        await self.invalidate_dependencies(source_key)


# Global cache instances
_embedding_cache = None
_verification_cache = None
_general_cache = None


def get_embedding_cache() -> EmbeddingCache:
    """Get global embedding cache instance."""
    global _embedding_cache
    if _embedding_cache is None:
        # Let the cache manage its own Redis connection
        _embedding_cache = EmbeddingCache()
    return _embedding_cache


def get_verification_cache() -> VerificationCache:
    """Get global verification cache instance."""
    global _verification_cache
    if _verification_cache is None:
        # Let the cache manage its own Redis connection
        _verification_cache = VerificationCache()
    return _verification_cache


def get_general_cache() -> IntelligentCache:
    """Get global general cache instance."""
    global _general_cache
    if _general_cache is None:
        # Let the cache manage its own Redis connection
        _general_cache = IntelligentCache()
    return _general_cache

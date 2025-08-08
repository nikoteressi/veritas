"""
Base interface for cache services.

Defines the contract that all cache services must implement to ensure
consistent API and behavior across different cache implementations.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from app.cache.core import CacheManager


class CacheServiceInterface(ABC):
    """
    Base interface for all cache services.

    This interface defines the minimum contract that all cache services
    must implement to ensure consistent behavior and API.
    """

    @abstractmethod
    def __init__(self, cache_manager: CacheManager):
        """
        Initialize cache service with cache manager.

        Args:
            cache_manager: Unified cache manager instance
        """

    @property
    @abstractmethod
    def cache_type(self) -> str:
        """
        Get cache type identifier.

        Returns:
            String identifier for this cache service type
        """

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache service statistics.

        Returns:
            Dictionary containing service statistics and metrics
        """

    @abstractmethod
    async def cleanup(self) -> int:
        """
        Clean up expired or invalid cache entries.

        Returns:
            Number of cleaned/removed entries
        """

    async def health_check(self) -> bool:
        """
        Perform health check for the cache service.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Basic health check - try to get stats
            stats = await self.get_stats()
            return stats is not None
        except Exception:
            return False

    async def reset(self) -> bool:
        """
        Reset/clear all data in this cache service.

        Returns:
            True if successful, False otherwise
        """
        try:
            await self.cleanup()
            return True
        except Exception:
            return False


class BatchCacheServiceInterface(CacheServiceInterface):
    """
    Extended interface for cache services that support batch operations.
    """

    @abstractmethod
    async def get_batch(self, keys: list) -> Dict[str, Any]:
        """
        Get multiple cache entries in a single operation.

        Args:
            keys: List of keys to retrieve

        Returns:
            Dictionary mapping keys to their values
        """

    @abstractmethod
    async def set_batch(self, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Set multiple cache entries in a single operation.

        Args:
            data: Dictionary mapping keys to values
            ttl: Time to live in seconds

        Returns:
            True if all operations successful, False otherwise
        """


class InvalidationCacheServiceInterface(CacheServiceInterface):
    """
    Extended interface for cache services that support intelligent invalidation.
    """

    @abstractmethod
    async def invalidate_by_pattern(self, pattern: str) -> int:
        """
        Invalidate cache entries matching a pattern.

        Args:
            pattern: Pattern to match for invalidation

        Returns:
            Number of invalidated entries
        """

    @abstractmethod
    async def invalidate_by_dependency(self, dependency: str) -> int:
        """
        Invalidate cache entries that depend on specific content.

        Args:
            dependency: Dependency identifier

        Returns:
            Number of invalidated entries
        """

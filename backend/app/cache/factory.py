"""
Cache factory for unified cache system initialization.

Provides a centralized way to initialize and manage all cache components
with proper dependency injection and lifecycle management.
"""
import logging
from typing import Optional

from app.exceptions import CacheError

from .cache_manager import CacheManager
from .config import CacheConfig, cache_config
from .services import EmbeddingCache, TemporalCache, VerificationCache

logger = logging.getLogger(__name__)


class CacheFactory:
    """
    Factory for creating and managing cache components.

    Provides centralized initialization, dependency injection,
    and lifecycle management for the unified cache system.
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize cache factory.

        Args:
            config: Cache configuration (uses default if None)
        """
        self.config = config or cache_config

        # Core components
        self._cache_manager: Optional[CacheManager] = None

        # Specialized services
        self._embedding_cache: Optional[EmbeddingCache] = None
        self._verification_cache: Optional[VerificationCache] = None
        self._temporal_cache: Optional[TemporalCache] = None

        # Initialization state
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all cache components."""
        if self._initialized:
            logger.warning("Cache factory already initialized")
            return

        try:
            # Initialize core cache manager
            self._cache_manager = CacheManager(self.config)
            await self._cache_manager.initialize()

            # Initialize specialized services
            self._embedding_cache = EmbeddingCache(self._cache_manager)
            self._verification_cache = VerificationCache(self._cache_manager)
            self._temporal_cache = TemporalCache(self._cache_manager)

            self._initialized = True
            logger.info("Cache factory initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize cache factory: %s", e)
            await self.close()
            raise

    async def close(self) -> None:
        """Close all cache components."""
        try:
            if self._cache_manager:
                await self._cache_manager.close()

            # Reset components
            self._cache_manager = None
            self._embedding_cache = None
            self._verification_cache = None
            self._temporal_cache = None

            self._initialized = False
            logger.info("Cache factory closed successfully")

        except Exception as e:
            raise CacheError(f"Error closing cache factory: {e}") from e

    @property
    def cache_manager(self) -> CacheManager:
        """Get cache manager instance."""
        if not self._initialized or not self._cache_manager:
            raise RuntimeError("Cache factory not initialized")
        return self._cache_manager

    @property
    def embedding_cache(self) -> EmbeddingCache:
        """Get embedding cache service."""
        if not self._initialized or not self._embedding_cache:
            raise RuntimeError("Cache factory not initialized")
        return self._embedding_cache

    @property
    def verification_cache(self) -> VerificationCache:
        """Get verification cache service."""
        if not self._initialized or not self._verification_cache:
            raise RuntimeError("Cache factory not initialized")
        return self._verification_cache

    @property
    def temporal_cache(self) -> TemporalCache:
        """Get temporal cache service."""
        if not self._initialized or not self._temporal_cache:
            raise RuntimeError("Cache factory not initialized")
        return self._temporal_cache



    def is_initialized(self) -> bool:
        """Check if factory is initialized."""
        return self._initialized

    async def health_check(self) -> dict:
        """Perform health check on all cache components."""
        if not self._initialized:
            return {'status': 'not_initialized'}

        try:
            # Check cache manager health
            cache_stats = self.cache_manager.get_stats()

            # Check specialized services
            embedding_stats = self.embedding_cache.get_embedding_stats()
            verification_stats = await self.verification_cache.get_verification_stats()
            temporal_stats = await self.temporal_cache.get_temporal_stats()

            return {
                'status': 'healthy',
                'cache_manager': cache_stats,
                'embedding_cache': embedding_stats,
                'verification_cache': verification_stats,
                'temporal_cache': temporal_stats
            }

        except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
            logger.error("Health check failed: %s", e)
            return {
                'status': 'unhealthy',
                'error': str(e)
            }


# Global cache factory instance
cache_factory = CacheFactory()


# Convenience functions for easy access
async def get_general_cache() -> CacheManager:
    """Get initialized general cache (cache manager)."""
    if not cache_factory.is_initialized():
        await cache_factory.initialize()
    return cache_factory.cache_manager


async def get_embedding_cache() -> EmbeddingCache:
    """Get initialized embedding cache."""
    if not cache_factory.is_initialized():
        await cache_factory.initialize()
    return cache_factory.embedding_cache


async def get_verification_cache() -> VerificationCache:
    """Get initialized verification cache."""
    if not cache_factory.is_initialized():
        await cache_factory.initialize()
    return cache_factory.verification_cache


async def get_temporal_cache() -> TemporalCache:
    """Get initialized temporal cache."""
    if not cache_factory.is_initialized():
        await cache_factory.initialize()
    return cache_factory.temporal_cache








async def initialize_cache_system() -> None:
    """Initialize the entire cache system."""
    await cache_factory.initialize()


async def close_cache_system() -> None:
    """Close the entire cache system."""
    await cache_factory.close()

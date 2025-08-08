"""
Cache factory for unified cache system initialization.

Provides a centralized way to initialize and manage all cache components
with proper dependency injection and lifecycle management.
"""
import logging
from typing import Optional

from app.clients.chroma_client import OllamaChromaClient
from app.exceptions import CacheError

from app.cache.core import CacheManager
from app.cache.config import CacheType, CacheConfig, cache_config
from app.cache.services import TemporalCache, VerificationCache
from app.cache.services.chroma_embedding_cache import ChromaEmbeddingCache

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
        self._chroma_client: Optional[OllamaChromaClient] = None

        # Specialized services
        self._chroma_embedding_cache: Optional[ChromaEmbeddingCache] = None
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

            # Initialize ChromaDB client
            self._chroma_client = OllamaChromaClient()

            # Initialize specialized services
            self._chroma_embedding_cache = ChromaEmbeddingCache(
                self._cache_manager,
                self._chroma_client
            )
            self._verification_cache = VerificationCache(self._cache_manager)
            self._temporal_cache = TemporalCache(self._cache_manager)

            self._initialized = True
            logger.info("Cache factory initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize cache factory: %s", e)
            await self.close()
            raise

    async def get_cache(self, cache_type: CacheType):
        """
        Универсальный метод получения кеша по типу.

        Args:
            cache_type: Тип кеша из enum CacheType

        Returns:
            Соответствующий экземпляр кеша

        Raises:
            RuntimeError: Если factory не инициализирован
            ValueError: Если передан неподдерживаемый тип кеша
        """
        if not self._initialized:
            await self.initialize()

        cache_mapping = {
            CacheType.CHROMA_EMBEDDING: self._chroma_embedding_cache,
            CacheType.VERIFICATION: self._verification_cache,
            CacheType.TEMPORAL: self._temporal_cache,
            CacheType.SIMILARITY: self._cache_manager,  # Пока используем общий менеджер
            CacheType.SESSION: self._cache_manager,
            CacheType.METADATA: self._cache_manager,
            CacheType.RELEVANCE: self._cache_manager,
            CacheType.SHORT: self._cache_manager,
            CacheType.MEDIUM: self._cache_manager,
            CacheType.LONG: self._cache_manager,
            CacheType.STATS: self._cache_manager,
            CacheType.HEALTH: self._cache_manager,
            # Legacy aliases
            CacheType.EVIDENCE: self._verification_cache,
            CacheType.EXPLANATION: self._verification_cache,
            CacheType.ADAPTIVE: self._cache_manager,
        }

        cache_instance = cache_mapping.get(cache_type)
        if cache_instance is None:
            # Для неспециализированных типов возвращаем общий менеджер
            logger.warning(
                "Unknown cache type %s, returning general cache manager", cache_type)
            return self._cache_manager

        return cache_instance

    async def close(self) -> None:
        """Close all cache components."""
        try:
            if self._cache_manager:
                await self._cache_manager.close()

            # Reset components
            self._cache_manager = None
            self._chroma_client = None
            self._chroma_embedding_cache = None
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
    def chroma_embedding_cache(self) -> ChromaEmbeddingCache:
        """Get ChromaDB embedding cache service."""
        if not self._initialized or not self._chroma_embedding_cache:
            raise RuntimeError("Cache factory not initialized")
        return self._chroma_embedding_cache

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
            chroma_embedding_stats = self.chroma_embedding_cache.get_embedding_stats()
            verification_stats = await self.verification_cache.get_verification_stats()
            temporal_stats = await self.temporal_cache.get_temporal_stats()

            return {
                'status': 'healthy',
                'cache_manager': cache_stats,
                'chroma_embedding_cache': chroma_embedding_stats,
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


# New unified convenience function
async def get_cache(cache_type: CacheType):
    return await cache_factory.get_cache(cache_type)


# Legacy convenience functions for backward compatibility
async def get_general_cache() -> CacheManager:
    """Get initialized general cache (cache manager)."""
    if not cache_factory.is_initialized():
        await cache_factory.initialize()
    return cache_factory.cache_manager


async def initialize_cache_system() -> None:
    """Initialize the entire cache system."""
    await cache_factory.initialize()


async def close_cache_system() -> None:
    """Close the entire cache system."""
    await cache_factory.close()

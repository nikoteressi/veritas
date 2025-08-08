"""Graph cache management service.

This service handles all caching operations for graph-based fact checking,
including verification results, graph structures, and metadata.
"""

import hashlib
import logging

from agent.models import FactCheckResult
from agent.models.fact import FactHierarchy
from app.cache import CacheType, get_cache, get_general_cache

logger = logging.getLogger(__name__)


class GraphCacheManager:
    """Manages caching for graph-based fact checking operations.

    Handles verification cache, general cache, and provides methods for
    generating cache keys and managing cache lifecycle.
    """

    def __init__(self):
        """Initialize the cache manager."""
        self.verification_cache = None
        self.general_cache = None
        self._cache_initialized = False
        self.logger = logging.getLogger(__name__)

    async def ensure_cache_initialized(self) -> None:
        """Ensure cache systems are initialized (lazy initialization)."""
        if not self._cache_initialized:
            try:
                self.logger.info(
                    "Initializing cache systems (lazy initialization)")
                self.verification_cache = await get_cache(CacheType.VERIFICATION)
                self.general_cache = await get_general_cache()
                self._cache_initialized = True
                self.logger.info("Cache systems initialized successfully")
            except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
                self.logger.error("Cache systems unavailable: %s", e)
                self.verification_cache = None
                self.general_cache = None
                self._cache_initialized = True  # Mark as attempted

    def generate_cache_key(self, fact_hierarchy: FactHierarchy) -> str:
        """Generate a cache key for a fact hierarchy.

        Args:
            fact_hierarchy: The fact hierarchy to generate a key for

        Returns:
            str: A unique cache key for the fact hierarchy
        """
        # Create a deterministic hash based on the fact hierarchy content
        content = f"{fact_hierarchy.primary_thesis}"
        for fact in fact_hierarchy.supporting_facts:
            content += f"|{fact.description}|{fact.context}"

        return hashlib.sha256(content.encode()).hexdigest()

    def generate_graph_id(self, fact_hierarchy: FactHierarchy) -> str:
        """Generate a unique graph ID for persistent storage.

        Args:
            fact_hierarchy: The fact hierarchy to generate an ID for

        Returns:
            str: A unique graph ID
        """
        # Use the same logic as cache key for consistency
        return f"graph_{self.generate_cache_key(fact_hierarchy)}"

    async def get_cached_result(self, cache_key: str) -> FactCheckResult | None:
        """Retrieve a cached verification result.

        Args:
            cache_key: The cache key to look up

        Returns:
            FactCheckResult | None: The cached result if found, None otherwise
        """
        await self.ensure_cache_initialized()

        if not self.general_cache:
            return None

        try:
            cached_result = await self.general_cache.get(cache_key, cache_type="graph_fact_check")
            if cached_result:
                self.logger.info("Found cached verification result")
                return cached_result
        except Exception as e:
            self.logger.error("Error retrieving cached result: %s", e)

        return None

    async def cache_result(self, cache_key: str, result: FactCheckResult, ttl: int = 3600) -> None:
        """Cache a verification result.

        Args:
            cache_key: The cache key to store under
            result: The result to cache
            ttl: Time to live in seconds (default: 1 hour)
        """
        await self.ensure_cache_initialized()

        if not self.general_cache:
            self.logger.warning(
                "General cache not available, skipping cache storage")
            return

        try:
            await self.general_cache.set(cache_key, result, ttl=ttl, cache_type="graph_fact_check")
            self.logger.info(
                "Cached verification result with key: %s", cache_key)
        except Exception as e:
            self.logger.error("Error caching result: %s", e)

    async def invalidate_cache(self, cache_key: str) -> None:
        """Invalidate a cached result.

        Args:
            cache_key: The cache key to invalidate
        """
        await self.ensure_cache_initialized()

        if not self.general_cache:
            return

        try:
            await self.general_cache.delete(cache_key, cache_type="graph_fact_check")
            self.logger.info("Invalidated cache for key: %s", cache_key)
        except Exception as e:
            self.logger.error("Error invalidating cache: %s", e)

    async def clear_all_cache(self) -> None:
        """Clear all cached verification results."""
        await self.ensure_cache_initialized()

        if not self.general_cache:
            return

        try:
            # Note: This would need to be implemented in the cache backend
            # For now, we'll log the intent
            self.logger.info("Request to clear all graph fact check cache")
        except Exception as e:
            self.logger.error("Error clearing cache: %s", e)

    def is_cache_available(self) -> bool:
        """Check if cache systems are available.

        Returns:
            bool: True if cache is available, False otherwise
        """
        return self._cache_initialized and (self.verification_cache is not None or self.general_cache is not None)

"""
Verification cache service.

Specialized cache for verification results with dependency tracking
and intelligent invalidation based on content changes.
"""
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Set

from app.exceptions import CacheError
from app.cache.core import CacheManager
from .base_cache_service import BatchCacheServiceInterface, InvalidationCacheServiceInterface

logger = logging.getLogger(__name__)


class VerificationCache(BatchCacheServiceInterface, InvalidationCacheServiceInterface):
    """
    Specialized cache for verification results.

    Features:
    - Verification result caching
    - Dependency tracking
    - Content-based invalidation
    - Batch verification operations
    - Performance metrics
    """

    def __init__(self, cache_manager: CacheManager):
        """
        Initialize verification cache.

        Args:
            cache_manager: Unified cache manager instance
        """
        self.cache_manager = cache_manager
        self._cache_type = 'verification'

        # Dependency tracking
        self._dependencies: Dict[str, Set[str]] = {}
        self._reverse_dependencies: Dict[str, Set[str]] = {}

    @property
    def cache_type(self) -> str:
        """Get cache type identifier."""
        return self._cache_type

    async def get_verification_result(
        self,
        content: str,
        verification_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached verification result.

        Args:
            content: Content to verify
            verification_type: Type of verification
            context: Additional context for verification

        Returns:
            Verification result or None if not cached
        """
        key = self._generate_verification_key(
            content, verification_type, context)
        result = await self.cache_manager.get(key, self._cache_type)

        if result:
            # Update access time for metrics
            result['last_accessed'] = time.time()
            await self.cache_manager.set(key, result, cache_type=self._cache_type)

        return result

    async def set_verification_result(
        self,
        content: str,
        verification_type: str,
        result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store verification result.

        Args:
            content: Content that was verified
            verification_type: Type of verification
            result: Verification result
            context: Additional context for verification
            dependencies: List of content dependencies
            ttl: Time to live in seconds

        Returns:
            True if successful, False otherwise
        """
        key = self._generate_verification_key(
            content, verification_type, context)

        # Prepare result with metadata
        cached_result = {
            'result': result,
            'content_hash': self._hash_content(content),
            'verification_type': verification_type,
            'context': context,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'dependencies': dependencies or []
        }

        # Store result
        success = await self.cache_manager.set(key, cached_result, ttl, self._cache_type)

        if success and dependencies:
            # Update dependency tracking
            self._update_dependencies(key, dependencies)

        return success

    async def get_verification_batch(
        self,
        requests: List[Dict[str, Any]]
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Get verification results for multiple requests.

        Args:
            requests: List of verification requests with 'content', 'type', and optional 'context'

        Returns:
            Dictionary mapping request key to verification result
        """
        keys = []
        request_map = {}

        for i, request in enumerate(requests):
            content = request['content']
            verification_type = request['type']
            context = request.get('context')

            key = self._generate_verification_key(
                content, verification_type, context)
            keys.append(key)
            request_map[key] = i

        # Get results from cache
        results = await self.cache_manager.get_many(keys, self._cache_type)

        # Map back to request indices
        batch_results = {}
        for key, result in results.items():
            if result:
                request_idx = request_map[key]
                batch_results[f"request_{request_idx}"] = result

                # Update access time
                result['last_accessed'] = time.time()
                await self.cache_manager.set(key, result, cache_type=self._cache_type)

        return batch_results

    async def invalidate_by_content(self, content: str) -> int:
        """
        Invalidate verification results that depend on specific content.

        Args:
            content: Content that has changed

        Returns:
            Number of invalidated results
        """
        content_hash = self._hash_content(content)

        # Find all verification results that depend on this content
        dependent_keys = self._reverse_dependencies.get(content_hash, set())

        invalidated_count = 0
        for key in list(dependent_keys):
            success = await self.cache_manager.delete(key, self._cache_type)
            if success:
                invalidated_count += 1
                self._remove_dependency_tracking(key)

        return invalidated_count

    async def invalidate_by_type(self, verification_type: str) -> int:
        """
        Invalidate all verification results of a specific type.

        Args:
            verification_type: Type of verification to invalidate

        Returns:
            Number of invalidated results
        """
        pattern = f"*:{verification_type}:*"

        # Clear from cache
        deleted_count = await self.cache_manager.clear_by_pattern(pattern, self._cache_type)

        # Clean up dependency tracking
        keys_to_remove = []
        for key in self._dependencies:
            if f":{verification_type}:" in key:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            self._remove_dependency_tracking(key)

        return deleted_count

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache service statistics (implements CacheServiceInterface)."""
        return await self.get_verification_stats()

    async def get_verification_stats(self) -> Dict[str, Any]:
        """Get verification cache statistics."""
        base_stats = self.cache_manager.get_stats()

        # Count verification types
        type_counts = {}
        dependency_count = len(self._dependencies)

        # Additional verification-specific metrics
        verification_stats = {
            'verification_types': type_counts,
            'total_dependencies': dependency_count,
            'reverse_dependencies': len(self._reverse_dependencies)
        }

        return {
            **base_stats,
            'verification_stats': verification_stats
        }

    async def cleanup(self) -> int:
        """Clean up expired cache entries (implements CacheServiceInterface)."""
        return await self.cleanup_expired_dependencies()

    async def cleanup_expired_dependencies(self) -> int:
        """
        Clean up dependency tracking for expired cache entries.

        Returns:
            Number of cleaned dependencies
        """
        cleaned_count = 0
        keys_to_remove = []

        # Check which keys still exist in cache
        for key in self._dependencies:
            exists = await self.cache_manager.exists(key, self._cache_type)
            if not exists:
                keys_to_remove.append(key)

        # Remove dependency tracking for non-existent keys
        for key in keys_to_remove:
            self._remove_dependency_tracking(key)
            cleaned_count += 1

        return cleaned_count

    def _generate_verification_key(
        self,
        content: str,
        verification_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate cache key for verification result.

        Args:
            content: Content to verify
            verification_type: Type of verification
            context: Additional context

        Returns:
            Cache key
        """
        # Create hash of content
        content_hash = self._hash_content(content)

        # Include context in key if provided
        context_hash = ""
        if context:
            context_str = str(sorted(context.items()))
            context_hash = hashlib.sha256(
                context_str.encode('utf-8')).hexdigest()[:8]

        return f"{content_hash}:{verification_type}:{context_hash}"

    def _hash_content(self, content: str) -> str:
        """Generate hash for content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

    def _update_dependencies(self, key: str, dependencies: List[str]) -> None:
        """Update dependency tracking."""
        # Store dependencies for this key
        dependency_hashes = {self._hash_content(dep) for dep in dependencies}
        self._dependencies[key] = dependency_hashes

        # Update reverse dependencies
        for dep_hash in dependency_hashes:
            if dep_hash not in self._reverse_dependencies:
                self._reverse_dependencies[dep_hash] = set()
            self._reverse_dependencies[dep_hash].add(key)

    def _remove_dependency_tracking(self, key: str) -> None:
        """Remove dependency tracking for a key."""
        # Get dependencies for this key
        dependencies = self._dependencies.pop(key, set())

        # Remove from reverse dependencies
        for dep_hash in dependencies:
            if dep_hash in self._reverse_dependencies:
                self._reverse_dependencies[dep_hash].discard(key)

                # Clean up empty reverse dependency sets
                if not self._reverse_dependencies[dep_hash]:
                    del self._reverse_dependencies[dep_hash]

    async def get_batch(self, keys: list) -> Dict[str, Any]:
        """Get multiple verification results (implements BatchCacheServiceInterface)."""
        # This maps to our existing get_verification_batch method
        # Keys should be verification request dictionaries
        if not keys:
            return {}

        # Convert keys to verification requests format
        requests = []
        for key in keys:
            if isinstance(key, dict):
                requests.append(key)
            else:
                # Assume key is a string and create basic request
                requests.append({'content': str(key), 'type': 'default'})

        return await self.get_verification_batch(requests)

    async def set_batch(self, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple verification results (implements BatchCacheServiceInterface)."""
        # Data should be in format: {key: (content, verification_type, result), ...}
        try:
            for key, value in data.items():
                if isinstance(value, tuple) and len(value) >= 3:
                    content, verification_type, result = value[:3]
                    context = value[3] if len(value) > 3 else None
                    dependencies = value[4] if len(value) > 4 else None

                    success = await self.set_verification_result(
                        content, verification_type, result, context, dependencies, ttl
                    )
                    if not success:
                        return False
            return True
        except CacheError as e:
            logger.error("Cache error in set_batch: %s", e)
            return False
        except Exception as e:
            logger.error("Error in set_batch: %s", e)
            return False

    async def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern (implements InvalidationCacheServiceInterface)."""
        # Use the existing invalidate_by_type method if pattern matches verification type
        if ':' in pattern:
            # Extract verification type from pattern
            parts = pattern.split(':')
            if len(parts) >= 2:
                verification_type = parts[1]
                return await self.invalidate_by_type(verification_type)

        # For other patterns, use cache manager's clear_by_pattern
        return await self.cache_manager.clear_by_pattern(pattern, self._cache_type)

    async def invalidate_by_dependency(self, dependency: str) -> int:
        """Invalidate cache entries that depend on specific content (implements InvalidationCacheServiceInterface)."""
        # This maps to our existing invalidate_by_content method
        return await self.invalidate_by_content(dependency)

"""
Verification repository implementation.

This module implements the VerificationRepository interface for managing
verification results and history.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from agent.models import VerificationResult
from agent.services.graph.interfaces import StorageStrategy, VerificationRepository

logger = logging.getLogger(__name__)


class VerificationRepositoryImpl(VerificationRepository):
    """
    Implementation of VerificationRepository interface.

    This repository manages verification results and provides
    methods for storing, retrieving, and analyzing verification history.
    """

    def __init__(self, storage_strategy: StorageStrategy, config: dict[str, Any] | None = None):
        """
        Initialize verification repository.

        Args:
            storage_strategy: Storage strategy for persistence
            config: Repository configuration
        """
        self._storage = storage_strategy
        self._config = config or {}

        # Configuration
        self._cache_ttl = self._config.get("cache_ttl", 3600)  # 1 hour
        self._max_cache_size = self._config.get("max_cache_size", 1000)
        self._enable_caching = self._config.get("enable_caching", True)
        self._history_retention_days = self._config.get("history_retention_days", 30)

        # In-memory cache for verification results
        self._result_cache: dict[str, VerificationResult] = {}
        self._cache_timestamps: dict[str, datetime] = {}

        # History tracking
        self._verification_history: dict[str, list[VerificationResult]] = defaultdict(list)

        # Statistics
        self._stats = {
            "total_verifications": 0,
            "successful_verifications": 0,
            "failed_verifications": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Locks for thread safety
        self._cache_lock = asyncio.Lock()
        self._history_lock = asyncio.Lock()

        logger.info("VerificationRepositoryImpl initialized")

    async def save_verification_result(self, result: VerificationResult) -> None:
        """
        Save a verification result.

        Args:
            result: Verification result to save
        """
        try:
            # Save to storage
            await self._save_result_to_storage(result)

            # Update cache
            if self._enable_caching:
                await self._update_cache(result)

            # Update history
            await self._update_history(result)

            # Update statistics
            self._stats["total_verifications"] += 1
            if result.is_verified:
                self._stats["successful_verifications"] += 1
            else:
                self._stats["failed_verifications"] += 1

            logger.debug(f"Saved verification result for node {result.node_id}")

        except Exception as e:
            logger.error(f"Failed to save verification result: {str(e)}")
            raise

    async def get_verification_result(self, node_id: str) -> VerificationResult | None:
        """
        Get the latest verification result for a node.

        Args:
            node_id: Node identifier

        Returns:
            Latest verification result or None if not found
        """
        try:
            # Check cache first
            if self._enable_caching:
                cached_result = await self._get_from_cache(node_id)
                if cached_result:
                    self._stats["cache_hits"] += 1
                    return cached_result
                self._stats["cache_misses"] += 1

            # Load from storage
            result = await self._load_result_from_storage(node_id)

            # Update cache
            if result and self._enable_caching:
                await self._update_cache(result)

            return result

        except Exception as e:
            logger.error(f"Failed to get verification result for node {node_id}: {str(e)}")
            return None

    async def get_verification_history(self, node_id: str, limit: int | None = None) -> list[VerificationResult]:
        """
        Get verification history for a node.

        Args:
            node_id: Node identifier
            limit: Maximum number of results to return

        Returns:
            List of verification results ordered by timestamp (newest first)
        """
        try:
            async with self._history_lock:
                history = self._verification_history.get(node_id, [])

                # Sort by timestamp (newest first)
                sorted_history = sorted(history, key=lambda r: r.verified_at, reverse=True)

                # Apply limit
                if limit:
                    sorted_history = sorted_history[:limit]

                return sorted_history

        except Exception as e:
            logger.error(f"Failed to get verification history for node {node_id}: {str(e)}")
            return []

    async def get_verification_summary(self, graph_id: str) -> dict[str, Any]:
        """
        Get verification summary for a graph.

        Args:
            graph_id: Graph identifier

        Returns:
            Verification summary statistics
        """
        try:
            # Load all results for the graph
            results = await self._load_graph_results(graph_id)

            if not results:
                return {
                    "graph_id": graph_id,
                    "total_nodes": 0,
                    "verified_nodes": 0,
                    "unverified_nodes": 0,
                    "verification_rate": 0.0,
                    "average_confidence": 0.0,
                    "last_verification": None,
                }

            # Calculate statistics
            total_nodes = len(results)
            verified_nodes = sum(1 for r in results if r.is_verified)
            unverified_nodes = total_nodes - verified_nodes
            verification_rate = verified_nodes / total_nodes if total_nodes > 0 else 0.0

            # Calculate average confidence for verified nodes
            verified_results = [r for r in results if r.is_verified]
            average_confidence = (
                sum(r.confidence for r in verified_results) / len(verified_results) if verified_results else 0.0
            )

            # Find last verification
            last_verification = max(r.verified_at for r in results) if results else None

            # Confidence distribution
            confidence_ranges = {
                "high": sum(1 for r in verified_results if r.confidence >= 0.8),
                "medium": sum(1 for r in verified_results if 0.5 <= r.confidence < 0.8),
                "low": sum(1 for r in verified_results if r.confidence < 0.5),
            }

            # Evidence statistics
            evidence_stats = {
                "total_evidence": sum(len(r.evidence) for r in results),
                "average_evidence_per_node": (
                    sum(len(r.evidence) for r in results) / total_nodes if total_nodes > 0 else 0.0
                ),
            }

            return {
                "graph_id": graph_id,
                "total_nodes": total_nodes,
                "verified_nodes": verified_nodes,
                "unverified_nodes": unverified_nodes,
                "verification_rate": verification_rate,
                "average_confidence": average_confidence,
                "last_verification": last_verification.isoformat() if last_verification else None,
                "confidence_distribution": confidence_ranges,
                "evidence_statistics": evidence_stats,
            }

        except Exception as e:
            logger.error(f"Failed to get verification summary for graph {graph_id}: {str(e)}")
            return {"error": str(e)}

    async def delete_verification_results(self, node_id: str) -> None:
        """
        Delete all verification results for a node.

        Args:
            node_id: Node identifier
        """
        try:
            # Remove from storage
            await self._delete_from_storage(node_id)

            # Remove from cache
            if self._enable_caching:
                async with self._cache_lock:
                    self._result_cache.pop(node_id, None)
                    self._cache_timestamps.pop(node_id, None)

            # Remove from history
            async with self._history_lock:
                self._verification_history.pop(node_id, None)

            logger.debug(f"Deleted verification results for node {node_id}")

        except Exception as e:
            logger.error(f"Failed to delete verification results for node {node_id}: {str(e)}")
            raise

    async def cleanup_old_results(self, older_than_days: int | None = None) -> int:
        """
        Clean up old verification results.

        Args:
            older_than_days: Remove results older than this many days

        Returns:
            Number of results cleaned up
        """
        try:
            days = older_than_days or self._history_retention_days
            cutoff_date = datetime.now() - timedelta(days=days)

            cleaned_count = 0

            # Clean up history
            async with self._history_lock:
                for node_id, history in list(self._verification_history.items()):
                    # Filter out old results
                    filtered_history = [r for r in history if r.verified_at > cutoff_date]
                    removed_count = len(history) - len(filtered_history)

                    if removed_count > 0:
                        if filtered_history:
                            self._verification_history[node_id] = filtered_history
                        else:
                            del self._verification_history[node_id]
                        cleaned_count += removed_count

            # Clean up cache (remove expired entries)
            if self._enable_caching:
                await self._cleanup_cache()

            logger.info(f"Cleaned up {cleaned_count} old verification results")
            return cleaned_count

        except Exception as e:
            logger.error(f"Failed to cleanup old results: {str(e)}")
            return 0

    async def get_repository_stats(self) -> dict[str, Any]:
        """Get repository statistics."""
        try:
            # Calculate cache statistics
            cache_size = len(self._result_cache) if self._enable_caching else 0
            cache_hit_rate = (
                self._stats["cache_hits"] / (self._stats["cache_hits"] + self._stats["cache_misses"])
                if (self._stats["cache_hits"] + self._stats["cache_misses"]) > 0
                else 0.0
            )

            # Calculate history statistics
            total_history_entries = sum(len(history) for history in self._verification_history.values())

            return {
                "repository_type": "verification_repository",
                "total_verifications": self._stats["total_verifications"],
                "successful_verifications": self._stats["successful_verifications"],
                "failed_verifications": self._stats["failed_verifications"],
                "success_rate": (
                    self._stats["successful_verifications"] / self._stats["total_verifications"]
                    if self._stats["total_verifications"] > 0
                    else 0.0
                ),
                "cache_enabled": self._enable_caching,
                "cache_size": cache_size,
                "cache_hit_rate": cache_hit_rate,
                "total_history_entries": total_history_entries,
                "unique_nodes_with_history": len(self._verification_history),
                "storage_strategy": self._storage.get_strategy_name(),
            }

        except Exception as e:
            logger.error(f"Failed to get repository stats: {str(e)}")
            return {"error": str(e)}

    def get_config(self) -> dict[str, Any]:
        """Get current configuration."""
        return self._config.copy()

    def update_config(self, config: dict[str, Any]) -> None:
        """
        Update repository configuration.

        Args:
            config: New configuration
        """
        if self._validate_config(config):
            self._config.update(config)
            self._cache_ttl = self._config.get("cache_ttl", 3600)
            self._max_cache_size = self._config.get("max_cache_size", 1000)
            self._enable_caching = self._config.get("enable_caching", True)
            self._history_retention_days = self._config.get("history_retention_days", 30)
            logger.info("Verification repository configuration updated")
        else:
            raise ValueError("Invalid configuration")

    def _validate_config(self, config: dict[str, Any]) -> bool:
        """Validate repository configuration."""
        try:
            cache_ttl = config.get("cache_ttl", 3600)
            if not isinstance(cache_ttl, int) or cache_ttl <= 0:
                return False

            max_cache_size = config.get("max_cache_size", 1000)
            if not isinstance(max_cache_size, int) or max_cache_size <= 0:
                return False

            enable_caching = config.get("enable_caching", True)
            if not isinstance(enable_caching, bool):
                return False

            history_retention_days = config.get("history_retention_days", 30)
            if not isinstance(history_retention_days, int) or history_retention_days <= 0:
                return False

            return True

        except Exception:
            return False

    async def _save_result_to_storage(self, result: VerificationResult) -> None:
        """Save result to storage strategy."""
        # Create a storage key for the result
        _storage_key = f"verification_result_{result.node_id}_{result.verified_at.isoformat()}"

        # Serialize result
        _result_data = {
            "node_id": result.node_id,
            "is_verified": result.is_verified,
            "confidence": result.confidence,
            "evidence": result.evidence,
            "reasoning": result.reasoning,
            "verified_at": result.verified_at.isoformat(),
            "verification_method": result.verification_method,
            "metadata": result.metadata,
        }

        # Save using storage strategy (this is a simplified approach)
        # In a real implementation, you might extend the storage interface
        # to handle verification results specifically
        pass  # Storage implementation depends on the specific storage strategy

    async def _load_result_from_storage(self, node_id: str) -> VerificationResult | None:
        """Load latest result from storage strategy."""
        # This is a simplified implementation
        # In a real implementation, you would query the storage for the latest result
        return None

    async def _load_graph_results(self, graph_id: str) -> list[VerificationResult]:
        """Load all results for a graph from storage."""
        # This is a simplified implementation
        # In a real implementation, you would query the storage for all results in a graph
        return []

    async def _delete_from_storage(self, node_id: str) -> None:
        """Delete results from storage strategy."""
        # This is a simplified implementation
        # In a real implementation, you would delete from the storage
        pass

    async def _update_cache(self, result: VerificationResult) -> None:
        """Update cache with verification result."""
        if not self._enable_caching:
            return

        async with self._cache_lock:
            # Check cache size limit
            if len(self._result_cache) >= self._max_cache_size:
                # Remove oldest entry
                oldest_key = min(self._cache_timestamps.keys(), key=lambda k: self._cache_timestamps[k])
                self._result_cache.pop(oldest_key, None)
                self._cache_timestamps.pop(oldest_key, None)

            # Add new result
            self._result_cache[result.node_id] = result
            self._cache_timestamps[result.node_id] = datetime.now()

    async def _get_from_cache(self, node_id: str) -> VerificationResult | None:
        """Get result from cache."""
        if not self._enable_caching:
            return None

        async with self._cache_lock:
            if node_id not in self._result_cache:
                return None

            # Check if cache entry is still valid
            cache_time = self._cache_timestamps.get(node_id)
            if cache_time and (datetime.now() - cache_time).total_seconds() > self._cache_ttl:
                # Cache entry expired
                self._result_cache.pop(node_id, None)
                self._cache_timestamps.pop(node_id, None)
                return None

            return self._result_cache[node_id]

    async def _update_history(self, result: VerificationResult) -> None:
        """Update verification history."""
        async with self._history_lock:
            self._verification_history[result.node_id].append(result)

            # Limit history size per node
            max_history_per_node = 50  # Configurable
            if len(self._verification_history[result.node_id]) > max_history_per_node:
                # Keep only the most recent entries
                self._verification_history[result.node_id] = sorted(
                    self._verification_history[result.node_id], key=lambda r: r.verified_at, reverse=True
                )[:max_history_per_node]

    async def _cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        if not self._enable_caching:
            return

        async with self._cache_lock:
            current_time = datetime.now()
            expired_keys = [
                key
                for key, timestamp in self._cache_timestamps.items()
                if (current_time - timestamp).total_seconds() > self._cache_ttl
            ]

            for key in expired_keys:
                self._result_cache.pop(key, None)
                self._cache_timestamps.pop(key, None)

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

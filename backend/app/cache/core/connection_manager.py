"""
Redis connection manager using RedisClient.
Provides singleton access to Redis operations with connection pooling.
"""
from __future__ import annotations

import logging
from typing import Optional

import redis.asyncio as async_redis

from app.clients.redis_client import RedisClient
from app.cache.config import cache_config

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Redis connection manager implementing singleton pattern.
    Uses RedisClient for actual Redis operations.
    """

    _instance: ConnectionManager | None = None

    def __new__(cls) -> ConnectionManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if not hasattr(self, "_initialized"):
            self._redis_client = RedisClient(
                redis_url=cache_config.redis_url,
                max_connections=cache_config.max_connections
            )
            self._initialized = True

    async def initialize(self) -> bool:
        """Initialize Redis client."""
        return await self._redis_client.initialize()

    def get_async_client(self) -> async_redis.Redis:
        """Get an async Redis client from the connection pool."""
        return self._redis_client.get_client()

    async def get_client(self) -> Optional[async_redis.Redis]:
        """Get Redis client with automatic initialization."""
        return await self._redis_client.get_client_safe()

    async def health_check(self) -> bool:
        """Check if Redis is available."""
        return await self._redis_client.health_check()

    def get_pool_stats(self) -> dict:
        """Get connection pool statistics."""
        return self._redis_client.get_pool_stats()

    async def close(self):
        """Close Redis connection pool."""
        await self._redis_client.close()

    @property
    def is_healthy(self) -> bool:
        """Check if Redis client is healthy."""
        return self._redis_client.is_healthy


# Global connection manager instance
connection_manager = ConnectionManager()

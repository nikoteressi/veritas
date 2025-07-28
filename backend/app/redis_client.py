"""
Redis client for caching and session management.
Supports both synchronous and asynchronous Redis clients.
"""

from __future__ import annotations

import logging

import redis.asyncio as async_redis
from redis.asyncio.connection import ConnectionPool as AsyncConnectionPool
from redis.exceptions import RedisError

from app.config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Unified Redis client manager supporting both sync and async operations.
    Implements singleton pattern for connection pool management.
    """

    _instance: RedisClient | None = None

    def __new__(cls) -> RedisClient:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._async_pool = None
        return cls._instance

    def __init__(self):
        # Only initialize once
        if not hasattr(self, "_initialized"):
            self._async_pool: AsyncConnectionPool | None = None
            self._initialized = True

    async def init_redis(self):
        """Initialize both async and sync Redis connection pools."""
        await self._init_async_pool()

    async def _init_async_pool(self):
        """Initialize the async Redis connection pool."""
        if self._async_pool is None:
            try:
                logger.info(
                    "Initializing async Redis connection pool at %s", settings.redis_url
                )
                self._async_pool = AsyncConnectionPool.from_url(
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=20,  # Limit concurrent connections
                    retry_on_timeout=True,
                )
                # Test connection
                client = self.get_async_client()
                await client.ping()
                logger.info("Async Redis connection pool initialized successfully.")
            except (ConnectionError, RedisError) as e:
                logger.error("Failed to initialize async Redis connection pool: %s", e)
                self._async_pool = None

    def get_async_client(self) -> async_redis.Redis:
        """Get an async Redis client from the connection pool."""
        if self._async_pool is None:
            raise ConnectionError(
                "Async Redis connection pool is not initialized. Call init_redis() first."
            )
        return async_redis.Redis(connection_pool=self._async_pool)

    def get_client(self) -> async_redis.Redis:
        """Get async Redis client (for backward compatibility)."""
        return self.get_async_client()

    async def close(self):
        """Close both Redis connection pools."""
        if self._async_pool:
            await self._async_pool.disconnect()
            self._async_pool = None
            logger.info("Async Redis connection pool closed.")

    async def is_async_available(self) -> bool:
        """Check if async Redis is available."""
        try:
            if self._async_pool:
                client = self.get_async_client()
                await client.ping()
                return True
            return False
        except (ConnectionError, RedisError):
            return False


# Global Redis manager instance
redis_manager = RedisClient()

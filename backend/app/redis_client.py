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
    Supports both text and binary data with separate connection pools.
    """

    _instance: RedisClient | None = None

    def __new__(cls) -> RedisClient:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._async_pool_text = None
            cls._instance._async_pool_binary = None
        return cls._instance

    def __init__(self):
        # Only initialize once
        if not hasattr(self, "_initialized"):
            self._async_pool_text: AsyncConnectionPool | None = None
            self._async_pool_binary: AsyncConnectionPool | None = None
            self._initialized = True

    async def init_redis(self):
        """Initialize both async and sync Redis connection pools."""
        await self._init_async_pools()

    async def _init_async_pools(self):
        """Initialize the async Redis connection pools for text and binary data."""
        if self._async_pool_text is None:
            try:
                logger.info("Initializing async Redis connection pools at %s", settings.redis_url)
                
                # Pool for text data (with UTF-8 decoding)
                self._async_pool_text = AsyncConnectionPool.from_url(
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=10,  # Limit concurrent connections
                    retry_on_timeout=True,
                )
                
                # Pool for binary data (no decoding)
                self._async_pool_binary = AsyncConnectionPool.from_url(
                    settings.redis_url,
                    decode_responses=False,
                    max_connections=10,  # Limit concurrent connections
                    retry_on_timeout=True,
                )
                
                # Test connections
                client_text = self.get_async_client_text()
                client_binary = self.get_async_client_binary()
                await client_text.ping()
                await client_binary.ping()
                logger.info("Async Redis connection pools initialized successfully.")
            except (ConnectionError, RedisError) as e:
                logger.error("Failed to initialize async Redis connection pools: %s", e)
                self._async_pool_text = None
                self._async_pool_binary = None

    def get_async_client_text(self) -> async_redis.Redis:
        """Get an async Redis client for text data from the connection pool."""
        if self._async_pool_text is None:
            raise ConnectionError("Async Redis text connection pool is not initialized. Call init_redis() first.")
        return async_redis.Redis(connection_pool=self._async_pool_text)

    def get_async_client_binary(self) -> async_redis.Redis:
        """Get an async Redis client for binary data from the connection pool."""
        if self._async_pool_binary is None:
            raise ConnectionError("Async Redis binary connection pool is not initialized. Call init_redis() first.")
        return async_redis.Redis(connection_pool=self._async_pool_binary)

    def get_async_client(self) -> async_redis.Redis:
        """Get async Redis client for text data (for backward compatibility)."""
        return self.get_async_client_text()

    def get_client(self) -> async_redis.Redis:
        """Get async Redis client for text data (for backward compatibility)."""
        return self.get_async_client_text()

    async def close(self):
        """Close both Redis connection pools."""
        if self._async_pool_text:
            await self._async_pool_text.disconnect()
            self._async_pool_text = None
            logger.info("Async Redis text connection pool closed.")
        
        if self._async_pool_binary:
            await self._async_pool_binary.disconnect()
            self._async_pool_binary = None
            logger.info("Async Redis binary connection pool closed.")

    async def is_async_available(self) -> bool:
        """Check if async Redis is available."""
        try:
            if self._async_pool_text and self._async_pool_binary:
                client_text = self.get_async_client_text()
                client_binary = self.get_async_client_binary()
                await client_text.ping()
                await client_binary.ping()
                return True
            return False
        except (ConnectionError, RedisError):
            return False


# Global Redis manager instance
redis_manager = RedisClient()

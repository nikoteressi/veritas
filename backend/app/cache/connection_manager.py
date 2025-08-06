"""
Redis connection manager based on proven working client.
Supports async Redis operations with connection pooling.
"""
from __future__ import annotations

import logging
from typing import Optional

import redis.asyncio as async_redis
from redis.asyncio.connection import ConnectionPool as AsyncConnectionPool
from redis.exceptions import RedisError

from .config import cache_config

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Redis connection manager implementing singleton pattern.
    Based on proven working RedisClient implementation.
    """

    _instance: ConnectionManager | None = None

    def __new__(cls) -> ConnectionManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._async_pool_binary = None
        return cls._instance

    def __init__(self):
        # Only initialize once
        if not hasattr(self, "_initialized"):
            self._async_pool_binary: AsyncConnectionPool | None = None
            self._initialized = True
            self.is_healthy = False

    async def initialize(self) -> bool:
        """Initialize async Redis connection pool."""
        await self._init_async_pool()
        return self.is_healthy

    async def _init_async_pool(self):
        """Initialize the async Redis connection pool for binary data."""
        if self._async_pool_binary is None:
            try:
                logger.info(
                    "Initializing async Redis connection pool at %s", cache_config.redis_url)

                # Pool for binary data (no decoding) - exactly like working client
                self._async_pool_binary = AsyncConnectionPool.from_url(
                    cache_config.redis_url,
                    decode_responses=False,
                    max_connections=cache_config.max_connections,
                    retry_on_timeout=True,
                )

                # Test connection
                client = self.get_async_client()
                await client.ping()
                self.is_healthy = True
                logger.info(
                    "Async Redis connection pool initialized successfully.")
            except (ConnectionError, RedisError) as e:
                logger.error(
                    "Failed to initialize async Redis connection pool: %s", e)
                self._async_pool_binary = None
                self.is_healthy = False

    def get_async_client(self) -> async_redis.Redis:
        """Get an async Redis client for binary data from the connection pool."""
        if self._async_pool_binary is None:
            raise ConnectionError(
                "Async Redis connection pool is not initialized. Call initialize() first.")
        return async_redis.Redis(connection_pool=self._async_pool_binary)

    async def get_client(self) -> Optional[async_redis.Redis]:
        """Get Redis client with automatic initialization."""
        if self._async_pool_binary is None:
            await self.initialize()

        if self.is_healthy:
            return self.get_async_client()
        return None

    async def health_check(self) -> bool:
        """Check if Redis is available."""
        try:
            if self._async_pool_binary:
                client = self.get_async_client()
                await client.ping()
                self.is_healthy = True
                return True
            self.is_healthy = False
            return False
        except (ConnectionError, RedisError):
            self.is_healthy = False
            return False

    def get_pool_stats(self) -> dict:
        """Get connection pool statistics using only public properties."""
        if not self._async_pool_binary:
            return {
                'pool_initialized': False,
                'max_connections': 0,
                'is_healthy': self.is_healthy
            }

        try:
            # Get only publicly available pool statistics
            pool = self._async_pool_binary
            return {
                'pool_initialized': True,
                'max_connections': pool.max_connections,
                'is_healthy': self.is_healthy
            }
        except Exception as e:
            logger.error("Error getting pool stats: %s", e)
            return {
                'pool_initialized': True,
                'max_connections': 0,
                'is_healthy': False,
                'error': str(e)
            }

    async def close(self):
        """Close Redis connection pool."""
        if self._async_pool_binary:
            await self._async_pool_binary.disconnect()
            self._async_pool_binary = None
            self.is_healthy = False
            logger.info("Async Redis connection pool closed.")


# Global connection manager instance
connection_manager = ConnectionManager()

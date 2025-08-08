"""
Redis client wrapper with connection pooling and health monitoring.
Provides async Redis operations with proper error handling.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import redis.asyncio as async_redis
from redis.asyncio.connection import ConnectionPool as AsyncConnectionPool
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Redis client wrapper that manages connection pooling and provides
    async Redis operations with health monitoring.
    """

    def __init__(self, redis_url: str = None, max_connections: int = None):
        """
        Initialize Redis client with connection pooling.

        Args:
            redis_url: Redis connection URL
            max_connections: Maximum number of connections in pool
        """
        self.redis_url = redis_url or os.getenv(
            'REDIS_URL', 'redis://localhost:6379/0')
        self.max_connections = max_connections or int(
            os.getenv('CACHE_MAX_CONNECTIONS', '20'))

        self._async_pool: AsyncConnectionPool | None = None
        self._initialized = False
        self.is_healthy = False

    async def initialize(self) -> bool:
        """Initialize async Redis connection pool."""
        if self._initialized:
            return self.is_healthy

        try:
            logger.info("Initializing Redis client at %s", self.redis_url)

            # Create connection pool for binary data (no decoding)
            self._async_pool = AsyncConnectionPool.from_url(
                self.redis_url,
                decode_responses=False,
                max_connections=self.max_connections,
                retry_on_timeout=True,
            )

            # Test connection
            client = self.get_client()
            await client.ping()

            self.is_healthy = True
            self._initialized = True
            logger.info("Redis client initialized successfully")
            return True

        except (ConnectionError, RedisError) as e:
            logger.error("Failed to initialize Redis client: %s", e)
            self._async_pool = None
            self.is_healthy = False
            self._initialized = False
            return False

    def get_client(self) -> async_redis.Redis:
        """
        Get an async Redis client from the connection pool.

        Returns:
            Redis client instance

        Raises:
            ConnectionError: If connection pool is not initialized
        """
        if self._async_pool is None:
            raise ConnectionError(
                "Redis connection pool is not initialized. Call initialize() first."
            )
        return async_redis.Redis(connection_pool=self._async_pool)

    async def get_client_safe(self) -> Optional[async_redis.Redis]:
        """
        Get Redis client with automatic initialization.

        Returns:
            Redis client if healthy, None otherwise
        """
        if not self._initialized:
            await self.initialize()

        if self.is_healthy:
            return self.get_client()
        return None

    async def health_check(self) -> bool:
        """
        Check if Redis is available and healthy.

        Returns:
            True if Redis is healthy, False otherwise
        """
        try:
            if self._async_pool:
                client = self.get_client()
                await client.ping()
                self.is_healthy = True
                return True

            self.is_healthy = False
            return False

        except (ConnectionError, RedisError) as e:
            logger.warning("Redis health check failed: %s", e)
            self.is_healthy = False
            return False

    def get_pool_stats(self) -> dict:
        """
        Get connection pool statistics.

        Returns:
            Dictionary with pool statistics
        """
        if not self._async_pool:
            return {
                'pool_initialized': False,
                'max_connections': 0,
                'is_healthy': self.is_healthy,
                'initialized': self._initialized
            }

        try:
            pool = self._async_pool
            return {
                'pool_initialized': True,
                'max_connections': pool.max_connections,
                'is_healthy': self.is_healthy,
                'initialized': self._initialized,
                'redis_url': self.redis_url
            }
        except (AttributeError, ValueError, RedisError) as e:
            logger.error("Error getting pool stats: %s", e)
            return {
                'pool_initialized': True,
                'max_connections': 0,
                'is_healthy': False,
                'error': str(e)
            }

    async def close(self):
        """Close Redis connection pool and cleanup resources."""
        if self._async_pool:
            try:
                await self._async_pool.disconnect()
                logger.info("Redis connection pool closed")
            except (RedisError, ConnectionError) as e:
                logger.error("Error closing Redis connection pool: %s", e)
            finally:
                self._async_pool = None
                self.is_healthy = False
                self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if the client is initialized."""
        return self._initialized

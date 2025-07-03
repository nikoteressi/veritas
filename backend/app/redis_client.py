"""
Redis client for caching and session management.
"""
from __future__ import annotations
import logging
from typing import Optional
import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import ConnectionError, RedisError

from app.config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """Singleton for Redis connection pool."""
    _pool: Optional[ConnectionPool] = None
    _instance: Optional['RedisClient'] = None

    def __new__(cls) -> RedisClient:
        if cls._instance is None:
            cls._instance = super(RedisClient, cls).__new__(cls)
        return cls._instance

    async def init_redis(self):
        """Initialize the Redis connection pool."""
        if self._pool is None:
            try:
                logger.info(f"Initializing Redis connection pool at {settings.redis_url}")
                self._pool = ConnectionPool.from_url(
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                # Test connection
                client = self.get_client()
                await client.ping()
                logger.info("Redis connection pool initialized successfully.")
            except (ConnectionError, RedisError) as e:
                logger.error(f"Failed to initialize Redis connection pool: {e}")
                self._pool = None
                raise

    def get_client(self) -> redis.Redis:
        """Get a Redis client from the connection pool."""
        if self._pool is None:
            raise ConnectionError("Redis connection pool is not initialized. Call init_redis() first.")
        return redis.Redis(connection_pool=self._pool)

    async def close(self):
        """Close the Redis connection pool."""
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
            logger.info("Redis connection pool closed.")

# Singleton instance
redis_manager = RedisClient() 
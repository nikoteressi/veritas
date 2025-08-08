"""
Cache configuration management.

Centralized configuration for the unified cache system
with environment-based settings and validation.
"""
import os
from dataclasses import dataclass
from typing import Dict, Optional

from .cache_types import CacheTypes


@dataclass
class CacheConfig:
    """Cache configuration settings."""

    # Redis connection
    redis_url: str
    redis_host: str
    redis_port: int
    redis_db: int
    redis_password: Optional[str]
    redis_ssl: bool

    # Connection pool
    max_connections: int
    min_connections: int
    connection_timeout: float
    socket_timeout: float
    socket_keepalive: bool
    socket_keepalive_options: Dict

    # Cache behavior
    default_ttl: int
    max_memory_size: int
    compression_threshold: int
    compression_level: int

    # Performance
    batch_size: int
    pipeline_size: int
    retry_attempts: int
    retry_delay: float

    # Monitoring
    enable_metrics: bool
    metrics_interval: int
    health_check_interval: int

    # Memory cache
    memory_cache_size: int
    memory_cache_ttl: int

    @classmethod
    def from_environment(cls) -> 'CacheConfig':
        """Create configuration from environment variables."""
        return cls(
            # Redis connection
            redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
            redis_host=os.getenv('REDIS_HOST', 'localhost'),
            redis_port=int(os.getenv('REDIS_PORT', '6379')),
            redis_db=int(os.getenv('REDIS_DB', '0')),
            redis_password=os.getenv('REDIS_PASSWORD'),
            redis_ssl=os.getenv('REDIS_SSL', 'false').lower() == 'true',

            # Connection pool
            max_connections=int(os.getenv('CACHE_MAX_CONNECTIONS', '20')),
            min_connections=int(os.getenv('CACHE_MIN_CONNECTIONS', '5')),
            connection_timeout=float(
                os.getenv('CACHE_CONNECTION_TIMEOUT', '5.0')),
            socket_timeout=float(os.getenv('CACHE_SOCKET_TIMEOUT', '5.0')),
            socket_keepalive=os.getenv(
                'CACHE_SOCKET_KEEPALIVE', 'true').lower() == 'true',
            socket_keepalive_options={
                'TCP_KEEPIDLE': int(os.getenv('CACHE_TCP_KEEPIDLE', '1')),
                'TCP_KEEPINTVL': int(os.getenv('CACHE_TCP_KEEPINTVL', '3')),
                'TCP_KEEPCNT': int(os.getenv('CACHE_TCP_KEEPCNT', '5'))
            },

            # Cache behavior
            default_ttl=int(os.getenv('CACHE_DEFAULT_TTL', '3600')),  # 1 hour
            max_memory_size=int(
                os.getenv('CACHE_MAX_MEMORY_SIZE', '104857600')),  # 100MB
            compression_threshold=int(
                os.getenv('CACHE_COMPRESSION_THRESHOLD', '1024')),  # 1KB
            compression_level=int(os.getenv('CACHE_COMPRESSION_LEVEL', '6')),

            # Performance
            batch_size=int(os.getenv('CACHE_BATCH_SIZE', '100')),
            pipeline_size=int(os.getenv('CACHE_PIPELINE_SIZE', '1000')),
            retry_attempts=int(os.getenv('CACHE_RETRY_ATTEMPTS', '3')),
            retry_delay=float(os.getenv('CACHE_RETRY_DELAY', '0.1')),

            # Monitoring
            enable_metrics=os.getenv(
                'CACHE_ENABLE_METRICS', 'true').lower() == 'true',
            metrics_interval=int(
                os.getenv('CACHE_METRICS_INTERVAL', '60')),  # 1 minute
            health_check_interval=int(
                os.getenv('CACHE_HEALTH_CHECK_INTERVAL', '30')),  # 30 seconds

            # Memory cache
            memory_cache_size=int(
                os.getenv('CACHE_MEMORY_SIZE', '10485760')),  # 10MB
            memory_cache_ttl=int(
                os.getenv('CACHE_MEMORY_TTL', '300'))  # 5 minutes
        )

    def validate(self) -> None:
        """Validate configuration settings."""
        if self.redis_port < 1 or self.redis_port > 65535:
            raise ValueError(f"Invalid Redis port: {self.redis_port}")

        if self.max_connections < self.min_connections:
            raise ValueError("max_connections must be >= min_connections")

        if self.default_ttl < 0:
            raise ValueError("default_ttl must be non-negative")

        if self.compression_level < 1 or self.compression_level > 9:
            raise ValueError("compression_level must be between 1 and 9")

        if self.retry_attempts < 0:
            raise ValueError("retry_attempts must be non-negative")


# TTL presets for different cache types
TTL_PRESETS = {
    # 24 hours - embeddings are expensive to compute
    CacheTypes.CHROMA_EMBEDDING: 86400,
    CacheTypes.VERIFICATION: 3600,    # 1 hour - verification results
    CacheTypes.TEMPORAL: 1800,        # 30 minutes - temporal analysis
    CacheTypes.RELEVANCE: 900,        # 15 minutes - relevance scores
    CacheTypes.SESSION: 7200,         # 2 hours - session data
    CacheTypes.METADATA: 21600,       # 6 hours - metadata
    CacheTypes.SHORT: 300,            # 5 minutes - short-term cache
    CacheTypes.MEDIUM: 1800,          # 30 minutes - medium-term cache
    CacheTypes.LONG: 86400,           # 24 hours - long-term cache
}

# Cache key prefixes for organization
KEY_PREFIXES = {
    CacheTypes.CHROMA_EMBEDDING: 'emb:',
    CacheTypes.VERIFICATION: 'ver:',
    CacheTypes.TEMPORAL: 'tmp:',
    CacheTypes.RELEVANCE: 'rel:',
    CacheTypes.SESSION: 'ses:',
    CacheTypes.METADATA: 'meta:',
    CacheTypes.SIMILARITY: 'sim:',
    CacheTypes.STATS: 'stats:',
    CacheTypes.HEALTH: 'health:',
}

# Default cache configuration instance
cache_config = CacheConfig.from_environment()

"""
Unified cache system for Veritas.

This package provides a complete replacement for all existing cache implementations
with a unified, high-performance Redis-based solution.

Main components:
- CacheManager: Central cache management
- ConnectionManager: Redis connection pooling
- SerializationManager: Intelligent serialization
- CacheConfig: Configuration management

Usage:
    from app.cache import cache_manager
    
    # Initialize cache
    await cache_manager.initialize()
    
    # Use cache
    await cache_manager.set('key', 'value', cache_type='embedding')
    value = await cache_manager.get('key', cache_type='embedding')
    
    # Cleanup
    await cache_manager.close()
"""

from .cache_manager import CacheManager, cache_manager
from .config import CacheConfig, TTL_PRESETS, KEY_PREFIXES, cache_config
from .connection_manager import ConnectionManager
from .serialization_manager import SerializationManager
from .factory import (
    CacheFactory,
    cache_factory,
    get_general_cache,
    get_embedding_cache,
    get_verification_cache,
    get_temporal_cache,
)
from .cache_monitor import CacheMonitor

__all__ = [
    'CacheManager',
    'cache_manager',
    'CacheConfig',
    'cache_config',
    'ConnectionManager',
    'SerializationManager',
    'CacheFactory',
    'cache_factory',
    'TTL_PRESETS',
    'KEY_PREFIXES',
    'CacheMonitor',
    'get_general_cache',
    'get_embedding_cache',
    'get_verification_cache',
    'get_temporal_cache',
]

"""Unified cache system.

This module replaces all existing cache implementations with a unified,
high-performance Redis-based solution.

Components:
- Core: CacheManager, ConnectionManager, SerializationManager, CircuitBreaker
- Config: Configuration and type definitions
- Utils: Factory and monitoring utilities
- Services: Specialized cache services

Usage:
    from app.cache import cache_manager, cache_factory
    
    # Use unified cache manager directly
    await cache_manager.set('key', 'value', ttl=3600)
    value = await cache_manager.get('key')
    
    # Or use specialized services
    embedding_cache = cache_factory.chroma_embedding_cache
    verification_cache = cache_factory.verification_cache
"""

# Core components
from .core import CacheManager, ConnectionManager, SerializationManager, CircuitBreaker

# Configuration and types
from .config import CacheConfig, cache_config, CacheType, CacheTypes

# Utilities and factory
from .utils import CacheFactory, cache_factory, get_cache, get_general_cache, CacheMonitor

# Services
from .services import ChromaEmbeddingCache, VerificationCache, TemporalCache

# Initialize global instances
cache_manager = CacheManager()
connection_manager = ConnectionManager()

__all__ = [
    # Core components
    'CacheManager',
    'ConnectionManager',
    'SerializationManager',
    'CircuitBreaker',
    'CacheFactory',
    'CacheMonitor',

    # Configuration and types
    'CacheConfig',
    'cache_config',
    'CacheType',
    'CacheTypes',

    # Services
    'ChromaEmbeddingCache',
    'VerificationCache',
    'TemporalCache',

    # Global instances
    'cache_manager',
    'connection_manager',
    'cache_factory',

    # Factory functions
    'get_cache',
    'get_general_cache',
]

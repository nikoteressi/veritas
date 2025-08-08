"""
Core cache infrastructure components.

This package contains the fundamental building blocks of the unified cache system:
- CacheManager: Central cache management
- ConnectionManager: Redis connection pooling
- SerializationManager: Intelligent serialization
- CircuitBreaker: Protection against cascading failures
"""

from .cache_manager import CacheManager, cache_manager
from .connection_manager import ConnectionManager, connection_manager
from .serialization_manager import SerializationManager
from .circuit_breaker import CircuitBreaker, CircuitBreakerError, CircuitState

__all__ = [
    'CacheManager',
    'cache_manager',
    'ConnectionManager',
    'connection_manager',
    'SerializationManager',
    'CircuitBreaker',
    'CircuitBreakerError',
    'CircuitState',
]

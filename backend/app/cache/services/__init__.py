"""
Cache services package.

Specialized cache services for different types of data and operations.
Each service provides optimized caching for specific use cases while
using the unified cache manager as the backend.

Available services:
- EmbeddingCache: Semantic similarity search and embedding storage
- VerificationCache: Verification results with dependency tracking
- TemporalCache: Time-aware caching and temporal analysis

Usage:
    from app.cache.services import EmbeddingCache, VerificationCache, TemporalCache
    from app.cache import cache_manager
    
    # Initialize services
    embedding_cache = EmbeddingCache(cache_manager)
    verification_cache = VerificationCache(cache_manager)
    temporal_cache = TemporalCache(cache_manager)
"""

from .embedding_cache import EmbeddingCache
from .temporal_cache import TemporalCache
from .verification_cache import VerificationCache

__all__ = [
    'EmbeddingCache',
    'VerificationCache',
    'TemporalCache',
]

"""
Cache type constants and enums.

Centralized definition of all cache types to prevent typos
and ensure consistency across the codebase.
"""
from enum import Enum
from typing import Final


class CacheType(Enum):
    """Enumeration of all cache types used in the system."""
    
    # Core cache types
    EMBEDDING = "embedding"
    VERIFICATION = "verification" 
    TEMPORAL = "temporal"
    RELEVANCE = "relevance"
    
    # Session and metadata
    SESSION = "session"
    METADATA = "metadata"
    
    # Time-based categories
    SHORT = "short"      # 5 minutes
    MEDIUM = "medium"    # 30 minutes  
    LONG = "long"        # 24 hours
    
    # Specialized caches
    SIMILARITY = "similarity"
    STATS = "stats"
    HEALTH = "health"
    
    # Legacy aliases (to be deprecated)
    EVIDENCE = "evidence"      # -> VERIFICATION
    EXPLANATION = "explanation" # -> VERIFICATION
    ADAPTIVE = "adaptive"      # -> RELEVANCE


# String constants for backward compatibility
class CacheTypes:
    """String constants for cache types."""
    
    # Core types
    EMBEDDING: Final[str] = CacheType.EMBEDDING.value
    VERIFICATION: Final[str] = CacheType.VERIFICATION.value
    TEMPORAL: Final[str] = CacheType.TEMPORAL.value
    RELEVANCE: Final[str] = CacheType.RELEVANCE.value
    
    # Session and metadata
    SESSION: Final[str] = CacheType.SESSION.value
    METADATA: Final[str] = CacheType.METADATA.value
    
    # Time-based
    SHORT: Final[str] = CacheType.SHORT.value
    MEDIUM: Final[str] = CacheType.MEDIUM.value
    LONG: Final[str] = CacheType.LONG.value
    
    # Specialized
    SIMILARITY: Final[str] = CacheType.SIMILARITY.value
    STATS: Final[str] = CacheType.STATS.value
    HEALTH: Final[str] = CacheType.HEALTH.value
    
    # Legacy (deprecated)
    EVIDENCE: Final[str] = CacheType.EVIDENCE.value
    EXPLANATION: Final[str] = CacheType.EXPLANATION.value
    ADAPTIVE: Final[str] = CacheType.ADAPTIVE.value


# Validation helpers
def is_valid_cache_type(cache_type: str) -> bool:
    """Check if a string is a valid cache type."""
    try:
        CacheType(cache_type)
        return True
    except ValueError:
        return False


def get_all_cache_types() -> list[str]:
    """Get all valid cache type strings."""
    return [cache_type.value for cache_type in CacheType]


def get_core_cache_types() -> list[str]:
    """Get only the core cache types (excluding legacy aliases)."""
    return [
        CacheTypes.EMBEDDING,
        CacheTypes.VERIFICATION,
        CacheTypes.TEMPORAL,
        CacheTypes.RELEVANCE
    ]


# Migration mapping for legacy types
LEGACY_TYPE_MAPPING = {
    CacheTypes.EVIDENCE: CacheTypes.VERIFICATION,
    CacheTypes.EXPLANATION: CacheTypes.VERIFICATION,
    CacheTypes.ADAPTIVE: CacheTypes.RELEVANCE,
}


def normalize_cache_type(cache_type: str) -> str:
    """
    Normalize legacy cache types to their modern equivalents.
    
    Args:
        cache_type: The cache type to normalize
        
    Returns:
        The normalized cache type
    """
    return LEGACY_TYPE_MAPPING.get(cache_type, cache_type)
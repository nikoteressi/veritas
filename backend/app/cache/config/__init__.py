"""
Cache configuration and type definitions.

This package contains configuration management and type definitions:
- CacheConfig: Configuration management with environment-based settings
- CacheType/CacheTypes: Type definitions and constants
- TTL_PRESETS: Time-to-live presets for different cache types
- KEY_PREFIXES: Cache key prefixes for organization
"""

from .config import CacheConfig, TTL_PRESETS, KEY_PREFIXES, cache_config
from .cache_types import CacheType, CacheTypes, is_valid_cache_type, get_all_cache_types, get_core_cache_types, normalize_cache_type

__all__ = [
    'CacheConfig',
    'cache_config',
    'TTL_PRESETS',
    'KEY_PREFIXES',
    'CacheType',
    'CacheTypes',
    'is_valid_cache_type',
    'get_all_cache_types',
    'get_core_cache_types',
    'normalize_cache_type',
]

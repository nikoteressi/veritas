"""
Enhanced graph verification module with intelligent caching and adaptive thresholds.

This module provides an enhanced modular approach to graph verification,
featuring intelligent caching, adaptive thresholds, and comprehensive performance monitoring.
"""

from ..adaptive_thresholds import AdaptiveThresholds
from ..intelligent_cache import IntelligentCache
from .cluster_analyzer import ClusterAnalyzer
from .engine import EnhancedGraphVerificationEngine, GraphVerificationEngine
from .evidence_gatherer import EnhancedEvidenceGatherer, EvidenceGatherer
from .response_parser import ResponseParser
from .result_compiler import ResultCompiler
from .source_manager import EnhancedSourceManager, SourceManager
from .utils import CacheManager, VerificationUtils
from .verification_processor import EnhancedVerificationProcessor, VerificationProcessor

__all__ = [
    # Enhanced components (recommended)
    "EnhancedGraphVerificationEngine",
    "EnhancedEvidenceGatherer",
    "EnhancedSourceManager",
    "EnhancedVerificationProcessor",
    # Original components (backward compatibility)
    "GraphVerificationEngine",
    "EvidenceGatherer",
    "SourceManager",
    "VerificationProcessor",
    # Shared components
    "ClusterAnalyzer",
    "ResultCompiler",
    "ResponseParser",
    # Utility components
    "IntelligentCache",
    "AdaptiveThresholds",
    "VerificationUtils",
    "CacheManager",
]

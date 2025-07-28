"""
Agent Services Module

Экспорт всех сервисов агента, включая новые компоненты релевантности.
"""

from .adaptive_thresholds import AdaptiveThresholds
from .cache_monitor import CacheMonitor
from .cached_hybrid_relevance_scorer import CachedHybridRelevanceScorer

# Компоненты релевантности
from .enhanced_ollama_embeddings import EnhancedOllamaEmbeddings
from .explainable_relevance_scorer import ExplainableRelevanceScorer
from .graph_verification.evidence_gatherer import (
    EnhancedEvidenceGatherer,
    EvidenceGatherer,
)

# Компоненты верификации графов
from .graph_verification.source_manager import EnhancedSourceManager, SourceManager
from .graph_verification.verification_processor import VerificationProcessor

# Базовые компоненты
from .intelligent_cache import IntelligentCache

# Интеграционный менеджер
from .relevance_integration import (
    RelevanceIntegrationManager,
    close_relevance_manager,
    get_relevance_manager,
)
from .temporal_analysis_cache import TemporalAnalysisCache

__all__ = [
    # Базовые компоненты
    "IntelligentCache",
    "AdaptiveThresholds",
    "CacheMonitor",
    # Компоненты релевантности
    "EnhancedOllamaEmbeddings",
    "CachedHybridRelevanceScorer",
    "TemporalAnalysisCache",
    "ExplainableRelevanceScorer",
    # Интеграционный менеджер
    "RelevanceIntegrationManager",
    "get_relevance_manager",
    "close_relevance_manager",
    # Компоненты верификации графов
    "EnhancedSourceManager",
    "SourceManager",
    "EnhancedEvidenceGatherer",
    "EvidenceGatherer",
    "VerificationProcessor",
]

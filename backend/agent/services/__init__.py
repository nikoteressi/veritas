"""
Agent Services Module

Экспорт всех сервисов агента, включая новые компоненты релевантности и рефакторированную архитектуру fact-checking.
"""

# Core orchestration and system management
from .core.agent_manager import AgentManager
from .core.component_manager import ComponentManager
from .core.fact_checking_orchestrator import FactCheckingOrchestrator
from .core.system_config import (
    SystemConfig,
    get_default_config,
    get_production_config,
    get_development_config,
)
from .core.system_health_monitor import SystemHealthMonitor

# Graph-based verification and storage
from .graph.graph_builder import GraphBuilder
from .graph.graph_config import VerificationConfig, ClusteringConfig
from .graph.graph_fact_checking import GraphFactCheckingService
from .graph.graph_storage import Neo4jGraphStorage
from .graph.verification.evidence_gatherer import (
    EnhancedEvidenceGatherer,
    EvidenceGatherer,
)
from .graph.verification.source_manager import EnhancedSourceManager, SourceManager
from .graph.verification.verification_processor import VerificationProcessor as GraphVerificationProcessor

# Caching and performance optimization
from .cache.cache_monitor import CacheMonitor
from .cache.intelligent_cache import IntelligentCache
from .cache.temporal_analysis_cache import TemporalAnalysisCache

# Analysis and scoring services
from .analysis.adaptive_thresholds import AdaptiveThresholds
from .analysis.advanced_clustering import AdvancedClusteringSystem
from .analysis.bayesian_uncertainty import BayesianUncertaintyHandler
from .analysis.post_analyzer import PostAnalyzerService
from .analysis.relationship_analysis import SemanticAnalyzer

# Relevance scoring and integration
from .relevance.cached_hybrid_relevance_scorer import CachedHybridRelevanceScorer
from .relevance.explainable_relevance_scorer import ExplainableRelevanceScorer
from .relevance.relevance_integration import (
    RelevanceIntegrationManager,
    close_relevance_manager,
    get_relevance_manager,
)

# Source and reputation management
from .reputation.reputation import ReputationService
from .reputation.source_reputation import SourceReputationSystem

# Data processing and verification
from .processing.validation_service import ValidationService
from .processing.verification_processor import VerificationProcessor as FactVerificationProcessor

# Final output generation
from .output.result_compiler import ResultCompiler
from .output.summarizer import SummarizerService
from .output.verdict import VerdictService

# Infrastructure and utility services
from .infrastructure.enhanced_ollama_embeddings import EnhancedOllamaEmbeddings
from .infrastructure.event_emission import EventEmissionService
from .infrastructure.screenshot_parser import ScreenshotParserService
from .infrastructure.storage import StorageService
from .infrastructure.web_scraper import WebScraper

__all__ = [
    # Core orchestration and system management
    "AgentManager",
    "ComponentManager",
    "FactCheckingOrchestrator",
    "SystemConfig",
    "get_default_config",
    "get_production_config",
    "get_development_config",
    "SystemHealthMonitor",

    # Graph-based verification and storage
    "GraphBuilder",
    "VerificationConfig",
    "ClusteringConfig",
    "GraphFactCheckingService",
    "Neo4jGraphStorage",
    "EnhancedEvidenceGatherer",
    "EvidenceGatherer",
    "EnhancedSourceManager",
    "SourceManager",
    "GraphVerificationProcessor",

    # Caching and performance optimization
    "CacheMonitor",
    "IntelligentCache",
    "TemporalAnalysisCache",

    # Analysis and scoring services
    "AdaptiveThresholds",
    "AdvancedClusteringSystem",
    "BayesianUncertaintyHandler",
    "PostAnalyzerService",
    "SemanticAnalyzer",

    # Relevance scoring and integration
    "CachedHybridRelevanceScorer",
    "ExplainableRelevanceScorer",
    "RelevanceIntegrationManager",
    "close_relevance_manager",
    "get_relevance_manager",

    # Source and reputation management
    "ReputationService",
    "SourceReputationSystem",

    # Data processing and verification
    "ValidationService",
    "FactVerificationProcessor",

    # Final output generation
    "ResultCompiler",
    "SummarizerService",
    "VerdictService",

    # Infrastructure and utility services
    "EnhancedOllamaEmbeddings",
    "EventEmissionService",
    "ScreenshotParserService",
    "StorageService",
    "WebScraper",
]

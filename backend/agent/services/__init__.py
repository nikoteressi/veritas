"""
Agent Services Module

Export all agent services, including new relevance components and refactored fact-checking architecture.
"""

# Core orchestration and system management
# Analysis and scoring services
from .analysis.adaptive_thresholds import AdaptiveThresholds
from .analysis.advanced_clustering import AdvancedClusteringSystem
from .analysis.bayesian_uncertainty import BayesianUncertaintyHandler
from .analysis.post_analyzer import PostAnalyzerService
from .analysis.relationship_analysis import SemanticAnalyzer

# Caching and performance optimization
from .cache.relevance_cache_monitor import (
    RelevanceCacheMonitor,
    close_relevance_cache_monitor,
    get_relevance_cache_monitor,
)

from .core.agent_manager import AgentManager
from .core.component_manager import ComponentManager
from .core.fact_checking_orchestrator import FactCheckingOrchestrator
from .core.relevance_component_manager import (
    RelevanceComponentManager,
    close_relevance_component_manager,
    get_relevance_component_manager,
)
from .core.system_config import (
    SystemConfig,
    get_default_config,
    get_development_config,
    get_production_config,
)

# Monitoring services
from .core.system_health_monitor import SystemHealthMonitor

# Graph-based verification and storage
from .graph.graph_builder import GraphBuilder
from .graph.graph_config import ClusteringConfig, VerificationConfig
from .graph.graph_fact_checking import GraphFactCheckingService
from .graph.graph_storage import Neo4jGraphStorage
from .graph.verification.evidence_gatherer import (
    EnhancedEvidenceGatherer,
    EvidenceGatherer,
)
from .graph.verification.source_manager import EnhancedSourceManager, SourceManager
from .graph.verification.verification_processor import (
    VerificationProcessor as GraphVerificationProcessor,
)

# Infrastructure and utility services
from .infrastructure.enhanced_ollama_embeddings import EnhancedOllamaEmbeddings
from .infrastructure.event_emission import EventEmissionService
from .infrastructure.screenshot_parser import ScreenshotParserService
from .infrastructure.storage import StorageService
from .infrastructure.web_scraper import WebScraper
from .monitoring.relevance_system_health_monitor import (
    RelevanceSystemHealthMonitor,
    close_relevance_health_monitor,
    get_relevance_health_monitor,
)

# Final output generation
from .output.result_compiler import ResultCompiler
from .output.summarizer import SummarizerService
from .output.verdict import VerdictService

# Data processing and verification
from .processing.validation_service import ValidationService
from .processing.verification_processor import (
    VerificationProcessor as FactVerificationProcessor,
)

# Relevance scoring and integration
from .relevance.cached_hybrid_relevance_scorer import CachedHybridRelevanceScorer
from .relevance.explainable_relevance_scorer import ExplainableRelevanceScorer
from .relevance.relevance_embeddings_coordinator import RelevanceEmbeddingsCoordinator
from .relevance.relevance_orchestrator import (
    RelevanceOrchestrator,
    close_relevance_manager,
    get_relevance_manager,
)

# Source and reputation management
from .reputation.reputation import ReputationService
from .reputation.source_reputation import SourceReputationSystem

__all__ = [
    # Core orchestration and system management
    "AgentManager",
    "ComponentManager",
    "RelevanceComponentManager",
    "get_relevance_component_manager",
    "close_relevance_component_manager",
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
    "RelevanceCacheMonitor",
    "get_relevance_cache_monitor",
    "close_relevance_cache_monitor",

    # Monitoring services
    "RelevanceSystemHealthMonitor",
    "get_relevance_health_monitor",
    "close_relevance_health_monitor",
    # Analysis and scoring services
    "AdaptiveThresholds",
    "AdvancedClusteringSystem",
    "BayesianUncertaintyHandler",
    "PostAnalyzerService",
    "SemanticAnalyzer",
    # Relevance scoring and integration
    "CachedHybridRelevanceScorer",
    "ExplainableRelevanceScorer",
    "RelevanceEmbeddingsCoordinator",
    "RelevanceOrchestrator",
    "get_relevance_manager",
    "close_relevance_manager",
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

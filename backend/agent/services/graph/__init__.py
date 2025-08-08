"""
Graph-based verification and storage services.

This package provides a comprehensive graph-based fact verification system
with clustering, verification, and storage capabilities.
"""

# Legacy components (existing)
from .cache_manager import GraphCacheManager

# Configuration management
from .config import (
    ClusteringConfig,
    ConfigManager,
    GraphServiceConfig,
    RepositoryConfig,
    StorageConfig,
    get_config,
    get_config_manager,
    initialize_config,
)
from .config import VerificationConfig as NewVerificationConfig

# Factory implementations
from .factories import GraphComponentFactory, StrategyFactory
from .graph_config import VerificationConfig
from .graph_fact_checking import GraphFactCheckingService

# Main service
from .graph_service import GraphService

# New graph system components
# Interfaces
from .interfaces import (
    ClusteringStrategy,
    GraphRepository,
    StorageStrategy,
    VerificationRepository,
    VerificationStrategy,
)

# Repository implementations
from .repositories import GraphRepositoryImpl, VerificationRepositoryImpl
from .result_compiler import VerificationResultCompiler

# Concrete strategy implementations
from .strategies import (
    BatchVerificationStrategy,
    CausalClusteringStrategy,
    CrossVerificationStrategy,
    DomainClusteringStrategy,
    # Verification strategies
    IndividualVerificationStrategy,
    InMemoryStorageStrategy,
    # Storage strategies
    Neo4jStorageStrategy,
    # Clustering strategies
    SimilarityClusteringStrategy,
    TemporalClusteringStrategy,
)
from .uncertainty_analyzer import UncertaintyAnalyzer
from .verification_orchestrator import FactVerificationOrchestrator

__all__ = [
    # Legacy components
    "GraphFactCheckingService",
    "FactVerificationOrchestrator",
    "GraphCacheManager",
    "VerificationResultCompiler",
    "UncertaintyAnalyzer",
    "VerificationConfig",
    # Interfaces
    "ClusteringStrategy",
    "VerificationStrategy",
    "StorageStrategy",
    "GraphRepository",
    "VerificationRepository",
    # Clustering strategies
    "SimilarityClusteringStrategy",
    "DomainClusteringStrategy",
    "TemporalClusteringStrategy",
    "CausalClusteringStrategy",
    # Verification strategies
    "IndividualVerificationStrategy",
    "BatchVerificationStrategy",
    "CrossVerificationStrategy",
    # Storage strategies
    "Neo4jStorageStrategy",
    "InMemoryStorageStrategy",
    # Repositories
    "GraphRepositoryImpl",
    "VerificationRepositoryImpl",
    # Factories
    "GraphComponentFactory",
    "StrategyFactory",
    # Configuration
    "ClusteringConfig",
    "NewVerificationConfig",
    "StorageConfig",
    "RepositoryConfig",
    "GraphServiceConfig",
    "ConfigManager",
    "get_config_manager",
    "get_config",
    "initialize_config",
    # Main service
    "GraphService",
]

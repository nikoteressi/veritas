"""
from __future__ import annotations

Enhanced configuration for the improved fact-checking system.
This module provides comprehensive configuration for all new components.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CacheType(Enum):
    """Cache type enumeration."""

    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"


class ClusteringAlgorithm(Enum):
    """Clustering algorithm enumeration."""

    ADAPTIVE = "adaptive"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"


class UncertaintyLevel(Enum):
    """Uncertainty level enumeration."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CacheConfig:
    """Configuration for intelligent caching system."""

    cache_types: list[CacheType] = field(default_factory=lambda: [CacheType.MEMORY, CacheType.REDIS])
    memory_cache_size: int = 1000
    redis_url: str | None = "redis://localhost:6379"
    disk_cache_dir: str = "./cache"
    default_ttl: int = 3600  # 1 hour
    max_ttl: int = 86400  # 24 hours
    compression_enabled: bool = True
    encryption_enabled: bool = False

    # Invalidation strategies
    enable_ttl_invalidation: bool = True
    enable_lru_invalidation: bool = True
    enable_similarity_invalidation: bool = True
    enable_dependency_invalidation: bool = True

    # Similarity thresholds
    similarity_threshold: float = 0.95
    embedding_similarity_threshold: float = 0.9


@dataclass
class Neo4jConfig:
    """Configuration for Neo4j graph storage."""

    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "veritas"
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50
    connection_timeout: int = 30

    # Graph storage settings
    enable_versioning: bool = True
    enable_compression: bool = True
    batch_size: int = 100


@dataclass
class SourceReputationConfig:
    """Configuration for source reputation system."""

    # Scoring weights
    accuracy_weight: float = 0.4
    reliability_weight: float = 0.3
    bias_weight: float = 0.2
    freshness_weight: float = 0.1

    # Thresholds
    high_reputation_threshold: float = 0.8
    medium_reputation_threshold: float = 0.6
    low_reputation_threshold: float = 0.4

    # Update settings
    auto_update_enabled: bool = True
    update_frequency_hours: int = 24
    min_verification_count: int = 5

    # Bias detection
    enable_bias_detection: bool = True
    bias_keywords_file: str | None = None
    political_bias_threshold: float = 0.7


@dataclass
class AdvancedClusteringConfig:
    """Configuration for advanced clustering system."""

    # GNN settings
    embedding_dim: int = 128
    hidden_dim: int = 64
    num_layers: int = 3
    dropout_rate: float = 0.1
    learning_rate: float = 0.001

    # Clustering algorithms
    default_algorithm: ClusteringAlgorithm = ClusteringAlgorithm.ADAPTIVE
    enable_multiple_algorithms: bool = True

    # Similarity metrics
    semantic_weight: float = 0.4
    temporal_weight: float = 0.2
    structural_weight: float = 0.2
    gnn_weight: float = 0.2

    # Thresholds
    min_cluster_size: int = 2
    max_cluster_size: int = 50
    similarity_threshold: float = 0.7

    # DBSCAN parameters
    dbscan_eps: float = 0.3
    dbscan_min_samples: int = 2

    # Hierarchical clustering
    hierarchical_linkage: str = "ward"
    hierarchical_distance_threshold: float = 0.5


@dataclass
class BayesianUncertaintyConfig:
    """Configuration for Bayesian uncertainty handling."""

    # Model parameters
    num_samples: int = 1000
    num_warmup: int = 500
    num_chains: int = 4

    # Priors
    verification_prior_alpha: float = 2.0
    verification_prior_beta: float = 2.0
    source_reliability_prior_alpha: float = 5.0
    source_reliability_prior_beta: float = 2.0

    # Uncertainty thresholds
    low_uncertainty_threshold: float = 0.1
    medium_uncertainty_threshold: float = 0.3
    high_uncertainty_threshold: float = 0.6

    # Evidence weighting
    enable_evidence_weighting: bool = True
    min_evidence_weight: float = 0.1
    max_evidence_weight: float = 1.0

    # Propagation settings
    enable_uncertainty_propagation: bool = True
    propagation_decay_factor: float = 0.9


@dataclass
class RelationshipAnalysisConfig:
    """Configuration for relationship analysis."""

    # Analysis types
    enable_semantic_analysis: bool = True
    enable_temporal_analysis: bool = True
    enable_causal_analysis: bool = True

    # Semantic analysis - using Ollama embeddings instead of transformers
    use_transformer_embeddings: bool = True  # Now refers to Ollama embeddings
    semantic_similarity_threshold: float = 0.7
    tfidf_max_features: int = 5000
    tfidf_ngram_range: tuple = (1, 3)

    # Temporal analysis
    temporal_window_hours: int = 24
    temporal_window_days: int = 7  # Added for temporal analysis
    temporal_decay_factor: float = 0.95
    enable_temporal_clustering: bool = True

    # Causal analysis
    causal_keywords_file: str | None = None
    causal_confidence_threshold: float = 0.6
    enable_statistical_inference: bool = True

    # Relationship thresholds
    min_relationship_strength: float = 0.3
    min_relationship_confidence: float = 0.5
    max_relationships_per_fact: int = 10

    # Graph construction
    enable_bidirectional_edges: bool = True
    edge_weight_threshold: float = 0.4


@dataclass
class SystemConfig:
    """Master configuration for the fact-checking system."""

    # Component configurations
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    neo4j_config: Neo4jConfig = field(default_factory=Neo4jConfig)
    source_reputation_config: SourceReputationConfig = field(default_factory=SourceReputationConfig)
    clustering_config: AdvancedClusteringConfig = field(default_factory=AdvancedClusteringConfig)
    uncertainty_config: BayesianUncertaintyConfig = field(default_factory=BayesianUncertaintyConfig)
    relationship_config: RelationshipAnalysisConfig = field(default_factory=RelationshipAnalysisConfig)

    # Global settings
    enable_all_features: bool = True
    debug_mode: bool = False
    log_level: str = "INFO"

    # Performance settings
    max_concurrent_verifications: int = 10
    verification_timeout_seconds: int = 300
    enable_parallel_processing: bool = True

    # Quality settings
    min_confidence_threshold: float = 0.5
    enable_quality_checks: bool = True
    enable_result_validation: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "cache": {
                "types": [ct.value for ct in self.cache_config.cache_types],
                "memory_size": self.cache_config.memory_cache_size,
                "redis_url": self.cache_config.redis_url,
                "disk_dir": self.cache_config.disk_cache_dir,
                "ttl": self.cache_config.default_ttl,
                "compression": self.cache_config.compression_enabled,
                "encryption": self.cache_config.encryption_enabled,
            },
            "neo4j": {
                "uri": self.neo4j_config.uri if self.neo4j_config else "",
                "database": self.neo4j_config.database if self.neo4j_config else "",
                "versioning": (self.neo4j_config.enable_versioning if self.neo4j_config else False),
                "compression": (self.neo4j_config.enable_compression if self.neo4j_config else False),
            },
            "source_reputation": {
                "weights": {
                    "accuracy": self.source_reputation_config.accuracy_weight,
                    "reliability": self.source_reputation_config.reliability_weight,
                    "bias": self.source_reputation_config.bias_weight,
                    "freshness": self.source_reputation_config.freshness_weight,
                },
                "auto_update": self.source_reputation_config.auto_update_enabled,
                "bias_detection": self.source_reputation_config.enable_bias_detection,
            },
            "clustering": {
                "algorithm": self.clustering_config.default_algorithm.value,
                "gnn_settings": {
                    "embedding_dim": self.clustering_config.embedding_dim,
                    "hidden_dim": self.clustering_config.hidden_dim,
                    "num_layers": self.clustering_config.num_layers,
                },
                "thresholds": {
                    "similarity": self.clustering_config.similarity_threshold,
                    "min_cluster_size": self.clustering_config.min_cluster_size,
                },
            },
            "uncertainty": {
                "model_params": {
                    "num_samples": self.uncertainty_config.num_samples,
                    "num_chains": self.uncertainty_config.num_chains,
                },
                "thresholds": {
                    "low": self.uncertainty_config.low_uncertainty_threshold,
                    "medium": self.uncertainty_config.medium_uncertainty_threshold,
                    "high": self.uncertainty_config.high_uncertainty_threshold,
                },
            },
            "relationships": {
                "semantic": self.relationship_config.enable_semantic_analysis,
                "temporal": self.relationship_config.enable_temporal_analysis,
                "causal": self.relationship_config.enable_causal_analysis,
            },
            "global": {
                "debug": self.debug_mode,
                "log_level": self.log_level,
                "max_concurrent": self.max_concurrent_verifications,
                "timeout": self.verification_timeout_seconds,
                "parallel": self.enable_parallel_processing,
            },
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SystemConfig":
        """Create configuration from dictionary."""
        config = cls()

        # Update cache config
        if "cache" in config_dict:
            cache_data = config_dict["cache"]
            config.cache_config.memory_cache_size = cache_data.get("memory_size", 1000)
            config.cache_config.redis_url = cache_data.get("redis_url")
            config.cache_config.disk_cache_dir = cache_data.get("disk_dir", "./cache")
            config.cache_config.default_ttl = cache_data.get("ttl", 3600)

        return config


def get_default_config() -> SystemConfig:
    """Get default configuration for the system."""
    return SystemConfig()


def get_production_config() -> SystemConfig:
    """Get production-ready configuration."""
    config = SystemConfig()

    # Production optimizations
    config.debug_mode = False
    config.log_level = "WARNING"
    config.max_concurrent_verifications = 20
    config.enable_parallel_processing = True

    # Cache optimizations
    config.cache_config.memory_cache_size = 5000
    config.cache_config.compression_enabled = True
    config.cache_config.encryption_enabled = True

    # Neo4j optimizations
    config.neo4j_config.max_connection_pool_size = 100
    config.neo4j_config.enable_compression = True

    # Clustering optimizations
    config.clustering_config.enable_multiple_algorithms = True
    config.clustering_config.max_cluster_size = 100

    # Uncertainty optimizations
    config.uncertainty_config.num_samples = 2000
    config.uncertainty_config.num_chains = 8

    return config


def get_development_config() -> SystemConfig:
    """Get development configuration."""
    config = SystemConfig()

    # Development settings
    config.debug_mode = True
    config.log_level = "DEBUG"
    config.max_concurrent_verifications = 5

    # Reduced resource usage
    config.cache_config.memory_cache_size = 500
    config.clustering_config.num_layers = 2
    config.uncertainty_config.num_samples = 500
    config.uncertainty_config.num_chains = 2

    return config


def validate_config(config: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate configuration structure and values."""
    errors = []

    # Check required top-level sections
    required_sections = [
        "clustering",
        "uncertainty",
        "relationships",
        "caching",
        "graph_storage",
    ]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")

    # Validate clustering config
    if "clustering" in config:
        clustering = config["clustering"]
        if clustering.get("batch_size", 0) <= 0:
            errors.append("clustering.batch_size must be positive")

    # Validate uncertainty config
    if "uncertainty" in config:
        uncertainty = config["uncertainty"]
        if not 0 <= uncertainty.get("confidence_threshold", 0.5) <= 1:
            errors.append("uncertainty.confidence_threshold must be between 0 and 1")

    # Validate caching config
    if "caching" in config:
        caching = config["caching"]
        if caching.get("memory_cache_size", 0) <= 0:
            errors.append("caching.memory_cache_size must be positive")

    return len(errors) == 0, errors

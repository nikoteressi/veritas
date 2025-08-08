"""
Configuration management for graph services.

This module provides centralized configuration management for the
graph-based fact verification system.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ClusteringConfig:
    """Configuration for clustering strategies."""

    default_strategy: str = "similarity_clustering"
    similarity_config: dict[str, Any] = field(
        default_factory=lambda: {
            "eps": 0.3,
            "min_samples": 2,
        }
    )
    domain_config: dict[str, Any] = field(
        default_factory=lambda: {
            "default_domain": "general",
            "max_cluster_size": 10,
            "batch_threshold": 5,
        }
    )
    temporal_config: dict[str, Any] = field(
        default_factory=lambda: {
            "window_type": "day",
            "custom_window_hours": 24,
            "batch_threshold": 5,
        }
    )
    causal_config: dict[str, Any] = field(
        default_factory=lambda: {
            "min_component_size": 2,
            "batch_threshold": 5,
            "bidirectional": True,
            "use_text_analysis": False,
        }
    )


@dataclass
class VerificationConfig:
    """Configuration for verification strategies."""

    default_strategy: str = "individual_verification"
    individual_config: dict[str, Any] = field(
        default_factory=lambda: {
            "parallel_processing": True,
            "max_concurrent": 3,
            "request_delay": 0.5,
            "use_search": True,
            "max_evidence_sources": 5,
        }
    )
    batch_config: dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 5,
            "parallel_batches": True,
            "max_concurrent_batches": 2,
            "use_search": True,
            "max_evidence_sources": 3,
        }
    )
    cross_verification_config: dict[str, Any] = field(
        default_factory=lambda: {
            "comparison_threshold": 0.7,
            "consensus_threshold": 0.6,
            "max_cross_checks": 10,
            "use_search": True,
        }
    )


@dataclass
class StorageConfig:
    """Configuration for storage strategies."""

    default_strategy: str = "neo4j_storage"
    neo4j_config: dict[str, Any] = field(
        default_factory=lambda: {
            "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            "username": os.getenv("NEO4J_USERNAME", "neo4j"),
            "password": os.getenv("NEO4J_PASSWORD", "password"),
            "max_connection_lifetime": 3600,
            "max_connection_pool_size": 50,
        }
    )
    in_memory_config: dict[str, Any] = field(
        default_factory=lambda: {
            "max_graphs": 100,
            "enable_persistence": False,
            "persistence_path": "./data/graphs",
        }
    )


@dataclass
class RepositoryConfig:
    """Configuration for repository implementations."""

    graph_repository_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enable_cache": True,
            "cache_ttl_seconds": 3600,
        }
    )
    verification_repository_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enable_cache": True,
            "cache_ttl_seconds": 1800,
            "max_history_entries": 1000,
        }
    )


@dataclass
class GraphServiceConfig:
    """Main configuration for graph services."""

    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    repository: RepositoryConfig = field(default_factory=RepositoryConfig)

    # Global settings
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_metrics: bool = True
    enable_health_checks: bool = True

    # Performance settings
    max_graph_size: int = 10000  # Maximum nodes per graph
    max_concurrent_operations: int = 10
    operation_timeout_seconds: int = 300

    # Feature flags
    enable_experimental_features: bool = False
    enable_auto_clustering: bool = True
    enable_auto_verification: bool = True


class ConfigManager:
    """
    Configuration manager for graph services.

    Provides centralized configuration management with environment
    variable support and validation.
    """

    def __init__(self, config_path: str | None = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Optional path to configuration file
        """
        self._config = GraphServiceConfig()
        self._config_path = config_path

        if config_path and Path(config_path).exists():
            self._load_from_file(config_path)

        self._load_from_environment()
        self._validate_config()

        logger.info("Configuration manager initialized")

    def get_config(self) -> GraphServiceConfig:
        """Get current configuration."""
        return self._config

    def get_clustering_config(self, strategy_name: str) -> dict[str, Any]:
        """Get configuration for specific clustering strategy."""
        strategy_configs = {
            "similarity_clustering": self._config.clustering.similarity_config,
            "domain_clustering": self._config.clustering.domain_config,
            "temporal_clustering": self._config.clustering.temporal_config,
            "causal_clustering": self._config.clustering.causal_config,
        }

        return strategy_configs.get(strategy_name, {})

    def get_verification_config(self, strategy_name: str) -> dict[str, Any]:
        """Get configuration for specific verification strategy."""
        strategy_configs = {
            "individual_verification": self._config.verification.individual_config,
            "batch_verification": self._config.verification.batch_config,
            "cross_verification": self._config.verification.cross_verification_config,
        }

        return strategy_configs.get(strategy_name, {})

    def get_storage_config(self, strategy_name: str) -> dict[str, Any]:
        """Get configuration for specific storage strategy."""
        strategy_configs = {
            "neo4j_storage": self._config.storage.neo4j_config,
            "in_memory_storage": self._config.storage.in_memory_config,
        }

        return strategy_configs.get(strategy_name, {})

    def update_config(self, updates: dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates
        """
        # This is a simplified update mechanism
        # In practice, you'd want more sophisticated merging
        for key, value in updates.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

        self._validate_config()
        logger.info("Configuration updated")

    def _load_from_file(self, config_path: str) -> None:
        """Load configuration from file."""
        try:
            import json

            with open(config_path) as f:
                config_data = json.load(f)

            # Update configuration with file data
            self._merge_config(config_data)
            logger.info(f"Loaded configuration from {config_path}")

        except Exception as e:
            logger.warning(f"Failed to load configuration from file: {str(e)}")

    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        # Neo4j configuration
        if os.getenv("NEO4J_URI"):
            self._config.storage.neo4j_config["uri"] = os.getenv("NEO4J_URI")
        if os.getenv("NEO4J_USERNAME"):
            self._config.storage.neo4j_config["username"] = os.getenv("NEO4J_USERNAME")
        if os.getenv("NEO4J_PASSWORD"):
            self._config.storage.neo4j_config["password"] = os.getenv("NEO4J_PASSWORD")

        # Logging configuration
        if os.getenv("GRAPH_LOG_LEVEL"):
            self._config.log_level = os.getenv("GRAPH_LOG_LEVEL")

        # Feature flags
        if os.getenv("ENABLE_EXPERIMENTAL_FEATURES"):
            self._config.enable_experimental_features = os.getenv("ENABLE_EXPERIMENTAL_FEATURES").lower() == "true"

        logger.debug("Loaded configuration from environment variables")

    def _merge_config(self, config_data: dict[str, Any]) -> None:
        """Merge configuration data into current config."""
        # Simplified merging - in practice, you'd want recursive merging
        for section, values in config_data.items():
            if hasattr(self._config, section) and isinstance(values, dict):
                section_obj = getattr(self._config, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)

    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Basic validation
        if self._config.max_graph_size <= 0:
            raise ValueError("max_graph_size must be positive")

        if self._config.max_concurrent_operations <= 0:
            raise ValueError("max_concurrent_operations must be positive")

        if self._config.operation_timeout_seconds <= 0:
            raise ValueError("operation_timeout_seconds must be positive")

        # Validate strategy names
        valid_clustering_strategies = [
            "similarity_clustering",
            "domain_clustering",
            "temporal_clustering",
            "causal_clustering",
        ]
        if self._config.clustering.default_strategy not in valid_clustering_strategies:
            raise ValueError(f"Invalid clustering strategy: {self._config.clustering.default_strategy}")

        valid_verification_strategies = ["individual_verification", "batch_verification", "cross_verification"]
        if self._config.verification.default_strategy not in valid_verification_strategies:
            raise ValueError(f"Invalid verification strategy: {self._config.verification.default_strategy}")

        valid_storage_strategies = ["neo4j_storage", "in_memory_storage"]
        if self._config.storage.default_strategy not in valid_storage_strategies:
            raise ValueError(f"Invalid storage strategy: {self._config.storage.default_strategy}")

        logger.debug("Configuration validation passed")

    def save_to_file(self, config_path: str) -> None:
        """Save current configuration to file."""
        try:
            import json
            from dataclasses import asdict

            config_dict = asdict(self._config)

            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)

            logger.info(f"Saved configuration to {config_path}")

        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")


# Global configuration manager instance
_config_manager: ConfigManager | None = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> GraphServiceConfig:
    """Get current graph service configuration."""
    return get_config_manager().get_config()


def initialize_config(config_path: str | None = None) -> None:
    """Initialize global configuration manager."""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    logger.info("Global configuration manager initialized")

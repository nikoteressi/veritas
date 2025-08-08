"""
Factory for creating graph components with proper dependency injection.

Implements the Factory pattern to create graph components while managing
dependencies and configuration in a centralized manner.
"""

import logging
from typing import Any

from agent.services.graph.interfaces import (
    ClusteringStrategy,
    GraphRepository,
    StorageStrategy,
    VerificationRepository,
    VerificationStrategy,
)

logger = logging.getLogger(__name__)


class GraphComponentFactory:
    """
    Factory for creating graph components with dependency injection.

    Manages the creation and configuration of graph components while
    ensuring proper dependency injection and configuration management.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the factory with configuration.

        Args:
            config: Factory configuration dictionary
        """
        self.config = config or {}
        self._clustering_strategies: dict[str, type[ClusteringStrategy]] = {}
        self._verification_strategies: dict[str, type[VerificationStrategy]] = {}
        self._storage_strategies: dict[str, type[StorageStrategy]] = {}
        self._graph_repositories: dict[str, type[GraphRepository]] = {}
        self._verification_repositories: dict[str, type[VerificationRepository]] = {}

        # Initialize default registrations
        self._register_default_components()

    def _register_default_components(self) -> None:
        """Register default component implementations."""
        # This will be populated as we create concrete implementations

    def register_clustering_strategy(self, name: str, strategy_class: type[ClusteringStrategy]) -> None:
        """
        Register a clustering strategy implementation.

        Args:
            name: Strategy name identifier
            strategy_class: Strategy implementation class
        """
        if not issubclass(strategy_class, ClusteringStrategy):
            raise ValueError("Strategy class must inherit from ClusteringStrategy")

        self._clustering_strategies[name] = strategy_class
        logger.info("Registered clustering strategy: %s", name)

    def register_verification_strategy(self, name: str, strategy_class: type[VerificationStrategy]) -> None:
        """
        Register a verification strategy implementation.

        Args:
            name: Strategy name identifier
            strategy_class: Strategy implementation class
        """
        if not issubclass(strategy_class, VerificationStrategy):
            raise ValueError("Strategy class must inherit from VerificationStrategy")

        self._verification_strategies[name] = strategy_class
        logger.info("Registered verification strategy: %s", name)

    def register_storage_strategy(self, name: str, strategy_class: type[StorageStrategy]) -> None:
        """
        Register a storage strategy implementation.

        Args:
            name: Strategy name identifier
            strategy_class: Strategy implementation class
        """
        if not issubclass(strategy_class, StorageStrategy):
            raise ValueError("Strategy class must inherit from StorageStrategy")

        self._storage_strategies[name] = strategy_class
        logger.info("Registered storage strategy: %s", name)

    def register_graph_repository(self, name: str, repository_class: type[GraphRepository]) -> None:
        """
        Register a graph repository implementation.

        Args:
            name: Repository name identifier
            repository_class: Repository implementation class
        """
        if not issubclass(repository_class, GraphRepository):
            raise ValueError("Repository class must inherit from GraphRepository")

        self._graph_repositories[name] = repository_class
        logger.info("Registered graph repository: %s", name)

    def register_verification_repository(self, name: str, repository_class: type[VerificationRepository]) -> None:
        """
        Register a verification repository implementation.

        Args:
            name: Repository name identifier
            repository_class: Repository implementation class
        """
        if not issubclass(repository_class, VerificationRepository):
            raise ValueError("Repository class must inherit from VerificationRepository")

        self._verification_repositories[name] = repository_class
        logger.info("Registered verification repository: %s", name)

    def create_clustering_strategy(
        self, strategy_name: str, config: dict[str, Any] | None = None
    ) -> ClusteringStrategy:
        """
        Create a clustering strategy instance.

        Args:
            strategy_name: Name of the strategy to create
            config: Optional strategy-specific configuration

        Returns:
            Configured clustering strategy instance

        Raises:
            ValueError: If strategy name is not registered
        """
        if strategy_name not in self._clustering_strategies:
            available = list(self._clustering_strategies.keys())
            raise ValueError(f"Unknown clustering strategy: {strategy_name}. Available: {available}")

        strategy_class = self._clustering_strategies[strategy_name]
        strategy_config = self._merge_config("clustering", strategy_name, config)

        try:
            strategy = strategy_class(strategy_config)
            logger.info("Created clustering strategy: %s", strategy_name)
            return strategy
        except Exception as e:
            logger.error("Failed to create clustering strategy %s: %s", strategy_name, e)
            raise

    def create_verification_strategy(
        self, strategy_name: str, config: dict[str, Any] | None = None
    ) -> VerificationStrategy:
        """
        Create a verification strategy instance.

        Args:
            strategy_name: Name of the strategy to create
            config: Optional strategy-specific configuration

        Returns:
            Configured verification strategy instance

        Raises:
            ValueError: If strategy name is not registered
        """
        if strategy_name not in self._verification_strategies:
            available = list(self._verification_strategies.keys())
            raise ValueError(f"Unknown verification strategy: {strategy_name}. Available: {available}")

        strategy_class = self._verification_strategies[strategy_name]
        strategy_config = self._merge_config("verification", strategy_name, config)

        try:
            strategy = strategy_class(strategy_config)
            logger.info("Created verification strategy: %s", strategy_name)
            return strategy
        except Exception as e:
            logger.error("Failed to create verification strategy %s: %s", strategy_name, e)
            raise

    def create_storage_strategy(self, strategy_name: str, config: dict[str, Any] | None = None) -> StorageStrategy:
        """
        Create a storage strategy instance.

        Args:
            strategy_name: Name of the strategy to create
            config: Optional strategy-specific configuration

        Returns:
            Configured storage strategy instance

        Raises:
            ValueError: If strategy name is not registered
        """
        if strategy_name not in self._storage_strategies:
            available = list(self._storage_strategies.keys())
            raise ValueError(f"Unknown storage strategy: {strategy_name}. Available: {available}")

        strategy_class = self._storage_strategies[strategy_name]
        strategy_config = self._merge_config("storage", strategy_name, config)

        try:
            strategy = strategy_class(strategy_config)
            logger.info("Created storage strategy: %s", strategy_name)
            return strategy
        except Exception as e:
            logger.error("Failed to create storage strategy %s: %s", strategy_name, e)
            raise

    def create_graph_repository(self, repository_name: str, config: dict[str, Any] | None = None) -> GraphRepository:
        """
        Create a graph repository instance.

        Args:
            repository_name: Name of the repository to create
            config: Optional repository-specific configuration

        Returns:
            Configured graph repository instance

        Raises:
            ValueError: If repository name is not registered
        """
        if repository_name not in self._graph_repositories:
            available = list(self._graph_repositories.keys())
            raise ValueError(f"Unknown graph repository: {repository_name}. Available: {available}")

        repository_class = self._graph_repositories[repository_name]
        repository_config = self._merge_config("graph_repository", repository_name, config)

        try:
            repository = repository_class(repository_config)
            logger.info("Created graph repository: %s", repository_name)
            return repository
        except Exception as e:
            logger.error("Failed to create graph repository %s: %s", repository_name, e)
            raise

    def create_verification_repository(
        self, repository_name: str, config: dict[str, Any] | None = None
    ) -> VerificationRepository:
        """
        Create a verification repository instance.

        Args:
            repository_name: Name of the repository to create
            config: Optional repository-specific configuration

        Returns:
            Configured verification repository instance

        Raises:
            ValueError: If repository name is not registered
        """
        if repository_name not in self._verification_repositories:
            available = list(self._verification_repositories.keys())
            raise ValueError(f"Unknown verification repository: {repository_name}. Available: {available}")

        repository_class = self._verification_repositories[repository_name]
        repository_config = self._merge_config("verification_repository", repository_name, config)

        try:
            repository = repository_class(repository_config)
            logger.info("Created verification repository: %s", repository_name)
            return repository
        except Exception as e:
            logger.error("Failed to create verification repository %s: %s", repository_name, e)
            raise

    def _merge_config(
        self,
        component_type: str,
        component_name: str,
        override_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Merge factory config with component-specific config.

        Args:
            component_type: Type of component (clustering, verification, etc.)
            component_name: Specific component name
            override_config: Override configuration

        Returns:
            Merged configuration dictionary
        """
        base_config = self.config.get(component_type, {}).get(component_name, {})
        if override_config:
            merged_config = base_config.copy()
            merged_config.update(override_config)
            return merged_config
        return base_config

    def get_registered_components(self) -> dict[str, list[str]]:
        """
        Get all registered component names by type.

        Returns:
            Dictionary mapping component types to lists of registered names
        """
        return {
            "clustering_strategies": list(self._clustering_strategies.keys()),
            "verification_strategies": list(self._verification_strategies.keys()),
            "storage_strategies": list(self._storage_strategies.keys()),
            "graph_repositories": list(self._graph_repositories.keys()),
            "verification_repositories": list(self._verification_repositories.keys()),
        }

    def validate_configuration(self) -> dict[str, Any]:
        """
        Validate factory configuration.

        Returns:
            Validation results dictionary
        """
        results = {"valid": True, "errors": [], "warnings": [], "component_counts": {}}

        components = self.get_registered_components()
        for component_type, component_list in components.items():
            results["component_counts"][component_type] = len(component_list)
            if len(component_list) == 0:
                results["warnings"].append(f"No {component_type} registered")

        return results

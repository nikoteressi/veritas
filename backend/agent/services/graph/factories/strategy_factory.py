"""
Specialized factory for creating strategy instances.

Provides a focused factory for strategy creation with advanced configuration
and dependency management capabilities.
"""

import logging
from typing import Any

from agent.services.graph.interfaces import ClusteringStrategy, StorageStrategy, VerificationStrategy

logger = logging.getLogger(__name__)


class StrategyFactory:
    """
    Specialized factory for creating strategy instances.

    Provides advanced strategy creation capabilities with configuration
    management, dependency injection, and strategy composition.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize strategy factory with configuration.

        Args:
            config: Factory configuration dictionary
        """
        self.config = config or {}
        self._strategy_registry = {"clustering": {}, "verification": {}, "storage": {}}
        self._default_strategies = {}
        self._strategy_dependencies = {}

    def register_strategy(
        self,
        strategy_type: str,
        strategy_name: str,
        strategy_class: type,
        dependencies: list[str] | None = None,
        is_default: bool = False,
    ) -> None:
        """
        Register a strategy implementation.

        Args:
            strategy_type: Type of strategy (clustering, verification, storage)
            strategy_name: Unique strategy name
            strategy_class: Strategy implementation class
            dependencies: Optional list of dependency strategy names
            is_default: Whether this should be the default strategy for its type
        """
        if strategy_type not in self._strategy_registry:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        # Validate strategy class inheritance
        expected_base_classes = {
            "clustering": ClusteringStrategy,
            "verification": VerificationStrategy,
            "storage": StorageStrategy,
        }

        expected_base = expected_base_classes[strategy_type]
        if not issubclass(strategy_class, expected_base):
            raise ValueError(f"Strategy class must inherit from {expected_base.__name__}")

        self._strategy_registry[strategy_type][strategy_name] = strategy_class

        if dependencies:
            self._strategy_dependencies[f"{strategy_type}:{strategy_name}"] = dependencies

        if is_default:
            self._default_strategies[strategy_type] = strategy_name

        logger.info("Registered %s  strategy: %s", strategy_type, strategy_name)

    def create_strategy(
        self, strategy_type: str, strategy_name: str | None = None, config: dict[str, Any] | None = None, **kwargs
    ) -> ClusteringStrategy | VerificationStrategy | StorageStrategy:
        """
        Create a strategy instance.

        Args:
            strategy_type: Type of strategy to create
            strategy_name: Specific strategy name (uses default if None)
            config: Strategy-specific configuration
            **kwargs: Additional arguments passed to strategy constructor

        Returns:
            Configured strategy instance
        """
        if strategy_type not in self._strategy_registry:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        # Use default strategy if none specified
        if strategy_name is None:
            strategy_name = self._default_strategies.get(strategy_type)
            if strategy_name is None:
                available = list(self._strategy_registry[strategy_type].keys())
                raise ValueError(f"No default {strategy_type} strategy set. Available: {available}")

        if strategy_name not in self._strategy_registry[strategy_type]:
            available = list(self._strategy_registry[strategy_type].keys())
            raise ValueError(f"Unknown {strategy_type} strategy: {strategy_name}. Available: {available}")

        strategy_class = self._strategy_registry[strategy_type][strategy_name]

        # Merge configuration
        merged_config = self._build_strategy_config(strategy_type, strategy_name, config)

        # Check and resolve dependencies
        self._resolve_dependencies(strategy_type, strategy_name)

        try:
            # Create strategy instance
            strategy = strategy_class(config=merged_config, **kwargs)

            # Validate configuration
            if hasattr(strategy, "validate_config") and not strategy.validate_config():
                raise ValueError(f"Invalid configuration for {strategy_type} strategy: {strategy_name}")

            logger.info("Created %s strategy: %s", strategy_type, strategy_name)
            return strategy

        except Exception as e:
            logger.error("Failed to create %s strategy %s: %s", strategy_type, strategy_name, e)
            raise

    def create_clustering_strategy(
        self, strategy_name: str | None = None, config: dict[str, Any] | None = None, **kwargs
    ) -> ClusteringStrategy:
        """
        Create a clustering strategy instance.

        Args:
            strategy_name: Specific strategy name (uses default if None)
            config: Strategy-specific configuration
            **kwargs: Additional arguments

        Returns:
            Configured clustering strategy instance
        """
        return self.create_strategy("clustering", strategy_name, config, **kwargs)

    def create_verification_strategy(
        self, strategy_name: str | None = None, config: dict[str, Any] | None = None, **kwargs
    ) -> VerificationStrategy:
        """
        Create a verification strategy instance.

        Args:
            strategy_name: Specific strategy name (uses default if None)
            config: Strategy-specific configuration
            **kwargs: Additional arguments

        Returns:
            Configured verification strategy instance
        """
        return self.create_strategy("verification", strategy_name, config, **kwargs)

    def create_storage_strategy(
        self, strategy_name: str | None = None, config: dict[str, Any] | None = None, **kwargs
    ) -> StorageStrategy:
        """
        Create a storage strategy instance.

        Args:
            strategy_name: Specific strategy name (uses default if None)
            config: Strategy-specific configuration
            **kwargs: Additional arguments

        Returns:
            Configured storage strategy instance
        """
        return self.create_strategy("storage", strategy_name, config, **kwargs)

    def create_strategy_chain(
        self, strategy_configs: list[dict[str, Any]]
    ) -> list[ClusteringStrategy | VerificationStrategy | StorageStrategy]:
        """
        Create a chain of strategies with dependencies.

        Args:
            strategy_configs: List of strategy configuration dictionaries
                             Each dict should contain 'type', 'name', and optional 'config'

        Returns:
            List of configured strategy instances in dependency order
        """
        strategies = []

        for strategy_config in strategy_configs:
            strategy_type = strategy_config.get("type")
            strategy_name = strategy_config.get("name")
            config = strategy_config.get("config", {})

            if not strategy_type or not strategy_name:
                raise ValueError("Each strategy config must specify 'type' and 'name'")

            strategy = self.create_strategy(strategy_type, strategy_name, config)
            strategies.append(strategy)

        return strategies

    def _build_strategy_config(
        self, strategy_type: str, strategy_name: str, override_config: dict[str, Any] | None
    ) -> dict[str, Any]:
        """
        Build complete strategy configuration.

        Args:
            strategy_type: Type of strategy
            strategy_name: Strategy name
            override_config: Override configuration

        Returns:
            Merged configuration dictionary
        """
        # Start with factory-level config
        base_config = self.config.get("strategies", {}).get(strategy_type, {})

        # Add strategy-specific config
        strategy_config = base_config.get(strategy_name, {})

        # Merge with override config
        if override_config:
            merged_config = strategy_config.copy()
            merged_config.update(override_config)
            return merged_config

        return strategy_config

    def _resolve_dependencies(self, strategy_type: str, strategy_name: str) -> None:
        """
        Resolve strategy dependencies.

        Args:
            strategy_type: Type of strategy
            strategy_name: Strategy name
        """
        dependency_key = f"{strategy_type}:{strategy_name}"
        dependencies = self._strategy_dependencies.get(dependency_key, [])

        for dependency in dependencies:
            # Check if dependency is registered
            dep_type, dep_name = dependency.split(":", 1)
            if dep_type not in self._strategy_registry:
                raise ValueError(f"Unknown dependency type: {dep_type}")

            if dep_name not in self._strategy_registry[dep_type]:
                raise ValueError(f"Dependency not registered: {dependency}")

        logger.debug("Resolved dependencies for %s: %s", dependency_key, dependencies)

    def get_available_strategies(self, strategy_type: str | None = None) -> dict[str, list[str]]:
        """
        Get available strategies by type.

        Args:
            strategy_type: Optional specific strategy type

        Returns:
            Dictionary mapping strategy types to available strategy names
        """
        if strategy_type:
            if strategy_type not in self._strategy_registry:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
            return {strategy_type: list(self._strategy_registry[strategy_type].keys())}

        return {strategy_type: list(strategies.keys()) for strategy_type, strategies in self._strategy_registry.items()}

    def get_default_strategies(self) -> dict[str, str]:
        """
        Get default strategy names by type.

        Returns:
            Dictionary mapping strategy types to default strategy names
        """
        return self._default_strategies.copy()

    def set_default_strategy(self, strategy_type: str, strategy_name: str) -> None:
        """
        Set default strategy for a type.

        Args:
            strategy_type: Strategy type
            strategy_name: Strategy name to set as default
        """
        if strategy_type not in self._strategy_registry:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        if strategy_name not in self._strategy_registry[strategy_type]:
            raise ValueError(f"Strategy not registered: {strategy_name}")

        self._default_strategies[strategy_type] = strategy_name
        logger.info("Set default %s strategy to: %s", strategy_type, strategy_name)

    def validate_factory_state(self) -> dict[str, Any]:
        """
        Validate factory state and configuration.

        Returns:
            Validation results dictionary
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "strategy_counts": {},
            "default_strategies": self._default_strategies.copy(),
            "dependencies": self._strategy_dependencies.copy(),
        }

        # Count strategies by type
        for strategy_type, strategies in self._strategy_registry.items():
            count = len(strategies)
            results["strategy_counts"][strategy_type] = count

            if count == 0:
                results["warnings"].append(f"No {strategy_type} strategies registered")

            # Check if default is set
            if strategy_type not in self._default_strategies:
                results["warnings"].append(f"No default {strategy_type} strategy set")

        # Validate dependencies
        for dep_key, deps in self._strategy_dependencies.items():
            for dep in deps:
                try:
                    dep_type, dep_name = dep.split(":", 1)
                    if dep_type not in self._strategy_registry:
                        results["errors"].append(f"Invalid dependency type in {dep_key}: {dep_type}")
                    elif dep_name not in self._strategy_registry[dep_type]:
                        results["errors"].append(f"Missing dependency in {dep_key}: {dep}")
                except ValueError:
                    results["errors"].append(f"Invalid dependency format in {dep_key}: {dep}")

        if results["errors"]:
            results["valid"] = False

        return results

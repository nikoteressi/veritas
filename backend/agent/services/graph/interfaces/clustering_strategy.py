"""
Abstract interface for clustering strategies.

Defines the contract for different clustering algorithms used in graph construction.
"""

from abc import ABC, abstractmethod
from typing import Any

from agent.models.graph import FactCluster, FactNode


class ClusteringStrategy(ABC):
    """
    Abstract base class for clustering strategies.

    Implements the Strategy pattern for different clustering algorithms
    such as similarity-based, domain-based, temporal, and causal clustering.
    """

    def __init__(self, config: dict[str, Any] = None):
        """
        Initialize clustering strategy with configuration.

        Args:
            config: Strategy-specific configuration parameters
        """
        self.config = config or {}

    @abstractmethod
    async def create_clusters(self, nodes: list[FactNode]) -> list[FactCluster]:
        """
        Create clusters from a list of fact nodes.

        Args:
            nodes: List of fact nodes to cluster

        Returns:
            List of fact clusters

        Raises:
            ClusteringError: If clustering fails
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        Get the name of this clustering strategy.

        Returns:
            Strategy name identifier
        """
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate the strategy configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        pass

    def get_config(self) -> dict[str, Any]:
        """
        Get current strategy configuration.

        Returns:
            Configuration dictionary
        """
        return self.config.copy()

    def update_config(self, new_config: dict[str, Any]) -> None:
        """
        Update strategy configuration.

        Args:
            new_config: New configuration parameters
        """
        self.config.update(new_config)
        if not self.validate_config():
            raise ValueError(f"Invalid configuration for {self.get_strategy_name()}")

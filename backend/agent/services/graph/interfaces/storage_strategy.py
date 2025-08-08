"""
Abstract interface for storage strategies.

Defines the contract for different storage backends used in graph persistence.
"""

from abc import ABC, abstractmethod
from typing import Any

from agent.models.graph import FactCluster, FactEdge, FactGraph, FactNode


class StorageStrategy(ABC):
    """
    Abstract base class for storage strategies.

    Implements the Strategy pattern for different storage backends
    such as Neo4j, PostgreSQL, MongoDB, in-memory, etc.
    """

    def __init__(self, config: dict[str, Any] = None):
        """
        Initialize storage strategy with configuration.

        Args:
            config: Strategy-specific configuration parameters
        """
        self.config = config or {}
        self._connected = False

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the storage backend.

        Raises:
            StorageError: If connection fails
        """

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close connection to the storage backend.

        Raises:
            StorageError: If disconnection fails
        """

    @abstractmethod
    async def save_graph(self, graph: FactGraph, graph_id: str | None = None) -> str:
        """
        Save a complete graph to storage.

        Args:
            graph: Fact graph to save
            graph_id: Optional graph identifier

        Returns:
            Graph identifier in storage

        Raises:
            StorageError: If save operation fails
        """

    @abstractmethod
    async def load_graph(self, graph_id: str) -> FactGraph | None:
        """
        Load a complete graph from storage.

        Args:
            graph_id: Graph identifier

        Returns:
            Loaded fact graph or None if not found

        Raises:
            StorageError: If load operation fails
        """

    @abstractmethod
    async def save_node(self, node: FactNode, graph_id: str) -> str:
        """
        Save a single node to storage.

        Args:
            node: Fact node to save
            graph_id: Graph identifier

        Returns:
            Node identifier in storage

        Raises:
            StorageError: If save operation fails
        """

    @abstractmethod
    async def save_edge(self, edge: FactEdge, graph_id: str) -> str:
        """
        Save a single edge to storage.

        Args:
            edge: Fact edge to save
            graph_id: Graph identifier

        Returns:
            Edge identifier in storage

        Raises:
            StorageError: If save operation fails
        """

    @abstractmethod
    async def save_cluster(self, cluster: FactCluster, graph_id: str) -> str:
        """
        Save a single cluster to storage.

        Args:
            cluster: Fact cluster to save
            graph_id: Graph identifier

        Returns:
            Cluster identifier in storage

        Raises:
            StorageError: If save operation fails
        """

    @abstractmethod
    async def delete_graph(self, graph_id: str) -> bool:
        """
        Delete a graph from storage.

        Args:
            graph_id: Graph identifier

        Returns:
            True if deletion was successful, False otherwise

        Raises:
            StorageError: If delete operation fails
        """

    @abstractmethod
    async def list_graphs(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        List all graphs in storage.

        Args:
            limit: Optional limit on number of results

        Returns:
            List of graph metadata dictionaries

        Raises:
            StorageError: If list operation fails
        """

    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        Get the name of this storage strategy.

        Returns:
            Strategy name identifier
        """

    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate the strategy configuration.

        Returns:
            True if configuration is valid, False otherwise
        """

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

    def is_connected(self) -> bool:
        """
        Check if storage is connected.

        Returns:
            True if connected, False otherwise
        """
        return self._connected

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check on storage backend.

        Returns:
            Health status information
        """
        return {
            "strategy": self.get_strategy_name(),
            "connected": self.is_connected(),
            "config_valid": self.validate_config(),
        }

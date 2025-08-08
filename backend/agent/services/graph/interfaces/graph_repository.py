"""
Abstract repository interface for graph operations.

Defines the contract for graph data access layer using Repository pattern.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from agent.models.graph import FactCluster, FactEdge, FactGraph, FactNode


class GraphRepository(ABC):
    """
    Abstract repository for graph operations.

    Implements the Repository pattern to abstract graph data access
    and provide a clean interface for graph persistence operations.
    """

    @abstractmethod
    async def save_graph(self, graph: FactGraph, metadata: dict[str, Any] | None = None) -> str:
        """
        Save a complete graph with optional metadata.

        Args:
            graph: Fact graph to save
            metadata: Optional metadata to associate with the graph

        Returns:
            Unique graph identifier

        Raises:
            RepositoryError: If save operation fails
        """

    @abstractmethod
    async def load_graph(self, graph_id: str) -> FactGraph | None:
        """
        Load a complete graph by identifier.

        Args:
            graph_id: Unique graph identifier

        Returns:
            Loaded fact graph or None if not found

        Raises:
            RepositoryError: If load operation fails
        """

    @abstractmethod
    async def update_graph(self, graph_id: str, graph: FactGraph) -> bool:
        """
        Update an existing graph.

        Args:
            graph_id: Unique graph identifier
            graph: Updated fact graph

        Returns:
            True if update was successful, False otherwise

        Raises:
            RepositoryError: If update operation fails
        """

    @abstractmethod
    async def delete_graph(self, graph_id: str) -> bool:
        """
        Delete a graph by identifier.

        Args:
            graph_id: Unique graph identifier

        Returns:
            True if deletion was successful, False otherwise

        Raises:
            RepositoryError: If delete operation fails
        """

    @abstractmethod
    async def find_graphs_by_metadata(self, metadata_filter: dict[str, Any]) -> list[str]:
        """
        Find graphs by metadata criteria.

        Args:
            metadata_filter: Dictionary of metadata key-value pairs to match

        Returns:
            List of graph identifiers matching the criteria

        Raises:
            RepositoryError: If search operation fails
        """

    @abstractmethod
    async def get_graph_metadata(self, graph_id: str) -> dict[str, Any] | None:
        """
        Get metadata for a specific graph.

        Args:
            graph_id: Unique graph identifier

        Returns:
            Graph metadata dictionary or None if not found

        Raises:
            RepositoryError: If metadata retrieval fails
        """

    @abstractmethod
    async def list_graphs(
        self,
        limit: int | None = None,
        offset: int | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        List graphs with optional filtering and pagination.

        Args:
            limit: Maximum number of results to return
            offset: Number of results to skip
            created_after: Only return graphs created after this date
            created_before: Only return graphs created before this date

        Returns:
            List of graph summary dictionaries

        Raises:
            RepositoryError: If list operation fails
        """

    @abstractmethod
    async def get_graph_statistics(self, graph_id: str) -> dict[str, Any] | None:
        """
        Get statistics for a specific graph.

        Args:
            graph_id: Unique graph identifier

        Returns:
            Graph statistics dictionary or None if not found

        Raises:
            RepositoryError: If statistics retrieval fails
        """

    @abstractmethod
    async def add_node_to_graph(self, graph_id: str, node: FactNode) -> bool:
        """
        Add a single node to an existing graph.

        Args:
            graph_id: Unique graph identifier
            node: Fact node to add

        Returns:
            True if addition was successful, False otherwise

        Raises:
            RepositoryError: If add operation fails
        """

    @abstractmethod
    async def add_edge_to_graph(self, graph_id: str, edge: FactEdge) -> bool:
        """
        Add a single edge to an existing graph.

        Args:
            graph_id: Unique graph identifier
            edge: Fact edge to add

        Returns:
            True if addition was successful, False otherwise

        Raises:
            RepositoryError: If add operation fails
        """

    @abstractmethod
    async def add_cluster_to_graph(self, graph_id: str, cluster: FactCluster) -> bool:
        """
        Add a single cluster to an existing graph.

        Args:
            graph_id: Unique graph identifier
            cluster: Fact cluster to add

        Returns:
            True if addition was successful, False otherwise

        Raises:
            RepositoryError: If add operation fails
        """

    @abstractmethod
    async def remove_node_from_graph(self, graph_id: str, node_id: str) -> bool:
        """
        Remove a node from an existing graph.

        Args:
            graph_id: Unique graph identifier
            node_id: Node identifier to remove

        Returns:
            True if removal was successful, False otherwise

        Raises:
            RepositoryError: If remove operation fails
        """

    @abstractmethod
    async def backup_graph(self, graph_id: str, backup_location: str) -> bool:
        """
        Create a backup of a specific graph.

        Args:
            graph_id: Unique graph identifier
            backup_location: Location to store the backup

        Returns:
            True if backup was successful, False otherwise

        Raises:
            RepositoryError: If backup operation fails
        """

    @abstractmethod
    async def restore_graph(self, backup_location: str, new_graph_id: str | None = None) -> str:
        """
        Restore a graph from backup.

        Args:
            backup_location: Location of the backup
            new_graph_id: Optional new identifier for the restored graph

        Returns:
            Identifier of the restored graph

        Raises:
            RepositoryError: If restore operation fails
        """

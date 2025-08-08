"""
Graph repository implementation.

This module implements the GraphRepository interface for managing
graph data access and persistence operations.
"""

import logging
from datetime import datetime
from typing import Any

from agent.models.graph import FactCluster, FactEdge, FactGraph, FactNode
from agent.services.graph.interfaces.graph_repository import GraphRepository
from agent.services.graph.interfaces.storage_strategy import StorageStrategy

logger = logging.getLogger(__name__)


class GraphRepositoryImpl(GraphRepository):
    """
    Concrete implementation of GraphRepository.

    Manages graph data access using configurable storage strategies
    and provides caching and validation capabilities.
    """

    def __init__(self, storage_strategy: StorageStrategy, config: dict[str, Any] | None = None):
        """
        Initialize graph repository.

        Args:
            storage_strategy: Storage strategy to use
            config: Optional configuration
        """
        self._storage_strategy = storage_strategy
        self._config = self._get_default_config()
        if config:
            self._config.update(config)

        self._cache: dict[str, FactGraph] = {}
        self._cache_timestamps: dict[str, datetime] = {}

        logger.info(f"Initialized GraphRepository with {storage_strategy.get_strategy_name()}")

    async def save_graph(self, graph: FactGraph) -> bool:
        """
        Save a complete graph.

        Args:
            graph: Graph to save

        Returns:
            True if save successful
        """
        try:
            # Validate graph before saving
            if not self._validate_graph(graph):
                logger.error(f"Graph validation failed for {graph.graph_id}")
                return False

            # Save using storage strategy
            success = await self._storage_strategy.save_graph(graph)

            if success:
                # Update cache
                if self._config["enable_cache"]:
                    self._cache[graph.graph_id] = graph
                    self._cache_timestamps[graph.graph_id] = datetime.now()

                logger.info(f"Successfully saved graph {graph.graph_id}")
                return True
            else:
                logger.error(f"Failed to save graph {graph.graph_id}")
                return False

        except Exception as e:
            logger.error(f"Error saving graph {graph.graph_id}: {str(e)}")
            return False

    async def load_graph(self, graph_id: str) -> FactGraph | None:
        """
        Load a graph by ID.

        Args:
            graph_id: ID of the graph to load

        Returns:
            Loaded graph or None if not found
        """
        try:
            # Check cache first
            if self._config["enable_cache"] and graph_id in self._cache:
                cache_time = self._cache_timestamps.get(graph_id)
                if cache_time and self._is_cache_valid(cache_time):
                    logger.debug(f"Returning cached graph {graph_id}")
                    return self._cache[graph_id]
                else:
                    # Remove expired cache entry
                    self._remove_from_cache(graph_id)

            # Load from storage
            graph = await self._storage_strategy.load_graph(graph_id)

            if graph:
                # Add to cache
                if self._config["enable_cache"]:
                    self._cache[graph_id] = graph
                    self._cache_timestamps[graph_id] = datetime.now()

                logger.info(f"Successfully loaded graph {graph_id}")
                return graph
            else:
                logger.warning(f"Graph {graph_id} not found")
                return None

        except Exception as e:
            logger.error(f"Error loading graph {graph_id}: {str(e)}")
            return None

    async def update_graph(self, graph: FactGraph) -> bool:
        """
        Update an existing graph.

        Args:
            graph: Graph with updates

        Returns:
            True if update successful
        """
        try:
            # For most storage strategies, update is the same as save
            success = await self.save_graph(graph)

            if success:
                logger.info(f"Successfully updated graph {graph.graph_id}")

            return success

        except Exception as e:
            logger.error(f"Error updating graph {graph.graph_id}: {str(e)}")
            return False

    async def delete_graph(self, graph_id: str) -> bool:
        """
        Delete a graph.

        Args:
            graph_id: ID of the graph to delete

        Returns:
            True if deletion successful
        """
        try:
            # Delete from storage
            success = await self._storage_strategy.delete_graph(graph_id)

            if success:
                # Remove from cache
                self._remove_from_cache(graph_id)
                logger.info(f"Successfully deleted graph {graph_id}")
                return True
            else:
                logger.error(f"Failed to delete graph {graph_id}")
                return False

        except Exception as e:
            logger.error(f"Error deleting graph {graph_id}: {str(e)}")
            return False

    async def find_graphs_by_criteria(self, criteria: dict[str, Any]) -> list[FactGraph]:
        """
        Find graphs matching criteria.

        Args:
            criteria: Search criteria

        Returns:
            List of matching graphs
        """
        try:
            # This would typically be implemented by the storage strategy
            # For now, we'll load all graphs and filter (not efficient for large datasets)
            all_graph_ids = await self.list_graphs()
            matching_graphs = []

            for graph_id in all_graph_ids:
                graph = await self.load_graph(graph_id)
                if graph and self._matches_criteria(graph, criteria):
                    matching_graphs.append(graph)

            logger.info(f"Found {len(matching_graphs)} graphs matching criteria")
            return matching_graphs

        except Exception as e:
            logger.error(f"Error finding graphs by criteria: {str(e)}")
            return []

    async def list_graphs(self) -> list[str]:
        """
        List all graph IDs.

        Returns:
            List of graph IDs
        """
        try:
            graph_ids = await self._storage_strategy.list_graphs()
            logger.debug(f"Found {len(graph_ids)} graphs")
            return graph_ids

        except Exception as e:
            logger.error(f"Error listing graphs: {str(e)}")
            return []

    async def save_node(self, node: FactNode, graph_id: str) -> bool:
        """
        Save a single node.

        Args:
            node: Node to save
            graph_id: ID of the parent graph

        Returns:
            True if save successful
        """
        try:
            success = await self._storage_strategy.save_node(node, graph_id)

            if success:
                # Update cache if graph is cached
                if self._config["enable_cache"] and graph_id in self._cache:
                    self._cache[graph_id].add_node(node)

                logger.debug(f"Successfully saved node {node.node_id}")
                return True
            else:
                logger.error(f"Failed to save node {node.node_id}")
                return False

        except Exception as e:
            logger.error(f"Error saving node {node.node_id}: {str(e)}")
            return False

    async def load_node(self, node_id: str, graph_id: str) -> FactNode | None:
        """
        Load a single node.

        Args:
            node_id: ID of the node to load
            graph_id: ID of the parent graph

        Returns:
            Loaded node or None if not found
        """
        try:
            # Check cache first
            if self._config["enable_cache"] and graph_id in self._cache:
                cached_graph = self._cache[graph_id]
                if node_id in cached_graph.nodes:
                    return cached_graph.nodes[node_id]

            # Load from storage
            node = await self._storage_strategy.load_node(node_id, graph_id)

            if node:
                logger.debug(f"Successfully loaded node {node_id}")
                return node
            else:
                logger.warning(f"Node {node_id} not found in graph {graph_id}")
                return None

        except Exception as e:
            logger.error(f"Error loading node {node_id}: {str(e)}")
            return None

    async def update_node(self, node: FactNode, graph_id: str) -> bool:
        """
        Update a single node.

        Args:
            node: Node with updates
            graph_id: ID of the parent graph

        Returns:
            True if update successful
        """
        return await self.save_node(node, graph_id)

    async def delete_node(self, node_id: str, graph_id: str) -> bool:
        """
        Delete a single node.

        Args:
            node_id: ID of the node to delete
            graph_id: ID of the parent graph

        Returns:
            True if deletion successful
        """
        try:
            success = await self._storage_strategy.delete_node(node_id, graph_id)

            if success:
                # Update cache
                if self._config["enable_cache"] and graph_id in self._cache:
                    cached_graph = self._cache[graph_id]
                    if node_id in cached_graph.nodes:
                        cached_graph.remove_node(node_id)

                logger.debug(f"Successfully deleted node {node_id}")
                return True
            else:
                logger.error(f"Failed to delete node {node_id}")
                return False

        except Exception as e:
            logger.error(f"Error deleting node {node_id}: {str(e)}")
            return False

    async def save_edge(self, edge: FactEdge, graph_id: str) -> bool:
        """
        Save a single edge.

        Args:
            edge: Edge to save
            graph_id: ID of the parent graph

        Returns:
            True if save successful
        """
        try:
            success = await self._storage_strategy.save_edge(edge, graph_id)

            if success:
                # Update cache
                if self._config["enable_cache"] and graph_id in self._cache:
                    self._cache[graph_id].add_edge(edge)

                logger.debug(f"Successfully saved edge {edge.source_id} -> {edge.target_id}")
                return True
            else:
                logger.error(f"Failed to save edge {edge.source_id} -> {edge.target_id}")
                return False

        except Exception as e:
            logger.error(f"Error saving edge {edge.source_id} -> {edge.target_id}: {str(e)}")
            return False

    async def save_cluster(self, cluster: FactCluster, graph_id: str) -> bool:
        """
        Save a single cluster.

        Args:
            cluster: Cluster to save
            graph_id: ID of the parent graph

        Returns:
            True if save successful
        """
        try:
            success = await self._storage_strategy.save_cluster(cluster, graph_id)

            if success:
                # Update cache
                if self._config["enable_cache"] and graph_id in self._cache:
                    self._cache[graph_id].add_cluster(cluster)

                logger.debug(f"Successfully saved cluster {cluster.cluster_id}")
                return True
            else:
                logger.error(f"Failed to save cluster {cluster.cluster_id}")
                return False

        except Exception as e:
            logger.error(f"Error saving cluster {cluster.cluster_id}: {str(e)}")
            return False

    def _validate_graph(self, graph: FactGraph) -> bool:
        """Validate graph before saving."""
        if not graph.graph_id:
            logger.error("Graph ID is required")
            return False

        if not graph.nodes:
            logger.warning(f"Graph {graph.graph_id} has no nodes")

        # Validate node references in edges
        for edge in graph.edges:
            if edge.source_id not in graph.nodes:
                logger.error(f"Edge references non-existent source node: {edge.source_id}")
                return False
            if edge.target_id not in graph.nodes:
                logger.error(f"Edge references non-existent target node: {edge.target_id}")
                return False

        # Validate node references in clusters
        for cluster in graph.clusters.values():
            for node in cluster.nodes:
                if node.node_id not in graph.nodes:
                    logger.error(f"Cluster references non-existent node: {node.node_id}")
                    return False

        return True

    def _matches_criteria(self, graph: FactGraph, criteria: dict[str, Any]) -> bool:
        """Check if graph matches search criteria."""
        # Simple criteria matching - can be extended
        for key, value in criteria.items():
            if key == "min_nodes" and len(graph.nodes) < value:
                return False
            elif key == "max_nodes" and len(graph.nodes) > value:
                return False
            elif key == "has_clusters" and value and not graph.clusters:
                return False

        return True

    def _is_cache_valid(self, cache_time: datetime) -> bool:
        """Check if cache entry is still valid."""
        cache_ttl = self._config["cache_ttl_seconds"]
        return (datetime.now() - cache_time).total_seconds() < cache_ttl

    def _remove_from_cache(self, graph_id: str) -> None:
        """Remove graph from cache."""
        if graph_id in self._cache:
            del self._cache[graph_id]
        if graph_id in self._cache_timestamps:
            del self._cache_timestamps[graph_id]

    def clear_cache(self) -> None:
        """Clear all cached graphs."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Cleared graph cache")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_graphs": len(self._cache),
            "cache_enabled": self._config["enable_cache"],
            "cache_ttl_seconds": self._config["cache_ttl_seconds"],
        }

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration."""
        return {
            "enable_cache": True,
            "cache_ttl_seconds": 3600,  # 1 hour
        }

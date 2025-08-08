"""
Main graph service for fact verification.

This module provides the main service class that orchestrates
clustering, verification, and storage of fact verification graphs.
"""

import logging
import uuid
from datetime import datetime
from typing import Any

from agent.models import FactCluster, FactEdge, FactGraph, FactNode, VerificationResult

from .config import GraphServiceConfig, get_config_manager
from .factories import GraphComponentFactory, StrategyFactory
from .interfaces import ClusteringStrategy, StorageStrategy, VerificationStrategy

logger = logging.getLogger(__name__)


class GraphService:
    """
    Main service for graph-based fact verification.

    This service orchestrates the entire fact verification pipeline:
    1. Creates and manages fact graphs
    2. Applies clustering strategies to group related facts
    3. Executes verification strategies on clusters
    4. Stores results and maintains verification history
    """

    def __init__(
        self,
        component_factory: GraphComponentFactory | None = None,
        strategy_factory: StrategyFactory | None = None,
        config: GraphServiceConfig | None = None,
    ):
        """
        Initialize graph service.

        Args:
            component_factory: Factory for creating graph components
            strategy_factory: Factory for creating strategies
            config: Service configuration
        """
        self._config = config or get_config_manager().get_config()
        self._component_factory = component_factory or GraphComponentFactory()
        self._strategy_factory = strategy_factory or StrategyFactory()

        # Initialize repositories
        self._graph_repository = self._component_factory.create_graph_repository()
        self._verification_repository = self._component_factory.create_verification_repository()

        # Current strategies
        self._clustering_strategy: ClusteringStrategy | None = None
        self._verification_strategy: VerificationStrategy | None = None
        self._storage_strategy: StorageStrategy | None = None

        # Active graphs cache
        self._active_graphs: dict[str, FactGraph] = {}

        # Initialize default strategies
        self._initialize_strategies()

        logger.info("GraphService initialized")

    async def create_graph(self, graph_id: str | None = None, metadata: dict[str, Any] | None = None) -> FactGraph:
        """
        Create a new fact graph.

        Args:
            graph_id: Optional graph identifier
            metadata: Optional graph metadata

        Returns:
            Created fact graph
        """
        if graph_id is None:
            graph_id = str(uuid.uuid4())

        graph = FactGraph()

        # Save graph
        await self._graph_repository.save_graph(graph)
        self._active_graphs[graph_id] = graph

        logger.info(f"Created graph: {graph_id}")
        return graph

    async def add_fact_node(
        self,
        graph_id: str,
        claim: str,
        source: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FactNode:
        """
        Add a fact node to a graph.

        Args:
            graph_id: Graph identifier
            claim: Fact claim text
            source: Optional source information
            confidence: Optional confidence score
            metadata: Optional node metadata

        Returns:
            Created fact node
        """
        graph = await self._get_graph(graph_id)

        node_id = str(uuid.uuid4())
        node = FactNode()

        graph.nodes[node_id] = node
        graph.updated_at = datetime.now()

        # Save updated graph
        await self._graph_repository.save_node(graph_id, node)

        logger.info(f"Added fact node {node_id} to graph {graph_id}")
        return node

    async def add_fact_edge(
        self,
        graph_id: str,
        source_node_id: str,
        target_node_id: str,
        relationship_type: str,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> FactEdge:
        """
        Add a fact edge to a graph.

        Args:
            graph_id: Graph identifier
            source_node_id: Source node identifier
            target_node_id: Target node identifier
            relationship_type: Type of relationship
            weight: Edge weight
            metadata: Optional edge metadata

        Returns:
            Created fact edge
        """
        graph = await self._get_graph(graph_id)

        # Validate nodes exist
        if source_node_id not in graph.nodes:
            raise ValueError(f"Source node {source_node_id} not found")
        if target_node_id not in graph.nodes:
            raise ValueError(f"Target node {target_node_id} not found")

        edge_id = str(uuid.uuid4())
        edge = FactEdge(
            id=edge_id,
            source_id=source_node_id,
            target_id=target_node_id,
            relationship_type=relationship_type,
            strength=weight,
            metadata=metadata or {},
            created_at=datetime.now(),
        )

        graph.edges[edge_id] = edge
        graph.updated_at = datetime.now()

        # Save updated graph
        await self._graph_repository.save_edge(graph_id, edge)

        logger.info(f"Added fact edge {edge_id} to graph {graph_id}")
        return edge

    async def cluster_graph(
        self, graph_id: str, strategy_name: str | None = None, strategy_config: dict[str, Any] | None = None
    ) -> list[FactCluster]:
        """
        Apply clustering to a graph.

        Args:
            graph_id: Graph identifier
            strategy_name: Optional clustering strategy name
            strategy_config: Optional strategy configuration

        Returns:
            List of created clusters
        """
        graph = await self._get_graph(graph_id)

        # Get clustering strategy
        if strategy_name:
            clustering_strategy = self._strategy_factory.create_clustering_strategy(
                strategy_name, strategy_config)
        else:
            clustering_strategy = self._clustering_strategy

        if not clustering_strategy:
            raise ValueError("No clustering strategy available")

        # Create clusters
        clusters = await clustering_strategy.create_clusters(list(graph.nodes.values()))

        # Add clusters to graph
        for cluster in clusters:
            graph.clusters[cluster.cluster_id] = cluster

        graph.updated_at = datetime.now()

        # Save clusters
        for cluster in clusters:
            await self._graph_repository.save_cluster(graph_id, cluster)

        logger.info(f"Created {len(clusters)} clusters for graph {graph_id}")
        return clusters

    async def verify_graph(
        self, graph_id: str, strategy_name: str | None = None, strategy_config: dict[str, Any] | None = None
    ) -> dict[str, VerificationResult]:
        """
        Verify facts in a graph.

        Args:
            graph_id: Graph identifier
            strategy_name: Optional verification strategy name
            strategy_config: Optional strategy configuration

        Returns:
            Dictionary mapping cluster IDs to verification results
        """
        graph = await self._get_graph(graph_id)

        # Ensure graph has clusters
        if not graph.clusters:
            await self.cluster_graph(graph_id)
            graph = await self._get_graph(graph_id)  # Refresh graph

        # Get verification strategy
        if strategy_name:
            verification_strategy = self._strategy_factory.create_verification_strategy(
                strategy_name, strategy_config)
        else:
            verification_strategy = self._verification_strategy

        if not verification_strategy:
            raise ValueError("No verification strategy available")

        # Verify clusters
        results = {}
        for cluster_id, cluster in graph.clusters.items():
            try:
                result = await verification_strategy.verify_cluster(cluster, list(graph.nodes.values()))
                results[cluster_id] = result

                # Save verification result
                await self._verification_repository.save_verification_result(graph_id, cluster_id, result)

            except Exception as e:
                logger.error(
                    f"Failed to verify cluster {cluster_id}: {str(e)}")
                # Create error result
                error_result = VerificationResult(
                    result_id=str(uuid.uuid4()),
                    node_id=None,
                    cluster_id=cluster_id,
                    verification_status="error",
                    confidence_score=0.0,
                    evidence=[],
                    reasoning=f"Verification failed: {str(e)}",
                    metadata={"error": str(e)},
                    verified_at=datetime.now(),
                )
                results[cluster_id] = error_result

        logger.info(f"Verified {len(results)} clusters for graph {graph_id}")
        return results

    async def get_graph(self, graph_id: str) -> FactGraph:
        """Get a graph by ID."""
        return await self._get_graph(graph_id)

    async def list_graphs(self) -> list[str]:
        """List all available graph IDs."""
        return await self._graph_repository.list_graphs()

    async def delete_graph(self, graph_id: str) -> None:
        """Delete a graph."""
        await self._graph_repository.delete_graph(graph_id)
        if graph_id in self._active_graphs:
            del self._active_graphs[graph_id]
        logger.info(f"Deleted graph: {graph_id}")

    async def get_verification_history(self, graph_id: str, limit: int | None = None) -> list[VerificationResult]:
        """Get verification history for a graph."""
        return await self._verification_repository.get_verification_history(graph_id, limit)

    async def get_verification_summary(self, graph_id: str) -> dict[str, Any]:
        """Get verification summary for a graph."""
        return await self._verification_repository.get_verification_summary(graph_id)

    def set_clustering_strategy(self, strategy_name: str, config: dict[str, Any] | None = None) -> None:
        """Set the default clustering strategy."""
        self._clustering_strategy = self._strategy_factory.create_clustering_strategy(
            strategy_name, config)
        logger.info(f"Set clustering strategy: {strategy_name}")

    def set_verification_strategy(self, strategy_name: str, config: dict[str, Any] | None = None) -> None:
        """Set the default verification strategy."""
        self._verification_strategy = self._strategy_factory.create_verification_strategy(
            strategy_name, config)
        logger.info(f"Set verification strategy: {strategy_name}")

    def set_storage_strategy(self, strategy_name: str, config: dict[str, Any] | None = None) -> None:
        """Set the default storage strategy."""
        self._storage_strategy = self._strategy_factory.create_storage_strategy(
            strategy_name, config)
        # Update repositories with new storage strategy
        self._graph_repository = self._component_factory.create_graph_repository(
            storage_strategy=self._storage_strategy
        )
        self._verification_repository = self._component_factory.create_verification_repository(
            storage_strategy=self._storage_strategy
        )
        logger.info(f"Set storage strategy: {strategy_name}")

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on the service."""
        health_status = {"service": "healthy",
                         "timestamp": datetime.now().isoformat(), "components": {}}

        try:
            # Check storage strategy
            if self._storage_strategy:
                await self._storage_strategy.connect()
                health_status["components"]["storage"] = "healthy"
            else:
                health_status["components"]["storage"] = "not_configured"

            # Check repositories
            graphs = await self._graph_repository.list_graphs()
            health_status["components"]["graph_repository"] = "healthy"
            health_status["active_graphs"] = len(self._active_graphs)
            health_status["total_graphs"] = len(graphs)

        except Exception as e:
            health_status["service"] = "unhealthy"
            health_status["error"] = str(e)
            logger.error(f"Health check failed: {str(e)}")

        return health_status

    async def _get_graph(self, graph_id: str) -> FactGraph:
        """Get graph from cache or repository."""
        if graph_id in self._active_graphs:
            return self._active_graphs[graph_id]

        graph = await self._graph_repository.load_graph(graph_id)
        if graph:
            self._active_graphs[graph_id] = graph
            return graph

        raise ValueError(f"Graph {graph_id} not found")

    def _initialize_strategies(self) -> None:
        """Initialize default strategies."""
        try:
            # Initialize clustering strategy
            clustering_strategy_name = self._config.clustering.default_strategy
            clustering_config = get_config_manager(
            ).get_clustering_config(clustering_strategy_name)
            self._clustering_strategy = self._strategy_factory.create_clustering_strategy(
                clustering_strategy_name, clustering_config
            )

            # Initialize verification strategy
            verification_strategy_name = self._config.verification.default_strategy
            verification_config = get_config_manager(
            ).get_verification_config(verification_strategy_name)
            self._verification_strategy = self._strategy_factory.create_verification_strategy(
                verification_strategy_name, verification_config
            )

            # Initialize storage strategy
            storage_strategy_name = self._config.storage.default_strategy
            storage_config = get_config_manager().get_storage_config(storage_strategy_name)
            self._storage_strategy = self._strategy_factory.create_storage_strategy(
                storage_strategy_name, storage_config
            )

            logger.info("Default strategies initialized")

        except Exception as e:
            logger.error(f"Failed to initialize strategies: {str(e)}")
            raise


# Global service instance
_graph_service: GraphService | None = None


def get_graph_service() -> GraphService:
    """Get global graph service instance."""
    global _graph_service
    if _graph_service is None:
        _graph_service = GraphService()
    return _graph_service


def initialize_graph_service(
    component_factory: GraphComponentFactory | None = None,
    strategy_factory: StrategyFactory | None = None,
    config: GraphServiceConfig | None = None,
) -> GraphService:
    """Initialize global graph service."""
    global _graph_service
    _graph_service = GraphService(component_factory, strategy_factory, config)
    logger.info("Global graph service initialized")
    return _graph_service

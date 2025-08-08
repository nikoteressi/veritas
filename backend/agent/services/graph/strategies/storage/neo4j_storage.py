"""
Neo4j storage strategy implementation.

This module implements persistent storage using Neo4j graph database.
"""

import json
import logging
from typing import Any

import numpy as np
from neo4j import Driver, GraphDatabase

from agent.models.graph import FactCluster, FactEdge, FactGraph, FactNode
from agent.services.graph.interfaces.storage_strategy import StorageStrategy

logger = logging.getLogger(__name__)


class Neo4jStorageStrategy(StorageStrategy):
    """
    Storage strategy using Neo4j graph database.

    Provides persistent storage for fact verification graphs
    with full CRUD operations and relationship management.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize Neo4j storage strategy."""
        self._config = self._get_default_config()
        if config:
            self._config.update(config)

        self._validate_config(self._config)
        self._driver: Driver | None = None

        logger.info(f"Initialized {self.get_strategy_name()} with config")

    async def connect(self) -> bool:
        """
        Connect to Neo4j database.

        Returns:
            True if connection successful
        """
        try:
            self._driver = GraphDatabase.driver(
                self._config["uri"],
                auth=(self._config["username"], self._config["password"]),
                max_connection_lifetime=self._config["max_connection_lifetime"],
                max_connection_pool_size=self._config["max_connection_pool_size"],
            )

            # Test connection
            with self._driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]

            if test_value == 1:
                logger.info("Successfully connected to Neo4j")
                await self._create_constraints()
                return True
            else:
                logger.error("Neo4j connection test failed")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Neo4j database."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Disconnected from Neo4j")

    async def _create_constraints(self) -> None:
        """Create necessary constraints and indexes."""
        constraints = [
            "CREATE CONSTRAINT fact_node_id IF NOT EXISTS FOR (n:FactNode) REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT fact_cluster_id IF NOT EXISTS FOR (c:FactCluster) REQUIRE c.cluster_id IS UNIQUE",
            "CREATE CONSTRAINT fact_graph_id IF NOT EXISTS FOR (g:FactGraph) REQUIRE g.graph_id IS UNIQUE",
        ]

        indexes = [
            "CREATE INDEX fact_node_domain IF NOT EXISTS FOR (n:FactNode) ON (n.domain)",
            "CREATE INDEX fact_node_status IF NOT EXISTS FOR (n:FactNode) ON (n.verification_status)",
            "CREATE INDEX fact_cluster_type IF NOT EXISTS FOR (c:FactCluster) ON (c.cluster_type)",
        ]

        with self._driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.warning(f"Constraint creation warning: {str(e)}")

            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    logger.warning(f"Index creation warning: {str(e)}")

    async def save_graph(self, graph: FactGraph) -> bool:
        """
        Save complete graph to Neo4j.

        Args:
            graph: Fact graph to save

        Returns:
            True if save successful
        """
        if not self._driver:
            logger.error("Not connected to Neo4j")
            return False

        try:
            with self._driver.session() as session:
                # Start transaction
                with session.begin_transaction() as tx:
                    # Save graph metadata
                    self._save_graph_metadata(tx, graph)

                    # Save nodes
                    for node in graph.nodes.values():
                        self._save_node_tx(tx, node, graph.graph_id)

                    # Save edges
                    for edge in graph.edges:
                        self._save_edge_tx(tx, edge, graph.graph_id)

                    # Save clusters
                    for cluster in graph.clusters.values():
                        self._save_cluster_tx(tx, cluster, graph.graph_id)

                    # Commit transaction
                    tx.commit()

            logger.info(f"Successfully saved graph {graph.graph_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save graph {graph.graph_id}: {str(e)}")
            return False

    def _save_graph_metadata(self, tx, graph: FactGraph) -> None:
        """Save graph metadata."""
        query = """
        MERGE (g:FactGraph {graph_id: $graph_id})
        SET g.created_at = datetime(),
            g.node_count = $node_count,
            g.edge_count = $edge_count,
            g.cluster_count = $cluster_count,
            g.metadata = $metadata
        """

        tx.run(
            query,
            {
                "graph_id": graph.graph_id,
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "cluster_count": len(graph.clusters),
                "metadata": json.dumps(graph.to_dict().get("metadata", {}), default=self._json_serializer),
            },
        )

    async def save_node(self, node: FactNode, graph_id: str) -> bool:
        """
        Save fact node to Neo4j.

        Args:
            node: Fact node to save
            graph_id: ID of the parent graph

        Returns:
            True if save successful
        """
        if not self._driver:
            logger.error("Not connected to Neo4j")
            return False

        try:
            with self._driver.session() as session:
                self._save_node_tx(session, node, graph_id)

            logger.debug(f"Successfully saved node {node.node_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save node {node.node_id}: {str(e)}")
            return False

    def _save_node_tx(self, tx, node: FactNode, graph_id: str) -> None:
        """Save node within transaction."""
        query = """
        MERGE (n:FactNode {node_id: $node_id})
        SET n.claim = $claim,
            n.domain = $domain,
            n.confidence = $confidence,
            n.verification_status = $verification_status,
            n.embedding = $embedding,
            n.metadata = $metadata,
            n.graph_id = $graph_id,
            n.updated_at = datetime()
        """

        tx.run(
            query,
            {
                "node_id": node.node_id,
                "claim": node.claim,
                "domain": node.domain,
                "confidence": node.confidence,
                "verification_status": node.verification_status.value if node.verification_status else None,
                "embedding": node.embedding.tolist() if node.embedding is not None else None,
                "metadata": json.dumps(node.metadata or {}, default=self._json_serializer),
                "graph_id": graph_id,
            },
        )

        # Link to graph
        link_query = """
        MATCH (g:FactGraph {graph_id: $graph_id})
        MATCH (n:FactNode {node_id: $node_id})
        MERGE (g)-[:CONTAINS_NODE]->(n)
        """

        tx.run(link_query, {"graph_id": graph_id, "node_id": node.node_id})

    async def save_edge(self, edge: FactEdge, graph_id: str) -> bool:
        """
        Save fact edge to Neo4j.

        Args:
            edge: Fact edge to save
            graph_id: ID of the parent graph

        Returns:
            True if save successful
        """
        if not self._driver:
            logger.error("Not connected to Neo4j")
            return False

        try:
            with self._driver.session() as session:
                self._save_edge_tx(session, edge, graph_id)

            logger.debug(f"Successfully saved edge {edge.source_id} -> {edge.target_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save edge {edge.source_id} -> {edge.target_id}: {str(e)}")
            return False

    def _save_edge_tx(self, tx, edge: FactEdge, graph_id: str) -> None:
        """Save edge within transaction."""
        query = """
        MATCH (source:FactNode {node_id: $source_id})
        MATCH (target:FactNode {node_id: $target_id})
        MERGE (source)-[r:RELATES_TO {
            relationship_type: $relationship_type,
            graph_id: $graph_id
        }]->(target)
        SET r.strength = $strength,
            r.metadata = $metadata,
            r.updated_at = datetime()
        """

        tx.run(
            query,
            {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relationship_type": edge.relationship_type.value if edge.relationship_type else None,
                "strength": edge.strength,
                "metadata": json.dumps(edge.metadata or {}, default=self._json_serializer),
                "graph_id": graph_id,
            },
        )

    async def save_cluster(self, cluster: FactCluster, graph_id: str) -> bool:
        """
        Save fact cluster to Neo4j.

        Args:
            cluster: Fact cluster to save
            graph_id: ID of the parent graph

        Returns:
            True if save successful
        """
        if not self._driver:
            logger.error("Not connected to Neo4j")
            return False

        try:
            with self._driver.session() as session:
                self._save_cluster_tx(session, cluster, graph_id)

            logger.debug(f"Successfully saved cluster {cluster.cluster_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save cluster {cluster.cluster_id}: {str(e)}")
            return False

    def _save_cluster_tx(self, tx, cluster: FactCluster, graph_id: str) -> None:
        """Save cluster within transaction."""
        # Create cluster node
        query = """
        MERGE (c:FactCluster {cluster_id: $cluster_id})
        SET c.cluster_type = $cluster_type,
            c.verification_strategy = $verification_strategy,
            c.node_count = $node_count,
            c.graph_id = $graph_id,
            c.updated_at = datetime()
        """

        tx.run(
            query,
            {
                "cluster_id": cluster.cluster_id,
                "cluster_type": cluster.cluster_type.value if cluster.cluster_type else None,
                "verification_strategy": cluster.verification_strategy,
                "node_count": len(cluster.nodes),
                "graph_id": graph_id,
            },
        )

        # Link cluster to nodes
        for node in cluster.nodes:
            link_query = """
            MATCH (c:FactCluster {cluster_id: $cluster_id})
            MATCH (n:FactNode {node_id: $node_id})
            MERGE (c)-[:CONTAINS_NODE]->(n)
            """

            tx.run(link_query, {"cluster_id": cluster.cluster_id, "node_id": node.node_id})

        # Link to graph
        graph_link_query = """
        MATCH (g:FactGraph {graph_id: $graph_id})
        MATCH (c:FactCluster {cluster_id: $cluster_id})
        MERGE (g)-[:CONTAINS_CLUSTER]->(c)
        """

        tx.run(graph_link_query, {"graph_id": graph_id, "cluster_id": cluster.cluster_id})

    async def load_graph(self, graph_id: str) -> FactGraph | None:
        """
        Load complete graph from Neo4j.

        Args:
            graph_id: ID of the graph to load

        Returns:
            Loaded fact graph or None if not found
        """
        if not self._driver:
            logger.error("Not connected to Neo4j")
            return None

        try:
            with self._driver.session() as session:
                # Load graph metadata
                graph_query = "MATCH (g:FactGraph {graph_id: $graph_id}) RETURN g"
                graph_result = session.run(graph_query, {"graph_id": graph_id})
                graph_record = graph_result.single()

                if not graph_record:
                    logger.warning(f"Graph {graph_id} not found")
                    return None

                # Create graph instance
                graph = FactGraph(graph_id=graph_id)

                # Load nodes
                nodes = await self._load_graph_nodes(session, graph_id)
                for node in nodes:
                    graph.add_node(node)

                # Load edges
                edges = await self._load_graph_edges(session, graph_id)
                for edge in edges:
                    graph.add_edge(edge)

                # Load clusters
                clusters = await self._load_graph_clusters(session, graph_id)
                for cluster in clusters:
                    graph.add_cluster(cluster)

                logger.info(f"Successfully loaded graph {graph_id}")
                return graph

        except Exception as e:
            logger.error(f"Failed to load graph {graph_id}: {str(e)}")
            return None

    async def _load_graph_nodes(self, session, graph_id: str) -> list[FactNode]:
        """Load all nodes for a graph."""
        query = """
        MATCH (g:FactGraph {graph_id: $graph_id})-[:CONTAINS_NODE]->(n:FactNode)
        RETURN n
        """

        result = session.run(query, {"graph_id": graph_id})
        nodes = []

        for record in result:
            node_data = record["n"]
            node = self._create_node_from_record(node_data)
            nodes.append(node)

        return nodes

    def _create_node_from_record(self, record) -> FactNode:
        """Create FactNode from Neo4j record."""
        from agent.models.graph import VerificationStatus

        embedding = None
        if record.get("embedding"):
            embedding = np.array(record["embedding"])

        verification_status = None
        if record.get("verification_status"):
            verification_status = VerificationStatus(record["verification_status"])

        metadata = {}
        if record.get("metadata"):
            try:
                metadata = json.loads(record["metadata"])
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse metadata for node {record.get('node_id')}")

        return FactNode(
            id=record["node_id"],
            claim=record["claim"],
            domain=record.get("domain"),
            confidence=record.get("confidence", 0.0),
            metadata=metadata,
            verification_status=verification_status,
            embedding=embedding,
        )

    def _json_serializer(self, obj):
        """JSON serializer for numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, "isoformat"):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "neo4j_storage"

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate configuration."""
        return self._validate_config(config)

    def _validate_config(self, config: dict[str, Any]) -> bool:
        """Internal config validation."""
        required_keys = ["uri", "username", "password"]

        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

        return True

    def get_config(self) -> dict[str, Any]:
        """Get current configuration."""
        # Don't expose sensitive information
        safe_config = self._config.copy()
        safe_config["password"] = "***"
        return safe_config

    def update_config(self, config: dict[str, Any]) -> None:
        """Update configuration."""
        new_config = self._config.copy()
        new_config.update(config)
        self._validate_config(new_config)
        self._config = new_config
        logger.info(f"Updated config for {self.get_strategy_name()}")

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration."""
        return {
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password",
            "max_connection_lifetime": 3600,
            "max_connection_pool_size": 50,
        }

    # Additional methods would be implemented here for:
    # - delete_graph, delete_node, delete_edge, delete_cluster
    # - list_graphs
    # - _load_graph_edges, _load_graph_clusters
    # etc.

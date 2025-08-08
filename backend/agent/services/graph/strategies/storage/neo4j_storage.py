"""
Neo4j storage strategy implementation.

This module implements persistent storage using Neo4j graph database.
"""

import json
import logging
import uuid
from typing import Any

import numpy as np
from neo4j import Driver, GraphDatabase

from agent.models.graph import (
    FactCluster,
    FactEdge,
    FactGraph,
    FactNode,
    ClusterType,
    VerificationStatus,
)
from agent.services.graph.dependency_injection import injectable
from agent.services.graph.interfaces.config_interface import ConfigInterface
from agent.services.graph.interfaces.storage_strategy import StorageStrategy

logger = logging.getLogger(__name__)


@injectable
class Neo4jStorageStrategy(StorageStrategy):
    """
    Storage strategy using Neo4j graph database.

    Provides persistent storage for fact verification graphs
    with full CRUD operations and relationship management.
    """

    def __init__(self, config_service: ConfigInterface, config: dict[str, Any] | None = None):
        """Initialize Neo4j storage strategy."""
        super().__init__(config)
        self.config_service = config_service
        self._config = self._get_default_config()
        if config:
            self._config.update(config)

        # Override with config from service if available
        try:
            storage_config = self.config_service.get_storage_config()
            if storage_config:
                self._config.update(storage_config)
        except Exception as e:
            logger.warning("Could not load storage config from service: %s", e)

        self._validate_config(self._config)
        self._driver: Driver | None = None

        logger.info("Initialized %s with config", self.get_strategy_name())

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
            logger.error("Failed to connect to Neo4j: %s", str(e))
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
                    logger.warning("Constraint creation warning: %s", str(e))

            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    logger.warning("Index creation warning: %s", str(e))

    async def save_graph(self, graph: FactGraph, graph_id: str = None) -> str:
        """
        Save complete graph to Neo4j.

        Args:
            graph: Fact graph to save
            graph_id: Optional graph identifier to use

        Returns:
            Graph identifier in storage
        """
        if not self._driver:
            logger.error("Not connected to Neo4j")
            raise RuntimeError("Not connected to Neo4j")

        # Use provided graph_id or graph's own id from metadata
        final_graph_id = graph_id or graph.metadata.get("graph_id")

        # If no graph_id provided and none in metadata, generate one
        if not final_graph_id:
            final_graph_id = str(uuid.uuid4())

        # Update graph's id in metadata
        graph.metadata["graph_id"] = final_graph_id

        try:
            with self._driver.session() as session:
                # Start transaction
                with session.begin_transaction() as tx:
                    # Save graph metadata
                    self._save_graph_metadata(tx, graph)

                    # Save nodes
                    for node in graph.nodes.values():
                        self._save_node_tx(tx, node, final_graph_id)

                    # Save edges
                    for edge in graph.edges:
                        self._save_edge_tx(tx, edge, final_graph_id)

                    # Save clusters
                    for cluster in graph.clusters.values():
                        self._save_cluster_tx(tx, cluster, final_graph_id)

                    # Commit transaction
                    tx.commit()

            logger.info("Successfully saved graph %s", final_graph_id)
            return final_graph_id

        except Exception as e:
            logger.error("Failed to save graph %s: %s", final_graph_id, str(e))
            raise

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
                "graph_id": graph.metadata.get("graph_id"),
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "cluster_count": len(graph.clusters),
                "metadata": json.dumps(graph.to_dict().get("metadata", {}), default=self._json_serializer),
            },
        )

    async def save_node(self, node: FactNode, graph_id: str) -> str:
        """
        Save fact node to Neo4j.

        Args:
            node: Fact node to save
            graph_id: ID of the parent graph

        Returns:
            Node identifier in storage
        """
        if not self._driver:
            logger.error("Not connected to Neo4j")
            raise RuntimeError("Not connected to Neo4j")

        try:
            with self._driver.session() as session:
                session.execute_write(self._save_node_tx, node, graph_id)
            logger.debug("Saved node %s", node.node_id)
            return node.node_id
        except Exception as e:
            logger.error("Failed to save node %s: %s", node.node_id, str(e))
            raise

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

    async def save_edge(self, edge: FactEdge, graph_id: str) -> str:
        """
        Save fact edge to Neo4j.

        Args:
            edge: Fact edge to save
            graph_id: ID of the parent graph

        Returns:
            Edge identifier in storage
        """
        if not self._driver:
            logger.error("Not connected to Neo4j")
            raise RuntimeError("Not connected to Neo4j")

        try:
            with self._driver.session() as session:
                session.execute_write(self._save_edge_tx, edge, graph_id)

            logger.debug(
                "Successfully saved edge %s -> %s", edge.source_id, edge.target_id)
            return f"{edge.source_id}->{edge.target_id}"

        except Exception as e:
            logger.error(
                "Failed to save edge %s -> %s: %s", edge.source_id, edge.target_id, str(e))
            raise

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

    async def save_cluster(self, cluster: FactCluster, graph_id: str) -> str:
        """
        Save fact cluster to Neo4j.

        Args:
            cluster: Fact cluster to save
            graph_id: ID of the parent graph

        Returns:
            Cluster identifier in storage
        """
        if not self._driver:
            logger.error("Not connected to Neo4j")
            raise RuntimeError("Not connected to Neo4j")

        try:
            with self._driver.session() as session:
                session.execute_write(self._save_cluster_tx, cluster, graph_id)
            logger.debug("Saved cluster %s", cluster.id)
            return cluster.id
        except Exception as e:
            logger.error(
                "Failed to save cluster %s: %s", cluster.id, str(e))
            raise

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
                "cluster_id": cluster.id,
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

            tx.run(link_query, {
                   "cluster_id": cluster.id, "node_id": node.id})

        # Link to graph
        graph_link_query = """
        MATCH (g:FactGraph {graph_id: $graph_id})
        MATCH (c:FactCluster {cluster_id: $cluster_id})
        MERGE (g)-[:CONTAINS_CLUSTER]->(c)
        """

        tx.run(graph_link_query, {"graph_id": graph_id,
               "cluster_id": cluster.id})

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
                    logger.warning("Graph %s not found", graph_id)
                    return None

                # Create graph instance
                graph = FactGraph()
                graph.metadata["graph_id"] = graph_id

                # Load nodes
                nodes = self._load_graph_nodes(session, graph_id)
                for node in nodes:
                    graph.add_node(node)

                # Load edges
                edges = self._load_graph_edges(session, graph_id)
                for edge in edges:
                    graph.add_edge(edge)

                # Load clusters
                clusters = self._load_graph_clusters(session, graph_id)
                for cluster in clusters:
                    graph.add_cluster(cluster)

                logger.info("Successfully loaded graph %s", graph_id)
                return graph

        except Exception as e:
            logger.error("Failed to load graph %s: %s", graph_id, str(e))
            return None

    def _load_graph_nodes(self, session, graph_id: str) -> list[FactNode]:
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
        embedding = None
        if record.get("embedding"):
            embedding = np.array(record["embedding"])

        verification_status = None
        if record.get("verification_status"):
            verification_status = VerificationStatus(
                record["verification_status"])

        metadata = {}
        if record.get("metadata"):
            try:
                metadata = json.loads(record["metadata"])
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse metadata for node %s", record.get('node_id'))

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

    def validate_config(self) -> bool:
        """Validate configuration."""
        return self._validate_config(self._config)

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

    def update_config(self, new_config: dict[str, Any]) -> None:
        """Update configuration."""
        updated_config = self._config.copy()
        updated_config.update(new_config)
        self._validate_config(updated_config)
        self._config = updated_config
        logger.info("Updated config for %s", self.get_strategy_name())

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration."""
        return {
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password",
            "max_connection_lifetime": 3600,
            "max_connection_pool_size": 50,
        }

    async def delete_graph(self, graph_id: str) -> bool:
        """
        Delete a graph from Neo4j.

        Args:
            graph_id: Graph identifier

        Returns:
            True if deletion was successful, False otherwise
        """
        if not self._driver:
            logger.error("Not connected to Neo4j")
            raise RuntimeError("Not connected to Neo4j")

        try:
            with self._driver.session() as session:
                # Check if graph exists
                check_query = "MATCH (g:FactGraph {graph_id: $graph_id}) RETURN g"
                result = session.run(check_query, {"graph_id": graph_id})
                if not result.single():
                    logger.warning("Graph %s not found for deletion", graph_id)
                    return False

                # Delete all related data
                delete_queries = [
                    "MATCH (g:FactGraph {graph_id: $graph_id})-[:CONTAINS_NODE]->(n:FactNode) DETACH DELETE n",
                    "MATCH (g:FactGraph {graph_id: $graph_id})-[:CONTAINS_CLUSTER]->(c:FactCluster) DETACH DELETE c",
                    "MATCH ()-[r:RELATES_TO {graph_id: $graph_id}]-() DELETE r",
                    "MATCH (g:FactGraph {graph_id: $graph_id}) DELETE g"
                ]

                for query in delete_queries:
                    session.run(query, {"graph_id": graph_id})

            logger.debug("Deleted graph %s", graph_id)
            return True
        except Exception as e:
            logger.error("Failed to delete graph %s: %s", graph_id, str(e))
            raise

    async def list_graphs(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        List all graphs in storage.

        Args:
            limit: Optional limit on number of graphs to return

        Returns:
            List of graph metadata dictionaries
        """
        if not self._driver:
            logger.error("Not connected to Neo4j")
            raise RuntimeError("Not connected to Neo4j")

        try:
            with self._driver.session() as session:
                query = "MATCH (g:FactGraph) RETURN g ORDER BY g.created_at DESC"
                if limit:
                    query += f" LIMIT {limit}"

                result = session.run(query)
                graphs = []

                for record in result:
                    graph_data = record["g"]
                    graphs.append({
                        "graph_id": graph_data["graph_id"],
                        "created_at": graph_data.get("created_at"),
                        "node_count": graph_data.get("node_count", 0),
                        "edge_count": graph_data.get("edge_count", 0),
                        "cluster_count": graph_data.get("cluster_count", 0),
                        "metadata": json.loads(graph_data.get("metadata", "{}"))
                    })

                return graphs
        except Exception as e:
            logger.error("Failed to list graphs: %s", str(e))
            raise

    def _load_graph_edges(self, session, graph_id: str) -> list[FactEdge]:
        """Load all edges for a graph."""
        query = """
        MATCH (g:FactGraph {graph_id: $graph_id})-[:CONTAINS_NODE]->(source:FactNode)
        MATCH (source)-[r:RELATES_TO {graph_id: $graph_id}]->(target:FactNode)
        RETURN source.node_id as source_id, target.node_id as target_id, r
        """

        result = session.run(query, {"graph_id": graph_id})
        edges = []

        for record in result:
            edge_data = record["r"]
            edge = FactEdge(
                source_id=record["source_id"],
                target_id=record["target_id"],
                relationship_type=edge_data.get("relationship_type"),
                strength=edge_data.get("strength", 0.0),
                metadata=json.loads(edge_data.get("metadata", "{}"))
            )
            edges.append(edge)

        return edges

    def _load_graph_clusters(self, session, graph_id: str) -> list[FactCluster]:
        """Load all clusters for a graph."""
        query = """
        MATCH (g:FactGraph {graph_id: $graph_id})-[:CONTAINS_CLUSTER]->(c:FactCluster)
        OPTIONAL MATCH (c)-[:CONTAINS_NODE]->(n:FactNode)
        RETURN c, collect(n.node_id) as node_ids
        """

        result = session.run(query, {"graph_id": graph_id})
        clusters = []

        for record in result:
            cluster_data = record["c"]
            node_ids = record["node_ids"]

            cluster_type = None
            if cluster_data.get("cluster_type"):
                cluster_type = ClusterType(cluster_data["cluster_type"])

            cluster = FactCluster(
                id=cluster_data["cluster_id"],
                cluster_type=cluster_type,
                verification_strategy=cluster_data.get(
                    "verification_strategy"),
                nodes=node_ids  # These will be resolved to actual nodes later
            )
            clusters.append(cluster)

        return clusters

"""
from __future__ import annotations

Neo4j-based persistent storage for fact verification graphs.

This module provides persistent storage capabilities for FactGraph objects
using Neo4j as the backend database, enabling incremental updates and
efficient querying of large graphs.
"""

import json
import logging
from datetime import datetime
from typing import Any

import numpy as np
from neo4j import Driver, GraphDatabase, Session
from neo4j.exceptions import ClientError, DatabaseError, ServiceUnavailable

from agent.models.graph import (
    ClusterType,
    FactCluster,
    FactEdge,
    FactGraph,
    FactNode,
    RelationshipType,
    VerificationStatus,
)
from app.config import Settings

settings = Settings()


def safe_json_dumps(obj: Any) -> str:
    """
    Safely serialize object to JSON, handling numpy types.

    Args:
        obj: Object to serialize

    Returns:
        JSON string
    """

    def convert_numpy_types(obj):
        """Convert numpy types to native Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        else:
            return obj

    try:
        converted_obj = convert_numpy_types(obj)
        return json.dumps(converted_obj)
    except (TypeError, ValueError) as e:
        logging.getLogger(__name__).error(f"Failed to serialize object to JSON: {e}")
        return "{}"


class Neo4jGraphStorage:
    """
    Neo4j-based persistent storage for fact verification graphs.

    Provides efficient storage, retrieval, and incremental updates
    of FactGraph objects with support for complex queries and analytics.
    """

    def __init__(self, uri: str = None, auth: tuple = None, database: str = None):
        """Initialize Neo4j connection."""
        self.uri = uri or settings.neo4j_uri
        self.auth = auth or (settings.neo4j_user, settings.neo4j_password)
        self.database = database or settings.neo4j_database
        self.driver: Driver | None = None
        self.logger = logging.getLogger(__name__)

        self._connect()
        self._create_constraints()

    def _connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=self.auth)
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            self.logger.info("Successfully connected to Neo4j")
        except ServiceUnavailable as e:
            self.logger.error("Failed to connect to Neo4j: %s", e)
            raise

    def _create_constraints(self):
        """Create necessary constraints and indexes."""
        constraints = [
            "CREATE CONSTRAINT fact_node_id IF NOT EXISTS FOR (n:FactNode) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT fact_edge_id IF NOT EXISTS FOR (e:FactEdge) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT fact_cluster_id IF NOT EXISTS FOR (c:FactCluster) REQUIRE c.id IS UNIQUE",
            "CREATE INDEX fact_node_domain IF NOT EXISTS FOR (n:FactNode) ON (n.domain)",
            "CREATE INDEX fact_node_confidence IF NOT EXISTS FOR (n:FactNode) ON (n.confidence)",
            "CREATE INDEX fact_edge_type IF NOT EXISTS FOR (e:FactEdge) ON (e.relationship_type)",
            "CREATE INDEX fact_cluster_type IF NOT EXISTS FOR (c:FactCluster) ON (c.cluster_type)",
        ]

        with self.driver.session(database=self.database) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except (ClientError, DatabaseError) as e:
                    self.logger.debug(
                        "Constraint/Index already exists or failed: %s", e
                    )

    def store_graph(self, graph: FactGraph, graph_id: str = None) -> str:
        """
        Store a complete graph in Neo4j with incremental updates.

        Args:
            graph: The FactGraph to store
            graph_id: Optional graph identifier

        Returns:
            The graph ID used for storage
        """
        if not graph_id:
            graph_id = f"graph_{datetime.now().isoformat()}"

        with self.driver.session(database=self.database) as session:
            # Store nodes
            self._store_nodes(session, graph.nodes.values(), graph_id)

            # Store edges
            self._store_edges(session, graph.edges.values(), graph_id)

            # Store clusters
            self._store_clusters(session, graph.clusters.values(), graph_id)

            # Store graph metadata
            self._store_graph_metadata(session, graph, graph_id)

        self.logger.info(
            "Stored graph %s with %s nodes, %s edges, %s clusters",
            graph_id,
            len(graph.nodes),
            len(graph.edges),
            len(graph.clusters),
        )
        return graph_id

    def _store_nodes(self, session: Session, nodes: list[FactNode], graph_id: str):
        """Store fact nodes in Neo4j."""
        query = """
        UNWIND $nodes AS node
        MERGE (n:FactNode {id: node.id})
        SET n.claim = node.claim,
            n.domain = node.domain,
            n.confidence = node.confidence,
            n.verification_status = node.verification_status,
            n.metadata = node.metadata,
            n.verification_results = node.verification_results,
            n.embedding = node.embedding,
            n.graph_id = $graph_id,
            n.updated_at = datetime()
        """

        node_data = []
        for node in nodes:
            # Safe enum handling for verification_status
            verification_status_value = None
            if node.verification_status:
                if hasattr(node.verification_status, "value"):
                    verification_status_value = node.verification_status.value
                elif isinstance(node.verification_status, str):
                    verification_status_value = node.verification_status
                else:
                    self.logger.error(
                        f"Invalid verification_status type for node {node.id}: {type(node.verification_status)}"
                    )
                    raise ValueError(
                        f"Invalid verification_status type for node {node.id}"
                    )

            # Serialize embedding if present
            embedding_data = None
            if node.embedding is not None:
                try:
                    # Convert numpy array to list for JSON serialization
                    if hasattr(node.embedding, "tolist"):
                        embedding_data = safe_json_dumps(node.embedding.tolist())
                    else:
                        embedding_data = safe_json_dumps(list(node.embedding))
                except Exception as e:
                    self.logger.warning(
                        f"Failed to serialize embedding for node {node.id}: {e}"
                    )
                    embedding_data = None

            node_data.append(
                {
                    "id": node.id,
                    "claim": node.claim,
                    "domain": node.domain,
                    "confidence": node.confidence,
                    "verification_status": verification_status_value,
                    "metadata": (
                        safe_json_dumps(node.metadata) if node.metadata else "{}"
                    ),
                    "verification_results": (
                        safe_json_dumps(node.verification_results)
                        if node.verification_results
                        else "{}"
                    ),
                    "embedding": embedding_data,
                }
            )

        session.run(query, nodes=node_data, graph_id=graph_id)

    def _store_edges(self, session: Session, edges: list[FactEdge], graph_id: str):
        """Store fact edges in Neo4j."""
        query = """
        UNWIND $edges AS edge
        MATCH (source:FactNode {id: edge.source_id})
        MATCH (target:FactNode {id: edge.target_id})
        MERGE (source)-[r:RELATES {id: edge.id}]->(target)
        SET r.relationship_type = edge.relationship_type,
            r.strength = edge.strength,
            r.metadata = edge.metadata,
            r.graph_id = $graph_id,
            r.updated_at = datetime()
        """

        edge_data = []
        for edge in edges:
            # Safe enum handling for relationship_type
            relationship_type_value = None
            if edge.relationship_type:
                if hasattr(edge.relationship_type, "value"):
                    relationship_type_value = edge.relationship_type.value
                elif isinstance(edge.relationship_type, str):
                    relationship_type_value = edge.relationship_type
                else:
                    self.logger.error(
                        f"Invalid relationship_type type for edge {edge.id}: {type(edge.relationship_type)}"
                    )
                    raise ValueError(
                        f"Invalid relationship_type type for edge {edge.id}"
                    )

            edge_data.append(
                {
                    "id": edge.id,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relationship_type": relationship_type_value,
                    "strength": edge.strength,
                    "metadata": (
                        safe_json_dumps(edge.metadata) if edge.metadata else "{}"
                    ),
                }
            )

        session.run(query, edges=edge_data, graph_id=graph_id)

    def _store_clusters(
        self, session: Session, clusters: list[FactCluster], graph_id: str
    ):
        """Store fact clusters in Neo4j."""
        for cluster in clusters:
            # Safe enum handling for cluster_type
            cluster_type_value = None
            if cluster.cluster_type:
                if hasattr(cluster.cluster_type, "value"):
                    cluster_type_value = cluster.cluster_type.value
                elif isinstance(cluster.cluster_type, str):
                    cluster_type_value = cluster.cluster_type
                else:
                    self.logger.error(
                        f"Invalid cluster_type type for cluster {cluster.id}: {type(cluster.cluster_type)}"
                    )
                    raise ValueError(
                        f"Invalid cluster_type type for cluster {cluster.id}"
                    )

            # Create cluster node
            cluster_query = """
            MERGE (c:FactCluster {id: $cluster_id})
            SET c.cluster_type = $cluster_type,
                c.shared_context = $shared_context,
                c.verification_strategy = $verification_strategy,
                c.metadata = $metadata,
                c.batch_queries = $batch_queries,
                c.shared_sources = $shared_sources,
                c.cluster_verification_result = $cluster_verification_result,
                c.created_at = $created_at,
                c.graph_id = $graph_id,
                c.updated_at = datetime()
            """

            session.run(
                cluster_query,
                cluster_id=cluster.id,
                cluster_type=cluster_type_value,
                shared_context=cluster.shared_context,
                verification_strategy=cluster.verification_strategy,
                metadata=(
                    safe_json_dumps(cluster.metadata) if cluster.metadata else "{}"
                ),
                batch_queries=safe_json_dumps(cluster.batch_queries),
                shared_sources=safe_json_dumps(cluster.shared_sources),
                cluster_verification_result=(
                    safe_json_dumps(cluster.cluster_verification_result)
                    if cluster.cluster_verification_result
                    else None
                ),
                created_at=cluster.created_at.isoformat(),
                graph_id=graph_id,
            )

            # Link cluster to nodes
            link_query = """
            MATCH (c:FactCluster {id: $cluster_id})
            MATCH (n:FactNode {id: $node_id})
            MERGE (c)-[:CONTAINS]->(n)
            """

            for node in cluster.nodes:
                session.run(link_query, cluster_id=cluster.id, node_id=node.id)

    def _store_graph_metadata(self, session: Session, graph: FactGraph, graph_id: str):
        """Store graph-level metadata."""
        stats = graph.get_stats()

        # Ensure all required keys are present with default values
        required_keys = {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "cluster_count": len(graph.clusters),
        }

        for key, default_value in required_keys.items():
            if key not in stats:
                self.logger.warning(
                    f"Missing key '{key}' in graph stats, using default value: {default_value}"
                )
                stats[key] = default_value

        query = """
        MERGE (g:Graph {id: $graph_id})
        SET g.node_count = $node_count,
            g.edge_count = $edge_count,
            g.cluster_count = $cluster_count,
            g.created_at = datetime(),
            g.stats = $stats
        """

        session.run(
            query,
            graph_id=graph_id,
            node_count=stats["node_count"],
            edge_count=stats["edge_count"],
            cluster_count=stats["cluster_count"],
            stats=safe_json_dumps(stats),
        )

    def load_graph(self, graph_id: str) -> FactGraph | None:
        """
        Load a complete graph from Neo4j.

        Args:
            graph_id: The graph identifier

        Returns:
            The loaded FactGraph or None if not found
        """
        with self.driver.session(database=self.database) as session:
            # Load nodes
            nodes = self._load_nodes(session, graph_id)
            if not nodes:
                return None

            # Load edges
            edges = self._load_edges(session, graph_id)

            # Load clusters
            clusters = self._load_clusters(session, graph_id, nodes)

            # Create graph
            graph = FactGraph()

            # Add nodes
            for node in nodes:
                graph.add_node(node)

            # Add edges
            for edge in edges:
                graph.add_edge(edge)

            # Add clusters
            for cluster in clusters:
                graph.add_cluster(cluster)

            return graph

    def _load_nodes(self, session: Session, graph_id: str) -> list[FactNode]:
        """Load fact nodes from Neo4j."""
        query = """
        MATCH (n:FactNode {graph_id: $graph_id})
        RETURN n.id as id, n.claim as claim, n.domain as domain,
               n.confidence as confidence, n.verification_status as verification_status,
               n.metadata as metadata, n.verification_results as verification_results,
               n.embedding as embedding
        """

        result = session.run(query, graph_id=graph_id)
        nodes = []

        for record in result:
            verification_status = None
            if record["verification_status"]:
                verification_status = VerificationStatus(record["verification_status"])

            metadata = json.loads(record["metadata"]) if record["metadata"] else {}
            verification_results = (
                json.loads(record["verification_results"])
                if record["verification_results"]
                else {}
            )

            # Deserialize embedding if present
            embedding = None
            if record["embedding"]:
                try:
                    embedding_list = json.loads(record["embedding"])
                    embedding = np.array(embedding_list)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to deserialize embedding for node {record['id']}: {e}"
                    )
                    embedding = None

            node = FactNode(
                id=record["id"],
                claim=record["claim"],
                domain=record["domain"],
                confidence=record["confidence"],
                verification_status=verification_status,
                metadata=metadata,
                verification_results=verification_results,
                embedding=embedding,
            )
            nodes.append(node)

        return nodes

    def _load_edges(self, session: Session, graph_id: str) -> list[FactEdge]:
        """Load fact edges from Neo4j."""
        query = """
        MATCH (source:FactNode)-[r:RELATES {graph_id: $graph_id}]->(target:FactNode)
        RETURN r.id as id, source.id as source_id, target.id as target_id,
               r.relationship_type as relationship_type, r.strength as strength,
               r.metadata as metadata
        """

        result = session.run(query, graph_id=graph_id)
        edges = []

        for record in result:
            metadata = json.loads(record["metadata"]) if record["metadata"] else {}

            edge = FactEdge(
                id=record["id"],
                source_id=record["source_id"],
                target_id=record["target_id"],
                relationship_type=RelationshipType(record["relationship_type"]),
                strength=record["strength"],
                metadata=metadata,
            )
            edges.append(edge)

        return edges

    def _load_clusters(
        self, session: Session, graph_id: str, nodes: list[FactNode]
    ) -> list[FactCluster]:
        """Load fact clusters from Neo4j."""
        query = """
        MATCH (c:FactCluster {graph_id: $graph_id})-[:CONTAINS]->(n:FactNode)
        RETURN c.id as cluster_id, c.cluster_type as cluster_type,
               c.shared_context as shared_context, c.verification_strategy as verification_strategy,
               c.metadata as metadata, c.batch_queries as batch_queries,
               c.shared_sources as shared_sources, c.cluster_verification_result as cluster_verification_result,
               c.created_at as created_at, collect(n.id) as node_ids
        """

        result = session.run(query, graph_id=graph_id)
        clusters = []
        node_map = {node.id: node for node in nodes}

        for record in result:
            cluster_nodes = [
                node_map[node_id]
                for node_id in record["node_ids"]
                if node_id in node_map
            ]
            metadata = json.loads(record["metadata"]) if record["metadata"] else {}
            batch_queries = (
                json.loads(record["batch_queries"]) if record["batch_queries"] else []
            )
            shared_sources = (
                json.loads(record["shared_sources"]) if record["shared_sources"] else []
            )
            cluster_verification_result = (
                json.loads(record["cluster_verification_result"])
                if record["cluster_verification_result"]
                else None
            )

            # Parse created_at
            created_at = datetime.now()  # Default fallback
            if record["created_at"]:
                try:
                    if isinstance(record["created_at"], str):
                        created_at = datetime.fromisoformat(record["created_at"])
                    else:
                        # Neo4j datetime object
                        created_at = record["created_at"].to_native()
                except (ValueError, TypeError):
                    created_at = datetime.now()

            cluster = FactCluster(
                id=record["cluster_id"],
                nodes=cluster_nodes,
                cluster_type=ClusterType(record["cluster_type"]),
                shared_context=record["shared_context"] or "",
                verification_strategy=record["verification_strategy"] or "batch",
                metadata=metadata,
                created_at=created_at,
                batch_queries=batch_queries,
                shared_sources=shared_sources,
                cluster_verification_result=cluster_verification_result,
            )
            clusters.append(cluster)

        return clusters

    def update_node_verification(
        self,
        node_id: str,
        verification_results: dict[str, Any],
        verification_status: VerificationStatus,
        confidence: float,
    ):
        """Update verification results for a specific node."""
        # Safe enum handling for verification_status
        verification_status_value = None
        if verification_status:
            if hasattr(verification_status, "value"):
                verification_status_value = verification_status.value
            elif isinstance(verification_status, str):
                verification_status_value = verification_status
            else:
                self.logger.error(
                    f"Invalid verification_status type for node {node_id}: {type(verification_status)}"
                )
                raise ValueError(f"Invalid verification_status type for node {node_id}")

        query = """
        MATCH (n:FactNode {id: $node_id})
        SET n.verification_results = $verification_results,
            n.verification_status = $verification_status,
            n.confidence = $confidence,
            n.updated_at = datetime()
        """

        with self.driver.session(database=self.database) as session:
            session.run(
                query,
                node_id=node_id,
                verification_results=json.dumps(verification_results),
                verification_status=verification_status_value,
                confidence=confidence,
            )

    def get_graph_stats(self, graph_id: str) -> dict[str, Any]:
        """Get statistics for a specific graph."""
        query = """
        MATCH (g:Graph {id: $graph_id})
        RETURN g.node_count as node_count, g.edge_count as edge_count,
               g.cluster_count as cluster_count, g.stats as stats
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, graph_id=graph_id)
            record = result.single()

            if record:
                stats = json.loads(record["stats"]) if record["stats"] else {}
                stats.update(
                    {
                        "node_count": record["node_count"],
                        "edge_count": record["edge_count"],
                        "cluster_count": record["cluster_count"],
                    }
                )
                return stats

            return {}

    def get_verification_history(
        self, fact_claim: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get verification history for nodes matching a specific fact claim."""
        query = """
        MATCH (n:FactNode)
        WHERE n.claim CONTAINS $fact_claim OR $fact_claim CONTAINS n.claim
        AND n.verification_results IS NOT NULL
        RETURN n.id as node_id, n.claim as claim, n.domain as domain,
               n.verification_status as verification_status, n.confidence as confidence,
               n.verification_results as verification_results, n.updated_at as updated_at,
               n.graph_id as graph_id
        ORDER BY n.updated_at DESC
        LIMIT $limit
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, fact_claim=fact_claim, limit=limit)
            history = []

            for record in result:
                verification_results = (
                    json.loads(record["verification_results"])
                    if record["verification_results"]
                    else {}
                )

                history_item = {
                    "node_id": record["node_id"],
                    "claim": record["claim"],
                    "domain": record["domain"],
                    "verification_status": record["verification_status"],
                    "confidence": record["confidence"],
                    "verification_results": verification_results,
                    "updated_at": (
                        record["updated_at"].isoformat()
                        if record["updated_at"]
                        else None
                    ),
                    "graph_id": record["graph_id"],
                }
                history.append(history_item)

            return history

    def delete_graph(self, graph_id: str):
        """Delete a complete graph from Neo4j."""
        queries = [
            "MATCH (n:FactNode {graph_id: $graph_id}) DETACH DELETE n",
            "MATCH ()-[r:RELATES {graph_id: $graph_id}]-() DELETE r",
            "MATCH (c:FactCluster {graph_id: $graph_id}) DETACH DELETE c",
            "MATCH (g:Graph {id: $graph_id}) DELETE g",
        ]

        with self.driver.session(database=self.database) as session:
            for query in queries:
                session.run(query, graph_id=graph_id)

        self.logger.info("Deleted graph %s", graph_id)

    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed")

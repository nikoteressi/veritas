"""
from __future__ import annotations

Graph-based verification system data structures.

This module contains the core data structures for the graph-based fact verification system,
including nodes, edges, clusters, and the main graph structure.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class VerificationStatus(Enum):
    """Status of fact verification."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VERIFIED = "verified"
    FAILED = "failed"
    SKIPPED = "skipped"


class RelationshipType(Enum):
    """Types of relationships between facts."""

    SIMILARITY = "similarity"  # Facts are semantically similar
    TEMPORAL = "temporal"  # Facts are related by time
    CAUSAL = "causal"  # One fact causes another
    CONTRADICTION = "contradiction"  # Facts contradict each other
    SUPPORT = "support"  # One fact supports another
    DOMAIN = "domain"  # Facts belong to same domain


class ClusterType(Enum):
    """Types of fact clusters."""

    SIMILARITY_CLUSTER = "similarity"
    DOMAIN_CLUSTER = "domain"
    TEMPORAL_CLUSTER = "temporal"
    CAUSAL_CLUSTER = "causal"


@dataclass
class FactNode:
    """
    Represents a single fact in the verification graph.

    Each node contains the fact information, metadata, and verification status.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    claim: str = ""
    domain: str = "general"
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    verification_status: VerificationStatus = VerificationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)

    # Context from original Fact model
    context: dict[str, Any] = field(default_factory=dict)

    # Verification results
    search_queries: list[str] = field(default_factory=list)
    sources_examined: list[str] = field(default_factory=list)
    verification_result: dict[str, Any] | None = None
    verification_results: dict[str, Any] | None = None  # Added for compatibility

    # Embedding for similarity calculations
    embedding: list[float] | None = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, FactNode):
            return False
        return self.id == other.id


@dataclass
class FactEdge:
    """
    Represents a relationship between two facts in the verification graph.

    Edges capture semantic, temporal, causal, and other relationships between facts.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relationship_type: RelationshipType = RelationshipType.SIMILARITY
    strength: float = 0.0  # Relationship strength (0.0 to 1.0)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, FactEdge):
            return False
        return self.id == other.id


@dataclass
class FactCluster:
    """
    Represents a cluster of related facts that can be verified together.

    Clusters enable batch processing and context-aware verification.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    nodes: list[FactNode] = field(default_factory=list)
    cluster_type: ClusterType = ClusterType.SIMILARITY_CLUSTER
    shared_context: str = ""
    verification_strategy: str = "batch"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    # Cluster-level verification results
    batch_queries: list[str] = field(default_factory=list)
    shared_sources: list[str] = field(default_factory=list)
    cluster_verification_result: dict[str, Any] | None = None

    def add_node(self, node: FactNode):
        """Add a node to the cluster."""
        if node not in self.nodes:
            self.nodes.append(node)

    def remove_node(self, node: FactNode):
        """Remove a node from the cluster."""
        if node in self.nodes:
            self.nodes.remove(node)

    def get_node_ids(self) -> set[str]:
        """Get set of node IDs in this cluster."""
        return {node.id for node in self.nodes}

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, FactCluster):
            return False
        return self.id == other.id


class FactGraph:
    """
    Main graph structure for fact verification.

    Manages nodes, edges, and clusters for efficient fact verification.
    """

    def __init__(self):
        self.nodes: dict[str, FactNode] = {}
        self.edges: dict[str, FactEdge] = {}
        self.clusters: dict[str, FactCluster] = {}
        # node_id -> cluster_ids
        self.node_to_clusters: dict[str, set[str]] = {}
        self.created_at = datetime.now()
        self.metadata: dict[str, Any] = {}

    def add_node(self, node: FactNode) -> str:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        if node.id not in self.node_to_clusters:
            self.node_to_clusters[node.id] = set()
        return node.id

    def add_edge(self, edge: FactEdge) -> str:
        """Add an edge to the graph."""
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError("Both source and target nodes must exist in the graph")
        self.edges[edge.id] = edge
        return edge.id

    def add_cluster(self, cluster: FactCluster) -> str:
        """Add a cluster to the graph."""
        self.clusters[cluster.id] = cluster

        # Update node-to-cluster mapping
        for node in cluster.nodes:
            if node.id in self.node_to_clusters:
                self.node_to_clusters[node.id].add(cluster.id)
            else:
                self.node_to_clusters[node.id] = {cluster.id}

        return cluster.id

    def get_node(self, node_id: str) -> FactNode | None:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_edge(self, edge_id: str) -> FactEdge | None:
        """Get an edge by ID."""
        return self.edges.get(edge_id)

    def get_cluster(self, cluster_id: str) -> FactCluster | None:
        """Get a cluster by ID."""
        return self.clusters.get(cluster_id)

    def get_node_neighbors(self, node_id: str) -> list[FactNode]:
        """Get all neighboring nodes of a given node."""
        neighbors = []
        for edge in self.edges.values():
            if edge.source_id == node_id:
                neighbor = self.get_node(edge.target_id)
                if neighbor:
                    neighbors.append(neighbor)
            elif edge.target_id == node_id:
                neighbor = self.get_node(edge.source_id)
                if neighbor:
                    neighbors.append(neighbor)
        return neighbors

    def get_node_edges(self, node_id: str) -> list[FactEdge]:
        """Get all edges connected to a node."""
        return [edge for edge in self.edges.values() if edge.source_id == node_id or edge.target_id == node_id]

    def get_node_clusters(self, node_id: str) -> list[FactCluster]:
        """Get all clusters containing a node."""
        cluster_ids = self.node_to_clusters.get(node_id, set())
        return [self.clusters[cluster_id] for cluster_id in cluster_ids if cluster_id in self.clusters]

    def get_edges_by_type(self, relationship_type: RelationshipType) -> list[FactEdge]:
        """Get all edges of a specific relationship type."""
        return [edge for edge in self.edges.values() if edge.relationship_type == relationship_type]

    def get_clusters_by_type(self, cluster_type: ClusterType) -> list[FactCluster]:
        """Get all clusters of a specific type."""
        return [cluster for cluster in self.clusters.values() if cluster.cluster_type == cluster_type]

    def remove_node(self, node_id: str):
        """Remove a node and all its edges from the graph."""
        if node_id not in self.nodes:
            return

        # Remove all edges connected to this node
        edges_to_remove = [
            edge_id for edge_id, edge in self.edges.items() if edge.source_id == node_id or edge.target_id == node_id
        ]
        for edge_id in edges_to_remove:
            del self.edges[edge_id]

        # Remove node from clusters
        cluster_ids = self.node_to_clusters.get(node_id, set())
        for cluster_id in cluster_ids:
            if cluster_id in self.clusters:
                cluster = self.clusters[cluster_id]
                node = self.nodes[node_id]
                cluster.remove_node(node)

        # Remove node
        del self.nodes[node_id]
        if node_id in self.node_to_clusters:
            del self.node_to_clusters[node_id]

    def remove_edge(self, edge_id: str):
        """Remove an edge from the graph."""
        if edge_id in self.edges:
            del self.edges[edge_id]

    def remove_cluster(self, cluster_id: str):
        """Remove a cluster from the graph."""
        if cluster_id not in self.clusters:
            return

        cluster = self.clusters[cluster_id]

        # Update node-to-cluster mapping
        for node in cluster.nodes:
            if node.id in self.node_to_clusters:
                self.node_to_clusters[node.id].discard(cluster_id)

        del self.clusters[cluster_id]

    def get_stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "cluster_count": len(self.clusters),
            "avg_cluster_size": (sum(len(c.nodes) for c in self.clusters.values()) / max(len(self.clusters), 1)),
            "relationship_types": {rt.value: len(self.get_edges_by_type(rt)) for rt in RelationshipType},
            "cluster_types": {ct.value: len(self.get_clusters_by_type(ct)) for ct in ClusterType},
            "created_at": self.created_at.isoformat(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert graph to dictionary for serialization."""
        return {
            "nodes": {
                node_id: {
                    "id": node.id,
                    "claim": node.claim,
                    "domain": node.domain,
                    "confidence": node.confidence,
                    "metadata": node.metadata,
                    "verification_status": node.verification_status.value,
                    "context": node.context,
                    "search_queries": node.search_queries,
                    "sources_examined": node.sources_examined,
                    "verification_result": node.verification_result,
                    "embedding": node.embedding,
                    "created_at": node.created_at.isoformat(),
                }
                for node_id, node in self.nodes.items()
            },
            "edges": {
                edge_id: {
                    "id": edge.id,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relationship_type": edge.relationship_type.value,
                    "strength": edge.strength,
                    "metadata": edge.metadata,
                    "created_at": edge.created_at.isoformat(),
                }
                for edge_id, edge in self.edges.items()
            },
            "clusters": {
                cluster_id: {
                    "id": cluster.id,
                    "node_ids": [node.id for node in cluster.nodes],
                    "cluster_type": cluster.cluster_type.value,
                    "shared_context": cluster.shared_context,
                    "verification_strategy": cluster.verification_strategy,
                    "metadata": cluster.metadata,
                    "batch_queries": cluster.batch_queries,
                    "shared_sources": cluster.shared_sources,
                    "cluster_verification_result": cluster.cluster_verification_result,
                    "created_at": cluster.created_at.isoformat(),
                }
                for cluster_id, cluster in self.clusters.items()
            },
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "stats": self.get_stats(),
        }


# Pydantic models for API serialization
class FactNodeModel(BaseModel):
    """Pydantic model for FactNode."""

    id: str
    claim: str
    domain: str = "general"
    confidence: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    verification_status: VerificationStatus = VerificationStatus.PENDING
    context: dict[str, Any] = Field(default_factory=dict)
    search_queries: list[str] = Field(default_factory=list)
    sources_examined: list[str] = Field(default_factory=list)
    verification_result: dict[str, Any] | None = None
    embedding: list[float] | None = None
    created_at: datetime


class FactEdgeModel(BaseModel):
    """Pydantic model for FactEdge."""

    id: str
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    strength: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class FactClusterModel(BaseModel):
    """Pydantic model for FactCluster."""

    id: str
    node_ids: list[str]
    cluster_type: ClusterType
    shared_context: str = ""
    verification_strategy: str = "batch"
    metadata: dict[str, Any] = Field(default_factory=dict)
    batch_queries: list[str] = Field(default_factory=list)
    shared_sources: list[str] = Field(default_factory=list)
    cluster_verification_result: dict[str, Any] | None = None
    created_at: datetime


class FactGraphModel(BaseModel):
    """Pydantic model for FactGraph."""

    nodes: dict[str, FactNodeModel]
    edges: dict[str, FactEdgeModel]
    clusters: dict[str, FactClusterModel]
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    stats: dict[str, Any]

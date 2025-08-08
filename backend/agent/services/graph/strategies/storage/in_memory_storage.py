"""
In-memory storage strategy implementation.

This module implements in-memory storage for fact verification graphs,
suitable for development, testing, and small-scale deployments.
"""

import asyncio
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.models import FactCluster, FactEdge, FactGraph, FactNode
from agent.services.graph.interfaces import StorageStrategy

logger = logging.getLogger(__name__)


class InMemoryStorageStrategy(StorageStrategy):
    """
    In-memory storage strategy.

    This strategy stores all graph data in memory, with optional
    persistence to disk for data durability between sessions.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize in-memory storage strategy.

        Args:
            config: Storage configuration
        """
        self._config = config or {}
        self._max_graphs = self._config.get("max_graphs", 100)
        self._enable_persistence = self._config.get("enable_persistence", False)
        self._persistence_path = Path(self._config.get("persistence_path", "./data/graphs"))

        # In-memory storage
        self._graphs: dict[str, FactGraph] = {}
        # graph_id -> node_id -> node
        self._nodes: dict[str, dict[str, FactNode]] = {}
        # graph_id -> edge_id -> edge
        self._edges: dict[str, dict[str, FactEdge]] = {}
        # graph_id -> cluster_id -> cluster
        self._clusters: dict[str, dict[str, FactCluster]] = {}

        # Connection state
        self._connected = False

        # Persistence lock
        self._persistence_lock = asyncio.Lock()

        logger.info("InMemoryStorageStrategy initialized")

    async def connect(self) -> None:
        """Connect to storage (initialize in-memory structures)."""
        if self._connected:
            return

        # Initialize storage structures
        self._graphs = {}
        self._nodes = {}
        self._edges = {}
        self._clusters = {}

        # Load from persistence if enabled
        if self._enable_persistence:
            await self._load_from_persistence()

        self._connected = True
        logger.info("Connected to in-memory storage")

    async def disconnect(self) -> None:
        """Disconnect from storage (save to persistence if enabled)."""
        if not self._connected:
            return

        # Save to persistence if enabled
        if self._enable_persistence:
            await self._save_to_persistence()

        # Clear in-memory data
        self._graphs.clear()
        self._nodes.clear()
        self._edges.clear()
        self._clusters.clear()

        self._connected = False
        logger.info("Disconnected from in-memory storage")

    async def save_graph(self, graph: FactGraph) -> None:
        """
        Save a graph to storage.

        Args:
            graph: Graph to save
        """
        self._ensure_connected()

        # Check capacity
        if len(self._graphs) >= self._max_graphs and graph.graph_id not in self._graphs:
            # Remove oldest graph to make space
            oldest_graph_id = min(self._graphs.keys(), key=lambda gid: self._graphs[gid].created_at)
            await self._remove_graph_data(oldest_graph_id)
            logger.warning(f"Removed oldest graph {oldest_graph_id} to make space")

        # Save graph
        self._graphs[graph.graph_id] = graph

        # Initialize collections for this graph if not exists
        if graph.graph_id not in self._nodes:
            self._nodes[graph.graph_id] = {}
        if graph.graph_id not in self._edges:
            self._edges[graph.graph_id] = {}
        if graph.graph_id not in self._clusters:
            self._clusters[graph.graph_id] = {}

        # Save nodes, edges, and clusters
        for node in graph.nodes.values():
            self._nodes[graph.graph_id][node.node_id] = node

        for edge in graph.edges.values():
            self._edges[graph.graph_id][edge.edge_id] = edge

        for cluster in graph.clusters.values():
            self._clusters[graph.graph_id][cluster.cluster_id] = cluster

        # Persist if enabled
        if self._enable_persistence:
            await self._persist_graph(graph)

        logger.debug(f"Saved graph {graph.graph_id}")

    async def load_graph(self, graph_id: str) -> FactGraph | None:
        """
        Load a graph from storage.

        Args:
            graph_id: Graph identifier

        Returns:
            Loaded graph or None if not found
        """
        self._ensure_connected()

        if graph_id not in self._graphs:
            return None

        # Get base graph
        graph = self._graphs[graph_id]

        # Reconstruct full graph with current data
        nodes = self._nodes.get(graph_id, {})
        edges = self._edges.get(graph_id, {})
        clusters = self._clusters.get(graph_id, {})

        # Create updated graph
        updated_graph = FactGraph(
            id=graph.graph_id,
            nodes=nodes.copy(),
            edges=edges.copy(),
            clusters=clusters.copy(),
            metadata=graph.metadata.copy(),
            created_at=graph.created_at,
            updated_at=graph.updated_at,
        )

        logger.debug(f"Loaded graph {graph_id}")
        return updated_graph

    async def delete_graph(self, graph_id: str) -> None:
        """
        Delete a graph from storage.

        Args:
            graph_id: Graph identifier
        """
        self._ensure_connected()

        await self._remove_graph_data(graph_id)

        # Remove from persistence if enabled
        if self._enable_persistence:
            await self._remove_from_persistence(graph_id)

        logger.debug(f"Deleted graph {graph_id}")

    async def save_node(self, graph_id: str, node: FactNode) -> None:
        """
        Save a node to storage.

        Args:
            graph_id: Graph identifier
            node: Node to save
        """
        self._ensure_connected()

        if graph_id not in self._nodes:
            self._nodes[graph_id] = {}

        self._nodes[graph_id][node.node_id] = node

        # Update graph's updated_at timestamp
        if graph_id in self._graphs:
            self._graphs[graph_id].updated_at = datetime.now()

        logger.debug(f"Saved node {node.node_id} to graph {graph_id}")

    async def save_edge(self, graph_id: str, edge: FactEdge) -> None:
        """
        Save an edge to storage.

        Args:
            graph_id: Graph identifier
            edge: Edge to save
        """
        self._ensure_connected()

        if graph_id not in self._edges:
            self._edges[graph_id] = {}

        self._edges[graph_id][edge.edge_id] = edge

        # Update graph's updated_at timestamp
        if graph_id in self._graphs:
            self._graphs[graph_id].updated_at = datetime.now()

        logger.debug(f"Saved edge {edge.edge_id} to graph {graph_id}")

    async def save_cluster(self, graph_id: str, cluster: FactCluster) -> None:
        """
        Save a cluster to storage.

        Args:
            graph_id: Graph identifier
            cluster: Cluster to save
        """
        self._ensure_connected()

        if graph_id not in self._clusters:
            self._clusters[graph_id] = {}

        self._clusters[graph_id][cluster.cluster_id] = cluster

        # Update graph's updated_at timestamp
        if graph_id in self._graphs:
            self._graphs[graph_id].updated_at = datetime.now()

        logger.debug(f"Saved cluster {cluster.cluster_id} to graph {graph_id}")

    async def list_graphs(self) -> list[str]:
        """
        List all available graph identifiers.

        Returns:
            List of graph identifiers
        """
        self._ensure_connected()
        return list(self._graphs.keys())

    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "in_memory_storage"

    def validate_config(self, config: dict[str, Any]) -> bool:
        """
        Validate storage configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if configuration is valid
        """
        try:
            max_graphs = config.get("max_graphs", 100)
            if not isinstance(max_graphs, int) or max_graphs <= 0:
                return False

            enable_persistence = config.get("enable_persistence", False)
            if not isinstance(enable_persistence, bool):
                return False

            persistence_path = config.get("persistence_path", "./data/graphs")
            if not isinstance(persistence_path, str):
                return False

            return True

        except Exception:
            return False

    def get_config(self) -> dict[str, Any]:
        """Get current configuration."""
        return self._config.copy()

    def update_config(self, config: dict[str, Any]) -> None:
        """
        Update storage configuration.

        Args:
            config: New configuration
        """
        if self.validate_config(config):
            self._config.update(config)
            self._max_graphs = self._config.get("max_graphs", 100)
            self._enable_persistence = self._config.get("enable_persistence", False)
            self._persistence_path = Path(self._config.get("persistence_path", "./data/graphs"))
            logger.info("In-memory storage configuration updated")
        else:
            raise ValueError("Invalid configuration")

    async def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        self._ensure_connected()

        total_nodes = sum(len(nodes) for nodes in self._nodes.values())
        total_edges = sum(len(edges) for edges in self._edges.values())
        total_clusters = sum(len(clusters) for clusters in self._clusters.values())

        stats = {
            "strategy": "in_memory_storage",
            "connected": self._connected,
            "total_graphs": len(self._graphs),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "total_clusters": total_clusters,
            "max_graphs": self._max_graphs,
            "persistence_enabled": self._enable_persistence,
            "persistence_path": str(self._persistence_path),
        }

        if self._enable_persistence:
            stats["persistence_files"] = await self._count_persistence_files()

        return stats

    def _ensure_connected(self) -> None:
        """Ensure storage is connected."""
        if not self._connected:
            raise RuntimeError("Storage not connected. Call connect() first.")

    async def _remove_graph_data(self, graph_id: str) -> None:
        """Remove all data for a graph."""
        # Remove from main collections
        self._graphs.pop(graph_id, None)
        self._nodes.pop(graph_id, None)
        self._edges.pop(graph_id, None)
        self._clusters.pop(graph_id, None)

    async def _load_from_persistence(self) -> None:
        """Load data from persistence files."""
        if not self._persistence_path.exists():
            logger.info("No persistence directory found, starting with empty storage")
            return

        async with self._persistence_lock:
            try:
                # Load graphs
                graphs_file = self._persistence_path / "graphs.pkl"
                if graphs_file.exists():
                    with open(graphs_file, "rb") as f:
                        self._graphs = pickle.load(f)

                # Load nodes
                nodes_file = self._persistence_path / "nodes.pkl"
                if nodes_file.exists():
                    with open(nodes_file, "rb") as f:
                        self._nodes = pickle.load(f)

                # Load edges
                edges_file = self._persistence_path / "edges.pkl"
                if edges_file.exists():
                    with open(edges_file, "rb") as f:
                        self._edges = pickle.load(f)

                # Load clusters
                clusters_file = self._persistence_path / "clusters.pkl"
                if clusters_file.exists():
                    with open(clusters_file, "rb") as f:
                        self._clusters = pickle.load(f)

                logger.info(f"Loaded {len(self._graphs)} graphs from persistence")

            except Exception as e:
                logger.error(f"Failed to load from persistence: {str(e)}")
                # Initialize empty collections on error
                self._graphs = {}
                self._nodes = {}
                self._edges = {}
                self._clusters = {}

    async def _save_to_persistence(self) -> None:
        """Save data to persistence files."""
        if not self._enable_persistence:
            return

        async with self._persistence_lock:
            try:
                # Create persistence directory
                self._persistence_path.mkdir(parents=True, exist_ok=True)

                # Save graphs
                graphs_file = self._persistence_path / "graphs.pkl"
                with open(graphs_file, "wb") as f:
                    pickle.dump(self._graphs, f)

                # Save nodes
                nodes_file = self._persistence_path / "nodes.pkl"
                with open(nodes_file, "wb") as f:
                    pickle.dump(self._nodes, f)

                # Save edges
                edges_file = self._persistence_path / "edges.pkl"
                with open(edges_file, "wb") as f:
                    pickle.dump(self._edges, f)

                # Save clusters
                clusters_file = self._persistence_path / "clusters.pkl"
                with open(clusters_file, "wb") as f:
                    pickle.dump(self._clusters, f)

                logger.debug("Saved data to persistence")

            except Exception as e:
                logger.error(f"Failed to save to persistence: {str(e)}")

    async def _persist_graph(self, graph: FactGraph) -> None:
        """Persist a single graph (incremental save)."""
        if not self._enable_persistence:
            return

        # For simplicity, save all data
        # In a real implementation, you might want incremental saves
        await self._save_to_persistence()

    async def _remove_from_persistence(self, graph_id: str) -> None:
        """Remove graph from persistence."""
        if not self._enable_persistence:
            return

        # For simplicity, save all remaining data
        # In a real implementation, you might want selective removal
        await self._save_to_persistence()

    async def _count_persistence_files(self) -> int:
        """Count persistence files."""
        if not self._persistence_path.exists():
            return 0

        return len(list(self._persistence_path.glob("*.pkl")))

    async def clear_all_data(self) -> None:
        """Clear all data from storage (useful for testing)."""
        self._ensure_connected()

        self._graphs.clear()
        self._nodes.clear()
        self._edges.clear()
        self._clusters.clear()

        # Clear persistence if enabled
        if self._enable_persistence:
            async with self._persistence_lock:
                try:
                    if self._persistence_path.exists():
                        for file in self._persistence_path.glob("*.pkl"):
                            file.unlink()
                        logger.info("Cleared persistence files")
                except Exception as e:
                    logger.error(f"Failed to clear persistence: {str(e)}")

        logger.info("Cleared all in-memory data")

    async def export_data(self, export_path: str) -> None:
        """
        Export all data to a file.

        Args:
            export_path: Path to export file
        """
        self._ensure_connected()

        export_data = {
            "graphs": {gid: self._serialize_graph(graph) for gid, graph in self._graphs.items()},
            "nodes": {
                gid: {nid: self._serialize_node(node) for nid, node in nodes.items()}
                for gid, nodes in self._nodes.items()
            },
            "edges": {
                gid: {eid: self._serialize_edge(edge) for eid, edge in edges.items()}
                for gid, edges in self._edges.items()
            },
            "clusters": {
                gid: {cid: self._serialize_cluster(cluster) for cid, cluster in clusters.items()}
                for gid, clusters in self._clusters.items()
            },
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "total_graphs": len(self._graphs),
                "strategy": "in_memory_storage",
            },
        }

        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported data to {export_path}")

    def _serialize_graph(self, graph: FactGraph) -> dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "graph_id": graph.graph_id,
            "metadata": graph.metadata,
            "created_at": graph.created_at.isoformat(),
            "updated_at": graph.updated_at.isoformat(),
        }

    def _serialize_node(self, node: FactNode) -> dict[str, Any]:
        """Serialize node to dictionary."""
        return {
            "node_id": node.node_id,
            "claim": node.claim,
            "source": node.source,
            "confidence": node.confidence,
            "metadata": node.metadata,
            "created_at": node.created_at.isoformat() if node.created_at else None,
            "embedding": node.embedding.tolist() if node.embedding is not None else None,
        }

    def _serialize_edge(self, edge: FactEdge) -> dict[str, Any]:
        """Serialize edge to dictionary."""
        return {
            "edge_id": edge.edge_id,
            "source_node_id": edge.source_node_id,
            "target_node_id": edge.target_node_id,
            "relationship_type": edge.relationship_type,
            "weight": edge.weight,
            "metadata": edge.metadata,
            "created_at": edge.created_at.isoformat() if edge.created_at else None,
        }

    def _serialize_cluster(self, cluster: FactCluster) -> dict[str, Any]:
        """Serialize cluster to dictionary."""
        return {
            "cluster_id": cluster.cluster_id,
            "node_ids": cluster.node_ids,
            "cluster_type": cluster.cluster_type,
            "verification_strategy": cluster.verification_strategy,
            "metadata": cluster.metadata,
            "created_at": cluster.created_at.isoformat() if cluster.created_at else None,
        }

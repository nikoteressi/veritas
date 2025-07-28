"""Advanced clustering system using Graph Neural Networks for fact verification.

from __future__ import annotations

This module implements sophisticated clustering algorithms that go beyond simple
domain-based clustering, using GNNs to understand complex relationships between facts.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dateutil import parser
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

logger = logging.getLogger(__name__)


@dataclass
class ClusteringConfig:
    """Configuration for advanced clustering."""

    use_gnn: bool = True
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 3
    attention_heads: int = 4
    clustering_algorithm: str = "adaptive"  # "dbscan", "hierarchical", "adaptive"
    min_cluster_size: int = 2
    max_cluster_size: int = 10
    similarity_threshold: float = 0.7
    temporal_weight: float = 0.3
    semantic_weight: float = 0.5
    structural_weight: float = 0.2


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for learning fact representations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        attention_heads: int = 4,
    ):
        super().__init__()
        self.num_layers = num_layers

        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Attention layer for final representation
        self.attention = GATConv(
            hidden_dim, output_dim, heads=attention_heads, concat=False
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index, batch=None):
        """Forward pass through the GNN."""
        # Apply graph convolutions with ReLU activation
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        # Apply attention mechanism
        x = self.attention(x, edge_index)

        # Global pooling if batch is provided (for graph-level representations)
        if batch is not None:
            x = global_mean_pool(x, batch)

        return x


class AdvancedClusteringSystem:
    """Advanced clustering system using GNNs and multiple similarity metrics."""

    def __init__(self, config: ClusteringConfig | None = None, graph_builder=None):
        self.config = config or ClusteringConfig()
        self.graph_builder = graph_builder
        self.gnn_model = None
        self.is_trained = False
        self.logger = logging.getLogger(__name__)

        # Initialize GNN if enabled
        if self.config.use_gnn:
            self._initialize_gnn()

    def _initialize_gnn(self):
        """Initialize the Graph Neural Network."""
        try:
            # We'll initialize GNN dynamically when we know the actual embedding dimension
            # For now, just mark that GNN should be used
            self.gnn_model = None
            self.logger.info(
                "GNN will be initialized dynamically based on embedding dimensions"
            )

        except (ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as e:
            self.logger.error("Failed to initialize GNN: %s", e)
            self.config.use_gnn = False

    def _initialize_gnn_with_embedding_dim(self, embedding_dim: int):
        """Initialize GNN with the actual embedding dimension."""
        try:
            hidden_dim = self.config.gnn_hidden_dim
            output_dim = 64  # Compressed representation

            self.gnn_model = GraphNeuralNetwork(
                input_dim=embedding_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=self.config.gnn_num_layers,
                attention_heads=self.config.attention_heads,
            )

            # Set to evaluation mode (we'll implement training later)
            self.gnn_model.eval()
            self.logger.info(
                "GNN model initialized with embedding dimension %d", embedding_dim
            )

        except (ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as e:
            self.logger.error("Failed to initialize GNN with embedding dim: %s", e)
            self.config.use_gnn = False

    async def cluster_facts(self, graph) -> dict[str, list[str]]:
        """
        Perform advanced clustering on facts using multiple similarity metrics.

        Args:
            graph: FactGraph containing nodes and edges

        Returns:
            Dictionary mapping cluster IDs to lists of node IDs
        """
        if not graph.nodes:
            return {}

        self.logger.info("Starting advanced clustering for %d facts", len(graph.nodes))

        # Step 1: Extract features and build similarity matrix
        similarity_matrix = await self._build_similarity_matrix(graph)

        # Step 2: Apply clustering algorithm
        clusters = await self._apply_clustering(
            similarity_matrix, list(graph.nodes.keys())
        )

        # Step 3: Validate and optimize clusters
        optimized_clusters = await self._optimize_clusters(clusters, graph)

        self.logger.info("Created %d clusters", len(optimized_clusters))
        return optimized_clusters

    async def _build_similarity_matrix(self, graph) -> np.ndarray:
        """Build a comprehensive similarity matrix using multiple metrics."""
        nodes = list(graph.nodes.values())
        n_nodes = len(nodes)
        similarity_matrix = np.zeros((n_nodes, n_nodes))

        # Check and generate embeddings if missing
        nodes_without_embeddings = []
        for node in nodes:
            if not hasattr(node, "embedding") or node.embedding is None:
                nodes_without_embeddings.append(node)

        # Generate missing embeddings
        if nodes_without_embeddings:
            self.logger.info(
                "Generating embeddings for %d nodes", len(nodes_without_embeddings)
            )
            try:
                # Extract claims for embedding generation
                claims = [node.claim for node in nodes_without_embeddings]

                # Generate embeddings using the provided graph_builder or create new one
                if self.graph_builder:
                    embeddings_array = await self.graph_builder._get_embeddings(claims)
                else:
                    # Fallback: create new GraphBuilder instance
                    from agent.services.graph_builder import GraphBuilder

                    graph_builder = GraphBuilder()
                    embeddings_array = await graph_builder._get_embeddings(claims)

                # Assign embeddings to nodes
                for i, node in enumerate(nodes_without_embeddings):
                    node.embedding = embeddings_array[i]

                self.logger.info(
                    "Successfully generated embeddings for %d nodes",
                    len(nodes_without_embeddings),
                )

            except Exception as e:
                self.logger.error("Failed to generate embeddings: %s", e)
                raise RuntimeError(f"Failed to generate missing embeddings: {e}") from e

        # Get embeddings with validation
        embeddings = []
        for node in nodes:
            if hasattr(node, "embedding") and node.embedding is not None:
                embedding = np.array(node.embedding)
                # Check for NaN or invalid values
                if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                    self.logger.error(
                        "Node %s has invalid embedding with NaN/Inf values", node.id
                    )
                    raise ValueError(
                        f"Invalid embedding for node {node.id}: contains NaN or Inf values"
                    )
                embeddings.append(embedding)
            else:
                self.logger.error("Node %s has no valid embedding", node.id)
                raise ValueError(f"Node {node.id} is missing embedding data")

        embeddings = np.array(embeddings)

        # Validate embeddings shape
        if embeddings.ndim != 2:
            self.logger.error(
                "Embeddings have invalid shape: %s, expected 2D array", embeddings.shape
            )
            raise ValueError(
                f"Embeddings must be 2D array, got shape: {embeddings.shape}"
            )

        if embeddings.shape[0] != n_nodes:
            self.logger.error(
                "Embeddings count mismatch: %d vs %d nodes",
                embeddings.shape[0],
                n_nodes,
            )
            raise ValueError(
                f"Embeddings count ({embeddings.shape[0]}) doesn't match nodes count ({n_nodes})"
            )

        # 1. Semantic similarity (cosine similarity of embeddings)
        try:
            semantic_sim = cosine_similarity(embeddings)
        except Exception as e:
            self.logger.error("Failed to compute semantic similarity: %s", e)
            raise RuntimeError(f"Semantic similarity computation failed: {e}") from e

        # 2. Temporal similarity
        temporal_sim = self._compute_temporal_similarity(nodes)

        # 3. Structural similarity (based on graph structure)
        structural_sim = self._compute_structural_similarity(graph, nodes)

        # 4. GNN-enhanced similarity (if available)
        gnn_sim = None
        if self.config.use_gnn and self.gnn_model:
            gnn_sim = await self._compute_gnn_similarity(graph, nodes)

        # Combine similarities with weights
        similarity_matrix = (
            self.config.semantic_weight * semantic_sim
            + self.config.temporal_weight * temporal_sim
            + self.config.structural_weight * structural_sim
        )

        if gnn_sim is not None:
            # Adjust weights to include GNN similarity
            total_weight = (
                self.config.semantic_weight
                + self.config.temporal_weight
                + self.config.structural_weight
            )
            gnn_weight = 0.3

            # Renormalize existing weights
            similarity_matrix *= total_weight / (total_weight + gnn_weight)
            similarity_matrix += gnn_weight * gnn_sim

        return similarity_matrix

    def _compute_temporal_similarity(self, nodes) -> np.ndarray:
        """Compute temporal similarity between nodes."""
        n_nodes = len(nodes)
        temporal_sim = np.ones((n_nodes, n_nodes))

        # Extract temporal information from claims
        timestamps = []
        for node in nodes:
            timestamp = self._extract_timestamp(node.claim)
            timestamps.append(timestamp)

        # Compute temporal similarity
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if timestamps[i] and timestamps[j]:
                    time_diff = abs((timestamps[i] - timestamps[j]).days)
                    # Exponential decay with time difference
                    sim = np.exp(-time_diff / 365.0)  # 1-year decay constant
                    temporal_sim[i, j] = temporal_sim[j, i] = sim

        return temporal_sim

    def _compute_structural_similarity(self, graph, nodes) -> np.ndarray:
        """Compute structural similarity based on graph topology."""
        n_nodes = len(nodes)
        structural_sim = np.zeros((n_nodes, n_nodes))

        # Create node ID to index mapping
        # Node to index mapping is handled in the edge processing section

        # Compute common neighbors and path similarities
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i == j:
                    structural_sim[i, j] = 1.0
                    continue

                # Find common neighbors
                neighbors_i = set()
                neighbors_j = set()

                for edge in graph.edges.values():
                    if edge.source_id == node_i.id:
                        neighbors_i.add(edge.target_id)
                    elif edge.target_id == node_i.id:
                        neighbors_i.add(edge.source_id)

                    if edge.source_id == node_j.id:
                        neighbors_j.add(edge.target_id)
                    elif edge.target_id == node_j.id:
                        neighbors_j.add(edge.source_id)

                # Jaccard similarity of neighborhoods
                if neighbors_i or neighbors_j:
                    intersection = len(neighbors_i & neighbors_j)
                    union = len(neighbors_i | neighbors_j)
                    structural_sim[i, j] = intersection / union if union > 0 else 0.0

        return structural_sim

    async def _compute_gnn_similarity(self, graph, nodes) -> np.ndarray:
        """Compute similarity using GNN embeddings."""
        try:
            # Initialize GNN with actual embedding dimension if not already done
            if self.gnn_model is None and self.config.use_gnn:
                embeddings_list = [node.embedding for node in nodes]
                embeddings_array = np.array(embeddings_list)
                embedding_dim = embeddings_array.shape[1]
                self._initialize_gnn_with_embedding_dim(embedding_dim)

            # If GNN is still None (initialization failed), return zeros
            if self.gnn_model is None:
                self.logger.warning(
                    "GNN not available, returning zero similarity matrix"
                )
                return np.zeros((len(nodes), len(nodes)))

            # Prepare data for GNN - convert to numpy array first for better performance
            embeddings_list = [node.embedding for node in nodes]
            embeddings_array = np.array(embeddings_list)
            node_features = torch.tensor(embeddings_array, dtype=torch.float32)

            # Build edge index from graph edges
            edge_list = []
            node_to_idx = {node.id: i for i, node in enumerate(nodes)}

            for edge in graph.edges.values():
                if edge.source_id in node_to_idx and edge.target_id in node_to_idx:
                    src_idx = node_to_idx[edge.source_id]
                    tgt_idx = node_to_idx[edge.target_id]
                    edge_list.append([src_idx, tgt_idx])
                    edge_list.append([tgt_idx, src_idx])  # Undirected

            if not edge_list:
                # No edges, return identity matrix
                return np.eye(len(nodes))

            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

            # Forward pass through GNN
            with torch.no_grad():
                gnn_embeddings = self.gnn_model(node_features, edge_index)

            # Compute cosine similarity of GNN embeddings
            gnn_embeddings_np = gnn_embeddings.numpy()
            gnn_similarity = cosine_similarity(gnn_embeddings_np)

            return gnn_similarity

        except (ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as e:
            self.logger.warning("GNN similarity computation failed: %s", e)
            return np.zeros((len(nodes), len(nodes)))

    async def _apply_clustering(
        self, similarity_matrix: np.ndarray, node_ids: list[str]
    ) -> dict[str, list[str]]:
        """Apply clustering algorithm to similarity matrix."""
        if self.config.clustering_algorithm == "adaptive":
            return await self._adaptive_clustering(similarity_matrix, node_ids)
        elif self.config.clustering_algorithm == "dbscan":
            return await self._dbscan_clustering(similarity_matrix, node_ids)
        elif self.config.clustering_algorithm == "hierarchical":
            return await self._hierarchical_clustering(similarity_matrix, node_ids)
        else:
            raise ValueError(
                f"Unknown clustering algorithm: {self.config.clustering_algorithm}"
            )

    async def _adaptive_clustering(
        self, similarity_matrix: np.ndarray, node_ids: list[str]
    ) -> dict[str, list[str]]:
        """Adaptive clustering that chooses the best algorithm based on data characteristics."""
        n_nodes = len(node_ids)

        # Choose algorithm based on data size and characteristics
        if n_nodes < 10:
            # Small dataset: use hierarchical clustering
            return await self._hierarchical_clustering(similarity_matrix, node_ids)
        elif n_nodes < 100:
            # Medium dataset: use DBSCAN
            return await self._dbscan_clustering(similarity_matrix, node_ids)
        else:
            # Large dataset: use a combination approach
            return await self._combined_clustering(similarity_matrix, node_ids)

    async def _dbscan_clustering(
        self, similarity_matrix: np.ndarray, node_ids: list[str]
    ) -> dict[str, list[str]]:
        """DBSCAN clustering using distance matrix."""
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix

        # Apply DBSCAN
        clustering = DBSCAN(
            metric="precomputed",
            eps=1 - self.config.similarity_threshold,
            min_samples=self.config.min_cluster_size,
        )

        cluster_labels = clustering.fit_predict(distance_matrix)

        # Convert to cluster dictionary
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label == -1:  # Noise point
                clusters[f"singleton_{i}"] = [node_ids[i]]
            else:
                cluster_id = f"cluster_{label}"
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(node_ids[i])

        return clusters

    async def _hierarchical_clustering(
        self, similarity_matrix: np.ndarray, node_ids: list[str]
    ) -> dict[str, list[str]]:
        """Hierarchical clustering using similarity matrix."""
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix

        # Apply hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - self.config.similarity_threshold,
            metric="precomputed",
            linkage="average",
        )

        cluster_labels = clustering.fit_predict(distance_matrix)

        # Convert to cluster dictionary
        clusters = {}
        for i, label in enumerate(cluster_labels):
            cluster_id = f"cluster_{label}"
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(node_ids[i])

        return clusters

    async def _combined_clustering(
        self, similarity_matrix: np.ndarray, node_ids: list[str]
    ) -> dict[str, list[str]]:
        """Combined clustering approach for large datasets."""
        # First pass: rough clustering with DBSCAN
        rough_clusters = await self._dbscan_clustering(similarity_matrix, node_ids)

        # Second pass: refine large clusters with hierarchical clustering
        refined_clusters = {}

        for cluster_id, cluster_nodes in rough_clusters.items():
            if len(cluster_nodes) <= self.config.max_cluster_size:
                refined_clusters[cluster_id] = cluster_nodes
            else:
                # Refine large cluster
                node_indices = [node_ids.index(node_id) for node_id in cluster_nodes]
                sub_similarity = similarity_matrix[np.ix_(node_indices, node_indices)]

                sub_clusters = await self._hierarchical_clustering(
                    sub_similarity, cluster_nodes
                )

                # Add refined clusters with new IDs
                for i, (_, sub_cluster_nodes) in enumerate(sub_clusters.items()):
                    refined_clusters[f"{cluster_id}_sub_{i}"] = sub_cluster_nodes

        return refined_clusters

    async def _optimize_clusters(
        self, clusters: dict[str, list[str]], graph
    ) -> dict[str, list[str]]:
        """Optimize clusters by validating and adjusting them."""
        optimized_clusters = {}

        for cluster_id, node_ids in clusters.items():
            # Skip empty clusters
            if not node_ids:
                continue

            # Validate cluster size
            if len(node_ids) < self.config.min_cluster_size:
                # Try to merge with similar cluster or make singleton
                merged = False
                for other_id, other_nodes in optimized_clusters.items():
                    if self._should_merge_clusters(node_ids, other_nodes, graph):
                        optimized_clusters[other_id].extend(node_ids)
                        merged = True
                        break

                if not merged:
                    # Create singleton clusters
                    for i, node_id in enumerate(node_ids):
                        optimized_clusters[f"{cluster_id}_singleton_{i}"] = [node_id]

            elif len(node_ids) > self.config.max_cluster_size:
                # Split large cluster
                sub_clusters = await self._split_large_cluster(node_ids, graph)
                for i, sub_cluster in enumerate(sub_clusters):
                    optimized_clusters[f"{cluster_id}_split_{i}"] = sub_cluster

            else:
                # Cluster size is acceptable
                optimized_clusters[cluster_id] = node_ids

        return optimized_clusters

    def _should_merge_clusters(
        self, cluster1: list[str], cluster2: list[str], graph
    ) -> bool:
        """Determine if two clusters should be merged."""
        # Simple heuristic: merge if clusters are small and have high inter-cluster similarity
        if len(cluster1) + len(cluster2) > self.config.max_cluster_size:
            return False

        # Calculate average similarity between clusters
        total_similarity = 0
        count = 0

        for node1_id in cluster1:
            for node2_id in cluster2:
                node1 = graph.nodes.get(node1_id)
                node2 = graph.nodes.get(node2_id)
                if node1 and node2:
                    sim = cosine_similarity([node1.embedding], [node2.embedding])[0, 0]
                    total_similarity += sim
                    count += 1

        if count == 0:
            return False

        avg_similarity = total_similarity / count
        return avg_similarity > self.config.similarity_threshold

    async def _split_large_cluster(self, node_ids: list[str], graph) -> list[list[str]]:
        """Split a large cluster into smaller ones."""
        # Extract embeddings for nodes in this cluster
        embeddings = []
        valid_node_ids = []

        for node_id in node_ids:
            node = graph.nodes.get(node_id)
            if node:
                embeddings.append(node.embedding)
                valid_node_ids.append(node_id)

        if len(embeddings) <= self.config.max_cluster_size:
            return [valid_node_ids]

        # Use hierarchical clustering to split
        similarity_matrix = cosine_similarity(embeddings)

        # Determine number of clusters
        target_clusters = max(2, len(valid_node_ids) // self.config.max_cluster_size)

        clustering = AgglomerativeClustering(
            n_clusters=target_clusters, metric="precomputed", linkage="average"
        )

        distance_matrix = 1 - similarity_matrix
        cluster_labels = clustering.fit_predict(distance_matrix)

        # Group nodes by cluster
        sub_clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in sub_clusters:
                sub_clusters[label] = []
            sub_clusters[label].append(valid_node_ids[i])

        return list(sub_clusters.values())

    def _extract_timestamp(self, claim: str) -> datetime | None:
        """Extract timestamp from claim text."""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated NLP techniques
        # Look for date patterns
        date_patterns = [
            r"\b\d{4}-\d{2}-\d{2}\b",  # YYYY-MM-DD
            r"\b\d{2}/\d{2}/\d{4}\b",  # MM/DD/YYYY
            r"\b\d{1,2}\s+\w+\s+\d{4}\b",  # DD Month YYYY
        ]

        for pattern in date_patterns:
            match = re.search(pattern, claim)
            if match:
                try:
                    return parser.parse(match.group())
                except (ValueError, TypeError, parser.ParserError):
                    continue

        return None

    def get_clustering_stats(self) -> dict[str, Any]:
        """Get statistics about the clustering system."""
        return {
            "config": {
                "use_gnn": self.config.use_gnn,
                "clustering_algorithm": self.config.clustering_algorithm,
                "similarity_threshold": self.config.similarity_threshold,
            },
            "gnn_initialized": self.gnn_model is not None,
            "is_trained": self.is_trained,
        }

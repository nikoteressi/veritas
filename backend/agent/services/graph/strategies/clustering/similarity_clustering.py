"""
Similarity-based clustering strategy implementation.

This module implements clustering based on semantic similarity of fact embeddings.
"""

import logging
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from agent.models.graph import ClusterType, FactCluster, FactNode
from agent.services.graph.interfaces.clustering_strategy import ClusteringStrategy

logger = logging.getLogger(__name__)


class SimilarityClusteringStrategy(ClusteringStrategy):
    """
    Clustering strategy based on semantic similarity of fact embeddings.

    Uses DBSCAN algorithm with cosine similarity to group semantically
    related facts together.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize similarity clustering strategy."""
        self._config = self._get_default_config()
        if config:
            self._config.update(config)

        self._validate_config(self._config)
        logger.info(f"Initialized {self.get_strategy_name()} with config: {self._config}")

    def create_clusters(self, nodes: list[FactNode]) -> list[FactCluster]:
        """
        Create similarity-based clusters from fact nodes.

        Args:
            nodes: List of fact nodes to cluster

        Returns:
            List of fact clusters based on semantic similarity
        """
        if not nodes:
            logger.warning("No nodes provided for clustering")
            return []

        # Filter nodes with embeddings
        nodes_with_embeddings = [node for node in nodes if node.embedding is not None]

        if len(nodes_with_embeddings) < 2:
            logger.info(f"Insufficient nodes with embeddings ({len(nodes_with_embeddings)}), creating single cluster")
            return [
                FactCluster(
                    id="similarity_cluster_0",
                    nodes=nodes,
                    cluster_type=ClusterType.SIMILARITY_CLUSTER,
                    verification_strategy="individual",
                )
            ]

        # Extract embeddings
        embeddings = np.array([node.embedding for node in nodes_with_embeddings])

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Convert similarity to distance for DBSCAN
        distance_matrix = 1 - similarity_matrix

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=self._config["eps"], min_samples=self._config["min_samples"], metric="precomputed")

        cluster_labels = clustering.fit_predict(distance_matrix)

        # Group nodes by cluster labels
        clusters = self._group_nodes_by_labels(nodes_with_embeddings, cluster_labels)

        # Add nodes without embeddings to separate cluster if any
        nodes_without_embeddings = [node for node in nodes if node.embedding is None]
        if nodes_without_embeddings:
            clusters.append(
                FactCluster(
                    id="similarity_cluster_no_embedding",
                    nodes=nodes_without_embeddings,
                    cluster_type=ClusterType.SIMILARITY_CLUSTER,
                    verification_strategy="individual",
                )
            )

        logger.info(f"Created {len(clusters)} similarity clusters from {len(nodes)} nodes")
        return clusters

    def _group_nodes_by_labels(self, nodes: list[FactNode], labels: np.ndarray) -> list[FactCluster]:
        """Group nodes by cluster labels."""
        clusters = []
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:  # Noise points in DBSCAN
                cluster_nodes = [nodes[i] for i, label_val in enumerate(labels) if label_val == label]
                clusters.append(
                    FactCluster(
                        id="similarity_cluster_noise",
                        nodes=cluster_nodes,
                        cluster_type=ClusterType.SIMILARITY_CLUSTER,
                        verification_strategy="individual",
                    )
                )
            else:
                cluster_nodes = [nodes[i] for i, label_val in enumerate(labels) if label_val == label]
                clusters.append(
                    FactCluster(
                        id=f"similarity_cluster_{label}",
                        nodes=cluster_nodes,
                        cluster_type=ClusterType.SIMILARITY_CLUSTER,
                        verification_strategy="batch" if len(cluster_nodes) > 1 else "individual",
                    )
                )

        return clusters

    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "similarity_clustering"

    def validate_config(self, config: dict[str, Any]) -> bool:
        """
        Validate configuration parameters.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        return self._validate_config(config)

    def _validate_config(self, config: dict[str, Any]) -> bool:
        """Internal config validation."""
        required_keys = ["eps", "min_samples"]

        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

        if not isinstance(config["eps"], int | float) or config["eps"] <= 0:
            raise ValueError("eps must be a positive number")

        if not isinstance(config["min_samples"], int) or config["min_samples"] < 1:
            raise ValueError("min_samples must be a positive integer")

        return True

    def get_config(self) -> dict[str, Any]:
        """Get current configuration."""
        return self._config.copy()

    def update_config(self, config: dict[str, Any]) -> None:
        """
        Update configuration.

        Args:
            config: New configuration parameters
        """
        new_config = self._config.copy()
        new_config.update(config)
        self._validate_config(new_config)
        self._config = new_config
        logger.info(f"Updated config for {self.get_strategy_name()}: {self._config}")

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration."""
        return {
            "eps": 0.3,  # Maximum distance between samples in a cluster
            "min_samples": 2,  # Minimum samples in a cluster
        }

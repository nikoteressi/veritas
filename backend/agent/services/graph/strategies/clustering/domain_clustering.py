"""
Domain-based clustering strategy implementation.

This module implements clustering based on fact domains/topics.
"""

import logging
from collections import defaultdict
from typing import Any

from agent.models.graph import ClusterType, FactCluster, FactNode
from agent.services.graph.interfaces.clustering_strategy import ClusteringStrategy

logger = logging.getLogger(__name__)


class DomainClusteringStrategy(ClusteringStrategy):
    """
    Clustering strategy based on fact domains/topics.

    Groups facts that belong to the same domain or topic area.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize domain clustering strategy."""
        self._config = self._get_default_config()
        if config:
            self._config.update(config)

        self._validate_config(self._config)
        logger.info(f"Initialized {self.get_strategy_name()} with config: {self._config}")

    def create_clusters(self, nodes: list[FactNode]) -> list[FactCluster]:
        """
        Create domain-based clusters from fact nodes.

        Args:
            nodes: List of fact nodes to cluster

        Returns:
            List of fact clusters based on domains
        """
        if not nodes:
            logger.warning("No nodes provided for clustering")
            return []

        # Group nodes by domain
        domain_groups = defaultdict(list)

        for node in nodes:
            domain = node.domain or self._config["default_domain"]
            domain_groups[domain].append(node)

        # Create clusters from domain groups
        clusters = []
        for domain, domain_nodes in domain_groups.items():
            # Further split large domains if configured
            if self._config["max_cluster_size"] > 0 and len(domain_nodes) > self._config["max_cluster_size"]:
                sub_clusters = self._split_large_cluster(domain, domain_nodes)
                clusters.extend(sub_clusters)
            else:
                cluster = FactCluster(
                    id=f"domain_cluster_{domain}",
                    nodes=domain_nodes,
                    cluster_type=ClusterType.DOMAIN_CLUSTER,
                    verification_strategy=self._get_verification_strategy(len(domain_nodes)),
                )
                clusters.append(cluster)

        logger.info(f"Created {len(clusters)} domain clusters from {len(nodes)} nodes")
        return clusters

    def _split_large_cluster(self, domain: str, nodes: list[FactNode]) -> list[FactCluster]:
        """Split large domain cluster into smaller sub-clusters."""
        clusters = []
        max_size = self._config["max_cluster_size"]

        for i in range(0, len(nodes), max_size):
            sub_nodes = nodes[i : i + max_size]
            cluster = FactCluster(
                id=f"domain_cluster_{domain}_{i // max_size}",
                nodes=sub_nodes,
                cluster_type=ClusterType.DOMAIN_CLUSTER,
                verification_strategy=self._get_verification_strategy(len(sub_nodes)),
            )
            clusters.append(cluster)

        logger.info(f"Split large domain '{domain}' into {len(clusters)} sub-clusters")
        return clusters

    def _get_verification_strategy(self, cluster_size: int) -> str:
        """Determine verification strategy based on cluster size."""
        if cluster_size == 1:
            return "individual"
        elif cluster_size <= self._config["batch_threshold"]:
            return "batch"
        else:
            return "cross_verification"

    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "domain_clustering"

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
        if "default_domain" not in config:
            raise ValueError("Missing required config key: default_domain")

        if not isinstance(config["default_domain"], str):
            raise ValueError("default_domain must be a string")

        if "max_cluster_size" in config:
            if not isinstance(config["max_cluster_size"], int) or config["max_cluster_size"] < 0:
                raise ValueError("max_cluster_size must be a non-negative integer")

        if "batch_threshold" in config:
            if not isinstance(config["batch_threshold"], int) or config["batch_threshold"] < 1:
                raise ValueError("batch_threshold must be a positive integer")

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
            "default_domain": "general",
            "max_cluster_size": 10,  # 0 means no limit
            "batch_threshold": 5,  # Switch to cross-verification above this size
        }

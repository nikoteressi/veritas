"""
Causal clustering strategy implementation.

This module implements clustering based on causal relationships between facts.
"""

import logging
from collections import defaultdict
from typing import Any

from agent.models.graph import ClusterType, FactCluster, FactNode
from agent.services.graph.interfaces.clustering_strategy import ClusteringStrategy

logger = logging.getLogger(__name__)


class CausalClusteringStrategy(ClusteringStrategy):
    """
    Clustering strategy based on causal relationships.

    Groups facts that are causally connected, forming chains
    of cause-and-effect relationships.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize causal clustering strategy."""
        self._config = self._get_default_config()
        if config:
            self._config.update(config)

        self._validate_config(self._config)
        logger.info(f"Initialized {self.get_strategy_name()} with config: {self._config}")

    def create_clusters(self, nodes: list[FactNode]) -> list[FactCluster]:
        """
        Create causal-based clusters from fact nodes.

        Args:
            nodes: List of fact nodes to cluster

        Returns:
            List of fact clusters based on causal relationships
        """
        if not nodes:
            logger.warning("No nodes provided for clustering")
            return []

        # Build causal graph from nodes
        causal_graph = self._build_causal_graph(nodes)

        # Find connected components in causal graph
        causal_components = self._find_causal_components(nodes, causal_graph)

        # Create clusters from components
        clusters = []
        for i, component_nodes in enumerate(causal_components):
            cluster = FactCluster(
                id=f"causal_cluster_{i}",
                nodes=component_nodes,
                cluster_type=ClusterType.CAUSAL_CLUSTER,
                verification_strategy=self._get_verification_strategy(len(component_nodes)),
            )
            clusters.append(cluster)

        logger.info(f"Created {len(clusters)} causal clusters from {len(nodes)} nodes")
        return clusters

    def _build_causal_graph(self, nodes: list[FactNode]) -> dict[str, list[str]]:
        """Build causal relationship graph from nodes."""
        causal_graph = defaultdict(list)

        # Extract causal relationships from node metadata
        for node in nodes:
            if not node.metadata:
                continue

            # Look for causal indicators in metadata
            causal_relations = self._extract_causal_relations(node)

            for related_node_id in causal_relations:
                causal_graph[node.node_id].append(related_node_id)

        return causal_graph

    def _extract_causal_relations(self, node: FactNode) -> list[str]:
        """Extract causal relationships from node metadata."""
        causal_relations = []

        if not node.metadata:
            return causal_relations

        # Check for explicit causal relationships
        causal_fields = ["causes", "caused_by", "leads_to", "results_from", "causal_links"]

        for field in causal_fields:
            if field in node.metadata:
                relations = node.metadata[field]
                if isinstance(relations, list):
                    causal_relations.extend(relations)
                elif isinstance(relations, str):
                    causal_relations.append(relations)

        # Check for causal keywords in claim text
        if self._config["use_text_analysis"]:
            causal_keywords = self._config["causal_keywords"]
            claim_text = node.claim.lower()

            for keyword in causal_keywords:
                if keyword in claim_text:
                    # This is a simplified approach - in practice, you'd use NLP
                    # to extract actual causal relationships
                    break

        return causal_relations

    def _find_causal_components(
        self, nodes: list[FactNode], causal_graph: dict[str, list[str]]
    ) -> list[list[FactNode]]:
        """Find connected components in causal graph using DFS."""
        node_map = {node.node_id: node for node in nodes}
        visited = set()
        components = []

        for node in nodes:
            if node.node_id not in visited:
                component = []
                self._dfs_component(node.node_id, causal_graph, visited, component, node_map)

                if len(component) >= self._config["min_component_size"]:
                    components.append(component)
                else:
                    # Add small components to a separate cluster
                    if not hasattr(self, "_small_components"):
                        self._small_components = []
                    self._small_components.extend(component)

        # Add small components as a single cluster if any
        if hasattr(self, "_small_components") and self._small_components:
            components.append(self._small_components)
            delattr(self, "_small_components")

        return components

    def _dfs_component(
        self,
        node_id: str,
        causal_graph: dict[str, list[str]],
        visited: set[str],
        component: list[FactNode],
        node_map: dict[str, FactNode],
    ) -> None:
        """Depth-first search to find connected component."""
        if node_id in visited or node_id not in node_map:
            return

        visited.add(node_id)
        component.append(node_map[node_id])

        # Visit all causally connected nodes
        for connected_id in causal_graph.get(node_id, []):
            self._dfs_component(connected_id, causal_graph, visited, component, node_map)

        # Also check reverse connections (bidirectional)
        if self._config["bidirectional"]:
            for other_id, connections in causal_graph.items():
                if node_id in connections:
                    self._dfs_component(other_id, causal_graph, visited, component, node_map)

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
        return "causal_clustering"

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
        if "min_component_size" in config:
            if not isinstance(config["min_component_size"], int) or config["min_component_size"] < 1:
                raise ValueError("min_component_size must be a positive integer")

        if "batch_threshold" in config:
            if not isinstance(config["batch_threshold"], int) or config["batch_threshold"] < 1:
                raise ValueError("batch_threshold must be a positive integer")

        if "bidirectional" in config:
            if not isinstance(config["bidirectional"], bool):
                raise ValueError("bidirectional must be a boolean")

        if "use_text_analysis" in config:
            if not isinstance(config["use_text_analysis"], bool):
                raise ValueError("use_text_analysis must be a boolean")

        if "causal_keywords" in config:
            if not isinstance(config["causal_keywords"], list):
                raise ValueError("causal_keywords must be a list")

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
            "min_component_size": 2,  # Minimum nodes in a causal component
            "batch_threshold": 5,  # Switch to cross-verification above this size
            "bidirectional": True,  # Consider causal relationships as bidirectional
            "use_text_analysis": False,  # Use text analysis for causal detection
            "causal_keywords": [
                "because",
                "due to",
                "caused by",
                "leads to",
                "results in",
                "triggers",
                "influences",
                "affects",
                "impacts",
                "consequently",
            ],
        }

"""Cluster manager for grouping related graph nodes.

This service handles clustering of graph nodes based on similarity,
domain, and other criteria to form coherent groups for analysis.
"""

import logging
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


class ClusterManager:
    """Manages clustering of graph nodes.

    Groups related nodes into clusters based on various criteria
    including semantic similarity and domain classification.
    """

    def __init__(self):
        """Initialize the cluster manager."""
        self.logger = logging.getLogger(__name__)

    async def form_clusters(self, nodes: dict[str, Any], relationships: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Form clusters from nodes and relationships.

        Args:
            nodes: Dictionary of node_id -> node_data
            relationships: List of relationships between nodes

        Returns:
            List[Dict[str, Any]]: List of clusters with metadata
        """
        try:
            self.logger.info("Forming clusters from %d nodes", len(nodes))

            clusters = []

            # Create similarity-based clusters
            similarity_clusters = await self._create_similarity_clusters(nodes, relationships)
            clusters.extend(similarity_clusters)

            # Create domain-based clusters
            domain_clusters = await self._create_domain_clusters(nodes)
            clusters.extend(domain_clusters)

            # Merge overlapping clusters
            merged_clusters = self._merge_overlapping_clusters(clusters)

            # Add metadata to clusters
            final_clusters = []
            for i, cluster in enumerate(merged_clusters):
                cluster_with_metadata = await self._add_cluster_metadata(cluster, nodes, relationships, f"cluster_{i}")
                final_clusters.append(cluster_with_metadata)

            self.logger.info("Formed %d clusters", len(final_clusters))
            return final_clusters

        except Exception as e:
            self.logger.error("Error forming clusters: %s", e)
            return []

    async def _create_similarity_clusters(
        self, nodes: dict[str, Any], relationships: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Create clusters based on semantic similarity.

        Args:
            nodes: Dictionary of node_id -> node_data
            relationships: List of relationships between nodes

        Returns:
            List[Dict[str, Any]]: Similarity-based clusters
        """
        try:
            if len(nodes) < 2:
                return []

            # Extract embeddings for clustering
            node_ids = list(nodes.keys())
            embeddings = []

            for node_id in node_ids:
                embedding = nodes[node_id].get("embedding", [])
                if embedding:
                    embeddings.append(embedding)
                else:
                    # Use zero vector for nodes without embeddings
                    # Assuming 384-dim embeddings
                    embeddings.append([0.0] * 384)

            if not embeddings:
                return []

            # Convert to numpy array
            embeddings_array = np.array(embeddings)

            # Use DBSCAN for clustering
            # eps: maximum distance between samples in the same cluster
            # min_samples: minimum number of samples in a cluster
            dbscan = DBSCAN(eps=0.3, min_samples=2, metric="cosine")
            cluster_labels = dbscan.fit_predict(embeddings_array)

            # Group nodes by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label == -1:  # Noise points
                    continue

                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(node_ids[i])

            # Convert to cluster format
            similarity_clusters = []
            for _label, node_list in clusters.items():
                if len(node_list) >= 2:  # Only keep clusters with multiple nodes
                    cluster = {
                        "type": "similarity",
                        "nodes": node_list,
                        "metadata": {
                            "cluster_method": "dbscan",
                            "similarity_threshold": 0.3,
                        },
                    }
                    similarity_clusters.append(cluster)

            self.logger.info("Created %d similarity clusters", len(similarity_clusters))
            return similarity_clusters

        except Exception as e:
            self.logger.error("Error creating similarity clusters: %s", e)
            return []

    async def _create_domain_clusters(self, nodes: dict[str, Any]) -> list[dict[str, Any]]:
        """Create clusters based on domain classification.

        Args:
            nodes: Dictionary of node_id -> node_data

        Returns:
            List[Dict[str, Any]]: Domain-based clusters
        """
        try:
            # Group nodes by domain
            domain_groups = {}

            for node_id, node_data in nodes.items():
                domain = node_data.get("domain", "general")

                # Skip general domain for clustering
                if domain == "general":
                    continue

                if domain not in domain_groups:
                    domain_groups[domain] = []
                domain_groups[domain].append(node_id)

            # Convert to cluster format
            domain_clusters = []
            for domain, node_list in domain_groups.items():
                if len(node_list) >= 2:  # Only keep clusters with multiple nodes
                    cluster = {
                        "type": "domain",
                        "nodes": node_list,
                        "metadata": {
                            "domain": domain,
                            "cluster_method": "domain_classification",
                        },
                    }
                    domain_clusters.append(cluster)

            self.logger.info("Created %d domain clusters", len(domain_clusters))
            return domain_clusters

        except Exception as e:
            self.logger.error("Error creating domain clusters: %s", e)
            return []

    def _merge_overlapping_clusters(self, clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge clusters that have significant overlap.

        Args:
            clusters: List of clusters to merge

        Returns:
            List[Dict[str, Any]]: Merged clusters
        """
        try:
            if len(clusters) <= 1:
                return clusters

            merged_clusters = []
            used_indices = set()

            for i, cluster_1 in enumerate(clusters):
                if i in used_indices:
                    continue

                nodes_1 = set(cluster_1["nodes"])
                merged_cluster = {
                    "type": cluster_1["type"],
                    "nodes": list(nodes_1),
                    "metadata": cluster_1["metadata"].copy(),
                }

                # Check for overlaps with remaining clusters
                for j, cluster_2 in enumerate(clusters[i + 1 :], i + 1):
                    if j in used_indices:
                        continue

                    nodes_2 = set(cluster_2["nodes"])
                    overlap = nodes_1.intersection(nodes_2)

                    # Merge if overlap is significant (>50% of smaller cluster)
                    min_size = min(len(nodes_1), len(nodes_2))
                    if len(overlap) > min_size * 0.5:
                        # Merge clusters
                        merged_cluster["nodes"] = list(nodes_1.union(nodes_2))
                        merged_cluster["type"] = "merged"
                        merged_cluster["metadata"]["merged_types"] = [cluster_1["type"], cluster_2["type"]]

                        nodes_1 = nodes_1.union(nodes_2)
                        used_indices.add(j)

                merged_clusters.append(merged_cluster)
                used_indices.add(i)

            self.logger.info("Merged clusters: %d -> %d", len(clusters), len(merged_clusters))
            return merged_clusters

        except Exception as e:
            self.logger.error("Error merging clusters: %s", e)
            return clusters

    async def _add_cluster_metadata(
        self,
        cluster: dict[str, Any],
        nodes: dict[str, Any],
        relationships: list[dict[str, Any]],
        cluster_id: str,
    ) -> dict[str, Any]:
        """Add comprehensive metadata to a cluster.

        Args:
            cluster: Cluster data
            nodes: All nodes data
            relationships: All relationships data
            cluster_id: Unique cluster identifier

        Returns:
            Dict[str, Any]: Cluster with added metadata
        """
        try:
            cluster_nodes = cluster["nodes"]
            cluster_node_set = set(cluster_nodes)

            # Calculate cluster statistics
            confidences = [nodes[node_id].get("confidence", 0.0) for node_id in cluster_nodes]

            # Find internal relationships (within cluster)
            internal_relationships = [
                rel
                for rel in relationships
                if (rel.get("source") in cluster_node_set and rel.get("target") in cluster_node_set)
            ]

            # Find external relationships (to other clusters)
            external_relationships = [
                rel
                for rel in relationships
                if (
                    (rel.get("source") in cluster_node_set and rel.get("target") not in cluster_node_set)
                    or (rel.get("source") not in cluster_node_set and rel.get("target") in cluster_node_set)
                )
            ]

            # Calculate cluster cohesion (average internal relationship weight)
            internal_weights = [rel.get("weight", 0.0) for rel in internal_relationships]
            cohesion = sum(internal_weights) / len(internal_weights) if internal_weights else 0.0

            # Determine dominant domain
            domains = [nodes[node_id].get("domain", "general") for node_id in cluster_nodes]
            domain_counts = {}
            for domain in domains:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            dominant_domain = max(domain_counts, key=domain_counts.get) if domain_counts else "general"

            # Determine dominant node type
            node_types = [nodes[node_id].get("type", "general") for node_id in cluster_nodes]
            type_counts = {}
            for node_type in node_types:
                type_counts[node_type] = type_counts.get(node_type, 0) + 1
            dominant_type = max(type_counts, key=type_counts.get) if type_counts else "general"

            # Enhanced metadata
            enhanced_metadata = {
                **cluster.get("metadata", {}),
                "cluster_id": cluster_id,
                "size": len(cluster_nodes),
                "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
                "min_confidence": min(confidences) if confidences else 0.0,
                "max_confidence": max(confidences) if confidences else 0.0,
                "cohesion": cohesion,
                "internal_relationships": len(internal_relationships),
                "external_relationships": len(external_relationships),
                "dominant_domain": dominant_domain,
                "dominant_type": dominant_type,
                "domain_distribution": domain_counts,
                "type_distribution": type_counts,
            }

            return {**cluster, "id": cluster_id, "metadata": enhanced_metadata}

        except Exception as e:
            self.logger.error("Error adding cluster metadata: %s", e)
            return {**cluster, "id": cluster_id, "metadata": {**cluster.get("metadata", {}), "error": str(e)}}

    def get_cluster_statistics(self, clusters: list[dict[str, Any]]) -> dict[str, Any]:
        """Get statistics about clusters.

        Args:
            clusters: List of clusters

        Returns:
            Dict[str, Any]: Cluster statistics
        """
        try:
            if not clusters:
                return {
                    "total_clusters": 0,
                    "avg_cluster_size": 0.0,
                    "cluster_types": {},
                    "total_nodes_clustered": 0,
                }

            # Calculate statistics
            cluster_sizes = [len(cluster["nodes"]) for cluster in clusters]
            cluster_types = {}
            total_nodes = sum(cluster_sizes)

            for cluster in clusters:
                cluster_type = cluster.get("type", "unknown")
                cluster_types[cluster_type] = cluster_types.get(cluster_type, 0) + 1

            return {
                "total_clusters": len(clusters),
                "avg_cluster_size": sum(cluster_sizes) / len(cluster_sizes),
                "min_cluster_size": min(cluster_sizes),
                "max_cluster_size": max(cluster_sizes),
                "cluster_types": cluster_types,
                "total_nodes_clustered": total_nodes,
                "avg_cohesion": sum(cluster.get("metadata", {}).get("cohesion", 0.0) for cluster in clusters)
                / len(clusters),
            }

        except Exception as e:
            self.logger.error("Error calculating cluster statistics: %s", e)
            return {"error": str(e)}

    def find_cluster_by_node(self, node_id: str, clusters: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Find the cluster containing a specific node.

        Args:
            node_id: ID of the node to find
            clusters: List of clusters to search

        Returns:
            Dict[str, Any] | None: Cluster containing the node or None
        """
        try:
            for cluster in clusters:
                if node_id in cluster.get("nodes", []):
                    return cluster
            return None

        except Exception as e:
            self.logger.error("Error finding cluster by node: %s", e)
            return None

    def get_cluster_relationships(
        self,
        cluster: dict[str, Any],
        relationships: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Get relationships for a specific cluster.

        Args:
            cluster: Cluster data
            relationships: All relationships

        Returns:
            Dict[str, List[Dict[str, Any]]]: Internal and external relationships
        """
        try:
            cluster_nodes = set(cluster.get("nodes", []))

            internal_relationships = []
            external_relationships = []

            for rel in relationships:
                source = rel.get("source")
                target = rel.get("target")

                if source in cluster_nodes and target in cluster_nodes:
                    internal_relationships.append(rel)
                elif source in cluster_nodes or target in cluster_nodes:
                    external_relationships.append(rel)

            return {
                "internal": internal_relationships,
                "external": external_relationships,
            }

        except Exception as e:
            self.logger.error("Error getting cluster relationships: %s", e)
            return {"internal": [], "external": []}

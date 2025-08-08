"""Graph assembler for final graph construction and optimization.

This service handles the final assembly of nodes, relationships, and clusters
into a complete graph structure with optimization and validation.
"""

import logging
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class GraphAssembler:
    """Assembles final graph from components.

    Takes nodes, relationships, and clusters to create an optimized
    graph structure suitable for fact-checking analysis.
    """

    def __init__(self):
        """Initialize the graph assembler."""
        self.logger = logging.getLogger(__name__)

    async def assemble_graph(
        self,
        nodes: dict[str, Any],
        relationships: list[dict[str, Any]],
        clusters: list[dict[str, Any]],
        optimization_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assemble the final graph from components.

        Args:
            nodes: Dictionary of node_id -> node_data
            relationships: List of relationships between nodes
            clusters: List of clusters with metadata
            optimization_config: Configuration for graph optimization

        Returns:
            Dict[str, Any]: Complete assembled graph
        """
        try:
            self.logger.info(
                "Assembling graph with %d nodes, %d relationships, %d clusters",
                len(nodes),
                len(relationships),
                len(clusters),
            )

            # Set default optimization config
            if optimization_config is None:
                optimization_config = {
                    "remove_weak_edges": True,
                    "merge_similar_nodes": True,
                    "prune_isolated_nodes": True,
                    "optimize_clusters": True,
                    "min_edge_weight": 0.1,
                    "node_similarity_threshold": 0.9,
                }

            # Optimize graph structure
            if optimization_config.get("remove_weak_edges", True):
                relationships = self._remove_weak_edges(relationships, optimization_config.get("min_edge_weight", 0.1))

            if optimization_config.get("merge_similar_nodes", True):
                nodes, relationships = await self._merge_similar_nodes(
                    nodes, relationships, optimization_config.get("node_similarity_threshold", 0.9)
                )

            if optimization_config.get("prune_isolated_nodes", True):
                nodes = self._prune_isolated_nodes(nodes, relationships)

            if optimization_config.get("optimize_clusters", True):
                clusters = await self._optimize_clusters(clusters, nodes, relationships)

            # Calculate graph metrics
            graph_metrics = self._calculate_graph_metrics(nodes, relationships, clusters)

            # Create final graph structure
            assembled_graph = {
                "nodes": nodes,
                "relationships": relationships,
                "clusters": clusters,
                "metadata": {
                    "assembly_timestamp": self._get_timestamp(),
                    "optimization_config": optimization_config,
                    "metrics": graph_metrics,
                    "node_count": len(nodes),
                    "relationship_count": len(relationships),
                    "cluster_count": len(clusters),
                },
                "structure": {
                    "adjacency_list": self._create_adjacency_list(relationships),
                    "cluster_map": self._create_cluster_map(clusters),
                    "node_degrees": self._calculate_node_degrees(nodes, relationships),
                },
            }

            # Validate graph integrity
            validation_result = self._validate_graph(assembled_graph)
            assembled_graph["metadata"]["validation"] = validation_result

            self.logger.info("Graph assembly completed successfully")
            return assembled_graph

        except Exception as e:
            self.logger.error("Error assembling graph: %s", e)
            return {
                "nodes": nodes,
                "relationships": relationships,
                "clusters": clusters,
                "metadata": {"error": str(e)},
                "structure": {},
            }

    def _create_networkx_graph(self, nodes: dict[str, Any], relationships: list[dict[str, Any]]) -> nx.Graph:
        """Create NetworkX graph for analysis.

        Args:
            nodes: Dictionary of node_id -> node_data
            relationships: List of relationships

        Returns:
            nx.Graph: NetworkX graph object
        """
        try:
            G = nx.Graph()

            # Add nodes
            for node_id, node_data in nodes.items():
                G.add_node(node_id, **node_data)

            # Add edges
            for rel in relationships:
                source = rel.get("source")
                target = rel.get("target")
                weight = rel.get("weight", 1.0)

                if source and target and source in nodes and target in nodes:
                    G.add_edge(source, target, weight=weight, **rel)

            return G

        except Exception as e:
            self.logger.error("Error creating NetworkX graph: %s", e)
            return nx.Graph()

    def _remove_weak_edges(self, relationships: list[dict[str, Any]], min_weight: float) -> list[dict[str, Any]]:
        """Remove relationships with weight below threshold.

        Args:
            relationships: List of relationships
            min_weight: Minimum weight threshold

        Returns:
            List[Dict[str, Any]]: Filtered relationships
        """
        try:
            filtered_relationships = [rel for rel in relationships if rel.get("weight", 0.0) >= min_weight]

            removed_count = len(relationships) - len(filtered_relationships)
            if removed_count > 0:
                self.logger.info("Removed %d weak edges", removed_count)

            return filtered_relationships

        except Exception as e:
            self.logger.error("Error removing weak edges: %s", e)
            return relationships

    async def _merge_similar_nodes(
        self,
        nodes: dict[str, Any],
        relationships: list[dict[str, Any]],
        similarity_threshold: float,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Merge nodes that are very similar.

        Args:
            nodes: Dictionary of nodes
            relationships: List of relationships
            similarity_threshold: Similarity threshold for merging

        Returns:
            Tuple[Dict[str, Any], List[Dict[str, Any]]]: Updated nodes and relationships
        """
        try:
            # Find similar node pairs
            similar_pairs = []
            node_ids = list(nodes.keys())

            for i, node_id_1 in enumerate(node_ids):
                for node_id_2 in node_ids[i + 1 :]:
                    similarity = self._calculate_node_similarity(nodes[node_id_1], nodes[node_id_2])

                    if similarity >= similarity_threshold:
                        similar_pairs.append((node_id_1, node_id_2, similarity))

            if not similar_pairs:
                return nodes, relationships

            # Sort by similarity (highest first)
            similar_pairs.sort(key=lambda x: x[2], reverse=True)

            # Merge nodes
            merged_nodes = nodes.copy()
            merged_relationships = relationships.copy()
            node_mapping = {}  # old_id -> new_id

            for node_id_1, node_id_2, similarity in similar_pairs:
                # Skip if either node already merged
                if node_id_1 not in merged_nodes or node_id_2 not in merged_nodes:
                    continue

                # Merge node_2 into node_1
                merged_node = self._merge_node_data(merged_nodes[node_id_1], merged_nodes[node_id_2])
                merged_nodes[node_id_1] = merged_node

                # Remove node_2
                del merged_nodes[node_id_2]
                node_mapping[node_id_2] = node_id_1

                self.logger.debug("Merged nodes %s -> %s (similarity: %.3f)", node_id_2, node_id_1, similarity)

            # Update relationships
            if node_mapping:
                updated_relationships = []
                for rel in merged_relationships:
                    source = rel.get("source")
                    target = rel.get("target")

                    # Update source and target IDs
                    new_source = node_mapping.get(source, source)
                    new_target = node_mapping.get(target, target)

                    # Skip self-loops
                    if new_source == new_target:
                        continue

                    updated_rel = {**rel, "source": new_source, "target": new_target}
                    updated_relationships.append(updated_rel)

                merged_relationships = updated_relationships

            merge_count = len(node_mapping)
            if merge_count > 0:
                self.logger.info("Merged %d similar nodes", merge_count)

            return merged_nodes, merged_relationships

        except Exception as e:
            self.logger.error("Error merging similar nodes: %s", e)
            return nodes, relationships

    def _calculate_node_similarity(self, node_1: dict[str, Any], node_2: dict[str, Any]) -> float:
        """Calculate similarity between two nodes.

        Args:
            node_1: First node data
            node_2: Second node data

        Returns:
            float: Similarity score (0.0 to 1.0)
        """
        try:
            # Compare embeddings if available
            embedding_1 = node_1.get("embedding", [])
            embedding_2 = node_2.get("embedding", [])

            if embedding_1 and embedding_2 and len(embedding_1) == len(embedding_2):
                # Cosine similarity
                import numpy as np

                vec_1 = np.array(embedding_1)
                vec_2 = np.array(embedding_2)

                dot_product = np.dot(vec_1, vec_2)
                norm_1 = np.linalg.norm(vec_1)
                norm_2 = np.linalg.norm(vec_2)

                if norm_1 > 0 and norm_2 > 0:
                    embedding_similarity = dot_product / (norm_1 * norm_2)
                else:
                    embedding_similarity = 0.0
            else:
                embedding_similarity = 0.0

            # Compare text content
            text_1 = node_1.get("text", "").lower()
            text_2 = node_2.get("text", "").lower()

            if text_1 and text_2:
                # Simple Jaccard similarity on words
                words_1 = set(text_1.split())
                words_2 = set(text_2.split())

                intersection = words_1.intersection(words_2)
                union = words_1.union(words_2)

                text_similarity = len(intersection) / len(union) if union else 0.0
            else:
                text_similarity = 0.0

            # Compare metadata
            domain_1 = node_1.get("domain", "")
            domain_2 = node_2.get("domain", "")
            domain_similarity = 1.0 if domain_1 == domain_2 else 0.0

            type_1 = node_1.get("type", "")
            type_2 = node_2.get("type", "")
            type_similarity = 1.0 if type_1 == type_2 else 0.0

            # Weighted combination
            similarity = (
                0.5 * embedding_similarity + 0.3 * text_similarity + 0.1 * domain_similarity + 0.1 * type_similarity
            )

            return max(0.0, min(1.0, similarity))

        except Exception as e:
            self.logger.error("Error calculating node similarity: %s", e)
            return 0.0

    def _merge_node_data(self, node_1: dict[str, Any], node_2: dict[str, Any]) -> dict[str, Any]:
        """Merge data from two nodes.

        Args:
            node_1: First node (primary)
            node_2: Second node (to merge into first)

        Returns:
            Dict[str, Any]: Merged node data
        """
        try:
            merged_node = node_1.copy()

            # Combine text content
            text_1 = node_1.get("text", "")
            text_2 = node_2.get("text", "")
            if text_1 and text_2 and text_1 != text_2:
                merged_node["text"] = f"{text_1} | {text_2}"

            # Average confidence
            conf_1 = node_1.get("confidence", 0.0)
            conf_2 = node_2.get("confidence", 0.0)
            merged_node["confidence"] = (conf_1 + conf_2) / 2.0

            # Combine sources
            sources_1 = node_1.get("sources", [])
            sources_2 = node_2.get("sources", [])
            merged_sources = list(set(sources_1 + sources_2))
            merged_node["sources"] = merged_sources

            # Combine entities
            entities_1 = node_1.get("entities", [])
            entities_2 = node_2.get("entities", [])
            merged_entities = list(set(entities_1 + entities_2))
            merged_node["entities"] = merged_entities

            # Add merge metadata
            merged_node["merged_from"] = [node_1.get("id", "unknown"), node_2.get("id", "unknown")]
            merged_node["merge_timestamp"] = self._get_timestamp()

            return merged_node

        except Exception as e:
            self.logger.error("Error merging node data: %s", e)
            return node_1

    def _prune_isolated_nodes(self, nodes: dict[str, Any], relationships: list[dict[str, Any]]) -> dict[str, Any]:
        """Remove nodes that have no relationships.

        Args:
            nodes: Dictionary of nodes
            relationships: List of relationships

        Returns:
            Dict[str, Any]: Pruned nodes
        """
        try:
            # Find connected nodes
            connected_nodes = set()
            for rel in relationships:
                source = rel.get("source")
                target = rel.get("target")
                if source:
                    connected_nodes.add(source)
                if target:
                    connected_nodes.add(target)

            # Keep only connected nodes
            pruned_nodes = {node_id: node_data for node_id, node_data in nodes.items() if node_id in connected_nodes}

            removed_count = len(nodes) - len(pruned_nodes)
            if removed_count > 0:
                self.logger.info("Pruned %d isolated nodes", removed_count)

            return pruned_nodes

        except Exception as e:
            self.logger.error("Error pruning isolated nodes: %s", e)
            return nodes

    async def _optimize_clusters(
        self,
        clusters: list[dict[str, Any]],
        nodes: dict[str, Any],
        relationships: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Optimize cluster structure.

        Args:
            clusters: List of clusters
            nodes: Dictionary of nodes
            relationships: List of relationships

        Returns:
            List[Dict[str, Any]]: Optimized clusters
        """
        try:
            optimized_clusters = []

            for cluster in clusters:
                cluster_nodes = cluster.get("nodes", [])

                # Remove nodes that no longer exist
                valid_nodes = [node_id for node_id in cluster_nodes if node_id in nodes]

                # Skip clusters that are too small after pruning
                if len(valid_nodes) < 2:
                    continue

                # Update cluster
                optimized_cluster = {
                    **cluster,
                    "nodes": valid_nodes,
                }

                # Recalculate cluster metrics
                cluster_relationships = [
                    rel
                    for rel in relationships
                    if (rel.get("source") in valid_nodes and rel.get("target") in valid_nodes)
                ]

                if cluster_relationships:
                    avg_weight = sum(rel.get("weight", 0.0) for rel in cluster_relationships) / len(
                        cluster_relationships
                    )

                    optimized_cluster["metadata"] = {
                        **cluster.get("metadata", {}),
                        "optimized": True,
                        "internal_avg_weight": avg_weight,
                        "internal_edge_count": len(cluster_relationships),
                    }

                optimized_clusters.append(optimized_cluster)

            removed_count = len(clusters) - len(optimized_clusters)
            if removed_count > 0:
                self.logger.info("Removed %d invalid clusters", removed_count)

            return optimized_clusters

        except Exception as e:
            self.logger.error("Error optimizing clusters: %s", e)
            return clusters

    def _calculate_graph_metrics(
        self,
        nodes: dict[str, Any],
        relationships: list[dict[str, Any]],
        clusters: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calculate comprehensive graph metrics.

        Args:
            nodes: Dictionary of nodes
            relationships: List of relationships
            clusters: List of clusters

        Returns:
            Dict[str, Any]: Graph metrics
        """
        try:
            # Basic metrics
            node_count = len(nodes)
            edge_count = len(relationships)
            cluster_count = len(clusters)

            # Density
            max_edges = node_count * (node_count - 1) / 2 if node_count > 1 else 0
            density = edge_count / max_edges if max_edges > 0 else 0.0

            # Average degree
            degrees = self._calculate_node_degrees(nodes, relationships)
            avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0.0

            # Confidence statistics
            confidences = [node.get("confidence", 0.0) for node in nodes.values()]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            # Weight statistics
            weights = [rel.get("weight", 0.0) for rel in relationships]
            avg_weight = sum(weights) / len(weights) if weights else 0.0

            # Cluster statistics
            cluster_sizes = [len(cluster.get("nodes", [])) for cluster in clusters]
            avg_cluster_size = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0.0

            return {
                "node_count": node_count,
                "edge_count": edge_count,
                "cluster_count": cluster_count,
                "density": density,
                "avg_degree": avg_degree,
                "avg_confidence": avg_confidence,
                "avg_edge_weight": avg_weight,
                "avg_cluster_size": avg_cluster_size,
                "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
                "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
            }

        except Exception as e:
            self.logger.error("Error calculating graph metrics: %s", e)
            return {"error": str(e)}

    def _create_adjacency_list(self, relationships: list[dict[str, Any]]) -> dict[str, list[str]]:
        """Create adjacency list representation.

        Args:
            relationships: List of relationships

        Returns:
            Dict[str, List[str]]: Adjacency list
        """
        try:
            adjacency_list = {}

            for rel in relationships:
                source = rel.get("source")
                target = rel.get("target")

                if source and target:
                    if source not in adjacency_list:
                        adjacency_list[source] = []
                    if target not in adjacency_list:
                        adjacency_list[target] = []

                    adjacency_list[source].append(target)
                    adjacency_list[target].append(source)

            return adjacency_list

        except Exception as e:
            self.logger.error("Error creating adjacency list: %s", e)
            return {}

    def _create_cluster_map(self, clusters: list[dict[str, Any]]) -> dict[str, str]:
        """Create mapping from node to cluster.

        Args:
            clusters: List of clusters

        Returns:
            Dict[str, str]: Node ID -> Cluster ID mapping
        """
        try:
            cluster_map = {}

            for cluster in clusters:
                cluster_id = cluster.get("id", cluster.get("metadata", {}).get("cluster_id", "unknown"))
                cluster_nodes = cluster.get("nodes", [])

                for node_id in cluster_nodes:
                    cluster_map[node_id] = cluster_id

            return cluster_map

        except Exception as e:
            self.logger.error("Error creating cluster map: %s", e)
            return {}

    def _calculate_node_degrees(self, nodes: dict[str, Any], relationships: list[dict[str, Any]]) -> dict[str, int]:
        """Calculate degree for each node.

        Args:
            nodes: Dictionary of nodes
            relationships: List of relationships

        Returns:
            Dict[str, int]: Node ID -> degree mapping
        """
        try:
            degrees = {node_id: 0 for node_id in nodes.keys()}

            for rel in relationships:
                source = rel.get("source")
                target = rel.get("target")

                if source in degrees:
                    degrees[source] += 1
                if target in degrees:
                    degrees[target] += 1

            return degrees

        except Exception as e:
            self.logger.error("Error calculating node degrees: %s", e)
            return {}

    def _validate_graph(self, graph: dict[str, Any]) -> dict[str, Any]:
        """Validate graph integrity.

        Args:
            graph: Complete graph structure

        Returns:
            Dict[str, Any]: Validation results
        """
        try:
            nodes = graph.get("nodes", {})
            relationships = graph.get("relationships", [])
            clusters = graph.get("clusters", [])

            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": [],
            }

            # Check node integrity
            for node_id, node_data in nodes.items():
                if not isinstance(node_data, dict):
                    validation_result["errors"].append(f"Node {node_id} has invalid data type")
                    validation_result["valid"] = False

                if not node_data.get("text"):
                    validation_result["warnings"].append(f"Node {node_id} has no text content")

            # Check relationship integrity
            for i, rel in enumerate(relationships):
                source = rel.get("source")
                target = rel.get("target")

                if not source or not target:
                    validation_result["errors"].append(f"Relationship {i} missing source or target")
                    validation_result["valid"] = False
                    continue

                if source not in nodes:
                    validation_result["errors"].append(f"Relationship {i} references unknown source: {source}")
                    validation_result["valid"] = False

                if target not in nodes:
                    validation_result["errors"].append(f"Relationship {i} references unknown target: {target}")
                    validation_result["valid"] = False

            # Check cluster integrity
            for i, cluster in enumerate(clusters):
                cluster_nodes = cluster.get("nodes", [])

                for node_id in cluster_nodes:
                    if node_id not in nodes:
                        validation_result["errors"].append(f"Cluster {i} references unknown node: {node_id}")
                        validation_result["valid"] = False

            return validation_result

        except Exception as e:
            self.logger.error("Error validating graph: %s", e)
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
            }

    def _get_timestamp(self) -> str:
        """Get current timestamp.

        Returns:
            str: ISO format timestamp
        """
        from datetime import datetime

        return datetime.now().isoformat()

    def get_graph_summary(self, graph: dict[str, Any]) -> dict[str, Any]:
        """Get a summary of the assembled graph.

        Args:
            graph: Complete graph structure

        Returns:
            Dict[str, Any]: Graph summary
        """
        try:
            metadata = graph.get("metadata", {})
            metrics = metadata.get("metrics", {})
            validation = metadata.get("validation", {})

            return {
                "summary": {
                    "nodes": metrics.get("node_count", 0),
                    "relationships": metrics.get("edge_count", 0),
                    "clusters": metrics.get("cluster_count", 0),
                    "density": round(metrics.get("density", 0.0), 3),
                    "avg_confidence": round(metrics.get("avg_confidence", 0.0), 3),
                    "valid": validation.get("valid", False),
                },
                "optimization": metadata.get("optimization_config", {}),
                "assembly_time": metadata.get("assembly_timestamp"),
                "validation_errors": len(validation.get("errors", [])),
                "validation_warnings": len(validation.get("warnings", [])),
            }

        except Exception as e:
            self.logger.error("Error getting graph summary: %s", e)
            return {"error": str(e)}

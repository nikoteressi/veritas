"""
from __future__ import annotations

Graph builder service for constructing fact verification graphs.

This service analyzes facts, detects relationships, and forms clusters
for efficient batch verification.
"""

import logging
import re
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from agent.models.fact import FactHierarchy
from agent.models.graph import (
    ClusterType,
    FactCluster,
    FactEdge,
    FactGraph,
    FactNode,
    RelationshipType,
)
from agent.ollama_embeddings import OllamaEmbeddingFunction
from agent.services.graph_config import ClusteringConfig
from app.config import settings

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Service for building fact verification graphs.

    Analyzes facts, detects relationships, and forms clusters for efficient verification.
    """

    def __init__(self, config: ClusteringConfig | None = None):
        self.config = config or ClusteringConfig()
        self.embeddings = OllamaEmbeddingFunction(
            ollama_url=settings.ollama_base_url, model_name="nomic-embed-text"
        )
        self.logger = logging.getLogger(__name__)

        # Cache for embeddings to avoid recomputation
        self._embedding_cache: dict[str, np.ndarray] = {}

        # Domain keywords for domain clustering
        self.domain_keywords = {
            "financial": [
                "money",
                "dollar",
                "price",
                "cost",
                "investment",
                "stock",
                "market",
                "bitcoin",
                "crypto",
                "finance",
                "bank",
                "economy",
            ],
            "political": [
                "government",
                "president",
                "election",
                "vote",
                "policy",
                "law",
                "congress",
                "senate",
                "democrat",
                "republican",
            ],
            "health": [
                "health",
                "medical",
                "doctor",
                "hospital",
                "disease",
                "treatment",
                "medicine",
                "vaccine",
                "covid",
                "virus",
            ],
            "technology": [
                "technology",
                "computer",
                "software",
                "internet",
                "ai",
                "artificial intelligence",
                "tech",
                "digital",
                "app",
            ],
            "sports": [
                "sport",
                "game",
                "team",
                "player",
                "match",
                "championship",
                "league",
                "football",
                "basketball",
                "soccer",
            ],
            "science": [
                "science",
                "research",
                "study",
                "experiment",
                "scientist",
                "discovery",
                "theory",
                "data",
                "analysis",
            ],
        }

    async def build_graph(self, fact_hierarchy: FactHierarchy) -> FactGraph:
        """
        Build a complete fact verification graph from a fact hierarchy.

        Args:
            fact_hierarchy: The fact hierarchy to convert to a graph

        Returns:
            FactGraph: Complete graph with nodes, edges, and clusters
        """
        self.logger.info(
            "Building graph from %d facts", len(fact_hierarchy.supporting_facts)
        )

        graph = FactGraph()

        # Step 1: Create nodes from facts
        nodes = await self._create_nodes(fact_hierarchy, graph)
        self.logger.info("Created %d nodes", len(nodes))

        # Step 2: Detect relationships and create edges
        edges = await self._detect_relationships(nodes, graph)
        self.logger.info("Created %d edges", len(edges))

        # Step 3: Form clusters
        clusters = await self._form_clusters(nodes, edges, graph)
        self.logger.info("Created %d clusters", len(clusters))

        # Step 4: Optimize clusters
        await self._optimize_clusters(graph)

        self.logger.info("Graph building complete: %s", graph.get_stats())
        return graph

    async def _create_nodes(
        self, fact_hierarchy: FactHierarchy, graph: FactGraph
    ) -> list[FactNode]:
        """Create fact nodes from the fact hierarchy."""
        nodes = []

        for fact in fact_hierarchy.supporting_facts:
            # Detect domain for the fact
            domain = await self._detect_domain(fact.description)

            node = FactNode(
                claim=fact.description,
                domain=domain,
                context=fact.context,
                metadata={
                    "primary_thesis": fact_hierarchy.primary_thesis,
                    "original_fact": fact.description,
                },
            )

            graph.add_node(node)
            nodes.append(node)

        return nodes

    async def _detect_domain(self, text: str) -> str:
        """Detect the domain of a fact based on keywords."""
        text_lower = text.lower()

        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return "general"

    async def _detect_relationships(
        self, nodes: list[FactNode], graph: FactGraph
    ) -> list[FactEdge]:
        """Detect relationships between facts and create edges."""
        edges = []

        # Get embeddings for all nodes
        embeddings = await self._get_embeddings([node.claim for node in nodes])

        # Save embeddings to nodes
        for i, node in enumerate(nodes):
            if i < len(embeddings):
                # Convert numpy array to list
                node.embedding = embeddings[i].tolist()
                self.logger.debug("Saved embedding for node %s", node.id)

        # Compare each pair of nodes
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i + 1 :], i + 1):
                relationships = await self._analyze_relationship(
                    node1, node2, embeddings[i], embeddings[j]
                )

                for rel_type, strength in relationships:
                    if strength >= self.config.similarity_threshold:
                        edge = FactEdge(
                            source_id=node1.id,
                            target_id=node2.id,
                            relationship_type=rel_type,
                            strength=strength,
                            metadata={
                                "detection_method": (
                                    "embedding_similarity"
                                    if rel_type == RelationshipType.SIMILARITY
                                    else "rule_based"
                                )
                            },
                        )
                        graph.add_edge(edge)
                        edges.append(edge)

        return edges

    async def _analyze_relationship(
        self,
        node1: FactNode,
        node2: FactNode,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> list[tuple[RelationshipType, float]]:
        """Analyze the relationship between two nodes."""
        relationships = []

        # Semantic similarity
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        if similarity > 0.5:  # Lower threshold for detection
            relationships.append((RelationshipType.SIMILARITY, float(similarity)))

        # Domain relationship
        if node1.domain == node2.domain and node1.domain != "general":
            relationships.append((RelationshipType.DOMAIN, 0.8))

        # Temporal relationship
        temporal_strength = await self._detect_temporal_relationship(node1, node2)
        if temporal_strength > 0.6:
            relationships.append((RelationshipType.TEMPORAL, temporal_strength))

        # Causal relationship
        causal_strength = await self._detect_causal_relationship(node1, node2)
        if causal_strength > 0.6:
            relationships.append((RelationshipType.CAUSAL, causal_strength))

        # Contradiction detection
        contradiction_strength = await self._detect_contradiction(
            node1, node2, similarity
        )
        if contradiction_strength > 0.7:
            relationships.append(
                (RelationshipType.CONTRADICTION, contradiction_strength)
            )

        # Support relationship
        support_strength = await self._detect_support_relationship(
            node1, node2, similarity
        )
        if support_strength > 0.7:
            relationships.append((RelationshipType.SUPPORT, support_strength))

        return relationships

    async def _detect_temporal_relationship(
        self, node1: FactNode, node2: FactNode
    ) -> float:
        """Detect temporal relationships between facts."""
        # Look for temporal indicators in the text
        temporal_keywords = [
            "before",
            "after",
            "during",
            "while",
            "when",
            "then",
            "next",
            "previous",
            "earlier",
            "later",
            "since",
            "until",
            "by",
            "from",
            "to",
        ]

        text1 = node1.claim.lower()
        text2 = node2.claim.lower()

        # Check for temporal keywords
        temporal_score = 0.0
        for keyword in temporal_keywords:
            if keyword in text1 or keyword in text2:
                temporal_score += 0.1

        # Check for dates in context
        if node1.context or node2.context:
            # Look for date patterns
            date_pattern = (
                r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b|"
                r"\b(january|february|march|april|may|june|july|august|"
                r"september|october|november|december)\b"
            )

            context1_str = str(node1.context)
            context2_str = str(node2.context)

            has_date1 = re.search(date_pattern, context1_str, re.IGNORECASE)
            has_date2 = re.search(date_pattern, context2_str, re.IGNORECASE)
            if has_date1 and has_date2:
                temporal_score += 0.3

        return min(temporal_score, 1.0)

    async def _detect_causal_relationship(
        self, node1: FactNode, node2: FactNode
    ) -> float:
        """Detect causal relationships between facts."""
        causal_keywords = [
            "because",
            "due to",
            "caused by",
            "results in",
            "leads to",
            "triggers",
            "consequently",
            "therefore",
            "thus",
            "hence",
            "as a result",
            "owing to",
        ]

        text1 = node1.claim.lower()
        text2 = node2.claim.lower()
        combined_text = f"{text1} {text2}"

        causal_score = 0.0
        for keyword in causal_keywords:
            if keyword in combined_text:
                causal_score += 0.2

        return min(causal_score, 1.0)

    async def _detect_contradiction(
        self, node1: FactNode, node2: FactNode, similarity: float
    ) -> float:
        """Detect contradictions between facts."""
        contradiction_keywords = [
            "not",
            "no",
            "never",
            "false",
            "incorrect",
            "wrong",
            "opposite",
            "contrary",
            "however",
            "but",
            "although",
            "despite",
            "contradicts",
        ]

        text1 = node1.claim.lower()
        text2 = node2.claim.lower()

        # High similarity but with negation words might indicate contradiction
        if similarity > 0.7:
            contradiction_score = 0.0
            for keyword in contradiction_keywords:
                if keyword in text1 or keyword in text2:
                    contradiction_score += 0.3

            return min(contradiction_score, 1.0)

        return 0.0

    async def _detect_support_relationship(
        self, node1: FactNode, node2: FactNode, similarity: float
    ) -> float:
        """Detect support relationships between facts."""
        support_keywords = [
            "supports",
            "confirms",
            "validates",
            "proves",
            "demonstrates",
            "shows",
            "indicates",
            "suggests",
            "evidence",
            "corroborates",
        ]

        text1 = node1.claim.lower()
        text2 = node2.claim.lower()
        combined_text = f"{text1} {text2}"

        support_score = similarity * 0.5  # Base on similarity

        for keyword in support_keywords:
            if keyword in combined_text:
                support_score += 0.2

        return min(support_score, 1.0)

    async def _form_clusters(
        self, nodes: list[FactNode], edges: list[FactEdge], graph: FactGraph
    ) -> list[FactCluster]:
        """Form clusters of related facts."""
        clusters = []

        # Similarity-based clustering
        similarity_clusters = await self._create_similarity_clusters(nodes, graph)
        clusters.extend(similarity_clusters)

        # Domain-based clustering
        if self.config.enable_domain_clustering:
            domain_clusters = await self._create_domain_clusters(nodes, graph)
            clusters.extend(domain_clusters)

        # Temporal clustering
        if self.config.enable_temporal_clustering:
            temporal_clusters = await self._create_temporal_clusters(
                nodes, edges, graph
            )
            clusters.extend(temporal_clusters)

        # Causal clustering
        if self.config.enable_causal_clustering:
            causal_clusters = await self._create_causal_clusters(nodes, edges, graph)
            clusters.extend(causal_clusters)

        return clusters

    async def _create_similarity_clusters(
        self, nodes: list[FactNode], graph: FactGraph
    ) -> list[FactCluster]:
        """Create clusters based on semantic similarity."""
        if len(nodes) < self.config.min_cluster_size:
            return []

        # Get embeddings
        embeddings = await self._get_embeddings([node.claim for node in nodes])

        # Use DBSCAN for clustering
        clustering = DBSCAN(
            eps=self.config.dbscan_eps,
            min_samples=self.config.dbscan_min_samples,
            metric="cosine",
        ).fit(embeddings)

        clusters = []
        cluster_labels = set(clustering.labels_)

        for label in cluster_labels:
            if label == -1:  # Noise points
                continue

            cluster_nodes = [
                nodes[i] for i, l in enumerate(clustering.labels_) if l == label
            ]

            if (
                len(cluster_nodes) >= self.config.min_cluster_size
                and len(cluster_nodes) <= self.config.max_cluster_size
            ):
                # Generate shared context
                shared_context = await self._generate_shared_context(cluster_nodes)

                cluster = FactCluster(
                    nodes=cluster_nodes,
                    cluster_type=ClusterType.SIMILARITY_CLUSTER,
                    shared_context=shared_context,
                    verification_strategy="batch_similarity",
                    metadata={
                        "clustering_method": "dbscan",
                        "similarity_threshold": self.config.similarity_threshold,
                    },
                )

                graph.add_cluster(cluster)
                clusters.append(cluster)

        return clusters

    async def _create_domain_clusters(
        self, nodes: list[FactNode], graph: FactGraph
    ) -> list[FactCluster]:
        """Create clusters based on domain similarity."""
        domain_groups = {}

        for node in nodes:
            if node.domain not in domain_groups:
                domain_groups[node.domain] = []
            domain_groups[node.domain].append(node)

        clusters = []
        for domain, domain_nodes in domain_groups.items():
            if (
                len(domain_nodes) >= self.config.min_cluster_size
                and len(domain_nodes) <= self.config.max_cluster_size
            ):
                shared_context = f"All facts relate to the {domain} domain"

                cluster = FactCluster(
                    nodes=domain_nodes,
                    cluster_type=ClusterType.DOMAIN_CLUSTER,
                    shared_context=shared_context,
                    verification_strategy="domain_specific",
                    metadata={"domain": domain, "clustering_method": "domain_based"},
                )

                graph.add_cluster(cluster)
                clusters.append(cluster)

        return clusters

    async def _create_temporal_clusters(
        self, nodes: list[FactNode], edges: list[FactEdge], graph: FactGraph
    ) -> list[FactCluster]:
        """Create clusters based on temporal relationships."""
        temporal_edges = [
            e for e in edges if e.relationship_type == RelationshipType.TEMPORAL
        ]

        if not temporal_edges:
            return []

        # Find connected components in temporal graph
        temporal_groups = self._find_connected_components(nodes, temporal_edges)

        clusters = []
        for group in temporal_groups:
            if (
                len(group) >= self.config.min_cluster_size
                and len(group) <= self.config.max_cluster_size
            ):
                shared_context = "Facts are temporally related"

                cluster = FactCluster(
                    nodes=group,
                    cluster_type=ClusterType.TEMPORAL_CLUSTER,
                    shared_context=shared_context,
                    verification_strategy="temporal_sequence",
                    metadata={"clustering_method": "temporal_relationships"},
                )

                graph.add_cluster(cluster)
                clusters.append(cluster)

        return clusters

    async def _create_causal_clusters(
        self, nodes: list[FactNode], edges: list[FactEdge], graph: FactGraph
    ) -> list[FactCluster]:
        """Create clusters based on causal relationships."""
        causal_edges = [
            e for e in edges if e.relationship_type == RelationshipType.CAUSAL
        ]

        if not causal_edges:
            return []

        # Find connected components in causal graph
        causal_groups = self._find_connected_components(nodes, causal_edges)

        clusters = []
        for group in causal_groups:
            if (
                len(group) >= self.config.min_cluster_size
                and len(group) <= self.config.max_cluster_size
            ):
                shared_context = "Facts are causally related"

                cluster = FactCluster(
                    nodes=group,
                    cluster_type=ClusterType.CAUSAL_CLUSTER,
                    shared_context=shared_context,
                    verification_strategy="causal_chain",
                    metadata={"clustering_method": "causal_relationships"},
                )

                graph.add_cluster(cluster)
                clusters.append(cluster)

        return clusters

    def _find_connected_components(
        self, nodes: list[FactNode], edges: list[FactEdge]
    ) -> list[list[FactNode]]:
        """Find connected components in a graph defined by nodes and edges."""
        node_map = {node.id: node for node in nodes}
        adjacency = {node.id: set() for node in nodes}

        # Build adjacency list
        for edge in edges:
            if edge.source_id in adjacency and edge.target_id in adjacency:
                adjacency[edge.source_id].add(edge.target_id)
                adjacency[edge.target_id].add(edge.source_id)

        visited = set()
        components = []

        def dfs(node_id, component):
            if node_id in visited:
                return
            visited.add(node_id)
            component.append(node_map[node_id])

            for neighbor_id in adjacency[node_id]:
                dfs(neighbor_id, component)

        for node in nodes:
            if node.id not in visited:
                component = []
                dfs(node.id, component)
                if component:
                    components.append(component)

        return components

    async def _generate_shared_context(self, nodes: list[FactNode]) -> str:
        """Generate shared context for a cluster of nodes."""
        claims = [node.claim for node in nodes]

        # Simple approach: find common keywords
        all_words = []
        for claim in claims:
            words = claim.lower().split()
            all_words.extend(words)

        # Count word frequency
        word_freq = {}
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Find common words (appearing in multiple claims)
        common_words = [
            word for word, freq in word_freq.items() if freq > 1 and len(word) > 3
        ]

        if common_words:
            # Limit to 3 most common words to keep context short
            return f"Themes: {', '.join(common_words[:3])}"
        else:
            return "Related facts"

    async def _optimize_clusters(self, graph: FactGraph):
        """Optimize clusters by removing overlaps and merging similar clusters."""
        clusters = list(graph.clusters.values())

        # Remove clusters that are too small or too large
        clusters_to_remove = []
        for cluster in clusters:
            if (
                len(cluster.nodes) < self.config.min_cluster_size
                or len(cluster.nodes) > self.config.max_cluster_size
            ):
                clusters_to_remove.append(cluster.id)

        for cluster_id in clusters_to_remove:
            graph.remove_cluster(cluster_id)

        # Get updated clusters list after removal
        clusters = list(graph.clusters.values())

        # Implement cluster merging logic for similar clusters
        await self._merge_similar_clusters(graph, clusters)

        # Implement overlap resolution
        await self._resolve_cluster_overlaps(graph)

    async def _merge_similar_clusters(
        self, graph: FactGraph, clusters: list[FactCluster]
    ):
        """Merge clusters that are very similar based on their shared context and nodes."""
        if len(clusters) < 2:
            return

        # Calculate similarity between clusters
        cluster_pairs_to_merge = []

        for i, cluster1 in enumerate(clusters):
            for cluster2 in clusters[i + 1 :]:
                # Skip if clusters are of different types (unless both are similarity-based)
                if (
                    cluster1.cluster_type != cluster2.cluster_type
                    and cluster1.cluster_type != ClusterType.SIMILARITY_CLUSTER
                    and cluster2.cluster_type != ClusterType.SIMILARITY_CLUSTER
                ):
                    continue

                # Calculate overlap ratio
                nodes1 = set(node.id for node in cluster1.nodes)
                nodes2 = set(node.id for node in cluster2.nodes)
                overlap = len(nodes1.intersection(nodes2))
                union = len(nodes1.union(nodes2))
                overlap_ratio = overlap / union if union > 0 else 0

                # Calculate context similarity if both have shared context
                context_similarity = 0.0
                if cluster1.shared_context and cluster2.shared_context:
                    try:
                        contexts = [cluster1.shared_context, cluster2.shared_context]
                        embeddings = await self._get_embeddings(contexts)
                        if len(embeddings) == 2:
                            similarity_matrix = cosine_similarity(
                                [embeddings[0]], [embeddings[1]]
                            )
                            context_similarity = similarity_matrix[0][0]
                    except (ValueError, TypeError, RuntimeError) as e:
                        self.logger.warning(
                            "Failed to calculate context similarity: %s", e
                        )

                # Merge if high overlap or high context similarity
                if (overlap_ratio > 0.7 or context_similarity > 0.8) and len(
                    nodes1.union(nodes2)
                ) <= self.config.max_cluster_size:
                    cluster_pairs_to_merge.append(
                        (cluster1, cluster2, overlap_ratio, context_similarity)
                    )

        # Sort by similarity and merge the most similar pairs first
        cluster_pairs_to_merge.sort(key=lambda x: max(x[2], x[3]), reverse=True)

        merged_cluster_ids = set()
        for (
            cluster1,
            cluster2,
            overlap_ratio,
            context_similarity,
        ) in cluster_pairs_to_merge:
            # Skip if either cluster was already merged
            if cluster1.id in merged_cluster_ids or cluster2.id in merged_cluster_ids:
                continue

            # Merge clusters
            merged_nodes = list(
                {node.id: node for node in cluster1.nodes + cluster2.nodes}.values()
            )

            # Create new shared context
            if cluster1.shared_context and cluster2.shared_context:
                # Extract key terms instead of combining full contexts
                context1_clean = (
                    cluster1.shared_context.replace("Combined:", "")
                    .replace("Common themes:", "")
                    .strip()
                )
                context2_clean = (
                    cluster2.shared_context.replace("Combined:", "")
                    .replace("Common themes:", "")
                    .strip()
                )

                # Take only first few words from each context
                words1 = context1_clean.split()[:3]
                words2 = context2_clean.split()[:3]

                # Combine key terms, avoiding duplicates
                combined_words = []
                for word in words1 + words2:
                    if word not in combined_words and len(word) > 3:
                        combined_words.append(word)

                merged_context = "Merged: " + " ".join(combined_words[:5])
            else:
                merged_context = (
                    cluster1.shared_context
                    or cluster2.shared_context
                    or "Merged cluster context"
                )

            # Create merged cluster
            merged_cluster = FactCluster(
                nodes=merged_nodes,
                # Default to similarity cluster for merged
                cluster_type=ClusterType.SIMILARITY_CLUSTER,
                shared_context=merged_context,
                verification_strategy="batch_similarity",
                metadata={
                    "merged_from": [cluster1.id, cluster2.id],
                    "overlap_ratio": overlap_ratio,
                    "context_similarity": context_similarity,
                    "clustering_method": "merged",
                },
            )

            # Remove old clusters and add merged cluster
            graph.remove_cluster(cluster1.id)
            graph.remove_cluster(cluster2.id)
            graph.add_cluster(merged_cluster)

            merged_cluster_ids.add(cluster1.id)
            merged_cluster_ids.add(cluster2.id)

            self.logger.info(
                "Merged clusters %s and %s into %s",
                cluster1.id,
                cluster2.id,
                merged_cluster.id,
            )

    async def _resolve_cluster_overlaps(self, graph: FactGraph):
        """Resolve overlaps between clusters by reassigning nodes to the most appropriate
        cluster."""
        clusters = list(graph.clusters.values())

        if len(clusters) < 2:
            return

        # Find all nodes that appear in multiple clusters
        node_to_clusters = {}
        for cluster in clusters:
            for node in cluster.nodes:
                if node.id not in node_to_clusters:
                    node_to_clusters[node.id] = []
                node_to_clusters[node.id].append(cluster)

        # Process overlapping nodes
        for node_id, containing_clusters in node_to_clusters.items():
            if len(containing_clusters) <= 1:
                continue  # No overlap

            # Find the best cluster for this node
            best_cluster = await self._find_best_cluster_for_node(
                node_id, containing_clusters
            )

            # Remove node from all other clusters
            for cluster in containing_clusters:
                if cluster.id != best_cluster.id:
                    # Remove node from cluster
                    cluster.nodes = [n for n in cluster.nodes if n.id != node_id]
                    self.logger.debug(
                        "Removed node %s from cluster %s", node_id, cluster.id
                    )

        # Remove empty clusters
        empty_clusters = [c for c in clusters if len(c.nodes) == 0]
        for cluster in empty_clusters:
            graph.remove_cluster(cluster.id)
            self.logger.info("Removed empty cluster %s", cluster.id)

    async def _find_best_cluster_for_node(
        self, node_id: str, clusters: list[FactCluster]
    ) -> FactCluster:
        """Find the best cluster for a node based on similarity and cluster characteristics."""
        if len(clusters) == 1:
            return clusters[0]

        # Find the node
        target_node = None
        for cluster in clusters:
            for node in cluster.nodes:
                if node.id == node_id:
                    target_node = node
                    break
            if target_node:
                break

        if not target_node:
            raise ValueError(f"Node {node_id} not found in any cluster")

        best_cluster = clusters[0]
        best_score = 0.0

        for cluster in clusters:
            score = 0.0

            # Calculate similarity with other nodes in cluster
            other_nodes = [n for n in cluster.nodes if n.id != node_id]
            if other_nodes:
                try:
                    # Get embeddings for target node and other nodes
                    # Limit to 5 for performance
                    texts = [target_node.claim] + [n.claim for n in other_nodes[:5]]
                    embeddings = await self._get_embeddings(texts)

                    if len(embeddings) > 1:
                        target_embedding = embeddings[0:1]
                        other_embeddings = embeddings[1:]
                        similarities = cosine_similarity(
                            target_embedding, other_embeddings
                        )
                        avg_similarity = np.mean(similarities)
                        score += (
                            avg_similarity * 0.7
                        )  # 70% weight for content similarity
                except (ValueError, TypeError, RuntimeError) as e:
                    self.logger.warning("Failed to calculate node similarity: %s", e)

            # Add bonus for domain match
            if other_nodes and target_node.domain:
                domain_matches = sum(
                    1 for n in other_nodes if n.domain == target_node.domain
                )
                domain_score = domain_matches / len(other_nodes)
                score += domain_score * 0.2  # 20% weight for domain similarity

            # Add bonus for cluster size (prefer larger, more stable clusters)
            size_score = min(len(cluster.nodes) / self.config.max_cluster_size, 1.0)
            score += size_score * 0.1  # 10% weight for cluster size

            if score > best_score:
                best_score = score
                best_cluster = cluster

        return best_cluster

    async def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        """Get embeddings for a list of texts with caching."""
        embeddings = []

        for text in texts:
            if text in self._embedding_cache:
                embeddings.append(self._embedding_cache[text])
            else:
                # Get embedding from Ollama using the __call__ method
                embedding_result = self.embeddings([text])  # Pass as list
                if embedding_result and len(embedding_result) > 0:
                    embedding_array = np.array(embedding_result[0])
                    self._embedding_cache[text] = embedding_array
                    embeddings.append(embedding_array)
                else:
                    # Raise error if embedding fails
                    self.logger.error(
                        "Failed to get embedding for text: %s...", text[:50]
                    )
                    raise RuntimeError(
                        f"Failed to get embedding for text: {text[:50]}..."
                    )

        return np.array(embeddings)

    async def clear_cache(self):
        """Clear the embedding cache."""
        await self._embedding_cache.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._embedding_cache),
            "cache_memory_mb": (
                sum(arr.nbytes for arr in self._embedding_cache.values())
                / (1024 * 1024)
            ),
        }

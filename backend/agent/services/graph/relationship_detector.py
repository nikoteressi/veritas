"""Relationship detector for identifying connections between graph nodes.

This service handles detection of various types of relationships between nodes
including semantic similarity, temporal, causal, contradiction, and support relationships.
"""

import logging
from typing import Any

from agent.services import EnhancedOllamaEmbeddings

logger = logging.getLogger(__name__)


class RelationshipDetector:
    """Detects relationships between graph nodes.

    Identifies various types of relationships including semantic similarity,
    temporal connections, causal relationships, contradictions, and support.
    """

    def __init__(self, embedding_service: EnhancedOllamaEmbeddings):
        """Initialize the relationship detector.

        Args:
            embedding_service: Service for calculating embeddings and similarity
        """
        self.embedding_service = embedding_service
        self.logger = logging.getLogger(__name__)

    async def detect_relationships(self, nodes: dict[str, Any]) -> list[dict[str, Any]]:
        """Detect relationships between all nodes.

        Args:
            nodes: Dictionary of node_id -> node_data

        Returns:
            List[Dict[str, Any]]: List of detected relationships
        """
        try:
            self.logger.info(
                "Detecting relationships between %d nodes", len(nodes))

            relationships = []
            node_ids = list(nodes.keys())

            # Compare each pair of nodes
            for i, node_id_1 in enumerate(node_ids):
                for _j, node_id_2 in enumerate(node_ids[i + 1:], i + 1):
                    node_1 = nodes[node_id_1]
                    node_2 = nodes[node_id_2]

                    # Detect various types of relationships
                    detected_relationships = await self._detect_node_pair_relationships(
                        node_id_1, node_1, node_id_2, node_2
                    )

                    relationships.extend(detected_relationships)

            self.logger.info("Detected %d relationships", len(relationships))
            return relationships

        except Exception as e:
            self.logger.error("Error detecting relationships: %s", e)
            return []

    async def _detect_node_pair_relationships(
        self,
        node_id_1: str,
        node_1: dict[str, Any],
        node_id_2: str,
        node_2: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Detect relationships between a pair of nodes.

        Args:
            node_id_1: ID of first node
            node_1: First node data
            node_id_2: ID of second node
            node_2: Second node data

        Returns:
            List[Dict[str, Any]]: List of detected relationships
        """
        relationships = []

        try:
            # Calculate semantic similarity
            similarity = self.embedding_service.calculate_similarity(
                node_1.get("embedding", []), node_2.get("embedding", [])
            )

            # Detect semantic similarity relationship
            if similarity > 0.7:
                relationships.append(
                    {
                        "source": node_id_1,
                        "target": node_id_2,
                        "type": "semantic_similarity",
                        "weight": similarity,
                        "metadata": {"similarity_score": similarity},
                    }
                )

            # Detect domain relationship
            if node_1.get("domain") == node_2.get("domain") and node_1.get("domain") != "general":
                relationships.append(
                    {
                        "source": node_id_1,
                        "target": node_id_2,
                        "type": "domain_similarity",
                        "weight": 0.6,
                        "metadata": {"domain": node_1.get("domain")},
                    }
                )

            # Detect temporal relationship
            temporal_rel = self._detect_temporal_relationship(node_1, node_2)
            if temporal_rel:
                relationships.append(
                    {
                        "source": node_id_1,
                        "target": node_id_2,
                        "type": "temporal",
                        "weight": temporal_rel["weight"],
                        "metadata": temporal_rel["metadata"],
                    }
                )

            # Detect causal relationship
            causal_rel = self._detect_causal_relationship(node_1, node_2)
            if causal_rel:
                relationships.append(
                    {
                        "source": node_id_1,
                        "target": node_id_2,
                        "type": "causal",
                        "weight": causal_rel["weight"],
                        "metadata": causal_rel["metadata"],
                    }
                )

            # Detect contradiction
            contradiction_rel = self._detect_contradiction(
                node_1, node_2, similarity)
            if contradiction_rel:
                relationships.append(
                    {
                        "source": node_id_1,
                        "target": node_id_2,
                        "type": "contradiction",
                        "weight": contradiction_rel["weight"],
                        "metadata": contradiction_rel["metadata"],
                    }
                )

            # Detect support relationship
            support_rel = self._detect_support_relationship(
                node_1, node_2, similarity)
            if support_rel:
                relationships.append(
                    {
                        "source": node_id_1,
                        "target": node_id_2,
                        "type": "support",
                        "weight": support_rel["weight"],
                        "metadata": support_rel["metadata"],
                    }
                )

        except Exception as e:
            self.logger.warning(
                "Error detecting relationships between %s and %s: %s", node_id_1, node_id_2, e)

        return relationships

    def _detect_temporal_relationship(self, node_1: dict[str, Any], node_2: dict[str, Any]) -> dict[str, Any] | None:
        """Detect temporal relationships between nodes.

        Args:
            node_1: First node data
            node_2: Second node data

        Returns:
            Dict[str, Any] | None: Temporal relationship data or None
        """
        try:
            text_1 = node_1.get("text", "").lower()
            text_2 = node_2.get("text", "").lower()

            temporal_1 = node_1.get("temporal_info", {})
            temporal_2 = node_2.get("temporal_info", {})

            # Check for explicit temporal keywords
            temporal_keywords = [
                "before",
                "after",
                "during",
                "while",
                "when",
                "then",
                "previously",
                "subsequently",
                "earlier",
                "later",
            ]

            has_temporal_keywords = any(
                keyword in text_1 or keyword in text_2 for keyword in temporal_keywords)

            # Check for temporal information in metadata
            has_temporal_info = bool(temporal_1 or temporal_2)

            if has_temporal_keywords or has_temporal_info:
                # Determine temporal relationship type
                if any(keyword in text_1 or keyword in text_2 for keyword in ["before", "earlier", "previously"]):
                    rel_type = "before"
                elif any(keyword in text_1 or keyword in text_2 for keyword in ["after", "later", "subsequently"]):
                    rel_type = "after"
                elif any(keyword in text_1 or keyword in text_2 for keyword in ["during", "while", "when"]):
                    rel_type = "concurrent"
                else:
                    rel_type = "temporal_related"

                return {
                    "weight": 0.5,
                    "metadata": {
                        "temporal_type": rel_type,
                        "temporal_info_1": temporal_1,
                        "temporal_info_2": temporal_2,
                    },
                }

            return None

        except Exception as e:
            self.logger.warning("Error detecting temporal relationship: %s", e)
            return None

    def _detect_causal_relationship(self, node_1: dict[str, Any], node_2: dict[str, Any]) -> dict[str, Any] | None:
        """Detect causal relationships between nodes.

        Args:
            node_1: First node data
            node_2: Second node data

        Returns:
            Dict[str, Any] | None: Causal relationship data or None
        """
        try:
            text_1 = node_1.get("text", "").lower()
            text_2 = node_2.get("text", "").lower()

            # Causal keywords
            causal_keywords = [
                "because",
                "due to",
                "caused by",
                "results in",
                "leads to",
                "therefore",
                "consequently",
                "as a result",
                "owing to",
                "triggers",
                "produces",
                "generates",
                "brings about",
            ]

            # Check for causal language
            has_causal_language = any(
                keyword in text_1 or keyword in text_2 for keyword in causal_keywords)

            if has_causal_language:
                # Determine causal direction (simplified)
                if any(keyword in text_1 for keyword in ["because", "due to", "caused by"]):
                    causal_direction = "node_2_causes_node_1"
                elif any(keyword in text_1 for keyword in ["results in", "leads to", "therefore"]):
                    causal_direction = "node_1_causes_node_2"
                else:
                    causal_direction = "bidirectional"

                return {
                    "weight": 0.6,
                    "metadata": {
                        "causal_direction": causal_direction,
                        "causal_indicators": [
                            keyword for keyword in causal_keywords if keyword in text_1 or keyword in text_2
                        ],
                    },
                }

            return None

        except Exception as e:
            self.logger.warning("Error detecting causal relationship: %s", e)
            return None

    def _detect_contradiction(
        self, node_1: dict[str, Any], node_2: dict[str, Any], similarity: float
    ) -> dict[str, Any] | None:
        """Detect contradictions between nodes.

        Args:
            node_1: First node data
            node_2: Second node data
            similarity: Semantic similarity score

        Returns:
            Dict[str, Any] | None: Contradiction relationship data or None
        """
        try:
            text_1 = node_1.get("text", "").lower()
            text_2 = node_2.get("text", "").lower()

            # Contradiction indicators
            contradiction_keywords = [
                "not",
                "no",
                "never",
                "none",
                "neither",
                "nor",
                "however",
                "but",
                "although",
                "despite",
                "contrary",
                "opposite",
                "different",
                "disagree",
                "dispute",
            ]

            # Check for negation and contradiction patterns
            has_negation_1 = any(keyword in text_1 for keyword in [
                                 "not", "no", "never", "none"])
            has_negation_2 = any(keyword in text_2 for keyword in [
                                 "not", "no", "never", "none"])

            has_contradiction_keywords = any(
                keyword in text_1 or keyword in text_2 for keyword in contradiction_keywords
            )

            # High similarity with negation suggests contradiction
            if similarity > 0.6 and (has_negation_1 or has_negation_2 or has_contradiction_keywords):
                # High similarity makes contradiction stronger
                contradiction_strength = similarity * 0.8

                return {
                    "weight": contradiction_strength,
                    "metadata": {
                        "contradiction_type": (
                            "semantic_negation" if (
                                has_negation_1 or has_negation_2) else "explicit_contradiction"
                        ),
                        "similarity_score": similarity,
                        "contradiction_indicators": [
                            keyword for keyword in contradiction_keywords if keyword in text_1 or keyword in text_2
                        ],
                    },
                }

            return None

        except Exception as e:
            self.logger.warning("Error detecting contradiction: %s", e)
            return None

    def _detect_support_relationship(
        self, node_1: dict[str, Any], node_2: dict[str, Any], similarity: float
    ) -> dict[str, Any] | None:
        """Detect support relationships between nodes.

        Args:
            node_1: First node data
            node_2: Second node data
            similarity: Semantic similarity score

        Returns:
            Dict[str, Any] | None: Support relationship data or None
        """
        try:
            text_1 = node_1.get("text", "").lower()
            text_2 = node_2.get("text", "").lower()

            # Support indicators
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
                "agrees",
                "consistent",
                "aligns",
                "reinforces",
            ]

            # Check for support language
            has_support_language = any(
                keyword in text_1 or keyword in text_2 for keyword in support_keywords)

            # High similarity without contradiction suggests support
            if similarity > 0.5 and (has_support_language or similarity > 0.7):
                # Check that there's no contradiction
                contradiction_keywords = [
                    "not", "no", "however", "but", "contrary", "opposite"]
                has_contradiction = any(
                    keyword in text_1 or keyword in text_2 for keyword in contradiction_keywords)

                if not has_contradiction:
                    support_strength = similarity * 0.9

                    return {
                        "weight": support_strength,
                        "metadata": {
                            "support_type": "explicit" if has_support_language else "implicit",
                            "similarity_score": similarity,
                            "support_indicators": [
                                keyword for keyword in support_keywords if keyword in text_1 or keyword in text_2
                            ],
                        },
                    }

            return None

        except Exception as e:
            self.logger.warning("Error detecting support relationship: %s", e)
            return None

    def filter_relationships_by_threshold(
        self, relationships: list[dict[str, Any]], threshold: float = 0.3
    ) -> list[dict[str, Any]]:
        """Filter relationships by weight threshold.

        Args:
            relationships: List of relationships
            threshold: Minimum weight threshold

        Returns:
            List[Dict[str, Any]]: Filtered relationships
        """
        try:
            filtered = [rel for rel in relationships if rel.get(
                "weight", 0.0) >= threshold]

            self.logger.info(
                "Filtered relationships: %d -> %d (threshold=%.2f)", len(
                    relationships), len(filtered), threshold
            )

            return filtered

        except Exception as e:
            self.logger.error("Error filtering relationships: %s", e)
            return relationships

    def get_relationship_statistics(self, relationships: list[dict[str, Any]]) -> dict[str, Any]:
        """Get statistics about detected relationships.

        Args:
            relationships: List of relationships

        Returns:
            Dict[str, Any]: Relationship statistics
        """
        try:
            if not relationships:
                return {"total": 0, "types": {}, "avg_weight": 0.0}

            # Count by type
            type_counts = {}
            total_weight = 0.0

            for rel in relationships:
                rel_type = rel.get("type", "unknown")
                type_counts[rel_type] = type_counts.get(rel_type, 0) + 1
                total_weight += rel.get("weight", 0.0)

            avg_weight = total_weight / len(relationships)

            return {
                "total": len(relationships),
                "types": type_counts,
                "avg_weight": avg_weight,
                "max_weight": max(rel.get("weight", 0.0) for rel in relationships),
                "min_weight": min(rel.get("weight", 0.0) for rel in relationships),
            }

        except Exception as e:
            self.logger.error(
                "Error calculating relationship statistics: %s", e)
            return {"total": 0, "types": {}, "avg_weight": 0.0, "error": str(e)}

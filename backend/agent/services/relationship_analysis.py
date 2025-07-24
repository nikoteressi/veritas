"""Advanced relationship analysis system for fact verification.

from __future__ import annotations

This module implements sophisticated analysis of relationships between facts,
including causal inference, temporal dependencies, and semantic connections.
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import networkx as nx
import numpy as np
import spacy
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agent.ollama_embeddings import create_ollama_embedding_function

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class RelationshipConfig:
    """Configuration for relationship analysis."""

    enable_causal_inference: bool = True
    enable_temporal_analysis: bool = True
    enable_semantic_analysis: bool = True
    semantic_similarity_threshold: float = 0.7
    temporal_window_days: int = 30
    causal_significance_level: float = 0.05
    max_relationship_depth: int = 3
    min_evidence_overlap: int = 2
    use_transformer_embeddings: bool = True
    transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class RelationshipType:
    """Types of relationships between facts."""

    CAUSAL = "causal"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    CONTRADICTORY = "contradictory"
    SUPPORTING = "supporting"
    DEPENDENT = "dependent"
    INDEPENDENT = "independent"


@dataclass
class FactRelationship:
    """Represents a relationship between two facts."""

    fact_id_1: str
    fact_id_2: str
    relationship_type: str
    strength: float
    confidence: float
    evidence: list[str]
    temporal_order: Optional[str] = None  # "before", "after", "concurrent"
    # "causes", "caused_by", "bidirectional"
    causal_direction: Optional[str] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SemanticAnalyzer:
    """Analyzes semantic relationships between facts."""

    def __init__(self, config: RelationshipConfig):
        self.config = config
        self.nlp = None
        self.ollama_embedding_function = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        self.logger = logging.getLogger(__name__)

        self._initialize_models()

    def _initialize_models(self):
        """Initialize NLP models."""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("spaCy model not found, using basic text processing")

        if self.config.use_transformer_embeddings:
            try:
                # Initialize Ollama embedding function
                self.ollama_embedding_function = create_ollama_embedding_function()
                self.logger.info("Initialized Ollama embeddings for semantic analysis")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Ollama embeddings: {e}")

    def analyze_semantic_similarity(
        self, fact1: dict[str, Any], fact2: dict[str, Any]
    ) -> float:
        """Analyze semantic similarity between two facts."""
        text1 = self._extract_fact_text(fact1)
        text2 = self._extract_fact_text(fact2)

        if (
            self.config.use_transformer_embeddings
            and self.ollama_embedding_function is not None
        ):
            return self._compute_ollama_similarity(text1, text2)
        else:
            return self._compute_tfidf_similarity(text1, text2)

    def _extract_fact_text(self, fact: dict[str, Any]) -> str:
        """Extract text content from fact for analysis."""
        text_parts = []

        # Extract claim text
        if "claim" in fact:
            text_parts.append(str(fact["claim"]))

        # Extract evidence text
        if "evidence" in fact:
            for evidence in fact["evidence"]:
                if isinstance(evidence, dict) and "text" in evidence:
                    text_parts.append(evidence["text"])
                elif isinstance(evidence, str):
                    text_parts.append(evidence)

        # Extract summary if available
        if "summary" in fact:
            text_parts.append(str(fact["summary"]))

        return " ".join(text_parts)

    def _compute_ollama_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity using Ollama embeddings."""
        try:
            # Get embeddings from Ollama
            embeddings1 = self.ollama_embedding_function([text1])
            embeddings2 = self.ollama_embedding_function([text2])

            if embeddings1 and embeddings2:
                # Convert to numpy arrays for cosine similarity
                import numpy as np

                vec1 = np.array(embeddings1[0])
                vec2 = np.array(embeddings2[0])

                # Compute cosine similarity
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)

                if norm1 > 0 and norm2 > 0:
                    similarity = dot_product / (norm1 * norm2)
                    return float(similarity)

            return 0.0

        except Exception as e:
            self.logger.warning(f"Ollama similarity computation failed: {e}")
            return self._compute_tfidf_similarity(text1, text2)

    def _compute_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity using TF-IDF vectors."""
        try:
            # Fit TF-IDF on both texts
            corpus = [text1, text2]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)

            # Compute cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            return float(similarity_matrix[0, 1])

        except Exception as e:
            self.logger.warning(f"TF-IDF similarity computation failed: {e}")
            return 0.0

    def extract_entities_and_concepts(
        self, fact: dict[str, Any]
    ) -> dict[str, list[str]]:
        """Extract entities and concepts from fact text."""
        text = self._extract_fact_text(fact)

        entities = {"PERSON": [], "ORG": [], "GPE": [], "EVENT": [], "PRODUCT": []}
        concepts = []

        if self.nlp is not None:
            try:
                doc = self.nlp(text)

                # Extract named entities
                for ent in doc.ents:
                    if ent.label_ in entities:
                        entities[ent.label_].append(ent.text)

                # Extract noun phrases as concepts
                for chunk in doc.noun_chunks:
                    concepts.append(chunk.text)

            except Exception as e:
                self.logger.warning(f"Entity extraction failed: {e}")

        return {"entities": entities, "concepts": concepts}


class TemporalAnalyzer:
    """Analyzes temporal relationships between facts."""

    def __init__(self, config: RelationshipConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def analyze_temporal_relationship(
        self, fact1: dict[str, Any], fact2: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze temporal relationship between two facts."""
        timestamp1 = self._extract_timestamp(fact1)
        timestamp2 = self._extract_timestamp(fact2)

        if timestamp1 is None or timestamp2 is None:
            return {"relationship": "unknown", "confidence": 0.0}

        # Calculate time difference
        time_diff = timestamp2 - timestamp1
        time_diff_days = abs(time_diff.total_seconds()) / (24 * 3600)

        # Determine temporal relationship
        if time_diff_days <= 1:
            relationship = "concurrent"
        elif time_diff.total_seconds() > 0:
            relationship = "after"  # fact2 is after fact1
        else:
            relationship = "before"  # fact2 is before fact1

        # Calculate confidence based on temporal proximity
        if time_diff_days <= self.config.temporal_window_days:
            confidence = 1.0 - (time_diff_days / self.config.temporal_window_days)
        else:
            confidence = 0.1  # Low confidence for distant events

        return {
            "relationship": relationship,
            "confidence": float(confidence),
            "time_difference_days": float(time_diff_days),
            "timestamp1": timestamp1.isoformat(),
            "timestamp2": timestamp2.isoformat(),
        }

    def _extract_timestamp(self, fact: dict[str, Any]) -> Optional[datetime]:
        """Extract timestamp from fact data."""
        # Try different timestamp fields
        timestamp_fields = ["timestamp", "created_at", "published_at", "date"]

        for field in timestamp_fields:
            if field in fact:
                timestamp_value = fact[field]
                if isinstance(timestamp_value, datetime):
                    return timestamp_value
                elif isinstance(timestamp_value, str):
                    try:
                        return datetime.fromisoformat(
                            timestamp_value.replace("Z", "+00:00")
                        )
                    except ValueError:
                        continue

        # Try to extract from evidence
        if "evidence" in fact:
            for evidence in fact["evidence"]:
                if isinstance(evidence, dict):
                    for field in timestamp_fields:
                        if field in evidence:
                            try:
                                timestamp_value = evidence[field]
                                if isinstance(timestamp_value, datetime):
                                    return timestamp_value
                                elif isinstance(timestamp_value, str):
                                    return datetime.fromisoformat(
                                        timestamp_value.replace("Z", "+00:00")
                                    )
                            except (ValueError, TypeError):
                                continue

        return None


class CausalAnalyzer:
    """Analyzes causal relationships between facts."""

    def __init__(self, config: RelationshipConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def analyze_causal_relationship(
        self,
        fact1: dict[str, Any],
        fact2: dict[str, Any],
        context_facts: list[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Analyze potential causal relationship between two facts."""
        # Extract features for causal analysis
        features1 = self._extract_causal_features(fact1)
        features2 = self._extract_causal_features(fact2)

        # Check for causal indicators
        causal_indicators = self._detect_causal_indicators(fact1, fact2)

        # Perform statistical causal inference if enough data
        if context_facts and len(context_facts) > 10:
            statistical_result = self._perform_statistical_causal_inference(
                fact1, fact2, context_facts
            )
        else:
            statistical_result = {"method": "insufficient_data", "p_value": 1.0}

        # Combine evidence
        causal_strength = self._compute_causal_strength(
            causal_indicators, statistical_result
        )
        causal_direction = self._determine_causal_direction(
            fact1, fact2, causal_indicators
        )

        return {
            "causal_strength": causal_strength,
            "causal_direction": causal_direction,
            "causal_indicators": causal_indicators,
            "statistical_result": statistical_result,
            "confidence": self._compute_causal_confidence(
                causal_indicators, statistical_result
            ),
        }

    def _extract_causal_features(self, fact: dict[str, Any]) -> dict[str, Any]:
        """Extract features relevant for causal analysis."""
        text = self._extract_fact_text(fact)

        # Look for causal keywords
        causal_keywords = [
            "because",
            "due to",
            "caused by",
            "results in",
            "leads to",
            "triggers",
            "influences",
            "affects",
            "impacts",
            "consequence",
        ]

        causal_score = sum(1 for keyword in causal_keywords if keyword in text.lower())

        return {
            "causal_keyword_count": causal_score,
            "text_length": len(text),
            "has_temporal_markers": any(
                marker in text.lower()
                for marker in ["before", "after", "then", "subsequently", "following"]
            ),
        }

    def _extract_fact_text(self, fact: dict[str, Any]) -> str:
        """Extract text content from fact."""
        text_parts = []
        if "claim" in fact:
            text_parts.append(str(fact["claim"]))
        if "summary" in fact:
            text_parts.append(str(fact["summary"]))
        return " ".join(text_parts)

    def _detect_causal_indicators(
        self, fact1: dict[str, Any], fact2: dict[str, Any]
    ) -> dict[str, Any]:
        """Detect linguistic and structural indicators of causality."""
        text1 = self._extract_fact_text(fact1)
        text2 = self._extract_fact_text(fact2)

        # Linguistic indicators
        causal_phrases_1_to_2 = [
            "leads to",
            "causes",
            "results in",
            "triggers",
            "brings about",
        ]
        causal_phrases_2_to_1 = ["because of", "due to", "caused by", "triggered by"]

        indicators = {
            "linguistic_1_to_2": any(
                phrase in text1.lower() for phrase in causal_phrases_1_to_2
            ),
            "linguistic_2_to_1": any(
                phrase in text2.lower() for phrase in causal_phrases_2_to_1
            ),
            "temporal_precedence": False,  # Will be set by temporal analyzer
            "mechanism_described": "mechanism" in (text1 + text2).lower(),
            "correlation_mentioned": any(
                word in (text1 + text2).lower()
                for word in ["correlation", "associated", "linked"]
            ),
        }

        return indicators

    def _perform_statistical_causal_inference(
        self,
        fact1: dict[str, Any],
        fact2: dict[str, Any],
        context_facts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Perform statistical causal inference using available data."""
        try:
            # This is a simplified implementation
            # In practice, you'd need more sophisticated causal inference methods

            # Extract numerical features if available
            features = []
            outcomes = []

            for fact in [fact1, fact2] + context_facts:
                # Extract numerical features (simplified)
                feature_vector = self._extract_numerical_features(fact)
                outcome = self._extract_outcome_variable(fact)

                if feature_vector is not None and outcome is not None:
                    features.append(feature_vector)
                    outcomes.append(outcome)

            if len(features) < 10:
                return {"method": "insufficient_data", "p_value": 1.0}

            # Perform correlation test as a proxy for causal inference
            features_array = np.array(features)
            outcomes_array = np.array(outcomes)

            if features_array.shape[1] > 0:
                correlation, p_value = stats.pearsonr(
                    features_array[:, 0], outcomes_array
                )
                return {
                    "method": "correlation_test",
                    "correlation": float(correlation),
                    "p_value": float(p_value),
                    "significant": p_value < self.config.causal_significance_level,
                }

        except Exception as e:
            self.logger.warning(f"Statistical causal inference failed: {e}")

        return {"method": "failed", "p_value": 1.0}

    def _extract_numerical_features(self, fact: dict[str, Any]) -> list[float] | None:
        """Extract numerical features from fact for statistical analysis."""
        # This is a placeholder - in practice, you'd extract meaningful numerical features
        features = []

        # Extract confidence scores
        if "confidence" in fact:
            features.append(float(fact["confidence"]))

        # Extract evidence count
        if "evidence" in fact:
            features.append(float(len(fact["evidence"])))

        # Extract verification score
        if "verification_score" in fact:
            features.append(float(fact["verification_score"]))

        return features if features else None

    def _extract_outcome_variable(self, fact: dict[str, Any]) -> Optional[float]:
        """Extract outcome variable for causal analysis."""
        # Use verification result as outcome
        if "verdict" in fact:
            verdict_map = {"TRUE": 1.0, "FALSE": 0.0, "MIXED": 0.5, "UNKNOWN": 0.5}
            return verdict_map.get(fact["verdict"], 0.5)

        if "verification_score" in fact:
            return float(fact["verification_score"])

        return None

    def _compute_causal_strength(
        self, indicators: dict[str, Any], statistical_result: dict[str, Any]
    ) -> float:
        """Compute overall causal strength."""
        strength = 0.0

        # Linguistic evidence
        if indicators.get("linguistic_1_to_2") or indicators.get("linguistic_2_to_1"):
            strength += 0.3

        # Temporal precedence
        if indicators.get("temporal_precedence"):
            strength += 0.2

        # Mechanism described
        if indicators.get("mechanism_described"):
            strength += 0.2

        # Statistical significance
        if statistical_result.get("significant"):
            strength += 0.3

        return min(1.0, strength)

    def _determine_causal_direction(
        self, fact1: dict[str, Any], fact2: dict[str, Any], indicators: dict[str, Any]
    ) -> str:
        """Determine direction of causal relationship."""
        if indicators.get("linguistic_1_to_2"):
            return "fact1_causes_fact2"
        elif indicators.get("linguistic_2_to_1"):
            return "fact2_causes_fact1"
        elif indicators.get("temporal_precedence"):
            return "temporal_order"
        else:
            return "bidirectional_or_unknown"

    def _compute_causal_confidence(
        self, indicators: dict[str, Any], statistical_result: dict[str, Any]
    ) -> float:
        """Compute confidence in causal relationship."""
        confidence = 0.0

        # Count supporting indicators
        indicator_count = sum(1 for v in indicators.values() if v)
        confidence += indicator_count * 0.15

        # Statistical evidence
        if statistical_result.get("method") != "insufficient_data":
            p_value = statistical_result.get("p_value", 1.0)
            confidence += (1 - p_value) * 0.4

        return min(1.0, confidence)


class RelationshipAnalysisEngine:
    """Main engine for analyzing relationships between facts."""

    def __init__(self, config: Optional[RelationshipConfig] = None):
        self.config = config or RelationshipConfig()
        self.semantic_analyzer = SemanticAnalyzer(self.config)
        self.temporal_analyzer = TemporalAnalyzer(self.config)
        self.causal_analyzer = CausalAnalyzer(self.config)
        self.logger = logging.getLogger(__name__)

    async def analyze_fact_relationships(
        self, facts: list[dict[str, Any]]
    ) -> list[FactRelationship]:
        """Analyze relationships between a collection of facts."""
        relationships = []

        # Analyze pairwise relationships
        for i, fact1 in enumerate(facts):
            for j, fact2 in enumerate(facts[i + 1 :], i + 1):
                relationship = await self._analyze_fact_pair(fact1, fact2, facts)
                if relationship:
                    relationships.append(relationship)

        # Filter and rank relationships
        filtered_relationships = self._filter_relationships(relationships)

        return filtered_relationships

    async def _analyze_fact_pair(
        self,
        fact1: dict[str, Any],
        fact2: dict[str, Any],
        context_facts: list[dict[str, Any]],
    ) -> Optional[FactRelationship]:
        """Analyze relationship between a pair of facts."""
        fact_id_1 = fact1.get("id", f"fact_{hash(str(fact1))}")
        fact_id_2 = fact2.get("id", f"fact_{hash(str(fact2))}")

        # Semantic analysis
        semantic_similarity = 0.0
        if self.config.enable_semantic_analysis:
            semantic_similarity = self.semantic_analyzer.analyze_semantic_similarity(
                fact1, fact2
            )

        # Temporal analysis
        temporal_result = {}
        if self.config.enable_temporal_analysis:
            temporal_result = self.temporal_analyzer.analyze_temporal_relationship(
                fact1, fact2
            )

        # Causal analysis
        causal_result = {}
        if self.config.enable_causal_inference:
            causal_result = self.causal_analyzer.analyze_causal_relationship(
                fact1, fact2, context_facts
            )

        # Determine primary relationship type and strength
        relationship_type, strength, confidence = self._determine_primary_relationship(
            semantic_similarity, temporal_result, causal_result
        )

        # Only create relationship if strength is above threshold
        if strength < 0.3:
            return None

        # Create relationship object
        relationship = FactRelationship(
            fact_id_1=fact_id_1,
            fact_id_2=fact_id_2,
            relationship_type=relationship_type,
            strength=strength,
            confidence=confidence,
            evidence=self._collect_relationship_evidence(
                semantic_similarity, temporal_result, causal_result
            ),
            temporal_order=temporal_result.get("relationship"),
            causal_direction=causal_result.get("causal_direction"),
            metadata={
                "semantic_similarity": semantic_similarity,
                "temporal_analysis": temporal_result,
                "causal_analysis": causal_result,
            },
        )

        return relationship

    def _determine_primary_relationship(
        self,
        semantic_similarity: float,
        temporal_result: dict[str, Any],
        causal_result: dict[str, Any],
    ) -> tuple[str, float, float]:
        """Determine the primary relationship type and its strength."""
        # Causal relationship (highest priority)
        causal_strength = causal_result.get("causal_strength", 0.0)
        if causal_strength > 0.5:
            return (
                RelationshipType.CAUSAL,
                causal_strength,
                causal_result.get("confidence", 0.5),
            )

        # Semantic relationship
        if semantic_similarity > self.config.semantic_similarity_threshold:
            # Check if facts contradict each other
            if self._detect_contradiction(
                semantic_similarity, temporal_result, causal_result
            ):
                return RelationshipType.CONTRADICTORY, semantic_similarity, 0.8
            else:
                return RelationshipType.SUPPORTING, semantic_similarity, 0.7

        # Temporal relationship
        temporal_confidence = temporal_result.get("confidence", 0.0)
        if temporal_confidence > 0.5:
            return RelationshipType.TEMPORAL, temporal_confidence, temporal_confidence

        # Weak semantic relationship
        if semantic_similarity > 0.3:
            return RelationshipType.SEMANTIC, semantic_similarity, 0.5

        # Default to independent
        return RelationshipType.INDEPENDENT, 0.1, 0.1

    def _detect_contradiction(
        self,
        semantic_similarity: float,
        temporal_result: dict[str, Any],
        causal_result: dict[str, Any],
    ) -> bool:
        """Detect if facts contradict each other."""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated contradiction detection

        # High semantic similarity but opposing causal directions might indicate contradiction
        causal_direction = causal_result.get("causal_direction", "")
        if semantic_similarity > 0.7 and "bidirectional" in causal_direction:
            return True

        return False

    def _collect_relationship_evidence(
        self,
        semantic_similarity: float,
        temporal_result: dict[str, Any],
        causal_result: dict[str, Any],
    ) -> list[str]:
        """Collect evidence supporting the relationship."""
        evidence = []

        if semantic_similarity > 0.5:
            evidence.append(f"High semantic similarity: {semantic_similarity:.2f}")

        if temporal_result.get("confidence", 0) > 0.5:
            evidence.append(
                f"Temporal relationship: {temporal_result.get('relationship', 'unknown')}"
            )

        if causal_result.get("causal_strength", 0) > 0.3:
            evidence.append("Causal indicators detected")

        return evidence

    def _filter_relationships(
        self, relationships: list[FactRelationship]
    ) -> list[FactRelationship]:
        """Filter and rank relationships by strength and confidence."""
        # Filter by minimum strength
        filtered = [r for r in relationships if r.strength > 0.3]

        # Sort by combined strength and confidence
        filtered.sort(key=lambda r: r.strength * r.confidence, reverse=True)

        return filtered

    def build_relationship_graph(
        self, relationships: list[FactRelationship]
    ) -> nx.Graph:
        """Build a NetworkX graph from relationships."""
        G = nx.Graph()

        for rel in relationships:
            G.add_edge(
                rel.fact_id_1,
                rel.fact_id_2,
                relationship_type=rel.relationship_type,
                strength=rel.strength,
                confidence=rel.confidence,
                evidence=rel.evidence,
            )

        return G

    def get_relationship_stats(self) -> dict[str, Any]:
        """Get statistics about the relationship analysis engine."""
        return {
            "config": {
                "semantic_threshold": self.config.semantic_similarity_threshold,
                "temporal_window_days": self.config.temporal_window_days,
                "causal_significance_level": self.config.causal_significance_level,
                "max_depth": self.config.max_relationship_depth,
            },
            "analyzers": {
                "semantic_initialized": self.semantic_analyzer is not None,
                "temporal_initialized": self.temporal_analyzer is not None,
                "causal_initialized": self.causal_analyzer is not None,
            },
        }

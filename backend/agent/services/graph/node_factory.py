"""Node factory for creating graph nodes from evidence texts.

This service handles the creation of graph nodes with embeddings,
confidence scores, and metadata extraction.
"""

import logging
import uuid
from typing import Any

from agent.services.embeddings.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class NodeFactory:
    """Factory for creating graph nodes from evidence texts.

    Handles node creation with embeddings, confidence calculation,
    and metadata extraction.
    """

    def __init__(self, embedding_service: EmbeddingService):
        """Initialize the node factory.

        Args:
            embedding_service: Service for generating embeddings
        """
        self.embedding_service = embedding_service
        self.logger = logging.getLogger(__name__)

    async def create_nodes_from_evidence(self, evidence_texts: list[str], claim: str) -> dict[str, Any]:
        """Create graph nodes from evidence texts.

        Args:
            evidence_texts: List of evidence text strings
            claim: The claim being verified

        Returns:
            Dict[str, Any]: Dictionary of node_id -> node_data
        """
        try:
            self.logger.info("Creating %d nodes from evidence texts", len(evidence_texts))

            nodes = {}
            claim_embedding = await self.embedding_service.get_embedding(claim)

            for i, text in enumerate(evidence_texts):
                if not text or not text.strip():
                    continue

                try:
                    # Generate unique node ID
                    node_id = f"node_{i}_{uuid.uuid4().hex[:8]}"

                    # Create node data
                    node_data = await self._create_single_node(text, claim, claim_embedding, node_id)

                    nodes[node_id] = node_data

                except Exception as e:
                    self.logger.warning("Error creating node for text %d: %s", i, e)
                    continue

            self.logger.info("Successfully created %d nodes", len(nodes))
            return nodes

        except Exception as e:
            self.logger.error("Error creating nodes from evidence: %s", e)
            return {}

    async def _create_single_node(
        self,
        text: str,
        claim: str,
        claim_embedding: list[float],
        node_id: str,
    ) -> dict[str, Any]:
        """Create a single graph node from text.

        Args:
            text: Evidence text
            claim: The claim being verified
            claim_embedding: Embedding of the claim
            node_id: Unique identifier for the node

        Returns:
            Dict[str, Any]: Node data
        """
        try:
            # Generate embedding for the text
            text_embedding = await self.embedding_service.get_embedding(text)

            # Calculate confidence based on similarity to claim
            confidence = self._calculate_node_confidence(text, claim, text_embedding, claim_embedding)

            # Extract metadata
            metadata = self._extract_node_metadata(text)

            # Create node data
            node_data = {
                "id": node_id,
                "text": text,
                "embedding": text_embedding,
                "confidence": confidence,
                "type": self._determine_node_type(text),
                "domain": self._extract_domain(text),
                "sources": self._extract_sources(text),
                "temporal_info": self._extract_temporal_info(text),
                "entities": self._extract_entities(text),
                "metadata": metadata,
            }

            self.logger.debug("Created node %s: type=%s, confidence=%.3f", node_id, node_data["type"], confidence)

            return node_data

        except Exception as e:
            self.logger.error("Error creating single node: %s", e)
            # Return minimal node on error
            return {
                "id": node_id,
                "text": text,
                "embedding": [],
                "confidence": 0.0,
                "type": "unknown",
                "domain": "unknown",
                "sources": [],
                "temporal_info": {},
                "entities": [],
                "metadata": {"error": str(e)},
            }

    def _calculate_node_confidence(
        self,
        text: str,
        claim: str,
        text_embedding: list[float],
        claim_embedding: list[float],
    ) -> float:
        """Calculate confidence score for a node.

        Args:
            text: Evidence text
            claim: The claim being verified
            text_embedding: Embedding of the text
            claim_embedding: Embedding of the claim

        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        try:
            # Calculate semantic similarity
            similarity = self.embedding_service.calculate_similarity(text_embedding, claim_embedding)

            # Adjust confidence based on text characteristics
            confidence = similarity

            # Boost confidence for longer, more detailed texts
            text_length_factor = min(len(text) / 500, 1.0)  # Cap at 500 chars
            confidence += text_length_factor * 0.1

            # Boost confidence for texts with specific indicators
            if any(
                indicator in text.lower()
                for indicator in ["study", "research", "data", "evidence", "analysis", "report"]
            ):
                confidence += 0.1

            # Reduce confidence for uncertain language
            if any(
                uncertain in text.lower() for uncertain in ["might", "could", "possibly", "allegedly", "reportedly"]
            ):
                confidence -= 0.1

            # Ensure confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))

            return confidence

        except Exception as e:
            self.logger.warning("Error calculating node confidence: %s", e)
            return 0.5  # Default confidence

    def _determine_node_type(self, text: str) -> str:
        """Determine the type of a node based on its content.

        Args:
            text: Evidence text

        Returns:
            str: Node type
        """
        text_lower = text.lower()

        # Check for different types of evidence
        if any(keyword in text_lower for keyword in ["study", "research", "experiment", "trial", "analysis"]):
            return "research"
        elif any(keyword in text_lower for keyword in ["news", "report", "article", "journalist", "media"]):
            return "news"
        elif any(keyword in text_lower for keyword in ["expert", "professor", "doctor", "scientist", "specialist"]):
            return "expert_opinion"
        elif any(keyword in text_lower for keyword in ["data", "statistics", "number", "percent", "figure"]):
            return "statistical"
        elif any(keyword in text_lower for keyword in ["law", "legal", "court", "regulation", "policy"]):
            return "legal"
        elif any(keyword in text_lower for keyword in ["history", "historical", "past", "previous", "before"]):
            return "historical"
        else:
            return "general"

    def _extract_domain(self, text: str) -> str:
        """Extract the domain/topic of the text.

        Args:
            text: Evidence text

        Returns:
            str: Domain identifier
        """
        text_lower = text.lower()

        # Define domain keywords
        domains = {
            "health": ["health", "medical", "disease", "treatment", "medicine", "doctor"],
            "science": ["science", "research", "study", "experiment", "data", "analysis"],
            "politics": ["politics", "government", "policy", "election", "politician"],
            "economics": ["economy", "economic", "finance", "money", "market", "business"],
            "technology": ["technology", "tech", "computer", "software", "digital", "internet"],
            "environment": ["environment", "climate", "nature", "pollution", "green", "ecology"],
            "sports": ["sport", "game", "team", "player", "match", "competition"],
            "entertainment": ["movie", "music", "celebrity", "entertainment", "show", "film"],
        }

        # Find matching domain
        for domain, keywords in domains.items():
            if any(keyword in text_lower for keyword in keywords):
                return domain

        return "general"

    def _extract_sources(self, text: str) -> list[str]:
        """Extract source information from text.

        Args:
            text: Evidence text

        Returns:
            List[str]: List of identified sources
        """
        sources = []
        text_lower = text.lower()

        # Look for common source indicators
        source_patterns = [
            "according to",
            "reported by",
            "published in",
            "study by",
            "research from",
            "data from",
        ]

        for pattern in source_patterns:
            if pattern in text_lower:
                # Extract text after the pattern (simplified extraction)
                start_idx = text_lower.find(pattern) + len(pattern)
                # Limit extraction length
                end_idx = min(start_idx + 100, len(text))
                potential_source = text[start_idx:end_idx].strip()

                # Clean up the extracted source
                if potential_source and len(potential_source) > 3:
                    # Take first sentence or phrase
                    for delimiter in [".", ",", ";", "\n"]:
                        if delimiter in potential_source:
                            potential_source = potential_source.split(delimiter)[0]
                            break

                    sources.append(potential_source.strip())

        return sources[:3]  # Limit to 3 sources per node

    def _extract_temporal_info(self, text: str) -> dict[str, Any]:
        """Extract temporal information from text.

        Args:
            text: Evidence text

        Returns:
            Dict[str, Any]: Temporal information
        """
        temporal_info = {}
        text_lower = text.lower()

        # Look for temporal indicators
        temporal_keywords = {
            "recent": {"type": "relative", "period": "recent"},
            "recently": {"type": "relative", "period": "recent"},
            "yesterday": {"type": "relative", "period": "yesterday"},
            "today": {"type": "relative", "period": "today"},
            "last week": {"type": "relative", "period": "last_week"},
            "last month": {"type": "relative", "period": "last_month"},
            "last year": {"type": "relative", "period": "last_year"},
            "years ago": {"type": "relative", "period": "years_ago"},
            "decades ago": {"type": "relative", "period": "decades_ago"},
        }

        for keyword, info in temporal_keywords.items():
            if keyword in text_lower:
                temporal_info.update(info)
                break

        # Look for specific years (simplified pattern)
        import re

        year_pattern = r"\b(19|20)\d{2}\b"
        years = re.findall(year_pattern, text)
        if years:
            temporal_info["years"] = years
            temporal_info["type"] = "absolute"

        return temporal_info

    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities from text (simplified).

        Args:
            text: Evidence text

        Returns:
            List[str]: List of identified entities
        """
        entities = []

        # Simple entity extraction based on capitalization patterns
        import re

        # Find capitalized words (potential proper nouns)
        capitalized_words = re.findall(r"\b[A-Z][a-z]+\b", text)

        # Filter out common words that are often capitalized
        common_words = {
            "The",
            "This",
            "That",
            "These",
            "Those",
            "A",
            "An",
            "In",
            "On",
            "At",
            "To",
            "For",
            "With",
            "By",
            "From",
            "Of",
            "As",
            "Is",
            "Are",
            "Was",
            "Were",
        }

        entities = [word for word in capitalized_words if word not in common_words]

        # Limit to unique entities
        entities = list(set(entities))[:10]  # Limit to 10 entities

        return entities

    def _extract_node_metadata(self, text: str) -> dict[str, Any]:
        """Extract additional metadata from text.

        Args:
            text: Evidence text

        Returns:
            Dict[str, Any]: Metadata dictionary
        """
        metadata = {
            "length": len(text),
            "word_count": len(text.split()),
            "has_numbers": any(char.isdigit() for char in text),
            "has_urls": "http" in text.lower() or "www." in text.lower(),
            "language_indicators": self._detect_language_indicators(text),
            "sentiment_indicators": self._detect_sentiment_indicators(text),
        }

        return metadata

    def _detect_language_indicators(self, text: str) -> dict[str, Any]:
        """Detect language-related indicators in text.

        Args:
            text: Evidence text

        Returns:
            Dict[str, Any]: Language indicators
        """
        text_lower = text.lower()

        indicators = {
            "formal": any(
                word in text_lower for word in ["furthermore", "however", "therefore", "consequently", "nevertheless"]
            ),
            "informal": any(word in text_lower for word in ["yeah", "okay", "stuff", "things", "kinda", "sorta"]),
            "technical": any(
                word in text_lower for word in ["analysis", "methodology", "hypothesis", "correlation", "significant"]
            ),
            "uncertain": any(
                word in text_lower for word in ["might", "could", "possibly", "perhaps", "allegedly", "reportedly"]
            ),
        }

        return indicators

    def _detect_sentiment_indicators(self, text: str) -> dict[str, Any]:
        """Detect sentiment indicators in text.

        Args:
            text: Evidence text

        Returns:
            Dict[str, Any]: Sentiment indicators
        """
        text_lower = text.lower()

        positive_words = [
            "good",
            "great",
            "excellent",
            "positive",
            "beneficial",
            "effective",
            "successful",
            "improved",
            "better",
            "advantage",
        ]

        negative_words = [
            "bad",
            "poor",
            "negative",
            "harmful",
            "ineffective",
            "failed",
            "worse",
            "problem",
            "issue",
            "concern",
            "risk",
            "danger",
        ]

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        return {
            "positive_indicators": positive_count,
            "negative_indicators": negative_count,
            "sentiment_balance": positive_count - negative_count,
        }

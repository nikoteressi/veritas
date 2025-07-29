"""
Explainable relevance scorer with detailed analysis and interpretability.

Provides detailed explanations for relevance scores using multiple analysis techniques
including linguistic analysis, attention mechanisms, and feature importance.
"""

import asyncio
import hashlib
import logging
import re
import time
from collections import Counter
from typing import Any
from app.exceptions import ValidationError

import numpy as np

from ..cache.intelligent_cache import CacheStrategy, IntelligentCache
from ..cache.temporal_analysis_cache import TemporalAnalysisCache
from .cached_hybrid_relevance_scorer import CachedHybridRelevanceScorer

logger = logging.getLogger(__name__)


class ExplainableRelevanceScorer:
    """Explainable relevance scorer with detailed analysis and interpretability."""

    def __init__(
        self,
        hybrid_scorer: CachedHybridRelevanceScorer | None = None,
        temporal_analyzer: TemporalAnalysisCache | None = None,
        cache_size: int = 300,
        explanation_depth: str = "detailed",  # "basic", "detailed", "comprehensive"
    ):
        """
        Initialize explainable relevance scorer.

        Args:
            hybrid_scorer: Hybrid relevance scorer instance
            temporal_analyzer: Temporal analysis cache instance
            cache_size: Cache size for explanations
            explanation_depth: Level of explanation detail
        """
        self.hybrid_scorer = hybrid_scorer or CachedHybridRelevanceScorer()
        self.temporal_analyzer = temporal_analyzer or TemporalAnalysisCache()
        self.explanation_depth = explanation_depth

        self.cache = IntelligentCache(max_memory_size=cache_size)

        # Linguistic patterns for analysis
        self.question_patterns = [
            r"\b(what|who|when|where|why|how|which)\b",
            r"\?",
            r"\b(is|are|was|were|do|does|did|can|could|will|would|should)\b",
        ]

        self.importance_indicators = [
            r"\b(important|critical|essential|key|main|primary|significant)\b",
            r"\b(must|need|require|necessary)\b",
            r"\b(urgent|immediate|priority)\b",
        ]

        # Performance metrics
        self.metrics = {
            "total_explanations": 0,
            "cache_hits": 0,
            "linguistic_analyses": 0,
            "feature_extractions": 0,
            "attention_calculations": 0,
            "processing_time": 0.0,
        }

        logger.info(
            "Explainable relevance scorer initialized with %s explanations", explanation_depth)

    def _generate_cache_key(self, query: str, text: str, depth: str = "detailed") -> str:
        """Generate cache key for explanation."""
        combined = f"{query}|{text}|{depth}"
        text_hash = hashlib.md5(combined.encode("utf-8")).hexdigest()
        return f"explanation:{depth}:{text_hash}"

    def _extract_linguistic_features(self, text: str) -> dict[str, Any]:
        """Extract linguistic features from text."""
        try:
            self.metrics["linguistic_analyses"] += 1

            # Basic text statistics
            words = text.lower().split()
            sentences = re.split(r"[.!?]+", text)

            # Word frequency analysis
            word_freq = Counter(words)

            # Pattern matching
            question_matches = sum(len(re.findall(pattern, text, re.IGNORECASE))
                                   for pattern in self.question_patterns)
            importance_matches = sum(
                len(re.findall(pattern, text, re.IGNORECASE)) for pattern in self.importance_indicators
            )

            # Text complexity metrics
            avg_word_length = np.mean([len(word)
                                      for word in words]) if words else 0
            avg_sentence_length = np.mean(
                [len(sent.split()) for sent in sentences if sent.strip()]) if sentences else 0

            # Named entity patterns (simple heuristic)
            capitalized_words = [word for word in words if word.istitle()]

            features = {
                "word_count": len(words),
                "sentence_count": len([s for s in sentences if s.strip()]),
                "avg_word_length": avg_word_length,
                "avg_sentence_length": avg_sentence_length,
                "unique_words": len(set(words)),
                "lexical_diversity": len(set(words)) / len(words) if words else 0,
                "question_indicators": question_matches,
                "importance_indicators": importance_matches,
                "capitalized_words": len(capitalized_words),
                "most_frequent_words": word_freq.most_common(5),
                "text_length": len(text),
            }

            return features

        except Exception as e:
            raise ValidationError(
                f"Failed to extract linguistic features: {e}") from e

    def _calculate_word_importance(self, query: str, text: str) -> dict[str, float]:
        """Calculate importance scores for words in text."""
        try:
            self.metrics["feature_extractions"] += 1

            query_words = set(query.lower().split())
            text_words = text.lower().split()

            word_importance = {}

            for word in set(text_words):
                importance = 0.0

                # Exact match with query
                if word in query_words:
                    importance += 1.0

                # Partial match with query words
                for query_word in query_words:
                    if word in query_word or query_word in word:
                        importance += 0.5

                # Position importance (earlier words are more important)
                first_occurrence = text_words.index(word)
                position_weight = 1.0 - (first_occurrence / len(text_words))
                importance += position_weight * 0.3

                # Frequency importance
                frequency = text_words.count(word)
                # Cap at 5 occurrences
                frequency_weight = min(frequency / 5.0, 1.0)
                importance += frequency_weight * 0.2

                # Capitalization importance
                if word.istitle():
                    importance += 0.2

                word_importance[word] = importance

            return word_importance

        except Exception as e:
            raise ValidationError(
                f"Failed to calculate word importance: {e}") from e

    def _calculate_attention_weights(self, query: str, text: str) -> dict[str, float]:
        """Calculate attention weights for text segments."""
        try:
            self.metrics["attention_calculations"] += 1

            # Split text into sentences
            sentences = re.split(r"[.!?]+", text)
            sentences = [s.strip() for s in sentences if s.strip()]

            query_words = set(query.lower().split())
            attention_weights = {}

            for i, sentence in enumerate(sentences):
                sentence_words = set(sentence.lower().split())

                # Calculate overlap with query
                overlap = len(query_words.intersection(sentence_words))
                overlap_ratio = overlap / \
                    len(query_words) if query_words else 0

                # Position weight (earlier sentences are more important)
                position_weight = 1.0 - (i / len(sentences))

                # Length normalization
                length_weight = min(len(sentence.split()) / 20.0, 1.0)

                # Combined attention weight
                attention = overlap_ratio * 0.6 + position_weight * 0.3 + length_weight * 0.1
                attention_weights[f"sentence_{i}"] = attention

            return attention_weights

        except Exception as e:
            raise ValidationError(
                f"Failed to calculate attention weights: {e}") from e

    def _generate_basic_explanation(self, score: float, components: dict[str, Any]) -> dict[str, Any]:
        """Generate basic explanation for relevance score."""
        explanation = {
            "score": score,
            "level": "basic",
            "summary": self._get_score_summary(score),
            "main_factors": [],
        }

        # Add main contributing factors
        if "bm25_score" in components:
            explanation["main_factors"].append(
                {
                    "factor": "keyword_matching",
                    "score": components["bm25_score"],
                    "description": "How well the text matches query keywords",
                }
            )

        if "semantic_score" in components:
            explanation["main_factors"].append(
                {
                    "factor": "semantic_similarity",
                    "score": components["semantic_score"],
                    "description": "How similar the text meaning is to the query",
                }
            )

        return explanation

    def _generate_detailed_explanation(
        self, query: str, text: str, score: float, components: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate detailed explanation for relevance score."""
        # Start with basic explanation
        explanation = self._generate_basic_explanation(score, components)
        explanation["level"] = "detailed"

        # Add linguistic analysis
        linguistic_features = self._extract_linguistic_features(text)
        explanation["linguistic_analysis"] = linguistic_features

        # Add word importance
        word_importance = self._calculate_word_importance(query, text)
        top_words = sorted(word_importance.items(),
                           key=lambda x: x[1], reverse=True)[:10]
        explanation["important_words"] = top_words

        # Add matching analysis
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())

        explanation["matching_analysis"] = {
            "exact_matches": list(query_words.intersection(text_words)),
            "query_coverage": (len(query_words.intersection(text_words)) / len(query_words) if query_words else 0),
            "text_relevance": (len(query_words.intersection(text_words)) / len(text_words) if text_words else 0),
        }

        return explanation

    def _generate_comprehensive_explanation(
        self, query: str, text: str, score: float, components: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate comprehensive explanation for relevance score."""
        # Start with detailed explanation
        explanation = self._generate_detailed_explanation(
            query, text, score, components)
        explanation["level"] = "comprehensive"

        # Add attention analysis
        attention_weights = self._calculate_attention_weights(query, text)
        explanation["attention_analysis"] = attention_weights

        # Add component breakdown
        explanation["component_breakdown"] = components

        # Add recommendations
        explanation["recommendations"] = self._generate_recommendations(
            query, text, score, components)

        # Add confidence metrics
        explanation["confidence_metrics"] = self._calculate_confidence_metrics(
            components)

        return explanation

    def _get_score_summary(self, score: float) -> str:
        """Get human-readable summary for score."""
        if score >= 0.8:
            return "Highly relevant - strong match with query"
        elif score >= 0.6:
            return "Relevant - good match with query"
        elif score >= 0.4:
            return "Moderately relevant - partial match with query"
        elif score >= 0.2:
            return "Somewhat relevant - weak match with query"
        else:
            return "Not relevant - poor match with query"

    def _generate_recommendations(self, query: str, text: str, score: float, components: dict[str, Any]) -> list[str]:
        """Generate recommendations for improving relevance."""
        recommendations = []

        if score < 0.5:
            recommendations.append(
                "Consider refining the query to better match the content")

            if components.get("bm25_score", 0) < 0.3:
                recommendations.append(
                    "Add more specific keywords that appear in the text")

            if components.get("semantic_score", 0) < 0.3:
                recommendations.append(
                    "Use synonyms or related terms to improve semantic matching")

        if len(text.split()) < 50:
            recommendations.append(
                "Text might be too short for comprehensive analysis")

        return recommendations

    def _calculate_confidence_metrics(self, components: dict[str, Any]) -> dict[str, float]:
        """Calculate confidence metrics for the relevance score."""
        confidence = {}

        # Score consistency
        scores = [v for k, v in components.items() if k.endswith(
            "_score") and isinstance(v, (int, float))]
        if scores:
            confidence["score_variance"] = float(np.var(scores))
            confidence["score_consistency"] = 1.0 - \
                min(confidence["score_variance"], 1.0)

        # Component agreement
        if "bm25_score" in components and "semantic_score" in components:
            score_diff = abs(
                components["bm25_score"] - components["semantic_score"])
            confidence["component_agreement"] = 1.0 - min(score_diff, 1.0)

        return confidence

    async def explain_relevance(
        self,
        query: str,
        text: str,
        content_date: str | None = None,
        use_cache: bool = True,
        depth: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate detailed explanation for relevance score.

        Args:
            query: Search query
            text: Text to analyze
            content_date: Date of content (for temporal analysis)
            use_cache: Whether to use caching
            depth: Explanation depth override

        Returns:
            Detailed explanation of relevance score
        """
        start_time = time.time()
        self.metrics["total_explanations"] += 1

        explanation_depth = depth or self.explanation_depth
        cache_key = self._generate_cache_key(query, text, explanation_depth)

        # Try cache first
        if use_cache:
            cached_explanation = await self.cache.get(cache_key)
            if cached_explanation is not None:
                self.metrics["cache_hits"] += 1
                return cached_explanation

        # Calculate hybrid relevance score with explanation
        hybrid_score, hybrid_components = await self.hybrid_scorer.calculate_hybrid_score(
            query, text, use_cache=True, explain=True
        )

        # Generate explanation based on depth
        if explanation_depth == "basic":
            explanation = self._generate_basic_explanation(
                hybrid_score, hybrid_components)
        elif explanation_depth == "detailed":
            explanation = self._generate_detailed_explanation(
                query, text, hybrid_score, hybrid_components)
        else:  # comprehensive
            explanation = self._generate_comprehensive_explanation(
                query, text, hybrid_score, hybrid_components)

        # Add temporal analysis if date provided
        if content_date:
            temporal_analysis = await self.temporal_analyzer.analyze_temporal_relevance(
                query, text, content_date, hybrid_score
            )
            explanation["temporal_analysis"] = temporal_analysis
            explanation["final_score"] = temporal_analysis["adjusted_relevance"]
        else:
            explanation["final_score"] = hybrid_score

        # Add metadata
        explanation["metadata"] = {
            "query": query,
            "text_preview": text[:200] + "..." if len(text) > 200 else text,
            "analysis_timestamp": time.time(),
            "explanation_depth": explanation_depth,
        }

        # Cache explanation
        if use_cache:
            dependencies = [
                f"query:{hashlib.md5(query.encode()).hexdigest()[:8]}"]
            await self.cache.set(
                cache_key,
                explanation,
                ttl_seconds=1800,  # 30 minutes
                level=CacheStrategy.LRU,
                dependencies=dependencies,
            )

        # Update metrics
        self.metrics["processing_time"] += time.time() - start_time

        return explanation

    async def batch_explain_relevance(
        self,
        query: str,
        texts: list[str],
        content_dates: list[str] | None = None,
        depth: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate explanations for multiple texts.

        Args:
            query: Search query
            texts: List of texts to analyze
            content_dates: Optional list of content dates
            depth: Explanation depth

        Returns:
            List of explanations
        """
        if content_dates is None:
            content_dates = [None] * len(texts)

        tasks = [
            self.explain_relevance(query, text, date, depth=depth)
            for text, date in zip(texts, content_dates, strict=False)
        ]

        explanations = await asyncio.gather(*tasks)
        return explanations

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics."""
        total_explanations = self.metrics["total_explanations"]
        if total_explanations == 0:
            return self.metrics.copy()

        cache_hit_rate = self.metrics["cache_hits"] / total_explanations
        avg_processing_time = self.metrics["processing_time"] / \
            total_explanations

        hybrid_metrics = self.hybrid_scorer.get_performance_metrics()
        temporal_metrics = self.temporal_analyzer.get_performance_metrics()

        return {
            **self.metrics,
            "cache_hit_rate": cache_hit_rate,
            "avg_processing_time": avg_processing_time,
            "hybrid_scorer_metrics": hybrid_metrics,
            "temporal_analyzer_metrics": temporal_metrics,
        }

    def clear_cache(self):
        """Clear explanation cache."""
        self.cache.clear()
        self.hybrid_scorer.clear_cache()
        self.temporal_analyzer.clear_cache()
        logger.info("Explanation cache cleared")

    def optimize_cache(self):
        """Optimize cache performance."""
        self.cache.optimize()
        self.hybrid_scorer.optimize_cache()
        self.temporal_analyzer.optimize_cache()
        logger.info("Explanation cache optimized")

    async def close(self):
        """Close and cleanup resources."""
        await self.cache.close()
        await self.hybrid_scorer.close()
        await self.temporal_analyzer.close()
        logger.info("Explainable relevance scorer closed")


# Factory function for easy initialization
def create_explainable_scorer(explanation_depth: str = "detailed", cache_size: int = 300) -> ExplainableRelevanceScorer:
    """
    Factory function to create explainable relevance scorer.

    Args:
        explanation_depth: Level of explanation detail
        cache_size: Cache size

    Returns:
        ExplainableRelevanceScorer instance
    """
    return ExplainableRelevanceScorer(explanation_depth=explanation_depth, cache_size=cache_size)

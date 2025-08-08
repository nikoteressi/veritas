"""
Cached hybrid relevance scorer combining BM25 and semantic similarity.

Provides advanced relevance scoring using both traditional keyword matching (BM25)
and modern semantic similarity with intelligent caching.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import traceback
from typing import Any

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import settings
from app.exceptions import ValidationError

from app.cache import CacheType, get_cache
from agent.services.infrastructure.enhanced_ollama_embeddings import EnhancedOllamaEmbeddings

logger = logging.getLogger(__name__)


class CachedHybridRelevanceScorer:
    """Hybrid relevance scorer with BM25 and semantic similarity caching."""

    def __init__(
        self,
        embeddings_model: str | None = None,
        embeddings_url: str | None = None,
        cache_size: int = 200,  # Reduced for Docker environment
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
        min_score_threshold: float = 0.1,
        shared_embeddings: EnhancedOllamaEmbeddings | None = None,
    ):
        """
        Initialize hybrid relevance scorer.

        Args:
            embeddings_model: Ollama model for embeddings (if None, uses settings)
            embeddings_url: Ollama server URL (if None, uses settings)
            cache_size: Cache size for relevance scores
            bm25_weight: Weight for BM25 score (0-1)
            semantic_weight: Weight for semantic score (0-1)
            min_score_threshold: Minimum relevance score threshold
            shared_embeddings: Optional shared embeddings instance to avoid duplication
        """
        # Use settings if parameters not provided
        if embeddings_model is None:
            embeddings_model = settings.embedding_model_name
        if embeddings_url is None:
            embeddings_url = settings.ollama_base_url

        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.min_score_threshold = min_score_threshold

        # Normalize weights
        total_weight = bm25_weight + semantic_weight
        if total_weight > 0:
            self.bm25_weight = bm25_weight / total_weight
            self.semantic_weight = semantic_weight / total_weight

        # Initialize components - use shared embeddings if provided
        if shared_embeddings is not None:
            self.embeddings = shared_embeddings
            logger.info(
                "Using shared embeddings instance to reduce memory usage")
        else:
            self.embeddings = EnhancedOllamaEmbeddings(
                model=embeddings_model, base_url=embeddings_url, cache_size=cache_size
            )

        # Initialize cache using new unified cache system
        self.cache = None
        self._cache_initialized = False

        # BM25 and TF-IDF components
        self.bm25_corpus = None
        self.bm25_scorer = None
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),  # Reduced for Docker
        )

        # Performance metrics
        self.metrics = {
            "total_scores": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "bm25_scores": 0,
            "semantic_scores": 0,
            "hybrid_scores": 0,
            "processing_time": 0.0,
        }

    async def _ensure_cache_initialized(self):
        """Ensure cache is initialized using the new unified cache system."""
        if not self._cache_initialized:
            try:
                self.cache = await get_cache(CacheType.RELEVANCE)

                self._cache_initialized = True
                logger.info("Relevance cache initialized successfully")
            except Exception as e:
                logger.error("Failed to initialize relevance cache: %s", e)
                raise

        logger.info("Cached hybrid relevance scorer initialized")

    def _generate_cache_key(self, query: str, text: str, score_type: str = "hybrid") -> str:
        """Generate cache key for relevance score."""
        combined = f"{query}|{text}|{score_type}"
        text_hash = hashlib.md5(combined.encode("utf-8")).hexdigest()
        return f"relevance:{score_type}:{text_hash}"

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for scoring."""
        return text.strip().lower()

    def prepare_corpus(self, documents: list[str]):
        """Prepare BM25 corpus for scoring."""
        try:
            if not documents:
                raise ValueError(
                    "Cannot prepare corpus: no documents provided")

            # Preprocess documents
            processed_docs = [self._preprocess_text(doc) for doc in documents]

            # Filter out empty documents after preprocessing
            non_empty_docs = [doc for doc in processed_docs if doc.strip()]

            if not non_empty_docs:
                raise ValueError(
                    "Cannot prepare corpus: all documents are empty after preprocessing")

            # Tokenize for BM25
            tokenized_docs = [doc.split() for doc in non_empty_docs]

            # Filter out documents with no tokens
            valid_tokenized_docs = [
                tokens for tokens in tokenized_docs if tokens]

            if not valid_tokenized_docs:
                raise ValueError(
                    "Cannot prepare corpus: no valid tokens found in documents")

            # Initialize BM25
            self.bm25_scorer = BM25Okapi(valid_tokenized_docs)
            self.bm25_corpus = [non_empty_docs[i]
                                for i, tokens in enumerate(tokenized_docs) if tokens]

            # Fit TF-IDF
            self.tfidf_vectorizer.fit(self.bm25_corpus)

            logger.info("Corpus prepared with %d valid documents",
                        len(self.bm25_corpus))

        except Exception as e:
            raise ValidationError(f"Failed to prepare corpus: {e}") from e

    def calculate_bm25_score(self, query: str, text: str) -> float:
        """Calculate BM25 relevance score."""
        try:
            query_tokens = self._preprocess_text(query).split()

            # Filter out empty query tokens
            if not query_tokens:
                logger.debug("Empty query tokens, returning 0.0")
                return 0.0

            # Require BM25 scorer to be properly initialized with corpus
            if self.bm25_scorer is None or not self.bm25_corpus:
                logger.error(
                    "BM25 scorer not initialized or corpus not prepared. Call prepare_corpus first.")
                return 0.0

            try:
                preprocessed_text = self._preprocess_text(text)
                text_tokens = preprocessed_text.split()

                # Filter out empty text tokens
                if not text_tokens:
                    logger.debug("Empty text tokens, returning 0.0")
                    return 0.0

                # Find the document index in the corpus
                doc_index = -1
                for i, corpus_doc in enumerate(self.bm25_corpus):
                    if corpus_doc == preprocessed_text:
                        doc_index = i
                        break

                if doc_index >= 0:
                    # Document found in corpus, get its score
                    scores = self.bm25_scorer.get_scores(query_tokens)
                    score = float(scores[doc_index]) if doc_index < len(
                        scores) else 0.0
                else:
                    # Document not in corpus, cannot score properly
                    logger.warning(
                        "Document not found in BM25 corpus. Corpus may need to be updated.")
                    return 0.0

            except Exception as e:
                logger.error("Error using BM25 scorer: %s", e)
                return 0.0

            # Normalize score (BM25 scores can be quite high)
            normalized_score = min(score / 10.0, 1.0)
            return max(normalized_score, 0.0)

        except (ValueError, RuntimeError) as e:
            logger.error("Failed to calculate BM25 score: %s", e)
            return 0.0

    async def calculate_semantic_score(self, query: str, text: str) -> float:
        """Calculate semantic similarity score using embeddings."""
        try:
            # Get embeddings for query and text
            query_embedding = await self.embeddings.embed_text(query)
            text_embedding = await self.embeddings.embed_text(text)
            # Calculate similarity
            similarity = self.embeddings.calculate_similarity(
                query_embedding, text_embedding)

            return max(similarity, 0.0)

        except (ValueError, RuntimeError, ImportError) as e:
            logger.error("Failed to calculate semantic score: %s", e)
            return 0.0

    async def calculate_hybrid_score(
        self, query: str, text: str, use_cache: bool = True, explain: bool = False
    ) -> float | tuple[float, dict[str, Any]]:
        """
        Calculate hybrid relevance score combining BM25 and semantic similarity.

        Args:
            query: Search query
            text: Text to score
            use_cache: Whether to use caching
            explain: Whether to return explanation

        Returns:
            Relevance score or (score, explanation) if explain=True
        """
        try:
            start_time = time.time()
            self.metrics["total_scores"] += 1

            # Ensure cache is initialized
            await self._ensure_cache_initialized()

            cache_key = self._generate_cache_key(query, text, "hybrid")
            logger.debug("Generated cache key: %s", cache_key)

            # Try cache first
            if use_cache:
                logger.debug("Checking cache...")
                cached_result = await self.cache.get(cache_key)
                if cached_result is not None:
                    logger.info("Cache hit!")
                    self.metrics["cache_hits"] += 1
                    if explain:
                        return cached_result["score"], cached_result.get("explanation", {})
                    return cached_result["score"]
                logger.info("Cache miss, proceeding with calculation...")

            # Calculate individual scores
            logger.debug("Calculating BM25 score for text: %s...", text[:50])
            bm25_score = self.calculate_bm25_score(query, text)
            logger.debug("BM25 score calculated: %s", bm25_score)

            logger.debug(
                "Calculating semantic score for text: %s...", text[:50])
            semantic_score = await self.calculate_semantic_score(query, text)
            logger.debug("Semantic score calculated: %s", semantic_score)

            logger.debug("BM25: %s, Semantic: %s", bm25_score, semantic_score)

            # Combine scores
            hybrid_score = self.bm25_weight * bm25_score + \
                self.semantic_weight * semantic_score

            # Apply minimum threshold
            final_score = max(hybrid_score, 0.0)
            if final_score < self.min_score_threshold:
                final_score = 0.0

            # Create explanation
            explanation = {
                "bm25_score": bm25_score,
                "semantic_score": semantic_score,
                "bm25_weight": self.bm25_weight,
                "semantic_weight": self.semantic_weight,
                "hybrid_score": hybrid_score,
                "final_score": final_score,
                "threshold_applied": final_score != hybrid_score,
            }

            # Cache result
            if use_cache:
                cache_data = {
                    "score": final_score,
                    "explanation": explanation,
                    "timestamp": time.time(),
                }

                await self.cache.set(
                    cache_key,
                    cache_data,
                    ttl=3600,  # 1 hour
                )

            # Update metrics
            self.metrics["hybrid_scores"] += 1
            self.metrics["processing_time"] += time.time() - start_time

            logger.debug("Hybrid score calculation completed: %s", final_score)

            if explain:
                return final_score, explanation
            return final_score

        except Exception as e:
            logger.error("Unexpected error in calculate_hybrid_score: %s", e)
            logger.error("Query: %s", query[:100])
            logger.error("Text: %s", text[:100])
            raise ValidationError(
                f"Failed to calculate hybrid score due to unexpected error: {e}") from e

    async def score_documents(
        self,
        query: str,
        documents: list[str],
        use_cache: bool = True,
        return_explanations: bool = False,
    ) -> list[float | tuple[float, dict[str, Any]]]:
        """
        Score multiple documents for relevance.

        Args:
            query: Search query
            documents: list of documents to score
            use_cache: Whether to use caching
            return_explanations: Whether to return explanations

        Returns:
            list of scores or (score, explanation) tuples
        """
        # Prepare corpus if not already done
        if self.bm25_scorer is None:
            logger.info("Preparing corpus for BM25 scoring")
            self.prepare_corpus(documents)
            logger.info("Corpus prepared successfully")

        # Score all documents
        logger.info("Creating tasks for %d documents", len(documents))

        # Create tasks one by one and log each
        tasks = []
        for i, doc in enumerate(documents):
            task = self.calculate_hybrid_score(
                query, doc, use_cache, return_explanations)
            tasks.append(task)

        logger.info("Created %d tasks, gathering results...", len(tasks))

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for exceptions or coroutines in results
            for i, result in enumerate(results):
                logger.debug("Checking result %d: type=%s", i, type(result))
                if isinstance(result, Exception):
                    logger.error("Exception in task %d: %s", i, result)
                    raise result
                else:
                    logger.info(
                        "Task %d result type: %s, value: %s", i, type(result), result)

            logger.info("Successfully scored %d documents", len(results))
            return results

        except Exception as e:
            logger.error("Unexpected error in score_documents: %s", e)
            logger.error("Error type: %s", type(e))

            logger.error("Traceback: %s", traceback.format_exc())
            raise ValidationError(
                f"Failed to score documents due to unexpected error: {e}") from e

    async def rank_documents(
        self,
        query: str,
        documents: list[str],
        max_results: int = 10,
        min_score: float | None = None,
    ) -> list[tuple[str, float]]:
        """
        Rank documents by relevance score.

        Args:
            query: Search query
            documents: list of documents to rank
            max_results: Maximum number of results
            min_score: Minimum score threshold

        Returns:
            list of (document, score) tuples sorted by relevance
        """
        scores = await self.score_documents(query, documents)

        # Create document-score pairs
        doc_scores = list(zip(documents, scores, strict=False))

        # Filter by minimum score
        if min_score is not None:
            doc_scores = [(doc, score)
                          for doc, score in doc_scores if score >= min_score]

        # Sort by score (descending)
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Limit results
        return doc_scores[:max_results]

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics."""
        total_scores = self.metrics["total_scores"]
        if total_scores == 0:
            return self.metrics.copy()

        cache_hit_rate = self.metrics["cache_hits"] / total_scores
        avg_processing_time = self.metrics["processing_time"] / total_scores

        embeddings_metrics = self.embeddings.get_performance_metrics()

        return {
            **self.metrics,
            "cache_hit_rate": cache_hit_rate,
            "avg_processing_time": avg_processing_time,
            "embeddings_metrics": embeddings_metrics,
        }

    async def update_weights(self, bm25_weight: float, semantic_weight: float):
        """Update scoring weights."""
        total_weight = bm25_weight + semantic_weight
        if total_weight > 0:
            self.bm25_weight = bm25_weight / total_weight
            self.semantic_weight = semantic_weight / total_weight

            # Clear cache as weights changed
            if self._cache_initialized:
                await self.cache.clear_by_pattern("*")
            logger.info(
                "Updated weights: BM25=%.2f, Semantic=%.2f",
                self.bm25_weight,
                self.semantic_weight,
            )

    async def clear_cache(self):
        """Clear component-specific cache (shared cache managed by factory)."""
        # Only clear embeddings cache - shared cache is managed by CacheFactory
        await self.embeddings.clear_cache()
        logger.info("Component-specific relevance cache cleared")

    async def optimize_cache(self):
        """Optimize component-specific cache (shared cache managed by factory)."""
        # Only optimize embeddings cache - shared cache optimization is automatic
        await self.embeddings.optimize_cache()
        logger.info("Component-specific relevance cache optimized")

    async def close(self):
        """Close and cleanup resources (cache is managed by factory)."""
        # Cache is managed by CacheFactory, no need to close here
        await self.embeddings.close()
        logger.info("Cached hybrid relevance scorer closed")


# Factory function for easy initialization
def create_cached_hybrid_scorer(
    embeddings_model: str = "nomic-embed-text",
    bm25_weight: float = 0.3,
    semantic_weight: float = 0.7,
) -> CachedHybridRelevanceScorer:
    """
    Factory function to create cached hybrid relevance scorer.

    Args:
        embeddings_model: Ollama model for embeddings
        bm25_weight: Weight for BM25 score
        semantic_weight: Weight for semantic score

    Returns:
        CachedHybridRelevanceScorer instance
    """
    return CachedHybridRelevanceScorer(
        embeddings_model=embeddings_model,
        bm25_weight=bm25_weight,
        semantic_weight=semantic_weight,
    )

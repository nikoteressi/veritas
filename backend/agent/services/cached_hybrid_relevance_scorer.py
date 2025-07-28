"""
Cached hybrid relevance scorer combining BM25 and semantic similarity.

Provides advanced relevance scoring using both traditional keyword matching (BM25)
and modern semantic similarity with intelligent caching.
"""

import asyncio
import hashlib
import logging
import time
from typing import Any, Optional

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import settings

from .enhanced_ollama_embeddings import EnhancedOllamaEmbeddings
from .intelligent_cache import CacheStrategy, IntelligentCache

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
        shared_embeddings: Optional["EnhancedOllamaEmbeddings"] = None,
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

        self.cache = IntelligentCache(
            max_memory_size=cache_size // 2
        )  # Further reduced for Docker

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
            "bm25_scores": 0,
            "semantic_scores": 0,
            "hybrid_scores": 0,
            "processing_time": 0.0,
        }

        logger.info("Cached hybrid relevance scorer initialized")

    def _generate_cache_key(
        self, query: str, text: str, score_type: str = "hybrid"
    ) -> str:
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
            logger.error("Failed to prepare corpus: %s", e)
            raise

    def calculate_bm25_score(self, query: str, text: str) -> float:
        """Calculate BM25 relevance score."""
        try:
            if self.bm25_scorer is None:
                logger.warning(
                    "BM25 scorer not initialized. Call prepare_corpus first."
                )
                return 0.0

            query_tokens = self._preprocess_text(query).split()
            text_tokens = self._preprocess_text(text).split()

            # Get BM25 score
            scores = self.bm25_scorer.get_scores(query_tokens)

            # Find the score for this specific text
            if self.bm25_corpus:
                try:
                    preprocessed_text = self._preprocess_text(text)
                    text_index = self.bm25_corpus.index(preprocessed_text)
                    score = float(scores[text_index]) if text_index < len(
                        scores) else 0.0
                except ValueError:
                    # Text not in corpus, calculate directly by adding to corpus temporarily
                    temp_corpus = self.bm25_corpus + \
                        [self._preprocess_text(text)]
                    temp_tokenized = [doc.split() for doc in temp_corpus]
                    temp_bm25 = BM25Okapi(temp_tokenized)
                    temp_scores = temp_bm25.get_scores(query_tokens)
                    score = float(
                        temp_scores[-1]) if len(temp_scores) > 0 else 0.0
            else:
                # No corpus available, create temporary one
                temp_corpus = [self._preprocess_text(text)]
                temp_tokenized = [doc.split() for doc in temp_corpus]
                temp_bm25 = BM25Okapi(temp_tokenized)
                temp_scores = temp_bm25.get_scores(query_tokens)
                score = float(temp_scores[0]) if len(temp_scores) > 0 else 0.0

            # Normalize score (BM25 scores can be quite high)
            normalized_score = min(score / 10.0, 1.0)
            return max(normalized_score, 0.0)

        except (ValueError, RuntimeError) as e:
            logger.error("Failed to calculate BM25 score: %s", e)
            return 0.0

    async def calculate_semantic_score(self, query: str, text: str) -> float:
        """Calculate semantic similarity score using embeddings."""
        try:
            logger.debug("=== ENTERING calculate_semantic_score ===")
            logger.debug(f"Query: {query[:50]}...")
            logger.debug(f"Text: {text[:50]}...")
            logger.debug(f"Embeddings object type: {type(self.embeddings)}")

            # Get embeddings for query and text
            logger.debug("About to call embed_text for query...")
            query_embedding = await self.embeddings.embed_text(query)
            logger.debug(f"Query embedding type: {type(query_embedding)}")
            logger.debug(
                f"Query embedding is coroutine: {asyncio.iscoroutine(query_embedding)}")
            if asyncio.iscoroutine(query_embedding):
                raise RuntimeError("Query embedding is still a coroutine!")

            logger.debug("About to call embed_text for text...")
            text_embedding = await self.embeddings.embed_text(text)
            logger.debug(f"Text embedding type: {type(text_embedding)}")
            logger.debug(
                f"Text embedding is coroutine: {asyncio.iscoroutine(text_embedding)}")
            if asyncio.iscoroutine(text_embedding):
                raise RuntimeError("Text embedding is still a coroutine!")

            logger.debug("About to call calculate_similarity...")
            # Calculate similarity
            similarity = self.embeddings.calculate_similarity(
                query_embedding, text_embedding
            )
            logger.info(f"Similarity calculated: {similarity}")
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
            logger.info("=== ENTERING calculate_hybrid_score ===")
            logger.info(f"Query: {query[:50]}...")
            logger.info(f"Text: {text[:50]}...")
            logger.info(f"Use cache: {use_cache}, Explain: {explain}")

            start_time = time.time()
            self.metrics["total_scores"] += 1

            cache_key = self._generate_cache_key(query, text, "hybrid")
            logger.info(f"Generated cache key: {cache_key}")

            # Try cache first
            if use_cache:
                logger.info("Checking cache...")
                cached_result = await self.cache.get(cache_key)
                if cached_result is not None:
                    logger.info("Cache hit!")
                    self.metrics["cache_hits"] += 1
                    if explain:
                        return cached_result["score"], cached_result.get("explanation", {})
                    return cached_result["score"]
                logger.info("Cache miss, proceeding with calculation...")

            # Calculate individual scores
            logger.info(f"Calculating BM25 score for text: {text[:50]}...")
            bm25_score = self.calculate_bm25_score(query, text)
            logger.info(f"BM25 score calculated: {bm25_score}")

            logger.info(f"Calculating semantic score for text: {text[:50]}...")
            semantic_score = await self.calculate_semantic_score(query, text)
            logger.info(f"Semantic score calculated: {semantic_score}")

            logger.info(f"BM25: {bm25_score}, Semantic: {semantic_score}")

            # Combine scores
            hybrid_score = (
                self.bm25_weight * bm25_score + self.semantic_weight * semantic_score
            )

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
                    ttl_seconds=3600,  # 1 hour
                    level=CacheStrategy.LRU,
                    dependencies=[
                        f"query:{hashlib.md5(query.encode()).hexdigest()[:8]}"],
                )

            # Update metrics
            self.metrics["hybrid_scores"] += 1
            self.metrics["processing_time"] += time.time() - start_time

            logger.info(f"Hybrid score calculation completed: {final_score}")

            if explain:
                return final_score, explanation
            return final_score

        except Exception as e:
            logger.error(f"Error in calculate_hybrid_score: {e}")
            logger.error(f"Query: {query[:100]}")
            logger.error(f"Text: {text[:100]}")
            raise

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
            documents: List of documents to score
            use_cache: Whether to use caching
            return_explanations: Whether to return explanations

        Returns:
            List of scores or (score, explanation) tuples
        """
        logger.debug("=== ENTERING score_documents ===")
        logger.info(
            f"Scoring {len(documents)} documents with query: {query[:50]}...")
        logger.info(
            f"Use cache: {use_cache}, Return explanations: {return_explanations}")

        # Prepare corpus if not already done
        if self.bm25_scorer is None:
            logger.info("Preparing corpus for BM25 scoring")
            self.prepare_corpus(documents)
            logger.info("Corpus prepared successfully")

        # Score all documents
        logger.info(f"Creating tasks for {len(documents)} documents")

        # Create tasks one by one and log each
        tasks = []
        for i, doc in enumerate(documents):
            logger.debug(f"Creating task {i} for document: {doc[:50]}...")
            task = self.calculate_hybrid_score(
                query, doc, use_cache, return_explanations)
            logger.debug(f"Task {i} created, type: {type(task)}")
            logger.debug(f"Task {i} is coroutine: {asyncio.iscoroutine(task)}")
            tasks.append(task)

        logger.info(f"Created {len(tasks)} tasks, gathering results...")

        try:
            logger.debug("About to call asyncio.gather...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(
                f"asyncio.gather completed, got {len(results)} results")

            # Check for exceptions or coroutines in results
            for i, result in enumerate(results):
                logger.debug(f"Checking result {i}: type={type(result)}")
                if isinstance(result, Exception):
                    logger.error(f"Exception in task {i}: {result}")
                    raise result
                elif asyncio.iscoroutine(result):
                    logger.error(
                        f"Task {i} returned a coroutine instead of result: {type(result)}")
                    raise RuntimeError(
                        f"Task {i} returned unawaited coroutine")
                else:
                    logger.info(
                        f"Task {i} result type: {type(result)}, value: {result}")

            logger.info(f"Successfully scored {len(results)} documents")
            return results

        except Exception as e:
            logger.error(f"Error in score_documents: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

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
            documents: List of documents to rank
            max_results: Maximum number of results
            min_score: Minimum score threshold

        Returns:
            List of (document, score) tuples sorted by relevance
        """
        scores = await self.score_documents(query, documents)

        # Create document-score pairs
        doc_scores = list(zip(documents, scores, strict=False))

        # Filter by minimum score
        if min_score is not None:
            doc_scores = [
                (doc, score) for doc, score in doc_scores if score >= min_score
            ]

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

    def update_weights(self, bm25_weight: float, semantic_weight: float):
        """Update scoring weights."""
        total_weight = bm25_weight + semantic_weight
        if total_weight > 0:
            self.bm25_weight = bm25_weight / total_weight
            self.semantic_weight = semantic_weight / total_weight

            # Clear cache as weights changed
            self.cache.clear()
            logger.info(
                "Updated weights: BM25=%.2f, Semantic=%.2f",
                self.bm25_weight,
                self.semantic_weight,
            )

    def clear_cache(self):
        """Clear relevance cache."""
        self.cache.clear()
        self.embeddings.clear_cache()
        logger.info("Relevance cache cleared")

    def optimize_cache(self):
        """Optimize cache performance."""
        self.cache.optimize()
        self.embeddings.optimize_cache()
        logger.info("Relevance cache optimized")

    async def close(self):
        """Close and cleanup resources."""
        await self.cache.close()
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

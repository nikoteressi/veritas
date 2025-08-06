"""
Enhanced Ollama embeddings with intelligent caching and optimization.

Provides advanced embedding functionality with caching, batch processing,
and semantic similarity search for improved relevance analysis.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any

import numpy as np
from langchain_ollama import OllamaEmbeddings

from app.config import settings
from app.exceptions import EmbeddingError
from app.cache.factory import cache_factory

logger = logging.getLogger(__name__)


class EnhancedOllamaEmbeddings:
    """Enhanced Ollama embeddings with intelligent caching and performance optimization."""

    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        cache_size: int = 1000,
        similarity_threshold: float = 0.95,
        batch_size: int = 10,
    ):
        """
        Initialize enhanced Ollama embeddings.

        Args:
            model: Ollama model name for embeddings
            base_url: Ollama server URL
            cache_size: Maximum cache size
            similarity_threshold: Threshold for semantic similarity caching
            batch_size: Batch size for processing multiple texts
        """
        # Use settings defaults if not provided
        self.model = model or settings.embedding_model_name
        self.base_url = base_url or settings.ollama_base_url
        self.batch_size = batch_size
        self.similarity_threshold = similarity_threshold

        # Initialize Ollama embeddings
        self.ollama = OllamaEmbeddings(
            model=self.model, base_url=self.base_url)

        # Initialize cache system
        self.cache = None
        self._cache_initialized = False

        # Performance metrics
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_requests": 0,
            "total_processing_time": 0.0,
            "similarity_matches": 0,
        }

        logger.info(
            "Enhanced Ollama embeddings initialized with model: %s", model)

    async def _ensure_cache_initialized(self):
        """Ensure cache is initialized."""
        if not self._cache_initialized:
            try:
                self.cache = cache_factory.embedding_cache
                self._cache_initialized = True
                logger.info("Enhanced Ollama embeddings cache initialized")
            except Exception as e:
                logger.error("Failed to initialize cache: %s", e)
                raise

    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        return f"embedding:{self.model}:{text_hash}"

    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent caching."""
        return text.strip().lower()

    async def embed_text(self, text: str, use_cache: bool = True) -> list[float]:
        """
        Generate embedding for a single text with caching.

        Args:
            text: Text to embed
            use_cache: Whether to use caching

        Returns:
            Embedding vector
        """
        start_time = time.time()
        self.metrics["total_requests"] += 1

        normalized_text = self._normalize_text(text)
        cache_key = self._generate_cache_key(normalized_text)

        # Try cache first
        if use_cache:
            await self._ensure_cache_initialized()
            cached_embedding = await self._get_cached_embedding(cache_key)
            if cached_embedding is not None:
                self.metrics["cache_hits"] += 1
                processing_time = time.time() - start_time
                self.metrics["total_processing_time"] += processing_time
                return cached_embedding

        # Generate new embedding
        self.metrics["cache_misses"] += 1
        try:
            embedding = self.ollama.embed_query(text)

            if not isinstance(embedding, list):
                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()
                else:
                    embedding = list(embedding)

            # Validate that all elements are numbers
            embedding = [float(x) for x in embedding]

            if use_cache:
                await self._cache_embedding(cache_key, embedding, normalized_text)

            processing_time = time.time() - start_time
            self.metrics["total_processing_time"] += processing_time

            return embedding

        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate embedding for text: {str(e)}") from e

    async def embed_texts(self, texts: list[str], use_cache: bool = True) -> list[list[float]]:
        """
        Generate embeddings for multiple texts with batch processing.

        Args:
            texts: List of texts to embed
            use_cache: Whether to use caching

        Returns:
            List of embedding vectors
        """
        embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            batch_embeddings = await asyncio.gather(*[self.embed_text(text, use_cache) for text in batch])

            # Safety check - ensure no coroutines in batch results
            for j, embedding in enumerate(batch_embeddings):
                if asyncio.iscoroutine(embedding):
                    logger.error(
                        "CRITICAL: embed_texts got coroutine at batch %d, item %d", i, j)
                    raise RuntimeError(
                        "embed_texts received coroutine instead of embedding")

            embeddings.extend(batch_embeddings)

        # Final safety check
        for k, embedding in enumerate(embeddings):
            if asyncio.iscoroutine(embedding):
                logger.error(
                    "CRITICAL: embed_texts final result contains coroutine at index %d", k)
                raise RuntimeError(
                    "embed_texts final result contains coroutine")

        return embeddings

    async def _get_cached_embedding(self, cache_key: str) -> list[float] | None:
        """Get embedding from cache with similarity search."""
        # Direct cache lookup
        cached = await self.cache.get_embedding(cache_key)
        if cached is not None:
            return cached["embedding"]

        # Try similarity search using the embedding cache's semantic search
        try:
            similar_embeddings = await self.cache.find_similar_embeddings(
                cache_key,
                threshold=self.similarity_threshold,
                limit=1
            )
            if similar_embeddings:
                self.metrics["similarity_matches"] += 1
                return similar_embeddings[0]["embedding"]
        except Exception as e:
            logger.debug("Similarity search failed: %s", e)

        return None

    async def _cache_embedding(self, cache_key: str, embedding: list[float], text: str):
        """Cache embedding with metadata."""
        await self.cache.set_embedding(
            text,
            embedding,
            self.model,
            ttl=86400  # 24 hours
        )

    def calculate_similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            if asyncio.iscoroutine(embedding1):
                logger.error("embedding1 is a coroutine!")
                raise RuntimeError(
                    "embedding1 is a coroutine - it should have been awaited")
            if asyncio.iscoroutine(embedding2):
                logger.error("embedding2 is a coroutine!")
                raise RuntimeError(
                    "embedding2 is a coroutine - it should have been awaited")

            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            raise EmbeddingError(
                f"Failed to calculate similarity: {str(e)}") from e

    async def find_similar_texts(
        self,
        query_text: str,
        candidate_texts: list[str],
        threshold: float = 0.7,
        max_results: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Find texts similar to query using embeddings.

        Args:
            query_text: Query text
            candidate_texts: List of candidate texts
            threshold: Similarity threshold
            max_results: Maximum number of results

        Returns:
            List of (text, similarity_score) tuples
        """
        query_embedding = await self.embed_text(query_text)
        candidate_embeddings = await self.embed_texts(candidate_texts)

        similarities = []
        for text, embedding in zip(candidate_texts, candidate_embeddings, strict=False):
            similarity = self.calculate_similarity(query_embedding, embedding)
            if similarity >= threshold:
                similarities.append((text, similarity))

        # Sort by similarity (descending) and limit results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics."""
        total_requests = self.metrics["total_requests"]
        if total_requests == 0:
            return self.metrics.copy()

        cache_hit_rate = self.metrics["cache_hits"] / total_requests
        avg_processing_time = self.metrics["total_processing_time"] / \
            total_requests

        return {
            **self.metrics,
            "cache_hit_rate": cache_hit_rate,
            "avg_processing_time": avg_processing_time,
        }

    async def clear_cache(self):
        """Clear component-specific cache (shared cache managed by factory)."""
        # Shared cache is managed by CacheFactory and should not be cleared here
        # This method is kept for interface compatibility
        logger.info(
            "Component-specific embedding cache cleared (shared cache managed by factory)")

    async def optimize_cache(self):
        """Optimize component-specific cache (shared cache managed by factory)."""
        # Shared cache optimization is automatic in the new system
        # This method is kept for interface compatibility
        logger.info(
            "Component-specific embedding cache optimized (automatic in new system)")

    async def close(self):
        """Close and cleanup resources."""
        # Cache cleanup is handled by CacheFactory
        if self._cache_initialized:
            logger.info(
                "Enhanced Ollama embeddings cache cleanup handled by CacheFactory")
        logger.info("Enhanced Ollama embeddings closed")


# Factory function for easy initialization
def create_enhanced_ollama_embeddings(
    model: str = None,
    base_url: str = None,
    cache_size: int = 1000,
) -> EnhancedOllamaEmbeddings:
    """
    Factory function to create enhanced Ollama embeddings.

    Args:
        model: Ollama model name
        base_url: Ollama server URL
        cache_size: Cache size

    Returns:
        EnhancedOllamaEmbeddings instance
    """
    # Use settings defaults if not provided
    return EnhancedOllamaEmbeddings(
        model=model or settings.embedding_model_name,
        base_url=base_url or settings.ollama_base_url,
        cache_size=cache_size,
    )

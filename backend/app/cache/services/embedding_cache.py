"""
Embedding cache service.

Specialized cache for embeddings with semantic similarity search
and optimized storage for vector data.
"""
import hashlib
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from app.cache.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Specialized cache for embeddings with similarity search.

    Features:
    - Optimized embedding storage
    - Semantic similarity search
    - Batch operations for embeddings
    - Vector similarity calculations
    - Efficient key generation
    """

    def __init__(self, cache_manager: CacheManager):
        """
        Initialize embedding cache.

        Args:
            cache_manager: Unified cache manager instance
        """
        self.cache_manager = cache_manager
        self.cache_type = 'embedding'

        # Similarity threshold for finding similar embeddings
        self.similarity_threshold = 0.85

        # Index for similarity search (in-memory for performance)
        self._similarity_index: Dict[str, Dict] = {}

    async def get_embedding(self, text: str, model: str = 'default') -> Optional[List[float]]:
        """
        Get embedding for text.

        Args:
            text: Input text
            model: Model name used for embedding

        Returns:
            Embedding vector or None if not cached
        """
        key = self._generate_embedding_key(text, model)
        return await self.cache_manager.get(key, self.cache_type)

    async def set_embedding(
        self,
        text: str,
        embedding: List[float],
        model: str = 'default',
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store embedding for text.

        Args:
            text: Input text
            embedding: Embedding vector
            model: Model name used for embedding
            ttl: Time to live in seconds

        Returns:
            True if successful, False otherwise
        """
        key = self._generate_embedding_key(text, model)

        # Store embedding
        success = await self.cache_manager.set(key, embedding, ttl, self.cache_type)

        if success:
            # Update similarity index
            self._update_similarity_index(key, text, embedding, model)

        return success

    async def get_embeddings_batch(
        self,
        texts: List[str],
        model: str = 'default'
    ) -> Dict[str, Optional[List[float]]]:
        """
        Get embeddings for multiple texts.

        Args:
            texts: List of input texts
            model: Model name used for embedding

        Returns:
            Dictionary mapping text to embedding (or None if not cached)
        """
        keys = [self._generate_embedding_key(text, model) for text in texts]
        results = await self.cache_manager.get_many(keys, self.cache_type)

        # Map back to original texts
        text_results = {}
        for text, key in zip(texts, keys):
            text_results[text] = results.get(key)

        return text_results

    async def set_embeddings_batch(
        self,
        text_embedding_pairs: List[Tuple[str, List[float]]],
        model: str = 'default',
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store embeddings for multiple texts.

        Args:
            text_embedding_pairs: List of (text, embedding) pairs
            model: Model name used for embedding
            ttl: Time to live in seconds

        Returns:
            True if all successful, False otherwise
        """
        mapping = {}

        for text, embedding in text_embedding_pairs:
            key = self._generate_embedding_key(text, model)
            mapping[key] = embedding

            # Update similarity index
            self._update_similarity_index(key, text, embedding, model)

        return await self.cache_manager.set_many(mapping, ttl, self.cache_type)

    async def find_similar_embeddings(
        self,
        query_embedding: List[float],
        model: str = 'default',
        limit: int = 10,
        threshold: Optional[float] = None
    ) -> List[Tuple[str, List[float], float]]:
        """
        Find similar embeddings using cosine similarity.

        Args:
            query_embedding: Query embedding vector
            model: Model name to search within
            limit: Maximum number of results
            threshold: Similarity threshold (uses default if None)

        Returns:
            List of (text, embedding, similarity_score) tuples
        """
        effective_threshold = threshold or self.similarity_threshold
        query_vector = np.array(query_embedding)

        similar_embeddings = []

        # Search in similarity index
        for key, entry in self._similarity_index.items():
            if entry['model'] != model:
                continue

            # Calculate cosine similarity
            cached_vector = np.array(entry['embedding'])
            similarity = self._cosine_similarity(query_vector, cached_vector)

            if similarity >= effective_threshold:
                similar_embeddings.append((
                    entry['text'],
                    entry['embedding'],
                    similarity
                ))

        # Sort by similarity (highest first) and limit results
        similar_embeddings.sort(key=lambda x: x[2], reverse=True)
        return similar_embeddings[:limit]

    async def find_similar_text(
        self,
        query_text: str,
        model: str = 'default',
        limit: int = 10,
        threshold: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Find texts with similar embeddings.

        Args:
            query_text: Query text
            model: Model name to search within
            limit: Maximum number of results
            threshold: Similarity threshold

        Returns:
            List of (text, similarity_score) tuples
        """
        # Get embedding for query text
        query_embedding = await self.get_embedding(query_text, model)
        if query_embedding is None:
            return []

        # Find similar embeddings
        similar_embeddings = await self.find_similar_embeddings(
            query_embedding, model, limit, threshold
        )

        # Return only text and similarity scores
        return [(text, score) for text, _, score in similar_embeddings]

    async def delete_embedding(self, text: str, model: str = 'default') -> bool:
        """
        Delete embedding for text.

        Args:
            text: Input text
            model: Model name used for embedding

        Returns:
            True if successful, False otherwise
        """
        key = self._generate_embedding_key(text, model)

        # Remove from similarity index
        self._similarity_index.pop(key, None)

        return await self.cache_manager.delete(key, self.cache_type)

    async def clear_model_embeddings(self, model: str) -> int:
        """
        Clear all embeddings for a specific model.

        Args:
            model: Model name

        Returns:
            Number of deleted embeddings
        """
        pattern = f"*:{model}:*"

        # Remove from similarity index
        keys_to_remove = [
            key for key, entry in self._similarity_index.items()
            if entry['model'] == model
        ]

        for key in keys_to_remove:
            self._similarity_index.pop(key, None)

        return await self.cache_manager.clear_by_pattern(pattern, self.cache_type)

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        model_counts = {}
        total_embeddings = len(self._similarity_index)

        for entry in self._similarity_index.values():
            model = entry['model']
            model_counts[model] = model_counts.get(model, 0) + 1

        return {
            'total_embeddings': total_embeddings,
            'models': model_counts,
            'similarity_threshold': self.similarity_threshold,
            'cache_stats': self.cache_manager.get_stats()
        }

    def _generate_embedding_key(self, text: str, model: str) -> str:
        """
        Generate cache key for embedding.

        Args:
            text: Input text
            model: Model name

        Returns:
            Cache key
        """
        # Create hash of text for consistent key generation
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        return f"{text_hash}:{model}:{len(text)}"

    def _update_similarity_index(
        self,
        key: str,
        text: str,
        embedding: List[float],
        model: str
    ) -> None:
        """Update in-memory similarity index."""
        self._similarity_index[key] = {
            'text': text,
            'embedding': embedding,
            'model': model
        }

        # Limit index size to prevent memory issues
        max_index_size = 10000
        if len(self._similarity_index) > max_index_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self._similarity_index.keys())[:1000]
            for old_key in keys_to_remove:
                self._similarity_index.pop(old_key, None)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)

            # Ensure result is in [0, 1] range
            return max(0.0, min(1.0, similarity))

        except Exception as e:
            logger.error("Error calculating cosine similarity: %s", e)
            return 0.0

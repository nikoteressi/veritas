"""
ChromaDB-based embedding cache with Redis caching for results.

This module provides an intelligent embedding cache that uses ChromaDB for
vector storage and similarity search, while leveraging Redis for caching
metadata and search results to improve performance.
"""
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from app.clients.chroma_client import OllamaChromaClient
from app.exceptions import CacheError, EmbeddingError

from app.cache.core import CacheManager
from .base_cache_service import BatchCacheServiceInterface

logger = logging.getLogger(__name__)


class ChromaEmbeddingCache(BatchCacheServiceInterface):
    """
    ChromaDB-based embedding cache with Redis caching for results.

    Features:
    - ChromaDB for vector storage and similarity search
    - Redis caching for metadata and search results
    - Model-based collection management
    - Intelligent caching strategies
    - Batch operations support
    """

    def __init__(
        self,
        cache_manager: CacheManager,
        chroma_client: OllamaChromaClient,
        similarity_threshold: float = 0.8,
        cache_ttl: int = 3600
    ):
        """
        Initialize ChromaDB embedding cache.

        Args:
            cache_manager: Redis cache manager for metadata caching
            chroma_client: ChromaDB client for vector operations
            similarity_threshold: Default similarity threshold for searches
            cache_ttl: TTL for cached search results in seconds
        """
        self.cache_manager = cache_manager
        self.chroma_client = chroma_client
        self.similarity_threshold = similarity_threshold
        self.cache_ttl = cache_ttl
        self._cache_type = "chroma_embedding"

        # Collection management
        self._collections: Dict[str, Any] = {}

        # Performance metrics
        self._metrics = {
            'chroma_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'embeddings_stored': 0,
            'embeddings_retrieved': 0
        }

    @property
    def cache_type(self) -> str:
        """Get cache type identifier."""
        return self._cache_type

    async def _get_collection(self, model: str):
        """Get or create ChromaDB collection for model."""
        if model not in self._collections:
            collection_name = f"embeddings_{model.replace('/', '_').replace(':', '_')}"
            self._collections[model] = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"model": model, "created_at": str(time.time())}
            )
        return self._collections[model]

    async def get_embedding(
        self,
        text: str,
        model: str = 'default'
    ) -> Optional[List[float]]:
        """
        Get embedding for text.

        Args:
            text: Input text
            model: Model name used for embedding

        Returns:
            Embedding vector or None if not found
        """
        try:
            # Generate cache key for metadata
            cache_key = self._generate_cache_key(text, model)

            # Try to get from Redis cache first
            cached_data = await self.cache_manager.get(cache_key, self._cache_type)
            if cached_data:
                self._metrics['cache_hits'] += 1
                return cached_data.get('embedding')

            # Try to get from ChromaDB
            collection = await self._get_collection(model)
            doc_id = self._generate_doc_id(text, model)

            try:
                result = collection.get(ids=[doc_id], include=['embeddings'])
                if result['embeddings'] and len(result['embeddings']) > 0:
                    embedding = result['embeddings'][0]

                    # Cache in Redis for faster future access
                    await self.cache_manager.set(
                        cache_key,
                        {
                            'embedding': embedding,
                            'text': text,
                            'model': model,
                            'timestamp': time.time()
                        },
                        ttl=self.cache_ttl,
                        cache_type=self._cache_type
                    )

                    self._metrics['embeddings_retrieved'] += 1
                    return embedding

            except EmbeddingError as e:
                logger.warning(
                    "Embedding error retrieving from ChromaDB: %s", e)
            except Exception as e:
                logger.warning("Error retrieving from ChromaDB: %s", e)

            self._metrics['cache_misses'] += 1
            return None

        except CacheError as e:
            logger.error("Cache error getting embedding: %s", e)
            return None
        except EmbeddingError as e:
            logger.error("Embedding error getting embedding: %s", e)
            return None
        except Exception as e:
            logger.error("Error getting embedding: %s", e)
            return None

    async def set_embedding(
        self,
        text: str,
        embedding: List[float],
        model: str = 'default'
    ) -> bool:
        """
        Store embedding in ChromaDB and cache metadata in Redis.

        Args:
            text: Input text
            embedding: Embedding vector
            model: Model name used for embedding

        Returns:
            True if successful, False otherwise
        """
        try:
            # Store in ChromaDB
            collection = await self._get_collection(model)
            doc_id = self._generate_doc_id(text, model)

            collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{
                    'model': model,
                    'text_length': len(text),
                    'timestamp': time.time()
                }]
            )

            # Cache metadata in Redis
            cache_key = self._generate_cache_key(text, model)
            await self.cache_manager.set(
                cache_key,
                {
                    'embedding': embedding,
                    'text': text,
                    'model': model,
                    'timestamp': time.time()
                },
                ttl=self.cache_ttl,
                cache_type=self._cache_type
            )

            self._metrics['embeddings_stored'] += 1
            return True

        except CacheError as e:
            logger.error("Cache error setting embedding: %s", e)
            return False
        except EmbeddingError as e:
            logger.error("Embedding error setting embedding: %s", e)
            return False
        except Exception as e:
            logger.error("Error setting embedding: %s", e)
            return False

    async def find_similar_embeddings(
        self,
        query_embedding: List[float],
        model: str = 'default',
        limit: int = 10,
        threshold: Optional[float] = None
    ) -> List[Tuple[str, List[float], float]]:
        """
        Find similar embeddings using ChromaDB with Redis caching.

        Args:
            query_embedding: Query embedding vector
            model: Model name to search within
            limit: Maximum number of results
            threshold: Similarity threshold (uses default if None)

        Returns:
            List of (text, embedding, similarity_score) tuples
        """
        effective_threshold = threshold or self.similarity_threshold

        try:
            # Generate cache key for search results
            search_key = self._generate_search_key(
                query_embedding, model, limit, effective_threshold)

            # Try to get cached results first
            cached_results = await self.cache_manager.get(search_key, self.cache_type)
            if cached_results:
                self._metrics['cache_hits'] += 1
                return cached_results

            # Query ChromaDB
            collection = await self._get_collection(model)

            # Perform similarity search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=['documents', 'embeddings', 'distances']
            )

            self._metrics['chroma_queries'] += 1

            # Process results and filter by threshold
            similar_embeddings = []
            if results['documents'] and len(results['documents']) > 0:
                documents = results['documents'][0]
                embeddings = results['embeddings'][0] if results['embeddings'] else [
                ]
                distances = results['distances'][0] if results['distances'] else [
                ]

                for i, (doc, emb, dist) in enumerate(zip(documents, embeddings, distances)):
                    # Convert distance to similarity (ChromaDB returns distances, not similarities)
                    similarity = 1.0 - dist if dist is not None else 0.0

                    if similarity >= effective_threshold:
                        similar_embeddings.append((doc, emb, similarity))

            # Cache results in Redis
            await self.cache_manager.set(
                search_key,
                similar_embeddings,
                ttl=self.cache_ttl // 2,  # Shorter TTL for search results
                cache_type=self._cache_type
            )

            self._metrics['cache_misses'] += 1
            return similar_embeddings

        except CacheError as e:
            logger.error("Cache error finding similar embeddings: %s", e)
            return []
        except EmbeddingError as e:
            logger.error("Embedding error finding similar embeddings: %s", e)
            return []
        except Exception as e:
            logger.error("Error finding similar embeddings: %s", e)
            return []

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
        try:
            # Delete from ChromaDB
            collection = await self._get_collection(model)
            doc_id = self._generate_doc_id(text, model)

            try:
                collection.delete(ids=[doc_id])
            except EmbeddingError as e:
                logger.warning("Embedding error deleting from ChromaDB: %s", e)
            except Exception as e:
                logger.warning("Error deleting from ChromaDB: %s", e)

            # Delete from Redis cache
            cache_key = self._generate_cache_key(text, model)
            await self.cache_manager.delete(cache_key, self._cache_type)

            return True

        except CacheError as e:
            logger.error("Cache error deleting embedding: %s", e)
            return False
        except EmbeddingError as e:
            logger.error("Embedding error deleting embedding: %s", e)
            return False
        except Exception as e:
            logger.error("Error deleting embedding: %s", e)
            return False

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
            Dictionary mapping text to embedding (or None if not found)
        """
        results = {}
        for text in texts:
            results[text] = await self.get_embedding(text, model)
        return results

    async def set_embeddings_batch(
        self,
        embeddings_data: Dict[str, List[float]],
        model: str = 'default'
    ) -> bool:
        """
        Store multiple embeddings.

        Args:
            embeddings_data: Dictionary mapping text to embedding
            model: Model name used for embedding

        Returns:
            True if all successful, False otherwise
        """
        try:
            collection = await self._get_collection(model)

            # Prepare batch data
            ids = []
            embeddings = []
            documents = []
            metadatas = []

            for text, embedding in embeddings_data.items():
                ids.append(self._generate_doc_id(text, model))
                embeddings.append(embedding)
                documents.append(text)
                metadatas.append({
                    'model': model,
                    'text_length': len(text),
                    'timestamp': time.time()
                })

            # Batch upsert to ChromaDB
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

            # Cache in Redis
            for text, embedding in embeddings_data.items():
                cache_key = self._generate_cache_key(text, model)
                await self.cache_manager.set(
                    cache_key,
                    {
                        'embedding': embedding,
                        'text': text,
                        'model': model,
                        'timestamp': time.time()
                    },
                    ttl=self.cache_ttl,
                    cache_type=self._cache_type
                )

            self._metrics['embeddings_stored'] += len(embeddings_data)
            return True

        except CacheError as e:
            logger.error("Cache error setting embeddings batch: %s", e)
            return False
        except EmbeddingError as e:
            logger.error("Embedding error setting embeddings batch: %s", e)
            return False
        except Exception as e:
            logger.error("Error setting embeddings batch: %s", e)
            return False

    async def clear_model_embeddings(self, model: str) -> int:
        """
        Clear all embeddings for a specific model.

        Args:
            model: Model name

        Returns:
            Number of deleted embeddings
        """
        try:
            # Clear ChromaDB collection
            collection = await self._get_collection(model)

            # Get all IDs in collection
            all_data = collection.get()
            if all_data['ids']:
                collection.delete(ids=all_data['ids'])
                deleted_count = len(all_data['ids'])
            else:
                deleted_count = 0

            # Clear Redis cache for this model
            pattern = f"*:{model}:*"
            await self.cache_manager.clear_by_pattern(pattern, self._cache_type)

            return deleted_count

        except CacheError as e:
            logger.error("Cache error clearing model embeddings: %s", e)
            return 0
        except EmbeddingError as e:
            logger.error("Embedding error clearing model embeddings: %s", e)
            return 0
        except Exception as e:
            logger.error("Error clearing model embeddings: %s", e)
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache service statistics (implements CacheServiceInterface)."""
        return self.get_embedding_stats()

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        try:
            # Get collection stats
            collection_stats = {}
            for model, collection in self._collections.items():
                try:
                    data = collection.get()
                    collection_stats[model] = len(
                        data['ids']) if data['ids'] else 0
                except Exception as e:
                    logger.warning(
                        "Error getting stats for model %s: %s", model, e)
                    collection_stats[model] = 0

            return {
                'cache_type': self._cache_type,
                'collections': collection_stats,
                'total_collections': len(self._collections),
                'similarity_threshold': self.similarity_threshold,
                'cache_ttl': self.cache_ttl,
                'metrics': self._metrics.copy(),
                'cache_manager_stats': self.cache_manager.get_stats()
            }

        except CacheError as e:
            logger.error("Cache error getting embedding stats: %s", e)
            return {
                'cache_type': self._cache_type,
                'error': str(e),
                'metrics': self._metrics.copy()
            }
        except EmbeddingError as e:
            logger.error("Embedding error getting embedding stats: %s", e)
            return {
                'cache_type': self._cache_type,
                'error': str(e),
                'metrics': self._metrics.copy()
            }
        except Exception as e:
            logger.error("Error getting embedding stats: %s", e)
            return {
                'cache_type': self._cache_type,
                'error': str(e),
                'metrics': self._metrics.copy()
            }

    def _generate_doc_id(self, text: str, model: str) -> str:
        """Generate unique document ID for ChromaDB."""
        # Create hash of text and model for consistent ID generation
        content = f"{text}:{model}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _generate_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for Redis."""
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        return f"emb:{model}:{text_hash}:{len(text)}"

    def _generate_search_key(
        self,
        query_embedding: List[float],
        model: str,
        limit: int,
        threshold: float
    ) -> str:
        """Generate cache key for search results."""
        # Create hash of query parameters
        query_str = json.dumps({
            'embedding_hash': hashlib.sha256(
                json.dumps(query_embedding).encode('utf-8')
            ).hexdigest()[:16],
            'model': model,
            'limit': limit,
            'threshold': threshold
        }, sort_keys=True)

        search_hash = hashlib.sha256(
            query_str.encode('utf-8')).hexdigest()[:16]
        return f"search:{model}:{search_hash}"

    async def cleanup(self) -> int:
        """Clean up expired cache entries (implements CacheServiceInterface)."""
        # For ChromaEmbeddingCache, we can clean up old collections or rely on TTL
        # This is a basic implementation - could be enhanced with more sophisticated cleanup
        cleaned_count = 0

        # Clean up empty collections or very old data
        for model in list(self._collections.keys()):
            try:
                collection = self._collections[model]
                # Get collection data to check if it's empty or very old
                data = collection.get()
                if not data['ids']:
                    # Remove empty collection reference
                    del self._collections[model]
                    cleaned_count += 1
            except Exception as e:
                logger.warning(
                    "Error during cleanup for model %s: %s", model, e)

        return cleaned_count

    async def get_batch(self, keys: list) -> Dict[str, Any]:
        """Get multiple embeddings (implements BatchCacheServiceInterface)."""
        # This maps to our existing get_embeddings_batch method
        # Keys should be in format: [(text, model), ...]
        results = {}
        for key in keys:
            if isinstance(key, tuple) and len(key) == 2:
                text, model = key
                embedding = await self.get_embedding(text, model)
                results[f"{text}:{model}"] = embedding
        return results

    async def set_batch(self, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple embeddings (implements BatchCacheServiceInterface)."""
        # Data should be in format: {text: (embedding, model), ...}
        try:
            for text, value in data.items():
                if isinstance(value, tuple) and len(value) == 2:
                    embedding, model = value
                    success = await self.set_embedding(text, embedding, model)
                    if not success:
                        return False
            return True
        except Exception as e:
            logger.error("Error in set_batch: %s", e)
            return False

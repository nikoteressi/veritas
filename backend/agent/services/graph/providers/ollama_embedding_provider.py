"""
Ollama Embedding Provider.

Implements EmbeddingInterface using Ollama embedding functionality.
"""

import asyncio
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor

from agent.services.graph.interfaces.embedding_interface import EmbeddingInterface
from agent.services.graph.interfaces.config_interface import ConfigInterface
from agent.services.graph.dependency_injection import injectable

logger = logging.getLogger(__name__)


@injectable
class OllamaEmbeddingProvider(EmbeddingInterface):
    """
    Ollama-based embedding provider.

    Provides document and query embeddings using Ollama's embedding models
    with support for both synchronous and asynchronous operations.
    """

    def __init__(self, config: ConfigInterface):
        """
        Initialize Ollama embedding provider.

        Args:
            config: Configuration interface for accessing settings
        """
        self.config = config
        self._embedding_function = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._logger = logging.getLogger(__name__)

        # Initialize embedding function
        self._initialize_embedding_function()

    def _initialize_embedding_function(self) -> None:
        """Initialize the Ollama embedding function."""
        try:
            # Import here to avoid circular dependencies
            from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

            # Get configuration
            embedding_config = self.config.get_embedding_config()
            base_url = embedding_config.get(
                'base_url', 'http://localhost:11434')
            model_name = embedding_config.get('model_name', 'nomic-embed-text')

            self._embedding_function = OllamaEmbeddingFunction(
                url=base_url,
                model_name=model_name
            )

            self._logger.info(
                "Initialized Ollama embedding function with model: %s", model_name)

        except Exception as e:
            self._logger.error(
                "Failed to initialize Ollama embedding function: %s", e)
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for documents.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If embedding generation fails
        """
        try:
            if not texts:
                return []

            self._logger.debug(
                "Generating embeddings for %d documents", len(texts))

            # Use the embedding function
            embeddings = self._embedding_function(texts)

            self._logger.debug("Generated %d embeddings", len(embeddings))
            return embeddings

        except Exception as e:
            self._logger.error("Failed to generate document embeddings: %s", e)
            raise RuntimeError(f"Document embedding failed: {e}")

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector

        Raises:
            RuntimeError: If embedding generation fails
        """
        try:
            self._logger.debug("Generating embedding for query")

            # Use the embedding function for single query
            embeddings = self._embedding_function([text])

            if not embeddings:
                raise RuntimeError("No embedding generated for query")

            return embeddings[0]

        except Exception as e:
            self._logger.error("Failed to generate query embedding: %s", e)
            raise RuntimeError(f"Query embedding failed: {e}")

    async def embed_documents_async(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for documents asynchronously.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If embedding generation fails
        """
        try:
            if not texts:
                return []

            self._logger.debug(
                "Generating embeddings for %d documents (async)", len(texts))

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                self._executor,
                self.embed_documents,
                texts
            )

            return embeddings

        except Exception as e:
            self._logger.error(
                "Failed to generate document embeddings (async): %s", e)
            raise RuntimeError(f"Async document embedding failed: {e}")

    async def embed_query_async(self, text: str) -> List[float]:
        """
        Generate embedding for a query asynchronously.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector

        Raises:
            RuntimeError: If embedding generation fails
        """
        try:
            self._logger.debug("Generating embedding for query (async)")

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self._executor,
                self.embed_query,
                text
            )

            return embedding

        except Exception as e:
            self._logger.error(
                "Failed to generate query embedding (async): %s", e)
            raise RuntimeError(f"Async query embedding failed: {e}")

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this provider.

        Returns:
            Embedding dimension
        """
        try:
            # Test with a simple text to get dimension
            test_embedding = self.embed_query("test")
            return len(test_embedding)

        except Exception as e:
            self._logger.warning(
                "Could not determine embedding dimension: %s", e)
            # Default dimension for nomic-embed-text
            return 768

    def get_model_name(self) -> str:
        """
        Get the name of the embedding model.

        Returns:
            Model name
        """
        embedding_config = self.config.get_embedding_config()
        return embedding_config.get('model_name', 'nomic-embed-text')

    def is_available(self) -> bool:
        """
        Check if the embedding service is available.

        Returns:
            True if service is available, False otherwise
        """
        try:
            # Test with a simple embedding
            self.embed_query("test")
            return True

        except Exception as e:
            self._logger.warning("Embedding service not available: %s", e)
            return False

    def dispose(self) -> None:
        """Dispose of resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        self._embedding_function = None
        self._logger.info("Ollama embedding provider disposed")

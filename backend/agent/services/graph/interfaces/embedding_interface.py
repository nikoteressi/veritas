"""
Embedding interface for dependency injection.

Defines the contract for embedding services used in graph construction and verification.
"""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingInterface(ABC):
    """
    Abstract interface for embedding services.

    This interface abstracts the embedding functionality to enable
    dependency injection and easier testing.
    """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            texts: List of text documents to embed

        Returns:
            List of embedding vectors (one per document)
        """

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector for the query
        """

    @abstractmethod
    async def embed_documents_async(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously generate embeddings for a list of documents.

        Args:
            texts: List of text documents to embed

        Returns:
            List of embedding vectors (one per document)
        """

    @abstractmethod
    async def embed_query_async(self, text: str) -> List[float]:
        """
        Asynchronously generate embedding for a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector for the query
        """

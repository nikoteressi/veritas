"""
Custom Ollama embedding function for ChromaDB integration.
This eliminates the need for external model downloads by using our existing Ollama server.
"""
from __future__ import annotations

import logging

import requests
from app.config import settings
from app.exceptions import EmbeddingError
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger(__name__)


class OllamaEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function that uses Ollama server for generating embeddings.
    This replaces ChromaDB's default embedding models to avoid external downloads.
    """

    def __init__(
        self,
        ollama_url: str = settings.ollama_base_url,
        model_name: str = "nomic-embed-text",
        timeout: int = 30,
    ):
        """
        Initialize Ollama embedding function.

        Args:
            ollama_url: URL of the Ollama server
            model_name: Name of the embedding model to use
            timeout: Request timeout in seconds
        """
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self._test_connection()

    def _test_connection(self):
        """Test connection to Ollama server and availability of the model."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()

            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]

            if not any(self.model_name in name for name in model_names):
                error_message = f"Model '{self.model_name}' not found on Ollama server at {self.ollama_url}. Available models: {model_names}"
                logger.error(error_message)
                raise ValueError(error_message)

            logger.info(
                "Ollama embedding function initialized with model: %s", self.model_name)

        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Could not connect to Ollama server at {self.ollama_url}") from e
        except Exception as e:
            raise EmbeddingError(
                f"Failed to test Ollama connection: {e}") from e

    def __call__(self, documents: Documents) -> Embeddings:
        """
        Generate embeddings for the given documents.

        Args:
            input: List of documents to embed

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If embedding generation fails for a document.
        """
        embeddings = []

        for document in documents:
            # Try embedding endpoint first (if available)
            embedding = self._get_embedding_via_embed_endpoint(document)

            if embedding is None:
                # Fallback to generate endpoint with special prompt
                logger.debug(
                    "Embedding with 'embed' endpoint failed, trying 'generate' endpoint.")
                embedding = self._get_embedding_via_generate_endpoint(document)

            if embedding is not None:
                embeddings.append(embedding)
            else:
                error_message = f"Failed to generate embedding for document: '{document[:100]}...'"
                logger.error(error_message)
                raise ValueError(error_message)

        logger.debug("Generated %d embeddings", len(embeddings))
        return embeddings

    def _get_embedding_via_embed_endpoint(self, text: str) -> list[float]:
        """Try to get embedding using Ollama's embed endpoint."""
        try:
            payload = {"model": self.model_name, "input": text}

            response = requests.post(
                f"{self.ollama_url}/api/embed", json=payload, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                if "embeddings" in data and data["embeddings"]:
                    return data["embeddings"][0]  # Return first embedding

        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {e}") from e

        return None

    def _get_embedding_via_generate_endpoint(self, text: str) -> list[float]:
        """Fallback: try to extract embeddings from generate endpoint."""
        try:
            # Some models might return embeddings in the response metadata
            payload = {
                "model": self.model_name,
                "prompt": f"Generate embedding for: {text}",
                "stream": False,
                "options": {"temperature": 0, "num_predict": 1},
            }

            response = requests.post(
                f"{self.ollama_url}/api/generate", json=payload, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                # Check if embeddings are in the response
                if "embedding" in data:
                    return data["embedding"]

        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {e}") from e

        return None


def create_ollama_embedding_function(
    ollama_url: str = settings.ollama_base_url, model_name: str = "nomic-embed-text"
) -> OllamaEmbeddingFunction:
    """
    Factory function to create Ollama embedding function.

    Args:
        ollama_url: URL of the Ollama server
        model_name: Name of the embedding model to use

    Returns:
        OllamaEmbeddingFunction instance
    """
    return OllamaEmbeddingFunction(ollama_url=ollama_url, model_name=model_name)
"""Client modules for external services."""

from .chroma_client import EmbeddingFunction, OllamaChromaClient
from .vector_store import VectorStore, vector_store

__all__ = [
    "OllamaChromaClient",
    "EmbeddingFunction",
    "VectorStore",
    "vector_store"
]

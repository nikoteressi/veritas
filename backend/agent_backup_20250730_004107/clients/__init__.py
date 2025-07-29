"""Client modules for external services."""

from .chroma_client import OllamaChromaClient, EmbeddingFunction
from .vector_store import VectorStore, vector_store

__all__ = [
    "OllamaChromaClient",
    "EmbeddingFunction",
    "VectorStore",
    "vector_store"
]

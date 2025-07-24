"""
from __future__ import annotations

Custom ChromaDB client wrapper that prevents external model downloads
and ensures all collections use Ollama embeddings.
"""

import logging
from typing import Optional

import chromadb
from chromadb.api.types import CollectionMetadata, EmbeddingFunction

from agent.ollama_embeddings import create_ollama_embedding_function
from app.config import settings

logger = logging.getLogger(__name__)


class OllamaChromaClient:
    """
    ChromaDB client wrapper that ensures all operations use Ollama embeddings
    and prevents fallback to default embedding functions.
    """

    def __init__(self, host: str = None, port: int = None):
        """
        Initialize ChromaDB client with Ollama embeddings only.

        Args:
            persist_directory: Directory to persist ChromaDB data
        """

        self.host = host or settings.chroma_host
        self.port = port or settings.chroma_port

        self.embedding_function: Optional[EmbeddingFunction] = None
        self.client: Optional[chromadb.HttpClient] = None
        self._initialized = False

    def _initialize(self):
        """Initialize the ChromaDB client and embedding function."""
        if self._initialized:
            return

        try:
            logger.info(
                "Initializing ChromaDB client for server at %s:%s", self.host, self.port
            )

            # Create Ollama embedding function first
            self.embedding_function = create_ollama_embedding_function()

            # Initialize ChromaDB HTTP client
            self.client = chromadb.HttpClient(host=self.host, port=self.port)

            self.client.heartbeat()

            self._initialized = True
            logger.info("ChromaDB client initialized successfully")

            # chroma_settings = Settings(
            #     anonymized_telemetry=False,
            #     allow_reset=True,
            #     # Use local implementations only
            #     chroma_api_impl="chromadb.api.segment.SegmentAPI",
            #     chroma_sysdb_impl="chromadb.db.impl.sqlite.SqliteDB",
            #     chroma_producer_impl="chromadb.db.impl.sqlite.SqliteDB",
            #     chroma_consumer_impl="chromadb.db.impl.sqlite.SqliteDB",
            #     chroma_segment_manager_impl="chromadb.segment.impl.manager.local.LocalSegmentManager"
            # )

            # Create persistent client
            # self.client = chromadb.PersistentClient(
            #     path=self.persist_directory,
            #     settings=chroma_settings
            # )

            # self._initialized = True
            # logger.info("ChromaDB client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            self._initialized = False
            raise

    def get_collection(self, name: str):
        """
        Get a collection with Ollama embedding function.

        Args:
            name: Collection name

        Returns:
            Collection with Ollama embedding function
        """
        if not self._initialized:
            self._initialize()

        return self.client.get_collection(
            name=name,
        )

    def create_collection(self, name: str, metadata: Optional[CollectionMetadata] = None):
        """
        Create a collection with Ollama embedding function.

        Args:
            name: Collection name
            metadata: Optional metadata

        Returns:
            Collection with Ollama embedding function
        """
        if not self._initialized:
            self._initialize()

        return self.client.create_collection(
            name=name,
            metadata=metadata,
        )

    def get_or_create_collection(
        self, name: str, metadata: Optional[CollectionMetadata] = None
    ):
        """
        Get or create a collection with Ollama embedding function.

        Args:
            name: Collection name
            metadata: Optional metadata

        Returns:
            Collection with Ollama embedding function
        """
        if not self._initialized:
            self._initialize()

        return self.client.get_or_create_collection(
            name=name,
            metadata=metadata,
        )

    def delete_collection(self, name: str):
        """
        Delete a collection.

        Args:
            name: Collection name
        """
        if not self._initialized:
            self._initialize()

        return self.client.delete_collection(name=name)

    def list_collections(self):
        """List all collections."""
        if not self._initialized:
            self._initialize()

        return self.client.list_collections()

    def reset(self):
        """Reset the ChromaDB client (for testing)."""
        if not self._initialized:
            self._initialize()

        return self.client.reset()

    def heartbeat(self):
        """Check if the client is alive."""
        if not self._initialized:
            self._initialize()

        return self.client.heartbeat()

    @property
    def is_initialized(self) -> bool:
        """Check if the client is initialized."""
        return self._initialized

"""
Custom ChromaDB client wrapper that prevents external model downloads
and ensures all collections use Ollama embeddings.
"""
import logging
from typing import Optional, Dict, Any
import chromadb
from chromadb.config import Settings
from chromadb.api.types import CollectionMetadata, EmbeddingFunction
from agent.ollama_embeddings import create_ollama_embedding_function

logger = logging.getLogger(__name__)


class OllamaChromaClient:
    """
    ChromaDB client wrapper that ensures all operations use Ollama embeddings
    and prevents fallback to default embedding functions.
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB client with Ollama embeddings only.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = persist_directory
        self.embedding_function = None
        self.client = None
        self._initialized = False
        
    def _initialize(self):
        """Initialize the ChromaDB client and embedding function."""
        if self._initialized:
            return
            
        try:
            logger.info("Initializing ChromaDB client with Ollama embeddings only...")
            
            # Create Ollama embedding function first
            self.embedding_function = create_ollama_embedding_function()
            
            # Configure ChromaDB settings to prevent external downloads
            chroma_settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                # Use local implementations only
                chroma_api_impl="chromadb.api.segment.SegmentAPI",
                chroma_sysdb_impl="chromadb.db.impl.sqlite.SqliteDB",
                chroma_producer_impl="chromadb.db.impl.sqlite.SqliteDB",
                chroma_consumer_impl="chromadb.db.impl.sqlite.SqliteDB",
                chroma_segment_manager_impl="chromadb.segment.impl.manager.local.LocalSegmentManager"
            )
            
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=chroma_settings
            )
            
            self._initialized = True
            logger.info("ChromaDB client initialized successfully with Ollama embeddings")
            
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
            embedding_function=self.embedding_function
        )
    
    def create_collection(
        self, 
        name: str, 
        metadata: Optional[CollectionMetadata] = None
    ):
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
            embedding_function=self.embedding_function
        )
    
    def get_or_create_collection(
        self, 
        name: str, 
        metadata: Optional[CollectionMetadata] = None
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
            embedding_function=self.embedding_function
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

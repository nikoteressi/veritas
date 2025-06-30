"""
Vector database integration for storing and retrieving verification results.
Uses Ollama server for embeddings to avoid external model downloads.
"""
import logging
import hashlib
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from agent.chroma_client import OllamaChromaClient
from app.json_utils import json_dumps, prepare_for_json_serialization
from app.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector database for storing and retrieving verification results."""

    def __init__(self, persist_directory: str = None, lazy_init: bool = True):
        """Initialize ChromaDB client with optional lazy initialization."""
        if persist_directory is None:
            persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        
        self.persist_directory = persist_directory
        self.lazy_init = lazy_init
        self.client = None
        self.verification_collection = None
        self.claims_collection = None
        self.sources_collection = None
        self._initialized = False

        if not lazy_init:
            self._initialize()

    def _initialize(self):
        """Initialize ChromaDB client and collections with Ollama embeddings."""
        if self._initialized:
            return

        try:
            logger.info("Initializing vector store with Ollama embeddings (no external downloads)...")

            # Create custom ChromaDB client that only uses Ollama embeddings
            self.client = OllamaChromaClient(persist_directory=self.persist_directory)

            # Create collections for different types of data with Ollama embeddings
            self.verification_collection = self.client.get_or_create_collection(
                "verification_results",
                metadata={"description": "Stores verification results with embeddings for similarity search"}
            )

            self.claims_collection = self.client.get_or_create_collection(
                "claims",
                metadata={"description": "Stores individual claims for pattern recognition"}
            )

            self.sources_collection = self.client.get_or_create_collection(
                "sources",
                metadata={"description": "Stores source information and credibility data"}
            )

            self._initialized = True
            logger.info("Vector store initialized successfully with Ollama embeddings")

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self._initialized = False
            # Don't raise - allow system to continue without vector store
    

    
    def store_verification_result(self, verification_data: Dict[str, Any]) -> str:
        """Store a complete verification result."""
        if not self._safe_initialize():
            return ""

        try:
            # Generate unique ID for this verification
            # Prepare data for JSON serialization to avoid datetime issues
            serializable_data = prepare_for_json_serialization(verification_data)
            content_hash = hashlib.md5(
                json_dumps(serializable_data, sort_keys=True).encode()
            ).hexdigest()

            verification_id = f"verification_{content_hash}_{int(datetime.now().timestamp())}"

            # Prepare document for embedding
            document_text = self._prepare_verification_document(verification_data)

            # Store in verification collection
            metadata = {
                "verification_id": verification_id,
                "timestamp": datetime.now().isoformat(),
                "username": verification_data.get("nickname", "unknown"),
                "verdict": verification_data.get("verdict", "unknown"),
                "confidence": verification_data.get("confidence_score", 0),
                "temporal_mismatch": verification_data.get("temporal_analysis", {}).get("temporal_mismatch", False),
                "mismatch_severity": verification_data.get("temporal_analysis", {}).get("mismatch_severity", "none"),
                "intent_analysis": verification_data.get("temporal_analysis", {}).get("intent_analysis", "unknown")
            }

            self.verification_collection.add(
                documents=[document_text],
                metadatas=[metadata],
                ids=[verification_id]
            )

            # Store individual claims
            self._store_claims(verification_data.get("claims", []), verification_id)

            # Store source information
            self._store_sources(verification_data.get("fact_check_results", {}), verification_id)

            logger.info(f"Stored verification result: {verification_id}")
            return verification_id

        except Exception as e:
            logger.error(f"Failed to store verification result: {e}", exc_info=True)
            raise VectorStoreError(f"Failed to store verification result: {e}") from e

    def _safe_initialize(self) -> bool:
        """Safely initialize vector store, returning False if it fails."""
        if self._initialized:
            return True

        try:
            self._initialize()
            return self._initialized
        except Exception as e:
            logger.warning(f"Vector store initialization failed, continuing without it: {e}")
            return False

    def reset_for_testing(self):
        """Reset vector store for testing purposes."""
        try:
            if self.client and self.client.is_initialized:
                # Delete existing collections
                collections = ["verification_results", "claims", "sources"]
                for collection_name in collections:
                    try:
                        self.client.delete_collection(collection_name)
                        logger.info(f"Deleted collection: {collection_name}")
                    except Exception:
                        pass  # Collection might not exist

            # Reset initialization state
            self._initialized = False
            self.client = None
            self.verification_collection = None
            self.claims_collection = None
            self.sources_collection = None

            logger.info("Vector store reset for testing")

        except Exception as e:
            logger.warning(f"Failed to reset vector store: {e}")
    
    def _prepare_verification_document(self, verification_data: Dict[str, Any]) -> str:
        """Prepare a document string for embedding."""
        parts = []
        
        # Add basic information
        if verification_data.get("nickname"):
            parts.append(f"User: {verification_data['nickname']}")
        
        # Add claims
        claims = verification_data.get("claims", [])
        if claims:
            parts.append(f"Claims: {' | '.join(claims)}")
        
        # Add extracted text
        if verification_data.get("extracted_text"):
            parts.append(f"Content: {verification_data['extracted_text'][:500]}")  # Limit length
        
        # Add temporal context
        temporal = verification_data.get("temporal_analysis", {})
        if temporal.get("temporal_flags"):
            parts.append(f"Temporal flags: {' | '.join(temporal['temporal_flags'])}")
        
        # Add verdict
        if verification_data.get("verdict"):
            parts.append(f"Verdict: {verification_data['verdict']}")
        
        return " | ".join(parts)
    
    def _store_claims(self, claims: List[str], verification_id: str):
        """Store individual claims for pattern recognition."""
        try:
            for i, claim in enumerate(claims):
                claim_id = f"{verification_id}_claim_{i}"

                metadata = {
                    "verification_id": verification_id,
                    "claim_index": i,
                    "timestamp": datetime.now().isoformat()
                }

                self.claims_collection.add(
                    documents=[claim],
                    metadatas=[metadata],
                    ids=[claim_id]
                )
        except Exception as e:
            logger.error(f"Failed to store claims: {e}")
    
    def _store_sources(self, fact_check_results: Dict[str, Any], verification_id: str):
        """Store source information."""
        try:
            examined_sources = fact_check_results.get("examined_sources", [])

            for i, source in enumerate(examined_sources):
                source_id = f"{verification_id}_source_{i}"

                # Create document from source URL and any available metadata
                source_doc = f"Source: {source}"

                metadata = {
                    "verification_id": verification_id,
                    "source_url": source,
                    "timestamp": datetime.now().isoformat()
                }

                self.sources_collection.add(
                    documents=[source_doc],
                    metadatas=[metadata],
                    ids=[source_id]
                )
        except Exception as e:
            logger.error(f"Failed to store sources: {e}")
    
    def find_similar_verifications(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar verification results."""
        if not self._safe_initialize():
            return []

        try:
            results = self.verification_collection.query(
                query_texts=[query_text],
                n_results=limit
            )

            similar_verifications = []
            if results["documents"]:
                for i in range(len(results["documents"][0])):
                    similar_verifications.append({
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if results.get("distances") else None,
                        "id": results["ids"][0][i]
                    })

            logger.info(f"Found {len(similar_verifications)} similar verifications")
            return similar_verifications

        except Exception as e:
            logger.error(f"Failed to find similar verifications: {e}")
            return []

    def find_similar_claims(self, claim: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar claims for pattern recognition."""
        if not self._safe_initialize():
            return []

        try:
            results = self.claims_collection.query(
                query_texts=[claim],
                n_results=limit
            )

            similar_claims = []
            if results["documents"]:
                for i in range(len(results["documents"][0])):
                    similar_claims.append({
                        "claim": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if results.get("distances") else None,
                        "id": results["ids"][0][i]
                    })

            return similar_claims

        except Exception as e:
            logger.error(f"Failed to find similar claims: {e}")
            return []
    
    def get_user_history(self, username: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get verification history for a specific user."""
        try:
            results = self.verification_collection.get(
                where={"username": username},
                limit=limit
            )
            
            user_history = []
            if results["documents"]:
                for i in range(len(results["documents"])):
                    user_history.append({
                        "document": results["documents"][i],
                        "metadata": results["metadatas"][i],
                        "id": results["ids"][i]
                    })
            
            return user_history
            
        except Exception as e:
            logger.error(f"Failed to get user history: {e}")
            return []
    
    def get_temporal_mismatch_patterns(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get verifications with temporal mismatches for pattern analysis."""
        try:
            results = self.verification_collection.get(
                where={"temporal_mismatch": True},
                limit=limit
            )
            
            patterns = []
            if results["documents"]:
                for i in range(len(results["documents"])):
                    patterns.append({
                        "document": results["documents"][i],
                        "metadata": results["metadatas"][i],
                        "id": results["ids"][i]
                    })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to get temporal mismatch patterns: {e}")
            return []


# Global vector store instance with lazy initialization to prevent blocking
vector_store = VectorStore(lazy_init=True)

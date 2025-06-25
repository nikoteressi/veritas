"""
Custom Ollama embedding function for ChromaDB integration.
This eliminates the need for external model downloads by using our existing Ollama server.
"""
import logging
import requests
import json
from typing import List, Dict, Any
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

logger = logging.getLogger(__name__)


class OllamaEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function that uses Ollama server for generating embeddings.
    This replaces ChromaDB's default embedding models to avoid external downloads.
    """
    
    def __init__(
        self, 
        ollama_url: str = "http://192.168.11.130:11434",
        model_name: str = "nomic-embed-text",
        timeout: int = 30
    ):
        """
        Initialize Ollama embedding function.
        
        Args:
            ollama_url: URL of the Ollama server
            model_name: Name of the embedding model to use
            timeout: Request timeout in seconds
        """
        self.ollama_url = ollama_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama server."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model.get('name', '') for model in models]
                
                # Check if our embedding model is available
                available_embedding_models = [
                    name for name in model_names 
                    if any(embed_keyword in name.lower() for embed_keyword in ['embed', 'embedding'])
                ]
                
                if available_embedding_models:
                    # Use the first available embedding model if our preferred one isn't available
                    if not any(self.model_name in name for name in model_names):
                        self.model_name = available_embedding_models[0]
                        logger.info(f"Using available embedding model: {self.model_name}")
                    
                    logger.info(f"Ollama embedding function initialized with model: {self.model_name}")
                else:
                    # Fallback to a general model that might support embeddings
                    if model_names:
                        self.model_name = model_names[0]  # Use first available model
                        logger.warning(f"No dedicated embedding model found, using: {self.model_name}")
                    else:
                        logger.error("No models available on Ollama server")
                        
        except Exception as e:
            logger.warning(f"Could not connect to Ollama server at {self.ollama_url}: {e}")
            logger.info("Will attempt to use embedding function anyway")
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for the given documents.
        
        Args:
            input: List of documents to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = []
            
            for document in input:
                # Prepare the request for Ollama
                payload = {
                    "model": self.model_name,
                    "prompt": document,
                    "options": {
                        "temperature": 0,  # Deterministic embeddings
                        "num_predict": 1   # We only need embeddings, not text generation
                    }
                }
                
                # Try embedding endpoint first (if available)
                embedding = self._get_embedding_via_embed_endpoint(document)
                
                if embedding is None:
                    # Fallback to generate endpoint with special prompt
                    embedding = self._get_embedding_via_generate_endpoint(document)
                
                if embedding is not None:
                    embeddings.append(embedding)
                else:
                    # Final fallback: create a simple hash-based embedding
                    logger.warning(f"Could not generate embedding for document, using fallback")
                    embeddings.append(self._create_fallback_embedding(document))
            
            logger.debug(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return fallback embeddings to prevent system failure
            return [self._create_fallback_embedding(doc) for doc in input]
    
    def _get_embedding_via_embed_endpoint(self, text: str) -> List[float]:
        """Try to get embedding using Ollama's embed endpoint."""
        try:
            payload = {
                "model": self.model_name,
                "input": text
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/embed",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'embeddings' in data and data['embeddings']:
                    return data['embeddings'][0]  # Return first embedding
                    
        except Exception as e:
            logger.debug(f"Embed endpoint failed: {e}")
        
        return None
    
    def _get_embedding_via_generate_endpoint(self, text: str) -> List[float]:
        """Fallback: try to extract embeddings from generate endpoint."""
        try:
            # Some models might return embeddings in the response metadata
            payload = {
                "model": self.model_name,
                "prompt": f"Generate embedding for: {text}",
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 1
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                # Check if embeddings are in the response
                if 'embedding' in data:
                    return data['embedding']
                    
        except Exception as e:
            logger.debug(f"Generate endpoint failed: {e}")
        
        return None
    
    def _create_fallback_embedding(self, text: str, dimension: int = 384) -> List[float]:
        """
        Create a simple fallback embedding based on text hash.
        This ensures the system continues working even if Ollama is unavailable.
        """
        import hashlib
        import struct
        
        # Create a deterministic hash of the text
        text_hash = hashlib.sha256(text.encode()).digest()
        
        # Convert hash to float values
        embedding = []
        for i in range(0, min(len(text_hash), dimension * 4), 4):
            if i + 4 <= len(text_hash):
                # Convert 4 bytes to float
                float_val = struct.unpack('f', text_hash[i:i+4])[0]
                # Normalize to [-1, 1] range
                embedding.append(max(-1.0, min(1.0, float_val)))
        
        # Pad or truncate to desired dimension
        while len(embedding) < dimension:
            embedding.append(0.0)
        
        return embedding[:dimension]


def create_ollama_embedding_function(
    ollama_url: str = "http://192.168.11.130:11434",
    model_name: str = "nomic-embed-text"
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

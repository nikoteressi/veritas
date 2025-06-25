#!/usr/bin/env python3
"""
Script to pre-initialize the vector store and download required models.
This should be run during system startup or deployment to avoid blocking during runtime.
"""
import sys
import os
import logging
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from agent.vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_vector_store():
    """Initialize vector store and download models."""
    try:
        logger.info("Starting vector store initialization...")
        
        # Create vector store with immediate initialization
        vector_store = VectorStore(lazy_init=False)
        
        # Test basic functionality
        test_data = {
            "nickname": "test_user",
            "extracted_text": "Test initialization",
            "claims": ["Test claim"],
            "temporal_analysis": {},
            "verdict": "test",
            "confidence_score": 50,
            "justification": "Test initialization",
            "fact_check_results": {}
        }
        
        # Store test data
        verification_id = vector_store.store_verification_result(test_data)
        if verification_id:
            logger.info(f"Test verification stored: {verification_id}")
        
        # Test similarity search
        similar = vector_store.find_similar_verifications("test", limit=1)
        logger.info(f"Found {len(similar)} similar verifications")
        
        logger.info("Vector store initialization completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Vector store initialization failed: {e}")
        return False


def main():
    """Main function."""
    logger.info("ChromaDB Vector Store Initialization")
    logger.info("=" * 50)
    
    success = initialize_vector_store()
    
    if success:
        logger.info("✅ Vector store is ready for use")
        sys.exit(0)
    else:
        logger.error("❌ Vector store initialization failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

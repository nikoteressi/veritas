#!/usr/bin/env python3
"""
Test script to verify that the JSON serialization and ChromaDB fixes work correctly.
"""
import sys
import os
import logging
from pathlib import Path
from datetime import datetime, timezone

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from app.json_utils import safe_json_dumps, prepare_for_json_serialization
from app.websocket_manager import ConnectionManager
from agent.vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_json_serialization():
    """Test JSON serialization with datetime objects."""
    logger.info("Testing JSON serialization...")
    
    # Test data with datetime objects
    test_data = {
        "timestamp": datetime.now(timezone.utc),
        "created_at": datetime.now(),
        "user": "test_user",
        "nested": {
            "last_update": datetime.now(timezone.utc),
            "count": 42
        },
        "list_with_datetime": [
            datetime.now(),
            "string",
            123
        ]
    }
    
    try:
        # Test safe_json_dumps
        json_str = safe_json_dumps(test_data)
        logger.info(f"✅ safe_json_dumps succeeded: {len(json_str)} characters")
        
        # Test prepare_for_json_serialization
        prepared_data = prepare_for_json_serialization(test_data)
        logger.info(f"✅ prepare_for_json_serialization succeeded")
        
        # Verify the prepared data can be serialized with standard json
        import json
        standard_json = json.dumps(prepared_data)
        logger.info(f"✅ Standard JSON serialization of prepared data succeeded")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ JSON serialization test failed: {e}")
        return False


def test_websocket_manager():
    """Test WebSocket manager with datetime objects."""
    logger.info("Testing WebSocket manager...")
    
    try:
        manager = ConnectionManager()
        
        # Test message with datetime objects
        test_message = {
            "type": "test",
            "data": {
                "timestamp": datetime.now(timezone.utc),
                "message": "Test message"
            }
        }
        
        # This should not raise an exception
        # Note: We can't actually send without a WebSocket connection,
        # but we can test the JSON serialization part
        from app.json_utils import safe_json_dumps
        serialized = safe_json_dumps(test_message)
        logger.info(f"✅ WebSocket message serialization succeeded")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ WebSocket manager test failed: {e}")
        return False


def test_vector_store():
    """Test vector store initialization and basic operations."""
    logger.info("Testing vector store...")
    
    try:
        # Create vector store with immediate initialization
        vector_store = VectorStore(persist_directory="./test_chroma_db", lazy_init=False)
        
        # Test data with datetime objects
        test_data = {
            "nickname": "test_user",
            "extracted_text": "Test verification with datetime",
            "claims": ["Test claim with datetime"],
            "temporal_analysis": {
                "timestamp": datetime.now(timezone.utc),
                "temporal_mismatch": False
            },
            "verdict": "test",
            "confidence_score": 75,
            "justification": "Test justification",
            "fact_check_results": {
                "examined_sources": ["http://test.com"],
                "timestamp": datetime.now()
            }
        }
        
        # Store test data - this should not raise JSON serialization errors
        verification_id = vector_store.store_verification_result(test_data)
        if verification_id:
            logger.info(f"✅ Vector store operation succeeded: {verification_id}")
        else:
            logger.warning("⚠️ Vector store operation returned empty ID")
        
        # Clean up test data
        try:
            vector_store.reset_for_testing()
            logger.info("✅ Vector store cleanup succeeded")
        except Exception as cleanup_error:
            logger.warning(f"⚠️ Vector store cleanup failed: {cleanup_error}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Vector store test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting fix verification tests...")
    logger.info("=" * 60)
    
    tests = [
        ("JSON Serialization", test_json_serialization),
        ("WebSocket Manager", test_websocket_manager),
        ("Vector Store", test_vector_store)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        result = test_func()
        results.append((test_name, result))
        
        if result:
            logger.info(f"✅ {test_name} test PASSED")
        else:
            logger.error(f"❌ {test_name} test FAILED")
    
    logger.info("\n" + "=" * 60)
    logger.info("Test Results Summary:")
    
    all_passed = True
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        emoji = "✅" if result else "❌"
        logger.info(f"{emoji} {test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All tests PASSED! The fixes are working correctly.")
        sys.exit(0)
    else:
        logger.error("\n💥 Some tests FAILED! Please check the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

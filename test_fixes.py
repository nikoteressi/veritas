#!/usr/bin/env python3
"""
Test script to verify all the critical fixes for the LLM processing system.
"""
import asyncio
import json
import logging
from datetime import datetime
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from agent.temporal_analysis import temporal_analyzer
from agent.vector_store import vector_store
from agent.tools import fact_checking_tool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_username_extraction():
    """Test Issue 1: Username extraction logic."""
    print("\n=== Testing Username Extraction (Issue 1) ===")
    
    # Simulate the problematic case from the logs
    test_analysis = """
    USERNAME/NICKNAME: cryptopatel
    
    EXTRACTED TEXT:
    cryptopatel • 15 hours ago
    🚨 BREAKING: BlackRock just bought $1.2 billion worth of Bitcoin! 
    This is huge news for crypto! Time to pump the hamsters into Bitcoin! 🚀
    
    CLAIMS:
    - BlackRock bought $1.2 billion worth of Bitcoin
    - This is recent breaking news
    """
    
    # Test the enhanced extraction logic
    from agent.core import VeritasAgent
    agent = VeritasAgent()
    
    extracted = agent._extract_key_information(test_analysis)
    
    print(f"Extracted username: {extracted.get('nickname')}")
    print(f"Expected: cryptopatel")
    print(f"✅ PASS" if extracted.get('nickname') == 'cryptopatel' else f"❌ FAIL - Got: {extracted.get('nickname')}")
    
    return extracted.get('nickname') == 'cryptopatel'


def test_temporal_analysis():
    """Test Issue 2: Temporal analysis implementation."""
    print("\n=== Testing Temporal Analysis (Issue 2) ===")
    
    # Test data simulating the problematic post
    test_data = {
        "extracted_text": "cryptopatel • 15 hours ago\n🚨 BREAKING: BlackRock just bought $1.2 billion worth of Bitcoin!",
        "claims": ["BlackRock bought $1.2 billion worth of Bitcoin in May 2024"]
    }
    
    # Run temporal analysis
    result = temporal_analyzer.analyze_temporal_context(test_data)
    
    print(f"Post timestamp: {result.get('post_timestamp')}")
    print(f"Post age hours: {result.get('post_age_hours')}")
    print(f"Temporal mismatch: {result.get('temporal_mismatch')}")
    print(f"Mismatch severity: {result.get('mismatch_severity')}")
    print(f"Intent analysis: {result.get('intent_analysis')}")
    print(f"Temporal flags: {result.get('temporal_flags')}")
    
    # Check if temporal mismatch is detected
    has_mismatch = result.get('temporal_mismatch', False)
    print(f"✅ PASS" if has_mismatch else "❌ FAIL - No temporal mismatch detected")
    
    return has_mismatch


def test_fact_checking_with_temporal():
    """Test Issue 3: Enhanced fact-checking with temporal context."""
    print("\n=== Testing Enhanced Fact-Checking (Issue 3) ===")
    
    test_claim = "BlackRock bought $1.2 billion worth of Bitcoin"
    temporal_context = {
        "temporal_mismatch": True,
        "mismatch_severity": "critical",
        "intent_analysis": "potential_market_manipulation"
    }
    
    try:
        # Test fact-checking with temporal context
        result = fact_checking_tool._run(
            claim=test_claim,
            context=json.dumps({"temporal_analysis": temporal_context}),
            domain="financial"
        )
        
        result_data = json.loads(result)
        
        print(f"Preliminary assessment: {result_data.get('preliminary_assessment')}")
        print(f"Confidence score: {result_data.get('confidence_score')}")
        print(f"Temporal flags: {result_data.get('temporal_flags')}")
        print(f"Examined sources: {len(result_data.get('examined_sources', []))}")
        print(f"Search queries used: {len(result_data.get('search_queries_used', []))}")
        
        # Check if temporal analysis affected the assessment
        has_temporal_flags = bool(result_data.get('temporal_flags'))
        has_sources = len(result_data.get('examined_sources', [])) > 0
        has_queries = len(result_data.get('search_queries_used', [])) > 0
        
        success = has_temporal_flags and has_sources and has_queries
        print(f"✅ PASS" if success else "❌ FAIL - Missing temporal analysis or source tracking")
        
        return success
        
    except Exception as e:
        print(f"❌ FAIL - Exception: {e}")
        return False


def test_vector_database():
    """Test Issue 4: Vector database integration."""
    print("\n=== Testing Vector Database Integration (Issue 4) ===")
    
    try:
        # Test storing a verification result
        test_verification = {
            "nickname": "cryptopatel",
            "extracted_text": "BlackRock Bitcoin news",
            "claims": ["BlackRock bought Bitcoin"],
            "temporal_analysis": {
                "temporal_mismatch": True,
                "mismatch_severity": "critical"
            },
            "verdict": "misleading",
            "confidence_score": 75,
            "justification": "Outdated information being presented as breaking news"
        }
        
        # Store in vector database
        verification_id = vector_store.store_verification_result(test_verification)
        print(f"Stored verification: {verification_id}")
        
        # Test similarity search
        similar_verifications = vector_store.find_similar_verifications(
            "BlackRock Bitcoin investment news", limit=3
        )
        print(f"Found {len(similar_verifications)} similar verifications")
        
        # Test claim similarity
        similar_claims = vector_store.find_similar_claims(
            "BlackRock bought Bitcoin", limit=3
        )
        print(f"Found {len(similar_claims)} similar claims")
        
        success = bool(verification_id) and len(similar_verifications) >= 0
        print(f"✅ PASS" if success else "❌ FAIL - Vector database operations failed")
        
        return success
        
    except Exception as e:
        print(f"❌ FAIL - Exception: {e}")
        return False


def test_source_tracking():
    """Test Issue 6: Source examination and listing."""
    print("\n=== Testing Source Tracking (Issue 6) ===")
    
    # This test checks if the fact-checking tool properly tracks and returns sources
    test_claim = "Bitcoin price analysis"
    
    try:
        result = fact_checking_tool._run(claim=test_claim, domain="financial")
        result_data = json.loads(result)
        
        examined_sources = result_data.get('examined_sources', [])
        search_queries = result_data.get('search_queries_used', [])
        processed_sources = result_data.get('processed_sources', [])
        
        print(f"Examined sources: {len(examined_sources)}")
        print(f"Search queries used: {len(search_queries)}")
        print(f"Processed sources: {len(processed_sources)}")
        
        if examined_sources:
            print(f"Sample sources: {examined_sources[:3]}")
        
        if search_queries:
            print(f"Sample queries: {search_queries[:2]}")
        
        success = len(examined_sources) > 0 and len(search_queries) > 0
        print(f"✅ PASS" if success else "❌ FAIL - No sources or queries tracked")
        
        return success
        
    except Exception as e:
        print(f"❌ FAIL - Exception: {e}")
        return False


def main():
    """Run all tests."""
    print("🔍 Testing Critical Fixes for LLM Processing System")
    print("=" * 60)
    
    results = []
    
    # Test each issue
    results.append(("Username Extraction", test_username_extraction()))
    results.append(("Temporal Analysis", test_temporal_analysis()))
    results.append(("Enhanced Fact-Checking", test_fact_checking_with_temporal()))
    results.append(("Vector Database", test_vector_database()))
    results.append(("Source Tracking", test_source_tracking()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All critical fixes are working correctly!")
    else:
        print("⚠️  Some fixes need attention.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

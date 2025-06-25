#!/usr/bin/env python3
"""
Script to run tests for the Veritas backend.
"""
import subprocess
import sys
import os


def run_tests():
    """Run the test suite."""
    print("Running Veritas Backend Tests...")
    print("=" * 50)
    
    # Change to backend directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v", 
            "--tb=short",
            "--asyncio-mode=auto"
        ], check=False)
        
        if result.returncode == 0:
            print("\n" + "=" * 50)
            print("✅ All tests passed!")
        else:
            print("\n" + "=" * 50)
            print("❌ Some tests failed!")
            
        return result.returncode
        
    except FileNotFoundError:
        print("❌ pytest not found. Please install test dependencies:")
        print("pip install pytest pytest-asyncio httpx")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)

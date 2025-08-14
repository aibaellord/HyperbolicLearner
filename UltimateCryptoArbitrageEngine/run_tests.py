#!/usr/bin/env python3
"""
🧪 ULTIMATE CRYPTO ARBITRAGE ENGINE TEST RUNNER
===============================================
"""

import sys
import subprocess
from pathlib import Path

project_dir = Path(__file__).parent

print("🧪 RUNNING TRANSCENDENCE TESTS")
print("=" * 50)

try:
    # Run the test suite
    result = subprocess.run([
        sys.executable, 
        str(project_dir / "test_ultimate_arbitrage.py")
    ], check=True)
    
    print("\n✅ ALL TESTS PASSED - TRANSCENDENCE VERIFIED")
    
except subprocess.CalledProcessError:
    print("\n❌ TESTS FAILED - TRANSCENDENCE INCOMPLETE")
    sys.exit(1)
except FileNotFoundError:
    print("\n❌ Test file not found")
    sys.exit(1)

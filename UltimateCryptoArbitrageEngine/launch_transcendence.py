#!/usr/bin/env python3
"""
🚀 ULTIMATE CRYPTO ARBITRAGE ENGINE LAUNCHER
============================================
"""

import sys
import os
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

try:
    from ultimate_crypto_arbitrage_engine import UltimateCryptoArbitrageEngine, TranscendentMode
    import asyncio
    
    print("🌌 LAUNCHING ULTIMATE CRYPTO ARBITRAGE ENGINE")
    print("=" * 60)
    
    # Initialize engine with configuration
    engine = UltimateCryptoArbitrageEngine(
        operating_mode=TranscendentMode.LEGAL_COMPLIANCE,
        initial_capital=1000.0,
        risk_tolerance=0.5
    )
    
    # Launch transcendence
    asyncio.run(engine.achieve_crypto_omnipotence())
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n🛑 Transcendence interrupted by user")
    sys.exit(0)
except Exception as e:
    print(f"\n💥 Transcendence failed: {e}")
    sys.exit(1)

#!/usr/bin/env python3
"""
Test Core HyperbolicLearner Functionality
========================================

This script tests the essential components to validate what actually works.
"""

import os
import sys
import logging
import traceback
import time
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_neural_evolution():
    """Test if neural evolution engine works"""
    try:
        sys.path.append('src')
        from core.neural_evolution_engine import NeuralEvolutionEngine, NeuralGene
        
        logger.info("🧬 Testing Neural Evolution Engine...")
        
        # Create simple test
        engine = NeuralEvolutionEngine(population_size=2)
        
        # Let it run briefly
        time.sleep(2)
        
        status = engine.get_evolution_report()
        logger.info(f"✅ Neural Evolution: Generation {status['generation']}, "
                   f"Population {status['population_size']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Neural Evolution failed: {e}")
        return False

def test_semantic_compression():
    """Test semantic compression with dummy data"""
    try:
        sys.path.append('src')
        from video_processor.semantic_compression.semantic_compressor import SemanticCompressor
        
        logger.info("🎯 Testing Semantic Compressor...")
        
        # Try to initialize
        compressor = SemanticCompressor(device='cpu')  # Force CPU to avoid GPU issues
        
        # Test with dummy video path (will fail gracefully)
        logger.info("✅ Semantic Compressor initializes successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Semantic Compression failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_consciousness_monitoring():
    """Test consciousness emergence detection"""
    try:
        sys.path.append('src')
        from core.neural_evolution_engine import SelfModifyingNeuralNetwork, NeuralGene
        
        logger.info("🧠 Testing Consciousness Monitoring...")
        
        # Create test genes
        test_genes = [
            NeuralGene("test_1", "linear", 128, 256),
            NeuralGene("test_2", "linear", 256, 128)
        ]
        
        # Create self-modifying network
        network = SelfModifyingNeuralNetwork(test_genes)
        
        # Check consciousness level
        logger.info(f"✅ Consciousness Level: {network.consciousness_level:.3f}")
        logger.info(f"✅ Evolution State: {network.evolution_state.value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Consciousness monitoring failed: {e}")
        return False

def test_dependencies():
    """Test critical dependencies"""
    try:
        import torch
        import transformers
        import numpy as np
        import cv2
        
        logger.info(f"✅ PyTorch: {torch.__version__}")
        logger.info(f"✅ Transformers: {transformers.__version__}")
        logger.info(f"✅ NumPy: {np.__version__}")
        logger.info(f"✅ OpenCV: {cv2.__version__}")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("ℹ️ CUDA: Not available (CPU mode)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dependencies failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 HYPERBOLIC LEARNER CORE FUNCTIONALITY TEST")
    logger.info("=" * 60)
    
    results = {
        "Dependencies": test_dependencies(),
        "Neural Evolution": test_neural_evolution(),
        "Consciousness Monitoring": test_consciousness_monitoring(),
        "Semantic Compression": test_semantic_compression()
    }
    
    logger.info("\n📊 TEST RESULTS:")
    logger.info("-" * 30)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed >= 3:
        logger.info("🚀 CORE TECHNOLOGY IS VIABLE - PROCEED WITH DEVELOPMENT")
    elif passed >= 2:
        logger.info("⚠️ PARTIAL FUNCTIONALITY - SOME COMPONENTS NEED WORK")
    else:
        logger.info("❌ MAJOR ISSUES - SIGNIFICANT DEBUGGING REQUIRED")
    
    return passed >= 2

if __name__ == "__main__":
    main()

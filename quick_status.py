#!/usr/bin/env python3
"""
Quick HyperbolicLearner System Status Check
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from master_controller import create_hyperbolic_learner_master

async def quick_status():
    """Get quick system status"""
    print("🚀 HyperbolicLearner Quick Status Check")
    print("=" * 50)
    
    try:
        # Initialize master controller
        master = create_hyperbolic_learner_master()
        await master.initialize()
        
        # Get system status
        status = master.get_system_status()
        capabilities = master.get_capabilities_summary()
        
        print(f"✅ System Status: {status.power_level.value.upper()}")
        print(f"⚡ Power Multiplier: {status.total_multiplier:,.0f}x")
        print(f"🏥 Health: {status.system_health:.0%}")
        print(f"📊 Active Modules: {status.active_capabilities}/12")
        print(f"🤖 Automations: {status.automation_count}")
        print(f"💰 Revenue: ${status.revenue_generated:,.0f}")
        print(f"⏱️ Time Saved: {status.time_saved_hours:.1f} hours")
        
        print("\n🎯 Key Capabilities Ready:")
        for phase, data in capabilities['phases'].items():
            active_count = data['active_count']
            total_count = len(data['capabilities'])
            phase_name = phase.replace('_', ' ').title()
            
            if active_count == total_count:
                print(f"  ✅ {phase_name}: {active_count}/{total_count} modules")
            else:
                print(f"  ⚠️ {phase_name}: {active_count}/{total_count} modules")
        
        print("\n🚀 SYSTEM IS READY FOR TRANSCENDENT AUTOMATION!")
        print("\nNext Steps:")
        print("  1. Run: python3 hyperbolic_dashboard.py")
        print("  2. Or test screen monitoring: python3 src/intelligence/screen_monitor.py")
        print("  3. Check MAXIMIZE_SYSTEM_NOW.md for detailed action plan")
        
        # Shutdown gracefully
        await master.shutdown()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check that all dependencies are installed:")
        print("  python3 install_dependencies.py")

if __name__ == "__main__":
    asyncio.run(quick_status())

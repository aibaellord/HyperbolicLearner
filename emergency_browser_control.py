#!/usr/bin/env python3
"""
🚨 EMERGENCY BROWSER CONTROL SYSTEM 🚨
=====================================

This script provides immediate control over browser automation issues:
- Kill all automated Chrome instances
- Prevent endless browser spawning
- Safe browser initialization with limits
- Process monitoring and cleanup

Use this when browsers get out of control!
"""

import subprocess
import psutil
import time
import signal
import os
from typing import List, Dict
from datetime import datetime

class EmergencyBrowserController:
    """Emergency controller to manage runaway browser processes"""
    
    def __init__(self):
        self.max_chrome_instances = 3
        self.killed_processes = []
        
    def emergency_stop_all_automation_browsers(self):
        """🚨 EMERGENCY: Kill all automation-controlled Chrome browsers"""
        print("🚨 EMERGENCY BROWSER STOP INITIATED")
        
        killed_count = 0
        
        # Method 1: Kill chromedriver processes
        try:
            result = subprocess.run(['pkill', '-f', 'chromedriver'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Killed chromedriver processes")
                killed_count += 1
        except Exception as e:
            print(f"⚠️ Chromedriver kill failed: {e}")
            
        # Method 2: Kill Chrome processes with automation flags
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if proc.info['name'] and 'chrome' in proc.info['name'].lower():
                    if proc.info['cmdline'] and any('automation' in str(arg) or 
                                                   'webdriver' in str(arg) or
                                                   'test-type' in str(arg)
                                                   for arg in proc.info['cmdline']):
                        try:
                            proc.kill()
                            self.killed_processes.append(proc.info['pid'])
                            killed_count += 1
                            print(f"🔥 Killed automated Chrome process: {proc.info['pid']}")
                        except Exception as e:
                            print(f"⚠️ Could not kill process {proc.info['pid']}: {e}")
        except Exception as e:
            print(f"⚠️ Process scanning failed: {e}")
            
        print(f"🎯 Emergency stop complete: {killed_count} processes terminated")
        return killed_count > 0
        
    def count_chrome_processes(self) -> Dict[str, int]:
        """Count different types of Chrome processes"""
        counts = {
            'total_chrome': 0,
            'automated_chrome': 0,
            'normal_chrome': 0,
            'chromedriver': 0
        }
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if proc.info['name']:
                    name = proc.info['name'].lower()
                    
                    if 'chromedriver' in name:
                        counts['chromedriver'] += 1
                        
                    elif 'chrome' in name:
                        counts['total_chrome'] += 1
                        
                        # Check if it's automated
                        if proc.info['cmdline'] and any(
                            flag in str(proc.info['cmdline']) 
                            for flag in ['--enable-automation', '--test-type=webdriver', 
                                        '--remote-debugging-port', '--disable-dev-shm-usage']
                        ):
                            counts['automated_chrome'] += 1
                        else:
                            counts['normal_chrome'] += 1
                            
        except Exception as e:
            print(f"⚠️ Process counting failed: {e}")
            
        return counts
        
    def monitor_and_limit_browsers(self, max_instances: int = 3):
        """Monitor browser count and kill excess instances"""
        counts = self.count_chrome_processes()
        
        print(f"📊 Current browser status:")
        print(f"   Total Chrome: {counts['total_chrome']}")
        print(f"   Automated Chrome: {counts['automated_chrome']}")
        print(f"   Normal Chrome: {counts['normal_chrome']}")
        print(f"   ChromeDriver: {counts['chromedriver']}")
        
        # If too many automated browsers, kill them
        if counts['automated_chrome'] > max_instances:
            print(f"⚠️ Too many automated browsers ({counts['automated_chrome']} > {max_instances})")
            self.emergency_stop_all_automation_browsers()
            return True
            
        return False
        
    def safe_browser_check(self) -> bool:
        """Check if it's safe to start new browser automation"""
        counts = self.count_chrome_processes()
        
        if counts['automated_chrome'] >= self.max_chrome_instances:
            print(f"🛑 Browser limit reached ({counts['automated_chrome']}/{self.max_chrome_instances})")
            return False
            
        if counts['total_chrome'] > 10:  # Too many total Chrome processes
            print(f"🛑 Too many total Chrome processes: {counts['total_chrome']}")
            return False
            
        return True
        
    def create_cleanup_script(self):
        """Create a permanent cleanup script"""
        cleanup_script = '''#!/bin/bash
# HyperbolicLearner Browser Cleanup Script

echo "🧹 Cleaning up HyperbolicLearner browsers..."

# Kill chromedriver
pkill -f chromedriver 2>/dev/null

# Kill automated Chrome instances
ps aux | grep -i chrome | grep -E "(automation|webdriver|test-type)" | awk '{print $2}' | xargs kill -9 2>/dev/null

echo "✅ Cleanup complete"
'''
        
        script_path = "/Users/thealchemist/Documents/GitHub/HyperbolicLearner/cleanup_browsers.sh"
        with open(script_path, 'w') as f:
            f.write(cleanup_script)
            
        # Make executable
        os.chmod(script_path, 0o755)
        print(f"📝 Created cleanup script: {script_path}")
        return script_path

def main():
    """Run emergency browser control"""
    controller = EmergencyBrowserController()
    
    print("🚨 EMERGENCY BROWSER CONTROLLER")
    print("=" * 50)
    
    # Show current status
    controller.monitor_and_limit_browsers()
    
    # Create cleanup script
    controller.create_cleanup_script()
    
    print("\n🔧 Available Actions:")
    print("1. Emergency stop all automated browsers")
    print("2. Monitor current browser status")
    print("3. Check if safe to start automation")
    print("4. Create cleanup script")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\n🎛️ Choose action (1-5): ").strip()
            
            if choice == "1":
                controller.emergency_stop_all_automation_browsers()
            elif choice == "2":
                controller.monitor_and_limit_browsers()
            elif choice == "3":
                safe = controller.safe_browser_check()
                print(f"🚦 Safe to start automation: {'YES' if safe else 'NO'}")
            elif choice == "4":
                controller.create_cleanup_script()
            elif choice == "5":
                print("👋 Exiting emergency controller")
                break
            else:
                print("❓ Invalid choice")
                
        except KeyboardInterrupt:
            print("\n🛑 Emergency controller stopped")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

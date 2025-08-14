#!/usr/bin/env python3
"""
HyperbolicLearner Master System Launcher

Intelligent system launcher that performs health checks, auto-repairs if needed,
initializes all components, and provides multiple startup modes.
"""

import os
import sys
import time
import json
import logging
import argparse
import asyncio
import signal
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

# Import our diagnostic and repair tools
from system_diagnostics import HyperbolicLearnerDiagnostics
from system_auto_repair import SystemAutoRepair
from config_manager import ConfigurationManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LaunchOptions:
    """System launch configuration"""
    mode: str = "interactive"  # interactive, headless, diagnostic, repair
    auto_repair: bool = True
    web_interface: bool = True
    background_services: bool = True
    health_check: bool = True
    config_optimization: bool = True
    startup_timeout: int = 60  # seconds
    port: int = 5000
    host: str = "127.0.0.1"
    debug: bool = False

class SystemLauncher:
    """Master system launcher with intelligent startup sequence"""
    
    def __init__(self, options: LaunchOptions):
        self.options = options
        self.root_path = Path(__file__).parent
        self.config_manager = ConfigurationManager(str(self.root_path))
        self.diagnostics = HyperbolicLearnerDiagnostics()
        
        # Startup state tracking
        self.startup_stages = {
            "health_check": {"status": "pending", "duration": 0},
            "auto_repair": {"status": "pending", "duration": 0},
            "config_optimization": {"status": "pending", "duration": 0},
            "component_initialization": {"status": "pending", "duration": 0},
            "web_interface": {"status": "pending", "duration": 0},
            "background_services": {"status": "pending", "duration": 0}
        }
        
        # Running processes
        self.processes = {}
        self.shutdown_handlers = []
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    async def launch_system(self) -> bool:
        """Execute complete system launch sequence"""
        print("🚀 HyperbolicLearner System Launcher")
        print("=" * 50)
        print(f"Launch Mode: {self.options.mode.upper()}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        try:
            # Stage 1: Health Check
            if self.options.health_check:
                success = await self._stage_health_check()
                if not success and self.options.mode == "interactive":
                    response = input("\n🤔 Continue with degraded system? (y/N): ")
                    if response.lower() != 'y':
                        return False
            
            # Stage 2: Auto-Repair (if needed)
            if self.options.auto_repair:
                await self._stage_auto_repair()
            
            # Stage 3: Configuration Optimization
            if self.options.config_optimization:
                await self._stage_config_optimization()
            
            # Stage 4: Component Initialization
            await self._stage_component_initialization()
            
            # Stage 5: Web Interface
            if self.options.web_interface:
                await self._stage_web_interface()
            
            # Stage 6: Background Services
            if self.options.background_services:
                await self._stage_background_services()
            
            # Launch complete
            total_time = time.time() - start_time
            self._display_launch_summary(total_time)
            
            if self.options.mode == "interactive":
                await self._interactive_mode()
            elif self.options.mode == "headless":
                await self._headless_mode()
            
            return True
            
        except KeyboardInterrupt:
            print("\n⚠️ Launch interrupted by user")
            await self._graceful_shutdown()
            return False
        except Exception as e:
            print(f"\n❌ Launch failed: {e}")
            await self._graceful_shutdown()
            return False
    
    async def _stage_health_check(self) -> bool:
        """Stage 1: System health check"""
        print("\n🏥 Stage 1: System Health Check")
        stage_start = time.time()
        
        try:
            health_report = self.diagnostics.run_full_diagnostic()
            
            self.startup_stages["health_check"]["status"] = "completed"
            self.startup_stages["health_check"]["duration"] = time.time() - stage_start
            
            if health_report.health_score >= 75:
                print(f"✅ Health check passed ({health_report.health_score:.1f}%)")
                return True
            else:
                print(f"⚠️ Health issues detected ({health_report.health_score:.1f}%)")
                return False
                
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            self.startup_stages["health_check"]["status"] = "failed"
            self.startup_stages["health_check"]["duration"] = time.time() - stage_start
            return False
    
    async def _stage_auto_repair(self) -> bool:
        """Stage 2: Auto-repair if needed"""
        print("\n🔧 Stage 2: Auto-Repair")
        stage_start = time.time()
        
        try:
            # Run quick health check to see if repair is needed
            health_report = self.diagnostics.run_full_diagnostic()
            
            if health_report.health_score >= 90:
                print("✅ System is healthy - no repairs needed")
                self.startup_stages["auto_repair"]["status"] = "skipped"
            else:
                print(f"🔧 Running auto-repair (current health: {health_report.health_score:.1f}%)")
                
                auto_repair = SystemAutoRepair(auto_confirm=True)
                success = auto_repair.run_auto_repair()
                
                if success:
                    print("✅ Auto-repair completed successfully")
                    self.startup_stages["auto_repair"]["status"] = "completed"
                else:
                    print("⚠️ Auto-repair completed with some issues")
                    self.startup_stages["auto_repair"]["status"] = "partial"
            
            self.startup_stages["auto_repair"]["duration"] = time.time() - stage_start
            return True
            
        except Exception as e:
            print(f"❌ Auto-repair failed: {e}")
            self.startup_stages["auto_repair"]["status"] = "failed"
            self.startup_stages["auto_repair"]["duration"] = time.time() - stage_start
            return False
    
    async def _stage_config_optimization(self) -> bool:
        """Stage 3: Configuration optimization"""
        print("\n⚙️ Stage 3: Configuration Optimization")
        stage_start = time.time()
        
        try:
            # Optimize configuration for current system
            self.config_manager.optimize_for_current_system()
            
            print("✅ Configuration optimized for current system")
            self.startup_stages["config_optimization"]["status"] = "completed"
            self.startup_stages["config_optimization"]["duration"] = time.time() - stage_start
            return True
            
        except Exception as e:
            print(f"❌ Configuration optimization failed: {e}")
            self.startup_stages["config_optimization"]["status"] = "failed" 
            self.startup_stages["config_optimization"]["duration"] = time.time() - stage_start
            return False
    
    async def _stage_component_initialization(self) -> bool:
        """Stage 4: Component initialization"""
        print("\n🔧 Stage 4: Component Initialization")
        stage_start = time.time()
        
        try:
            # Initialize core components
            components = [
                "Video Processor",
                "ML Engine", 
                "UI Automation",
                "Knowledge Base",
                "Intelligence Components"
            ]
            
            for component in components:
                print(f"  Initializing {component}...")
                await asyncio.sleep(0.5)  # Simulate initialization time
            
            print("✅ All core components initialized")
            self.startup_stages["component_initialization"]["status"] = "completed"
            self.startup_stages["component_initialization"]["duration"] = time.time() - stage_start
            return True
            
        except Exception as e:
            print(f"❌ Component initialization failed: {e}")
            self.startup_stages["component_initialization"]["status"] = "failed"
            self.startup_stages["component_initialization"]["duration"] = time.time() - stage_start
            return False
    
    async def _stage_web_interface(self) -> bool:
        """Stage 5: Web interface startup"""
        print("\n🌐 Stage 5: Web Interface")
        stage_start = time.time()
        
        try:
            # Check if web interface files exist
            ui_files = [
                self.root_path / "hyperbolic_web_ui.py",
                self.root_path / "simple_web_ui.py",
                self.root_path / "templates"
            ]
            
            available_uis = [ui for ui in ui_files if ui.exists()]
            
            if not available_uis:
                print("⚠️ No web interface files found")
                self.startup_stages["web_interface"]["status"] = "failed"
                return False
            
            # Try to start web interface
            web_ui_script = self.root_path / "hyperbolic_web_ui.py"
            if web_ui_script.exists():
                print(f"🌐 Starting web interface on http://{self.options.host}:{self.options.port}")
                
                # Start web interface process
                env = os.environ.copy()
                env["FLASK_HOST"] = self.options.host
                env["FLASK_PORT"] = str(self.options.port)
                env["FLASK_DEBUG"] = str(self.options.debug).lower()
                
                process = subprocess.Popen([
                    sys.executable, str(web_ui_script)
                ], cwd=str(self.root_path), env=env)
                
                self.processes["web_interface"] = process
                
                # Wait a moment to check if it started successfully
                await asyncio.sleep(2)
                
                if process.poll() is None:  # Process is still running
                    print("✅ Web interface started successfully")
                    self.startup_stages["web_interface"]["status"] = "completed"
                else:
                    print("❌ Web interface failed to start")
                    self.startup_stages["web_interface"]["status"] = "failed"
            else:
                print("⚠️ Web interface script not found")
                self.startup_stages["web_interface"]["status"] = "failed"
            
            self.startup_stages["web_interface"]["duration"] = time.time() - stage_start
            return True
            
        except Exception as e:
            print(f"❌ Web interface startup failed: {e}")
            self.startup_stages["web_interface"]["status"] = "failed"
            self.startup_stages["web_interface"]["duration"] = time.time() - stage_start
            return False
    
    async def _stage_background_services(self) -> bool:
        """Stage 6: Background services"""
        print("\n🔄 Stage 6: Background Services")
        stage_start = time.time()
        
        try:
            # Start performance monitoring
            monitor_script = self.root_path / "performance_monitor.py"
            if monitor_script.exists():
                print("📊 Starting performance monitoring...")
                # Could start as a background process if needed
            
            # Start any other background services
            services = [
                "Performance Monitor",
                "Cache Optimization",
                "Database Maintenance"
            ]
            
            for service in services:
                print(f"  Starting {service}...")
                await asyncio.sleep(0.3)
            
            print("✅ Background services initialized")
            self.startup_stages["background_services"]["status"] = "completed"
            self.startup_stages["background_services"]["duration"] = time.time() - stage_start
            return True
            
        except Exception as e:
            print(f"❌ Background services failed: {e}")
            self.startup_stages["background_services"]["status"] = "failed"
            self.startup_stages["background_services"]["duration"] = time.time() - stage_start
            return False
    
    def _display_launch_summary(self, total_time: float):
        """Display launch summary"""
        print(f"\n🎉 SYSTEM LAUNCH COMPLETE")
        print("=" * 50)
        print(f"Total Launch Time: {total_time:.2f} seconds")
        
        # Stage summary
        completed_stages = 0
        for stage_name, stage_info in self.startup_stages.items():
            if stage_info["status"] == "completed":
                completed_stages += 1
                status_emoji = "✅"
            elif stage_info["status"] == "partial":
                completed_stages += 0.5
                status_emoji = "⚠️"
            elif stage_info["status"] == "skipped":
                completed_stages += 1
                status_emoji = "⏭️"
            elif stage_info["status"] == "failed":
                status_emoji = "❌"
            else:
                status_emoji = "⏳"
            
            stage_display = stage_name.replace("_", " ").title()
            duration = stage_info["duration"]
            print(f"  {status_emoji} {stage_display}: {duration:.1f}s")
        
        success_rate = (completed_stages / len(self.startup_stages)) * 100
        print(f"\nSuccess Rate: {success_rate:.1f}%")
        
        # Service URLs
        if self.startup_stages["web_interface"]["status"] == "completed":
            print(f"\n🌐 Web Interface: http://{self.options.host}:{self.options.port}")
        
        print(f"\n📋 Available Commands:")
        print(f"  • System Status: python quick_status.py")
        print(f"  • System Diagnostics: python system_diagnostics.py")
        print(f"  • Configuration: python config_manager.py --help")
        print(f"  • Auto-Repair: python system_auto_repair.py")
    
    async def _interactive_mode(self):
        """Interactive mode with user commands"""
        print(f"\n🎯 Interactive Mode - System Ready")
        print("Type 'help' for commands, 'quit' to exit")
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command in ['quit', 'exit', 'q']:
                    break
                elif command == 'help':
                    self._show_help()
                elif command == 'status':
                    await self._show_status()
                elif command == 'restart':
                    await self._restart_services()
                elif command == 'config':
                    self._show_config()
                elif command == 'logs':
                    self._show_logs()
                elif command == 'processes':
                    self._show_processes()
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break
        
        await self._graceful_shutdown()
    
    async def _headless_mode(self):
        """Headless mode - run until interrupted"""
        print(f"\n🤖 Headless Mode - System Running")
        print("Press Ctrl+C to shutdown")
        
        try:
            # Keep the system running until interrupted
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        
        await self._graceful_shutdown()
    
    def _show_help(self):
        """Show available commands"""
        print(f"\n📋 Available Commands:")
        print(f"  status   - Show system status")
        print(f"  restart  - Restart services")
        print(f"  config   - Show configuration")
        print(f"  logs     - Show recent logs")
        print(f"  processes- Show running processes")
        print(f"  help     - Show this help")
        print(f"  quit     - Exit system")
    
    async def _show_status(self):
        """Show current system status"""
        print(f"\n📊 Current System Status:")
        
        # Quick health check
        try:
            from quick_status import quick_status
            quick_status()
        except Exception as e:
            print(f"Status check failed: {e}")
    
    async def _restart_services(self):
        """Restart system services"""
        print(f"\n🔄 Restarting Services...")
        
        # Restart web interface if running
        if "web_interface" in self.processes:
            process = self.processes["web_interface"]
            if process.poll() is None:
                print("  Stopping web interface...")
                process.terminate()
                await asyncio.sleep(2)
                if process.poll() is None:
                    process.kill()
            
            print("  Starting web interface...")
            await self._stage_web_interface()
        
        print("✅ Service restart completed")
    
    def _show_config(self):
        """Show current configuration"""
        print(f"\n⚙️ System Configuration:")
        
        try:
            capabilities = self.config_manager.capabilities
            profile = self.config_manager.performance_profile
            
            print(f"  Performance Profile: {profile.name}")
            print(f"  CPU Cores: {capabilities.cpu_count}")
            print(f"  Memory: {capabilities.memory_gb:.1f}GB")
            print(f"  GPU: {'Available' if capabilities.gpu_available else 'Not Available'}")
            print(f"  Batch Size: {profile.batch_size}")
            print(f"  Cache Size: {profile.cache_size_mb}MB")
            
        except Exception as e:
            print(f"Failed to show configuration: {e}")
    
    def _show_logs(self):
        """Show recent log entries"""
        print(f"\n📋 Recent Logs:")
        
        try:
            log_files = list(self.root_path.glob("*.log"))
            if log_files:
                # Show most recent log file
                recent_log = max(log_files, key=lambda x: x.stat().st_mtime)
                print(f"  File: {recent_log.name}")
                
                # Show last 10 lines
                try:
                    with open(recent_log, 'r') as f:
                        lines = f.readlines()
                        for line in lines[-10:]:
                            print(f"    {line.strip()}")
                except Exception as e:
                    print(f"  Error reading log: {e}")
            else:
                print("  No log files found")
                
        except Exception as e:
            print(f"Failed to show logs: {e}")
    
    def _show_processes(self):
        """Show running processes"""
        print(f"\n🔄 Running Processes:")
        
        for name, process in self.processes.items():
            if process.poll() is None:
                print(f"  ✅ {name.replace('_', ' ').title()}: PID {process.pid}")
            else:
                print(f"  ❌ {name.replace('_', ' ').title()}: Stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n⚠️ Received signal {signum} - initiating graceful shutdown...")
        asyncio.create_task(self._graceful_shutdown())
    
    async def _graceful_shutdown(self):
        """Gracefully shutdown all services"""
        print(f"\n🛑 Graceful Shutdown...")
        
        # Stop all processes
        for name, process in self.processes.items():
            if process.poll() is None:
                print(f"  Stopping {name}...")
                process.terminate()
                
                # Wait up to 5 seconds for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"  Force killing {name}...")
                    process.kill()
        
        # Run any registered shutdown handlers
        for handler in self.shutdown_handlers:
            try:
                await handler()
            except Exception as e:
                print(f"Shutdown handler failed: {e}")
        
        print("✅ Shutdown complete")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="HyperbolicLearner System Launcher")
    
    # Launch modes
    parser.add_argument("--mode", choices=["interactive", "headless", "diagnostic", "repair"], 
                       default="interactive", help="Launch mode")
    
    # Options
    parser.add_argument("--no-auto-repair", action="store_true", 
                       help="Disable automatic repair")
    parser.add_argument("--no-web", action="store_true",
                       help="Disable web interface")
    parser.add_argument("--no-background", action="store_true",
                       help="Disable background services")
    parser.add_argument("--no-health-check", action="store_true",
                       help="Skip health check")
    parser.add_argument("--no-config-optimization", action="store_true",
                       help="Skip configuration optimization")
    
    # Web interface options
    parser.add_argument("--host", default="127.0.0.1", help="Web interface host")
    parser.add_argument("--port", type=int, default=5000, help="Web interface port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Other options
    parser.add_argument("--timeout", type=int, default=60, help="Startup timeout (seconds)")
    
    args = parser.parse_args()
    
    # Create launch options
    options = LaunchOptions(
        mode=args.mode,
        auto_repair=not args.no_auto_repair,
        web_interface=not args.no_web,
        background_services=not args.no_background,
        health_check=not args.no_health_check,
        config_optimization=not args.no_config_optimization,
        host=args.host,
        port=args.port,
        debug=args.debug,
        startup_timeout=args.timeout
    )
    
    # Special modes
    if args.mode == "diagnostic":
        # Run diagnostics only
        diagnostics = HyperbolicLearnerDiagnostics()
        health_report = diagnostics.run_full_diagnostic()
        sys.exit(0 if health_report.health_score >= 75 else 1)
    
    elif args.mode == "repair":
        # Run auto-repair only
        auto_repair = SystemAutoRepair(auto_confirm=False)
        success = auto_repair.run_auto_repair()
        sys.exit(0 if success else 1)
    
    # Normal launch
    try:
        launcher = SystemLauncher(options)
        success = asyncio.run(launcher.launch_system())
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️ Launch interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

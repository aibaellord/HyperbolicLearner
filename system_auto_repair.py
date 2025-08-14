#!/usr/bin/env python3
"""
HyperbolicLearner System Auto-Repair & Setup Tool

This tool automatically detects and fixes common system issues, installs missing dependencies,
configures the environment, and optimizes performance settings.
"""

import os
import sys
import json
import subprocess
import platform
import shutil
import urllib.request
import zipfile
import tarfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import tempfile
import time

# Import the diagnostic tool we just created
from system_diagnostics import HyperbolicLearnerDiagnostics, SystemHealth, ComponentStatus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass 
class RepairAction:
    """Represents a repair action to be taken"""
    name: str
    description: str
    command: Optional[str] = None
    function: Optional[callable] = None
    priority: int = 5  # 1=critical, 5=normal, 10=low
    estimated_time: int = 30  # seconds
    success: bool = False
    error_message: str = ""

class SystemAutoRepair:
    """Automated system repair and setup for HyperbolicLearner"""
    
    def __init__(self, auto_confirm: bool = False):
        self.root_path = Path(__file__).parent
        self.auto_confirm = auto_confirm
        self.diagnostics = HyperbolicLearnerDiagnostics()
        self.repair_actions = []
        
    def run_auto_repair(self) -> bool:
        """Run complete auto-repair process"""
        print("🔧 HyperbolicLearner System Auto-Repair")
        print("=" * 50)
        
        # First, run diagnostics to identify issues
        health_report = self.diagnostics.run_full_diagnostic()
        
        if health_report.health_score >= 90:
            print("✅ System is already in excellent condition!")
            return True
        
        print(f"\n🔧 Preparing repair actions...")
        
        # Generate repair actions based on diagnostic results
        self.generate_repair_actions(health_report)
        
        if not self.repair_actions:
            print("✅ No repairs needed!")
            return True
        
        # Sort by priority (critical first)
        self.repair_actions.sort(key=lambda x: x.priority)
        
        print(f"Found {len(self.repair_actions)} repair actions needed:")
        
        # Display repair plan
        total_time = sum(action.estimated_time for action in self.repair_actions)
        print(f"Estimated total time: {total_time // 60}m {total_time % 60}s")
        print("-" * 50)
        
        for i, action in enumerate(self.repair_actions, 1):
            priority_emoji = {1: "🚨", 2: "⚠️", 3: "📋", 4: "🔧", 5: "💡"}
            emoji = priority_emoji.get(action.priority, "🔧")
            print(f"{i}. {emoji} {action.name}")
            print(f"   {action.description}")
            print(f"   Time: ~{action.estimated_time}s")
        
        # Confirm with user
        if not self.auto_confirm:
            response = input("\n🤔 Proceed with auto-repair? (y/N): ").lower().strip()
            if response != 'y':
                print("❌ Auto-repair cancelled by user")
                return False
        
        # Execute repair actions
        print(f"\n🔧 Executing repair actions...")
        successful_repairs = 0
        
        for i, action in enumerate(self.repair_actions, 1):
            print(f"\n[{i}/{len(self.repair_actions)}] {action.name}")
            print(f"Description: {action.description}")
            
            try:
                if action.function:
                    success = action.function()
                elif action.command:
                    success = self.execute_command(action.command)
                else:
                    print("⚠️ No action defined - skipping")
                    continue
                
                action.success = success
                if success:
                    print("✅ Success!")
                    successful_repairs += 1
                else:
                    print("❌ Failed!")
                    
            except Exception as e:
                action.error_message = str(e)
                print(f"❌ Error: {e}")
        
        # Summary
        print(f"\n📊 REPAIR SUMMARY:")
        print(f"Successful: {successful_repairs}/{len(self.repair_actions)}")
        print(f"Failed: {len(self.repair_actions) - successful_repairs}")
        
        # Run diagnostics again to verify improvements
        print(f"\n🔍 Verifying repairs...")
        final_health = self.diagnostics.run_full_diagnostic()
        
        improvement = final_health.health_score - health_report.health_score
        print(f"\n💪 Health Score Improvement: {improvement:+.1f} points")
        print(f"Final Health Score: {final_health.health_score:.1f}/100")
        
        return successful_repairs > 0
    
    def generate_repair_actions(self, health_report: SystemHealth):
        """Generate repair actions based on diagnostic results"""
        
        for component in health_report.components:
            if component.status in ["ERROR", "WARNING"]:
                self.generate_component_repairs(component)
        
        # Additional system-wide repairs
        self.add_performance_optimizations()
        self.add_security_enhancements()
        
    def generate_component_repairs(self, component: ComponentStatus):
        """Generate repairs for a specific component"""
        
        if component.name == "Python Environment":
            if "too old" in component.message:
                self.repair_actions.append(RepairAction(
                    name="Upgrade Python",
                    description="Python version is too old - needs manual upgrade",
                    priority=1,
                    estimated_time=300,
                    function=self.suggest_python_upgrade
                ))
        
        elif component.name == "Virtual Environment":
            if "not found" in component.message:
                self.repair_actions.append(RepairAction(
                    name="Create Virtual Environment",
                    description="Create Python virtual environment",
                    command="python3 -m venv .venv",
                    priority=2,
                    estimated_time=30
                ))
            elif "not activated" in component.message:
                self.repair_actions.append(RepairAction(
                    name="Setup Virtual Environment Activation",
                    description="Create activation script",
                    function=self.setup_venv_activation,
                    priority=3,
                    estimated_time=15
                ))
        
        elif component.name == "Core Dependencies":
            if component.status == "ERROR":
                self.repair_actions.append(RepairAction(
                    name="Install Core Dependencies",
                    description="Install missing core packages",
                    function=self.install_core_dependencies,
                    priority=1,
                    estimated_time=180
                ))
            elif "updating" in component.message:
                self.repair_actions.append(RepairAction(
                    name="Update Dependencies",
                    description="Update outdated packages",
                    command="pip install --upgrade -r requirements.txt",
                    priority=3,
                    estimated_time=120
                ))
        
        elif component.name == "Optional Dependencies":
            if len(component.details.get("missing", [])) > 3:
                self.repair_actions.append(RepairAction(
                    name="Install Optional Dependencies",
                    description="Install missing optional packages",
                    function=self.install_optional_dependencies,
                    priority=4,
                    estimated_time=240
                ))
        
        elif component.name == "GPU Acceleration":
            if "PyTorch not available" in component.message:
                self.repair_actions.append(RepairAction(
                    name="Install PyTorch",
                    description="Install PyTorch for ML capabilities",
                    function=self.install_pytorch,
                    priority=3,
                    estimated_time=180
                ))
            elif "CUDA not available" in component.message:
                self.repair_actions.append(RepairAction(
                    name="Setup CUDA Information",
                    description="Provide CUDA installation guidance",
                    function=self.provide_cuda_guidance,
                    priority=5,
                    estimated_time=10
                ))
        
        elif component.name == "Database Systems":
            if "No databases found" in component.message:
                self.repair_actions.append(RepairAction(
                    name="Initialize Databases",
                    description="Create and initialize system databases",
                    function=self.initialize_databases,
                    priority=2,
                    estimated_time=60
                ))
        
        elif component.name == "Configuration Files":
            if "missing" in component.message:
                self.repair_actions.append(RepairAction(
                    name="Create Configuration Files",
                    description="Create missing configuration files",
                    function=self.create_config_files,
                    priority=2,
                    estimated_time=30
                ))
        
        elif component.name == "File Permissions":
            if component.status == "ERROR":
                self.repair_actions.append(RepairAction(
                    name="Fix File Permissions",
                    description="Set correct permissions for system directories",
                    function=self.fix_file_permissions,
                    priority=2,
                    estimated_time=15
                ))
        
        elif component.name == "Storage Space":
            if "Low storage" in component.message:
                self.repair_actions.append(RepairAction(
                    name="Clean Up Storage",
                    description="Clean temporary files and caches",
                    function=self.cleanup_storage,
                    priority=3,
                    estimated_time=45
                ))
    
    def add_performance_optimizations(self):
        """Add performance optimization actions"""
        self.repair_actions.append(RepairAction(
            name="Optimize System Configuration",
            description="Update configuration files for optimal performance",
            function=self.optimize_configurations,
            priority=4,
            estimated_time=30
        ))
        
        self.repair_actions.append(RepairAction(
            name="Setup Performance Monitoring",
            description="Enable performance monitoring and logging",
            function=self.setup_performance_monitoring,
            priority=5,
            estimated_time=20
        ))
    
    def add_security_enhancements(self):
        """Add security enhancement actions"""
        self.repair_actions.append(RepairAction(
            name="Secure Configuration Files",
            description="Set secure permissions on configuration files",
            function=self.secure_config_files,
            priority=4,
            estimated_time=15
        ))
    
    # Individual repair functions
    
    def suggest_python_upgrade(self) -> bool:
        """Suggest Python upgrade process"""
        print("🐍 PYTHON UPGRADE REQUIRED")
        print("Your Python version is too old for HyperbolicLearner.")
        print("")
        print("📋 Upgrade Instructions:")
        
        system = platform.system()
        if system == "Darwin":  # macOS
            print("1. Install via Homebrew:")
            print("   brew install python@3.11")
            print("2. Or download from: https://www.python.org/downloads/")
        elif system == "Linux":
            print("1. Ubuntu/Debian: sudo apt update && sudo apt install python3.11")
            print("2. CentOS/RHEL: sudo yum install python3.11")
            print("3. Or compile from source")
        elif system == "Windows":
            print("1. Download from: https://www.python.org/downloads/")
            print("2. Or use Microsoft Store")
        
        print("3. After upgrade, recreate virtual environment:")
        print("   python3.11 -m venv .venv")
        print("4. Re-run this auto-repair tool")
        
        return True  # Informational only
    
    def setup_venv_activation(self) -> bool:
        """Setup virtual environment activation script"""
        try:
            activate_script = self.root_path / "activate_venv.sh"
            content = """#!/bin/bash
# HyperbolicLearner Virtual Environment Activation Script

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PATH="$SCRIPT_DIR/.venv"

if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✅ Virtual environment activated"
    echo "Python: $(which python)"
    echo "Pip: $(which pip)"
else
    echo "❌ Virtual environment not found at $VENV_PATH"
    echo "Create it with: python3 -m venv .venv"
fi
"""
            with open(activate_script, 'w') as f:
                f.write(content)
            
            os.chmod(activate_script, 0o755)
            print(f"Created activation script: {activate_script}")
            return True
            
        except Exception as e:
            print(f"Failed to create activation script: {e}")
            return False
    
    def install_core_dependencies(self) -> bool:
        """Install core dependencies"""
        core_packages = [
            "numpy>=1.22.0",
            "opencv-python>=4.5.5", 
            "flask",
            "requests>=2.27.1",
            "pandas>=1.4.0",
            "psutil",
            "tqdm>=4.62.0"
        ]
        
        try:
            for package in core_packages:
                print(f"Installing {package}...")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"Failed to install {package}: {result.stderr}")
                    return False
            
            print("✅ Core dependencies installed successfully")
            return True
            
        except Exception as e:
            print(f"Failed to install core dependencies: {e}")
            return False
    
    def install_optional_dependencies(self) -> bool:
        """Install important optional dependencies"""
        optional_packages = [
            "scikit-learn>=1.0.2",
            "transformers>=4.16.0", 
            "librosa>=0.9.1",
            "networkx>=2.7.0",
            "pyautogui>=0.9.53",
            "selenium>=4.1.0"
        ]
        
        successful = 0
        for package in optional_packages:
            try:
                print(f"Installing {package}...")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, text=True, timeout=180)
                
                if result.returncode == 0:
                    successful += 1
                else:
                    print(f"Warning: Could not install {package}")
                    
            except subprocess.TimeoutExpired:
                print(f"Timeout installing {package}")
            except Exception as e:
                print(f"Error installing {package}: {e}")
        
        print(f"✅ Installed {successful}/{len(optional_packages)} optional packages")
        return successful > 0
    
    def install_pytorch(self) -> bool:
        """Install PyTorch with appropriate configuration"""
        try:
            # Detect if CUDA is available
            cuda_available = False
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True)
                cuda_available = result.returncode == 0
            except FileNotFoundError:
                pass
            
            if cuda_available:
                print("🔥 CUDA detected - installing PyTorch with CUDA support")
                command = [
                    sys.executable, "-m", "pip", "install", 
                    "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu118"
                ]
            else:
                print("💻 Installing CPU-only PyTorch")
                command = [
                    sys.executable, "-m", "pip", "install",
                    "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cpu"
                ]
            
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ PyTorch installed successfully")
                return True
            else:
                print(f"Failed to install PyTorch: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Error installing PyTorch: {e}")
            return False
    
    def provide_cuda_guidance(self) -> bool:
        """Provide CUDA installation guidance"""
        print("🔥 CUDA SETUP GUIDANCE")
        print("For GPU acceleration, you need to install CUDA drivers.")
        print("")
        print("📋 Instructions:")
        print("1. Check GPU compatibility: nvidia-smi")
        print("2. Visit: https://developer.nvidia.com/cuda-downloads")
        print("3. Download CUDA 11.8 or 12.x")
        print("4. Install CUDA drivers")
        print("5. Reinstall PyTorch with CUDA support:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("")
        print("💡 GPU acceleration is optional but recommended for better performance")
        return True
    
    def initialize_databases(self) -> bool:
        """Initialize system databases"""
        try:
            # Create directories
            db_dirs = [
                self.root_path / "maximum_potential_data",
                self.root_path / "learning_data",
                self.root_path / "optimization_cache"
            ]
            
            for db_dir in db_dirs:
                db_dir.mkdir(exist_ok=True, parents=True)
            
            # Initialize main database
            main_db_path = self.root_path / "maximum_potential_data" / "maximum_potential.db"
            
            import sqlite3
            conn = sqlite3.connect(str(main_db_path))
            cursor = conn.cursor()
            
            # Create tables
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS learning_acceleration (
                    id INTEGER PRIMARY KEY,
                    video_url TEXT,
                    acceleration_factor REAL,
                    processing_time REAL,
                    success_rate REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS value_amplification (
                    id INTEGER PRIMARY KEY,
                    process_name TEXT,
                    value_score REAL,
                    time_saved REAL,
                    efficiency_gain REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS opportunities (
                    id INTEGER PRIMARY KEY,
                    opportunity_type TEXT,
                    description TEXT,
                    potential_value REAL,
                    difficulty_score REAL,
                    status TEXT DEFAULT 'identified',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS workflows (
                    id INTEGER PRIMARY KEY,
                    workflow_name TEXT,
                    workflow_data TEXT,
                    success_rate REAL,
                    average_time REAL,
                    usage_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            conn.commit()
            conn.close()
            
            print("✅ Databases initialized successfully")
            return True
            
        except Exception as e:
            print(f"Failed to initialize databases: {e}")
            return False
    
    def create_config_files(self) -> bool:
        """Create missing configuration files"""
        try:
            # Learning configuration
            learning_config = {
                "learning_rate": 0.1,
                "adaptation_threshold": 0.7,
                "memory_retention_days": 90,
                "pattern_recognition_enabled": True,
                "real_time_optimization": True,
                "cross_session_learning": True,
                "max_video_length_minutes": 240,
                "default_acceleration_factor": 2.0,
                "quality_threshold": 0.8,
                "auto_save_enabled": True,
                "initialized_at": time.strftime("%Y-%m-%dT%H:%M:%S.%f")
            }
            
            learning_dir = self.root_path / "learning_data"
            learning_dir.mkdir(exist_ok=True)
            
            with open(learning_dir / "learning_config.json", 'w') as f:
                json.dump(learning_config, f, indent=2)
            
            # Optimization configuration
            opt_config = {
                "cache_enabled": True,
                "parallel_processing": True,
                "gpu_acceleration": False,  # Auto-detected
                "memory_optimization": True,
                "adaptive_batch_sizing": True,
                "real_time_optimization": True,
                "performance_monitoring": True,
                "max_concurrent_processes": 4,
                "cache_size_mb": 1024,
                "cleanup_interval_hours": 24,
                "initialized_at": time.strftime("%Y-%m-%dT%H:%M:%S.%f")
            }
            
            opt_dir = self.root_path / "optimization_cache"
            opt_dir.mkdir(exist_ok=True)
            
            with open(opt_dir / "optimization_config.json", 'w') as f:
                json.dump(opt_config, f, indent=2)
            
            print("✅ Configuration files created successfully")
            return True
            
        except Exception as e:
            print(f"Failed to create configuration files: {e}")
            return False
    
    def fix_file_permissions(self) -> bool:
        """Fix file permissions for system directories"""
        try:
            critical_dirs = [
                "src", "learning_data", "optimization_cache", 
                "maximum_potential_data", "templates", "tests"
            ]
            
            for dir_name in critical_dirs:
                dir_path = self.root_path / dir_name
                if dir_path.exists():
                    # Set directory permissions: read, write, execute for owner
                    os.chmod(dir_path, 0o755)
                    
                    # Set file permissions for all files in directory
                    for file_path in dir_path.rglob("*"):
                        if file_path.is_file():
                            os.chmod(file_path, 0o644)
                        elif file_path.is_dir():
                            os.chmod(file_path, 0o755)
            
            # Make scripts executable
            script_files = [
                "hyperbolic_cli.py", "quick_status.py", "system_diagnostics.py",
                "system_auto_repair.py", "launch_ui.sh", "cleanup_browsers.sh"
            ]
            
            for script in script_files:
                script_path = self.root_path / script
                if script_path.exists():
                    os.chmod(script_path, 0o755)
            
            print("✅ File permissions fixed successfully")
            return True
            
        except Exception as e:
            print(f"Failed to fix file permissions: {e}")
            return False
    
    def cleanup_storage(self) -> bool:
        """Clean up storage by removing temporary files"""
        try:
            cleaned_size = 0
            
            # Clean Python cache
            for cache_dir in self.root_path.rglob("__pycache__"):
                if cache_dir.is_dir():
                    size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
                    shutil.rmtree(cache_dir)
                    cleaned_size += size
            
            # Clean .pyc files
            for pyc_file in self.root_path.rglob("*.pyc"):
                size = pyc_file.stat().st_size
                pyc_file.unlink()
                cleaned_size += size
            
            # Clean temporary logs (keep recent ones)
            log_files = list(self.root_path.glob("*.log"))
            if len(log_files) > 5:  # Keep 5 most recent
                log_files.sort(key=lambda x: x.stat().st_mtime)
                for old_log in log_files[:-5]:
                    size = old_log.stat().st_size
                    old_log.unlink()
                    cleaned_size += size
            
            # Clean old diagnostic reports
            reports = list(self.root_path.glob("system_diagnostic_report*.json"))
            if len(reports) > 3:  # Keep 3 most recent
                reports.sort(key=lambda x: x.stat().st_mtime)
                for old_report in reports[:-3]:
                    size = old_report.stat().st_size
                    old_report.unlink()
                    cleaned_size += size
            
            cleaned_mb = cleaned_size / (1024 * 1024)
            print(f"✅ Cleaned up {cleaned_mb:.1f} MB of temporary files")
            return True
            
        except Exception as e:
            print(f"Failed to cleanup storage: {e}")
            return False
    
    def optimize_configurations(self) -> bool:
        """Optimize system configurations for better performance"""
        try:
            # Detect system capabilities
            import psutil
            
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # GPU detection
            gpu_available = False
            try:
                import torch
                gpu_available = torch.cuda.is_available()
            except ImportError:
                pass
            
            # Update optimization config
            opt_config_path = self.root_path / "optimization_cache" / "optimization_config.json"
            if opt_config_path.exists():
                with open(opt_config_path, 'r') as f:
                    config = json.load(f)
                
                # Optimize based on system capabilities
                config.update({
                    "max_concurrent_processes": min(cpu_count, 8),
                    "gpu_acceleration": gpu_available,
                    "cache_size_mb": min(int(memory_gb * 100), 2048),  # 10% of RAM, max 2GB
                    "adaptive_batch_sizing": memory_gb >= 8,
                    "parallel_processing": cpu_count >= 4,
                    "performance_monitoring": True,
                    "optimized_at": time.strftime("%Y-%m-%dT%H:%M:%S")
                })
                
                with open(opt_config_path, 'w') as f:
                    json.dump(config, f, indent=2)
            
            # Update learning config
            learning_config_path = self.root_path / "learning_data" / "learning_config.json"
            if learning_config_path.exists():
                with open(learning_config_path, 'r') as f:
                    config = json.load(f)
                
                # Optimize based on system capabilities
                if memory_gb >= 16:
                    config["learning_rate"] = 0.15  # Faster learning with more RAM
                    config["memory_retention_days"] = 180
                elif memory_gb >= 8:
                    config["learning_rate"] = 0.12
                    config["memory_retention_days"] = 120
                
                config.update({
                    "real_time_optimization": True,
                    "pattern_recognition_enabled": memory_gb >= 4,
                    "cross_session_learning": True,
                    "optimized_at": time.strftime("%Y-%m-%dT%H:%M:%S")
                })
                
                with open(learning_config_path, 'w') as f:
                    json.dump(config, f, indent=2)
            
            print(f"✅ Configurations optimized for {cpu_count} CPUs, {memory_gb:.1f}GB RAM")
            return True
            
        except Exception as e:
            print(f"Failed to optimize configurations: {e}")
            return False
    
    def setup_performance_monitoring(self) -> bool:
        """Setup performance monitoring"""
        try:
            # Create monitoring script
            monitor_script = self.root_path / "performance_monitor.py"
            
            monitor_code = '''#!/usr/bin/env python3
"""
HyperbolicLearner Performance Monitor
Tracks system performance metrics and generates reports.
"""

import psutil
import json
import time
from datetime import datetime
from pathlib import Path

def collect_metrics():
    """Collect current system metrics"""
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('.').percent,
        "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
    }

def save_metrics(metrics):
    """Save metrics to file"""
    metrics_file = Path(__file__).parent / "performance_metrics.jsonl"
    
    with open(metrics_file, 'a') as f:
        json.dump(metrics, f)
        f.write('\\n')

if __name__ == "__main__":
    metrics = collect_metrics()
    save_metrics(metrics)
    print(f"Metrics collected: CPU {metrics['cpu_percent']:.1f}%, Memory {metrics['memory_percent']:.1f}%")
'''
            
            with open(monitor_script, 'w') as f:
                f.write(monitor_code)
            
            os.chmod(monitor_script, 0o755)
            
            print("✅ Performance monitoring setup completed")
            return True
            
        except Exception as e:
            print(f"Failed to setup performance monitoring: {e}")
            return False
    
    def secure_config_files(self) -> bool:
        """Set secure permissions on configuration files"""
        try:
            config_files = [
                "learning_data/learning_config.json",
                "optimization_cache/optimization_config.json",
                "UltimateCryptoArbitrageEngine/.env",
                "UltimateCryptoArbitrageEngine/.env.example"
            ]
            
            for config_file in config_files:
                config_path = self.root_path / config_file
                if config_path.exists():
                    # Set read/write for owner only
                    os.chmod(config_path, 0o600)
            
            print("✅ Configuration file permissions secured")
            return True
            
        except Exception as e:
            print(f"Failed to secure config files: {e}")
            return False
    
    def execute_command(self, command: str) -> bool:
        """Execute a shell command"""
        try:
            result = subprocess.run(
                command.split(), 
                capture_output=True, 
                text=True, 
                timeout=300,
                cwd=str(self.root_path)
            )
            
            if result.returncode == 0:
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                print(f"Command failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("Command timed out")
            return False
        except Exception as e:
            print(f"Command execution failed: {e}")
            return False

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HyperbolicLearner Auto-Repair Tool")
    parser.add_argument("-y", "--yes", action="store_true", 
                       help="Auto-confirm all repair actions")
    parser.add_argument("--diagnostics-only", action="store_true",
                       help="Run diagnostics only, no repairs")
    
    args = parser.parse_args()
    
    if args.diagnostics_only:
        diagnostics = HyperbolicLearnerDiagnostics()
        health_report = diagnostics.run_full_diagnostic()
        sys.exit(0 if health_report.health_score >= 75 else 1)
    
    try:
        auto_repair = SystemAutoRepair(auto_confirm=args.yes)
        success = auto_repair.run_auto_repair()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️ Auto-repair interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Auto-repair failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

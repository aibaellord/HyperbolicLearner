#!/usr/bin/env python3
"""
HyperbolicLearner Complete Installation Script

Comprehensive installation script that handles all dependencies, system setup,
configuration, and initialization. Provides multiple installation modes and
intelligent dependency resolution.
"""

import os
import sys
import json
import subprocess
import platform
import shutil
import urllib.request
import tarfile
import zipfile
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import tempfile
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class InstallOptions:
    """Installation configuration options"""
    mode: str = "complete"  # minimal, standard, complete, development
    auto_confirm: bool = False
    include_optional: bool = True
    setup_gpu: bool = True
    create_venv: bool = True
    run_tests: bool = True
    initialize_system: bool = True
    skip_system_packages: bool = False
    force_reinstall: bool = False

class HyperbolicLearnerInstaller:
    """Complete installation manager for HyperbolicLearner"""
    
    def __init__(self, options: InstallOptions):
        self.options = options
        self.root_path = Path(__file__).parent
        self.python_executable = sys.executable
        self.platform = platform.system().lower()
        
        # Package categories
        self.package_categories = {
            "essential": [
                "pip>=21.0",
                "setuptools>=60.0",
                "wheel>=0.37.0",
                "numpy>=1.22.0",
                "psutil>=5.9.0"
            ],
            "core": [
                "opencv-python>=4.5.5",
                "flask>=2.0.0",
                "requests>=2.27.1",
                "pandas>=1.4.0",
                "tqdm>=4.62.0",
                "matplotlib>=3.5.0",
                "pillow>=9.0.0",
                "pyyaml>=6.0",
                "click>=8.0.3",
                "python-dotenv>=0.19.2",
                "rich>=12.0.0",
                "colorama>=0.4.4"
            ],
            "ml": [
                "torch>=1.10.0",
                "torchvision>=0.11.0",
                "torchaudio>=0.10.0",
                "scikit-learn>=1.0.2",
                "transformers>=4.16.0",
                "sentence-transformers>=2.2.0"
            ],
            "video_processing": [
                "moviepy>=1.0.3",
                "pytube>=12.0.0",
                "ffmpeg-python>=0.2.0",
                "pydub>=0.25.1",
                "librosa>=0.9.1",
                "imageio>=2.16.0"
            ],
            "automation": [
                "selenium>=4.1.0",
                "pyautogui>=0.9.53",
                "pynput>=1.7.6",
                "playwright>=1.19.0",
                "webdriver-manager>=3.5.2"
            ],
            "nlp": [
                "nltk>=3.7.0",
                "spacy>=3.2.0",
                "gensim>=4.1.2",
                "textblob>=0.17.1",
                "keybert>=0.5.0",
                "yake>=0.4.8"
            ],
            "database": [
                "sqlalchemy>=1.4.31",
                "networkx>=2.7.0",
                "neo4j>=4.4.0",
                "pymongo>=4.0.1",
                "sqlite3"  # Built-in
            ],
            "web": [
                "fastapi>=0.75.0",
                "uvicorn>=0.17.0",
                "websockets>=10.2",
                "aiohttp>=3.8.0",
                "jinja2>=3.0.0"
            ],
            "audio": [
                "speechrecognition>=3.8.1",
                "whisper>=1.0.0",
                "vosk>=0.3.42",
                "pyaudio"  # Platform-specific installation
            ],
            "development": [
                "pytest>=7.0.0",
                "black>=22.1.0",
                "flake8>=4.0.1",
                "mypy>=0.931",
                "isort>=5.10.1",
                "pre-commit>=2.17.0"
            ]
        }
        
        # System packages (platform-specific)
        self.system_packages = self._get_system_packages()
        
        # Installation status tracking
        self.installation_status = {
            "system_packages": {"installed": [], "failed": []},
            "python_packages": {"installed": [], "failed": []},
            "optional_packages": {"installed": [], "failed": []},
            "gpu_setup": {"status": "pending", "details": ""},
            "configuration": {"status": "pending", "details": ""},
            "tests": {"status": "pending", "details": ""}
        }
    
    def _get_system_packages(self) -> Dict[str, List[str]]:
        """Get platform-specific system packages"""
        if self.platform == "darwin":  # macOS
            return {
                "package_manager": "brew",
                "packages": [
                    "ffmpeg",
                    "portaudio",
                    "pkg-config",
                    "cmake"
                ],
                "optional": [
                    "chromedriver",
                    "geckodriver"
                ]
            }
        elif self.platform == "linux":
            return {
                "package_manager": "apt",  # Assuming Ubuntu/Debian
                "packages": [
                    "ffmpeg",
                    "portaudio19-dev",
                    "python3-dev",
                    "build-essential",
                    "pkg-config",
                    "cmake",
                    "libsm6",
                    "libxext6",
                    "libxrender-dev",
                    "libgl1-mesa-glx"
                ],
                "optional": [
                    "chromium-chromedriver",
                    "firefox-geckodriver"
                ]
            }
        elif self.platform == "windows":
            return {
                "package_manager": "winget",  # or chocolatey
                "packages": [
                    "FFmpeg",
                    "Git.Git",
                    "Microsoft.VisualStudio.2019.BuildTools"
                ],
                "optional": [
                    "Google.Chrome",
                    "Mozilla.Firefox"
                ]
            }
        else:
            return {"package_manager": None, "packages": [], "optional": []}
    
    async def run_installation(self) -> bool:
        """Run complete installation process"""
        print("🚀 HyperbolicLearner Complete Installation")
        print("=" * 60)
        print(f"Platform: {platform.platform()}")
        print(f"Python: {sys.version}")
        print(f"Installation Mode: {self.options.mode.upper()}")
        
        start_time = time.time()
        
        try:
            # Step 1: Pre-installation checks
            print("\n🔍 Step 1: Pre-installation Checks")
            if not await self._pre_installation_checks():
                return False
            
            # Step 2: Create virtual environment
            if self.options.create_venv:
                print("\n🐍 Step 2: Virtual Environment Setup")
                if not await self._setup_virtual_environment():
                    return False
            
            # Step 3: Install system packages
            if not self.options.skip_system_packages:
                print("\n📦 Step 3: System Packages")
                await self._install_system_packages()
            
            # Step 4: Install Python packages
            print("\n🐍 Step 4: Python Packages")
            if not await self._install_python_packages():
                return False
            
            # Step 5: GPU setup
            if self.options.setup_gpu:
                print("\n🔥 Step 5: GPU Setup")
                await self._setup_gpu()
            
            # Step 6: System configuration
            print("\n⚙️ Step 6: System Configuration")
            if not await self._configure_system():
                return False
            
            # Step 7: Run tests
            if self.options.run_tests:
                print("\n🧪 Step 7: System Tests")
                await self._run_tests()
            
            # Step 8: System initialization
            if self.options.initialize_system:
                print("\n🚀 Step 8: System Initialization")
                await self._initialize_system()
            
            # Installation complete
            total_time = time.time() - start_time
            self._display_installation_summary(total_time)
            
            return True
            
        except KeyboardInterrupt:
            print("\n⚠️ Installation interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Installation failed: {e}")
            logger.exception("Installation failed")
            return False
    
    async def _pre_installation_checks(self) -> bool:
        """Pre-installation system checks"""
        checks_passed = True
        
        # Python version check
        if sys.version_info < (3, 8):
            print(f"❌ Python 3.8+ required (current: {sys.version_info.major}.{sys.version_info.minor})")
            checks_passed = False
        else:
            print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        
        # Disk space check
        try:
            import shutil
            free_space = shutil.disk_usage(self.root_path).free / (1024**3)  # GB
            required_space = 5  # GB
            
            if free_space < required_space:
                print(f"❌ Insufficient disk space: {free_space:.1f}GB available, {required_space}GB required")
                checks_passed = False
            else:
                print(f"✅ Disk space: {free_space:.1f}GB available")
        except Exception as e:
            print(f"⚠️ Could not check disk space: {e}")
        
        # Internet connectivity check
        try:
            import urllib.request
            urllib.request.urlopen("https://pypi.org", timeout=10)
            print("✅ Internet connectivity: OK")
        except Exception:
            print("❌ No internet connectivity - some packages may fail to install")
            if not self.options.auto_confirm:
                response = input("Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    checks_passed = False
        
        # Write permissions check
        try:
            test_file = self.root_path / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            print("✅ Write permissions: OK")
        except Exception as e:
            print(f"❌ No write permissions: {e}")
            checks_passed = False
        
        return checks_passed
    
    async def _setup_virtual_environment(self) -> bool:
        """Set up Python virtual environment"""
        venv_path = self.root_path / ".venv"
        
        if venv_path.exists() and not self.options.force_reinstall:
            print(f"✅ Virtual environment exists: {venv_path}")
            
            # Check if it's activated
            if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                print("✅ Virtual environment is active")
            else:
                print("⚠️ Virtual environment not activated")
                print(f"   Activate with: source {venv_path}/bin/activate")
            
            return True
        
        try:
            print(f"🔧 Creating virtual environment: {venv_path}")
            
            # Remove existing venv if force reinstall
            if venv_path.exists() and self.options.force_reinstall:
                shutil.rmtree(venv_path)
            
            # Create new virtual environment
            result = subprocess.run([
                self.python_executable, "-m", "venv", str(venv_path)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Failed to create virtual environment: {result.stderr}")
                return False
            
            print("✅ Virtual environment created successfully")
            
            # Update python executable to use venv
            if self.platform == "windows":
                self.python_executable = str(venv_path / "Scripts" / "python.exe")
            else:
                self.python_executable = str(venv_path / "bin" / "python")
            
            # Upgrade pip in virtual environment
            print("🔧 Upgrading pip in virtual environment...")
            result = subprocess.run([
                self.python_executable, "-m", "pip", "install", "--upgrade", "pip"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Pip upgraded successfully")
            else:
                print(f"⚠️ Pip upgrade failed: {result.stderr}")
            
            return True
            
        except Exception as e:
            print(f"❌ Virtual environment setup failed: {e}")
            return False
    
    async def _install_system_packages(self):
        """Install system-level packages"""
        if not self.system_packages["package_manager"]:
            print("⚠️ No system package manager detected - skipping system packages")
            return
        
        pm = self.system_packages["package_manager"]
        packages = self.system_packages["packages"]
        
        if not packages:
            print("📦 No system packages required")
            return
        
        print(f"📦 Installing system packages via {pm}...")
        
        for package in packages:
            try:
                print(f"  Installing {package}...")
                
                if pm == "brew":
                    cmd = ["brew", "install", package]
                elif pm == "apt":
                    cmd = ["sudo", "apt-get", "install", "-y", package]
                elif pm == "winget":
                    cmd = ["winget", "install", package]
                else:
                    print(f"⚠️ Unknown package manager: {pm}")
                    continue
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print(f"  ✅ {package}")
                    self.installation_status["system_packages"]["installed"].append(package)
                else:
                    print(f"  ❌ {package}: {result.stderr}")
                    self.installation_status["system_packages"]["failed"].append(package)
                    
            except subprocess.TimeoutExpired:
                print(f"  ⏱️ {package}: Installation timeout")
                self.installation_status["system_packages"]["failed"].append(package)
            except Exception as e:
                print(f"  ❌ {package}: {e}")
                self.installation_status["system_packages"]["failed"].append(package)
    
    async def _install_python_packages(self) -> bool:
        """Install Python packages based on installation mode"""
        categories_to_install = self._get_package_categories_for_mode()
        
        print(f"📦 Installing Python packages for {self.options.mode} mode...")
        print(f"Categories: {', '.join(categories_to_install)}")
        
        # Install essential packages first
        if "essential" in categories_to_install:
            if not await self._install_package_category("essential"):
                print("❌ Failed to install essential packages - aborting")
                return False
        
        # Install other categories
        for category in categories_to_install:
            if category != "essential":
                await self._install_package_category(category)
        
        # Install from requirements.txt if it exists
        requirements_file = self.root_path / "requirements.txt"
        if requirements_file.exists():
            print("📦 Installing from requirements.txt...")
            result = subprocess.run([
                self.python_executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print("✅ Requirements.txt packages installed")
            else:
                print(f"⚠️ Some requirements.txt packages failed: {result.stderr}")
        
        return True
    
    def _get_package_categories_for_mode(self) -> List[str]:
        """Get package categories based on installation mode"""
        if self.options.mode == "minimal":
            return ["essential", "core"]
        elif self.options.mode == "standard":
            return ["essential", "core", "ml", "video_processing", "web"]
        elif self.options.mode == "complete":
            return ["essential", "core", "ml", "video_processing", "automation", "nlp", "database", "web"]
        elif self.options.mode == "development":
            return ["essential", "core", "ml", "video_processing", "automation", "nlp", "database", "web", "development"]
        else:
            return ["essential", "core"]
    
    async def _install_package_category(self, category: str) -> bool:
        """Install packages from a specific category"""
        packages = self.package_categories.get(category, [])
        if not packages:
            return True
        
        print(f"\n  📂 Installing {category} packages...")
        
        success_count = 0
        
        for package in packages:
            try:
                print(f"    Installing {package}...")
                
                # Handle special cases
                if package == "sqlite3":
                    print(f"    ✅ {package} (built-in)")
                    success_count += 1
                    continue
                
                if package == "pyaudio":
                    # Platform-specific pyaudio installation
                    if not await self._install_pyaudio():
                        self.installation_status["python_packages"]["failed"].append(package)
                        continue
                    else:
                        success_count += 1
                        continue
                
                # Standard pip installation
                result = subprocess.run([
                    self.python_executable, "-m", "pip", "install", package
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print(f"    ✅ {package}")
                    self.installation_status["python_packages"]["installed"].append(package)
                    success_count += 1
                else:
                    print(f"    ❌ {package}: {result.stderr}")
                    self.installation_status["python_packages"]["failed"].append(package)
                    
            except subprocess.TimeoutExpired:
                print(f"    ⏱️ {package}: Installation timeout")
                self.installation_status["python_packages"]["failed"].append(package)
            except Exception as e:
                print(f"    ❌ {package}: {e}")
                self.installation_status["python_packages"]["failed"].append(package)
        
        success_rate = success_count / len(packages)
        print(f"  📊 {category}: {success_count}/{len(packages)} packages installed ({success_rate:.1%})")
        
        return success_rate > 0.5  # At least 50% success rate required for essential packages
    
    async def _install_pyaudio(self) -> bool:
        """Install pyaudio with platform-specific handling"""
        try:
            if self.platform == "darwin":
                # macOS: Install via pip (should work if portaudio is installed)
                result = subprocess.run([
                    self.python_executable, "-m", "pip", "install", "pyaudio"
                ], capture_output=True, text=True, timeout=180)
                
                return result.returncode == 0
                
            elif self.platform == "linux":
                # Linux: Install via pip (should work if portaudio19-dev is installed)
                result = subprocess.run([
                    self.python_executable, "-m", "pip", "install", "pyaudio"
                ], capture_output=True, text=True, timeout=180)
                
                return result.returncode == 0
                
            elif self.platform == "windows":
                # Windows: Try wheel first, then pip
                result = subprocess.run([
                    self.python_executable, "-m", "pip", "install", "pipwin"
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    result = subprocess.run([
                        self.python_executable, "-m", "pipwin", "install", "pyaudio"
                    ], capture_output=True, text=True, timeout=180)
                    
                    if result.returncode == 0:
                        return True
                
                # Fallback to regular pip
                result = subprocess.run([
                    self.python_executable, "-m", "pip", "install", "pyaudio"
                ], capture_output=True, text=True, timeout=180)
                
                return result.returncode == 0
            
        except Exception as e:
            print(f"    ❌ PyAudio installation failed: {e}")
            
        return False
    
    async def _setup_gpu(self):
        """Set up GPU acceleration if available"""
        print("🔥 Checking for GPU support...")
        
        # Check for NVIDIA GPU
        nvidia_available = False
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ NVIDIA GPU detected")
                nvidia_available = True
            else:
                print("ℹ️ No NVIDIA GPU detected")
        except FileNotFoundError:
            print("ℹ️ nvidia-smi not found - no NVIDIA GPU")
        
        # Install appropriate PyTorch version
        try:
            if nvidia_available:
                print("🔧 Installing PyTorch with CUDA support...")
                result = subprocess.run([
                    self.python_executable, "-m", "pip", "install",
                    "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu118"
                ], capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0:
                    print("✅ PyTorch with CUDA installed successfully")
                    self.installation_status["gpu_setup"]["status"] = "completed"
                    self.installation_status["gpu_setup"]["details"] = "CUDA-enabled PyTorch"
                else:
                    print(f"⚠️ CUDA PyTorch installation failed: {result.stderr}")
                    print("🔧 Installing CPU-only PyTorch as fallback...")
                    await self._install_cpu_pytorch()
            else:
                print("🔧 Installing CPU-only PyTorch...")
                await self._install_cpu_pytorch()
                
        except Exception as e:
            print(f"❌ GPU setup failed: {e}")
            self.installation_status["gpu_setup"]["status"] = "failed"
            self.installation_status["gpu_setup"]["details"] = str(e)
    
    async def _install_cpu_pytorch(self):
        """Install CPU-only PyTorch"""
        result = subprocess.run([
            self.python_executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ CPU-only PyTorch installed successfully")
            self.installation_status["gpu_setup"]["status"] = "completed"
            self.installation_status["gpu_setup"]["details"] = "CPU-only PyTorch"
        else:
            print(f"❌ CPU PyTorch installation failed: {result.stderr}")
            self.installation_status["gpu_setup"]["status"] = "failed"
    
    async def _configure_system(self) -> bool:
        """Configure system settings and initialize components"""
        try:
            # Initialize configuration manager
            print("⚙️ Initializing configuration manager...")
            from config_manager import ConfigurationManager
            config_manager = ConfigurationManager(str(self.root_path))
            
            # This will create default configuration based on system capabilities
            print("✅ Configuration manager initialized")
            
            # Set up directories
            print("📁 Creating system directories...")
            directories = [
                "learning_data",
                "optimization_cache", 
                "maximum_potential_data",
                "temp",
                "cache",
                "logs"
            ]
            
            for directory in directories:
                dir_path = self.root_path / directory
                dir_path.mkdir(exist_ok=True, parents=True)
                print(f"  ✅ {directory}/")
            
            # Initialize databases
            print("🗄️ Initializing databases...")
            from system_auto_repair import SystemAutoRepair
            auto_repair = SystemAutoRepair(auto_confirm=True)
            
            # Initialize databases through auto-repair
            success = auto_repair.initialize_databases()
            if success:
                print("✅ Databases initialized")
            else:
                print("⚠️ Database initialization had some issues")
            
            self.installation_status["configuration"]["status"] = "completed"
            self.installation_status["configuration"]["details"] = "System configured"
            
            return True
            
        except Exception as e:
            print(f"❌ System configuration failed: {e}")
            self.installation_status["configuration"]["status"] = "failed"
            self.installation_status["configuration"]["details"] = str(e)
            return False
    
    async def _run_tests(self):
        """Run system tests to verify installation"""
        print("🧪 Running system tests...")
        
        test_results = []
        
        # Test 1: Import test
        print("  Testing core imports...")
        try:
            import numpy
            import cv2
            import flask
            import requests
            print("    ✅ Core imports successful")
            test_results.append(("Core imports", True))
        except ImportError as e:
            print(f"    ❌ Core import failed: {e}")
            test_results.append(("Core imports", False))
        
        # Test 2: PyTorch test
        print("  Testing PyTorch...")
        try:
            import torch
            print(f"    ✅ PyTorch {torch.__version__}")
            if torch.cuda.is_available():
                print(f"    ✅ CUDA available: {torch.cuda.device_count()} device(s)")
            else:
                print("    ℹ️ CUDA not available (CPU only)")
            test_results.append(("PyTorch", True))
        except ImportError as e:
            print(f"    ❌ PyTorch test failed: {e}")
            test_results.append(("PyTorch", False))
        
        # Test 3: Configuration test
        print("  Testing configuration system...")
        try:
            from config_manager import ConfigurationManager
            config = ConfigurationManager(str(self.root_path))
            print("    ✅ Configuration system working")
            test_results.append(("Configuration", True))
        except Exception as e:
            print(f"    ❌ Configuration test failed: {e}")
            test_results.append(("Configuration", False))
        
        # Test 4: Database test
        print("  Testing database connectivity...")
        try:
            import sqlite3
            db_path = self.root_path / "maximum_potential_data" / "maximum_potential.db"
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                conn.execute("SELECT 1")
                conn.close()
                print("    ✅ Database connectivity working")
                test_results.append(("Database", True))
            else:
                print("    ⚠️ Database not found - may need initialization")
                test_results.append(("Database", False))
        except Exception as e:
            print(f"    ❌ Database test failed: {e}")
            test_results.append(("Database", False))
        
        # Test summary
        passed_tests = len([t for t in test_results if t[1]])
        total_tests = len(test_results)
        success_rate = (passed_tests / total_tests) * 100
        
        print(f"\n🧪 Test Results: {passed_tests}/{total_tests} passed ({success_rate:.1f}%)")
        
        self.installation_status["tests"]["status"] = "completed"
        self.installation_status["tests"]["details"] = f"{passed_tests}/{total_tests} tests passed"
        
        if success_rate < 50:
            print("⚠️ Many tests failed - installation may have issues")
    
    async def _initialize_system(self):
        """Initialize the HyperbolicLearner system"""
        print("🚀 Initializing HyperbolicLearner system...")
        
        try:
            # Run system diagnostics
            print("  Running system diagnostics...")
            from system_diagnostics import HyperbolicLearnerDiagnostics
            
            diagnostics = HyperbolicLearnerDiagnostics()
            health_report = diagnostics.run_full_diagnostic()
            
            print(f"  📊 System Health Score: {health_report.health_score:.1f}%")
            
            if health_report.health_score >= 75:
                print("  ✅ System is healthy and ready")
            else:
                print("  ⚠️ System has some issues - may need manual fixes")
            
            print("✅ System initialization completed")
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
    
    def _display_installation_summary(self, total_time: float):
        """Display installation summary"""
        print(f"\n🎉 INSTALLATION COMPLETE")
        print("=" * 60)
        print(f"Total Installation Time: {total_time:.1f} seconds")
        
        # Package summary
        installed_python = len(self.installation_status["python_packages"]["installed"])
        failed_python = len(self.installation_status["python_packages"]["failed"])
        total_python = installed_python + failed_python
        
        installed_system = len(self.installation_status["system_packages"]["installed"])
        failed_system = len(self.installation_status["system_packages"]["failed"])
        
        print(f"\n📦 Package Installation:")
        print(f"  Python Packages: {installed_python}/{total_python} installed")
        print(f"  System Packages: {installed_system} installed, {failed_system} failed")
        
        # Component status
        print(f"\n🔧 Component Status:")
        for component, status in self.installation_status.items():
            if isinstance(status, dict) and "status" in status:
                component_name = component.replace("_", " ").title()
                status_emoji = {"completed": "✅", "failed": "❌", "pending": "⏳"}
                emoji = status_emoji.get(status["status"], "❓")
                print(f"  {emoji} {component_name}: {status['status'].title()}")
        
        # Next steps
        print(f"\n🎯 Next Steps:")
        print(f"  1. Activate virtual environment: source .venv/bin/activate")
        print(f"  2. Run system diagnostics: python system_diagnostics.py")
        print(f"  3. Launch system: python system_launcher.py")
        print(f"  4. Access web interface: python hyperbolic_web_ui.py")
        
        # Troubleshooting
        if failed_python > 0 or failed_system > 0:
            print(f"\n🛠️ Troubleshooting:")
            print(f"  • Run auto-repair: python system_auto_repair.py")
            print(f"  • Check installation logs above for specific errors")
            print(f"  • Some optional packages may fail on certain systems")

def main():
    """Main installation entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HyperbolicLearner Complete Installation")
    parser.add_argument("--mode", choices=["minimal", "standard", "complete", "development"],
                       default="complete", help="Installation mode")
    parser.add_argument("-y", "--yes", action="store_true", help="Auto-confirm all prompts")
    parser.add_argument("--no-optional", action="store_true", help="Skip optional packages")
    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU setup")
    parser.add_argument("--no-venv", action="store_true", help="Skip virtual environment creation")
    parser.add_argument("--no-tests", action="store_true", help="Skip test execution")
    parser.add_argument("--no-init", action="store_true", help="Skip system initialization")
    parser.add_argument("--skip-system", action="store_true", help="Skip system packages")
    parser.add_argument("--force", action="store_true", help="Force reinstall existing components")
    
    args = parser.parse_args()
    
    # Create installation options
    options = InstallOptions(
        mode=args.mode,
        auto_confirm=args.yes,
        include_optional=not args.no_optional,
        setup_gpu=not args.no_gpu,
        create_venv=not args.no_venv,
        run_tests=not args.no_tests,
        initialize_system=not args.no_init,
        skip_system_packages=args.skip_system,
        force_reinstall=args.force
    )
    
    try:
        installer = HyperbolicLearnerInstaller(options)
        import asyncio
        success = asyncio.run(installer.run_installation())
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️ Installation interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

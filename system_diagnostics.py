#!/usr/bin/env python3
"""
HyperbolicLearner System Diagnostics & Health Monitor

This comprehensive diagnostic tool checks all system components, dependencies,
configurations, and provides detailed health reports with actionable recommendations.
"""

import os
import sys
import json
import time
import psutil
import sqlite3
import subprocess
import platform
import importlib
import traceback
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import pkg_resources

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ComponentStatus:
    """Status of a system component"""
    name: str
    status: str  # "OK", "WARNING", "ERROR", "MISSING"
    version: Optional[str] = None
    message: str = ""
    details: Dict[str, Any] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.recommendations is None:
            self.recommendations = []

@dataclass
class SystemHealth:
    """Overall system health report"""
    timestamp: str
    overall_status: str
    health_score: float  # 0-100
    components: List[ComponentStatus]
    system_info: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]
    
class HyperbolicLearnerDiagnostics:
    """Comprehensive system diagnostics for HyperbolicLearner"""
    
    def __init__(self):
        self.root_path = Path(__file__).parent
        self.results = []
        
    def run_full_diagnostic(self) -> SystemHealth:
        """Run complete system diagnostic"""
        print("🔍 Running HyperbolicLearner System Diagnostics...")
        print("=" * 60)
        
        start_time = time.time()
        
        # Core system checks
        self.check_python_environment()
        self.check_virtual_environment()
        self.check_core_dependencies()
        self.check_optional_dependencies()
        self.check_gpu_acceleration()
        self.check_databases()
        self.check_configuration_files()
        self.check_system_resources()
        self.check_file_permissions()
        self.check_network_connectivity()
        self.check_storage_space()
        self.run_component_tests()
        
        # Generate health report
        health_score = self.calculate_health_score()
        overall_status = self.determine_overall_status(health_score)
        
        system_health = SystemHealth(
            timestamp=datetime.now().isoformat(),
            overall_status=overall_status,
            health_score=health_score,
            components=self.results,
            system_info=self.get_system_info(),
            performance_metrics=self.get_performance_metrics(),
            recommendations=self.generate_recommendations()
        )
        
        # Display results
        self.display_results(system_health, time.time() - start_time)
        
        # Save detailed report
        self.save_diagnostic_report(system_health)
        
        return system_health
    
    def check_python_environment(self):
        """Check Python version and environment"""
        try:
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            
            if sys.version_info >= (3, 8):
                status = "OK"
                message = f"Python {python_version} is compatible"
                recommendations = []
            else:
                status = "ERROR"
                message = f"Python {python_version} is too old (requires 3.8+)"
                recommendations = ["Upgrade to Python 3.8 or newer"]
                
        except Exception as e:
            status = "ERROR"
            message = f"Failed to check Python version: {str(e)}"
            recommendations = ["Reinstall Python"]
            
        self.results.append(ComponentStatus(
            name="Python Environment",
            status=status,
            version=python_version if 'python_version' in locals() else "Unknown",
            message=message,
            details={"executable": sys.executable, "platform": platform.platform()},
            recommendations=recommendations
        ))
    
    def check_virtual_environment(self):
        """Check virtual environment status"""
        try:
            venv_path = self.root_path / ".venv"
            
            if venv_path.exists():
                # Check if we're in the virtual environment
                in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
                
                if in_venv:
                    status = "OK"
                    message = "Virtual environment is active"
                    recommendations = []
                else:
                    status = "WARNING"
                    message = "Virtual environment exists but not activated"
                    recommendations = ["Activate virtual environment: source .venv/bin/activate"]
            else:
                status = "ERROR"
                message = "Virtual environment not found"
                recommendations = ["Create virtual environment: python -m venv .venv"]
                
        except Exception as e:
            status = "ERROR"
            message = f"Failed to check virtual environment: {str(e)}"
            recommendations = ["Create new virtual environment"]
            
        self.results.append(ComponentStatus(
            name="Virtual Environment",
            status=status,
            message=message,
            details={"venv_path": str(venv_path), "in_venv": in_venv if 'in_venv' in locals() else False},
            recommendations=recommendations
        ))
    
    def check_core_dependencies(self):
        """Check essential dependencies"""
        core_packages = {
            'numpy': '1.22.0',
            'opencv-python': '4.5.5',
            'torch': '1.10.0',
            'flask': None,
            'pytube': '12.0.0',
            'sqlite3': None,  # Built-in
            'requests': '2.27.1',
            'pandas': '1.4.0'
        }
        
        missing_packages = []
        outdated_packages = []
        working_packages = []
        
        for package, min_version in core_packages.items():
            try:
                if package == 'sqlite3':
                    import sqlite3
                    working_packages.append(f"sqlite3 (built-in)")
                    continue
                    
                # Try to import and check version
                module = importlib.import_module(package.replace('-', '_'))
                
                if hasattr(module, '__version__'):
                    version = module.__version__
                    if min_version and pkg_resources.parse_version(version) < pkg_resources.parse_version(min_version):
                        outdated_packages.append(f"{package} {version} (requires {min_version}+)")
                    else:
                        working_packages.append(f"{package} {version}")
                else:
                    working_packages.append(f"{package} (version unknown)")
                    
            except ImportError:
                missing_packages.append(package)
            except Exception as e:
                missing_packages.append(f"{package} (error: {str(e)})")
        
        if not missing_packages and not outdated_packages:
            status = "OK"
            message = f"All {len(working_packages)} core dependencies available"
            recommendations = []
        elif outdated_packages:
            status = "WARNING"
            message = f"Some dependencies need updating"
            recommendations = [f"Update packages: pip install --upgrade {' '.join([p.split()[0] for p in outdated_packages])}"]
        else:
            status = "ERROR"
            message = f"Missing critical dependencies: {len(missing_packages)}"
            recommendations = ["Install missing packages: pip install -r requirements.txt"]
        
        self.results.append(ComponentStatus(
            name="Core Dependencies",
            status=status,
            message=message,
            details={
                "working": working_packages,
                "missing": missing_packages,
                "outdated": outdated_packages
            },
            recommendations=recommendations
        ))
    
    def check_optional_dependencies(self):
        """Check optional but recommended dependencies"""
        optional_packages = {
            'transformers': 'HuggingFace models',
            'scikit-learn': 'Machine learning algorithms',
            'librosa': 'Audio processing',
            'spacy': 'Advanced NLP',
            'networkx': 'Knowledge graphs',
            'selenium': 'Web automation',
            'pyautogui': 'UI automation',
            'neo4j': 'Graph database',
            'whisper': 'Speech recognition'
        }
        
        available = []
        missing = []
        
        for package, description in optional_packages.items():
            try:
                module = importlib.import_module(package.replace('-', '_'))
                version = getattr(module, '__version__', 'unknown')
                available.append(f"{package} {version} ({description})")
            except ImportError:
                missing.append(f"{package} ({description})")
        
        if len(available) >= len(optional_packages) * 0.8:  # 80%+ available
            status = "OK"
            message = f"Most optional dependencies available ({len(available)}/{len(optional_packages)})"
            recommendations = []
        elif len(available) >= len(optional_packages) * 0.5:  # 50%+ available
            status = "WARNING"
            message = f"Some optional features may be limited"
            recommendations = ["Consider installing missing packages for full functionality"]
        else:
            status = "WARNING"
            message = f"Many optional dependencies missing"
            recommendations = ["Install additional packages: pip install -r requirements.txt"]
        
        self.results.append(ComponentStatus(
            name="Optional Dependencies",
            status=status,
            message=message,
            details={"available": available, "missing": missing},
            recommendations=recommendations
        ))
    
    def check_gpu_acceleration(self):
        """Check GPU and CUDA availability"""
        gpu_info = {"cuda_available": False, "gpu_count": 0, "gpu_memory": []}
        
        try:
            import torch
            gpu_info["cuda_available"] = torch.cuda.is_available()
            gpu_info["gpu_count"] = torch.cuda.device_count()
            
            if gpu_info["cuda_available"]:
                for i in range(gpu_info["gpu_count"]):
                    gpu_name = torch.cuda.get_device_name(i)
                    total_memory = torch.cuda.get_device_properties(i).total_memory // (1024**3)  # GB
                    gpu_info["gpu_memory"].append({"name": gpu_name, "memory_gb": total_memory})
                    
                status = "OK"
                message = f"CUDA available with {gpu_info['gpu_count']} GPU(s)"
                recommendations = ["GPU acceleration is ready to use"]
            else:
                status = "WARNING"
                message = "CUDA not available - using CPU only"
                recommendations = ["Install CUDA drivers for GPU acceleration", "Performance will be limited to CPU"]
                
        except ImportError:
            status = "WARNING"
            message = "PyTorch not available - cannot check GPU"
            recommendations = ["Install PyTorch to enable GPU acceleration"]
        except Exception as e:
            status = "ERROR"
            message = f"GPU check failed: {str(e)}"
            recommendations = ["Check CUDA installation"]
        
        self.results.append(ComponentStatus(
            name="GPU Acceleration",
            status=status,
            message=message,
            details=gpu_info,
            recommendations=recommendations
        ))
    
    def check_databases(self):
        """Check database files and schemas"""
        databases = [
            "maximum_potential_data/maximum_potential.db",
            "UltimateCryptoArbitrageEngine/transcendent_arbitrage.db"
        ]
        
        db_status = []
        working_dbs = 0
        
        for db_path in databases:
            full_path = self.root_path / db_path
            
            if full_path.exists():
                try:
                    conn = sqlite3.connect(str(full_path))
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    conn.close()
                    
                    db_status.append({
                        "path": db_path,
                        "status": "OK",
                        "tables": len(tables),
                        "size_kb": full_path.stat().st_size // 1024
                    })
                    working_dbs += 1
                    
                except Exception as e:
                    db_status.append({
                        "path": db_path,
                        "status": "ERROR",
                        "error": str(e)
                    })
            else:
                db_status.append({
                    "path": db_path,
                    "status": "MISSING"
                })
        
        if working_dbs == len(databases):
            status = "OK"
            message = f"All {working_dbs} databases operational"
            recommendations = []
        elif working_dbs > 0:
            status = "WARNING"
            message = f"{working_dbs}/{len(databases)} databases working"
            recommendations = ["Initialize missing databases"]
        else:
            status = "ERROR"
            message = "No databases found"
            recommendations = ["Run system initialization to create databases"]
        
        self.results.append(ComponentStatus(
            name="Database Systems",
            status=status,
            message=message,
            details={"databases": db_status},
            recommendations=recommendations
        ))
    
    def check_configuration_files(self):
        """Check configuration files"""
        config_files = [
            "learning_data/learning_config.json",
            "optimization_cache/optimization_config.json",
            "UltimateCryptoArbitrageEngine/.env.example"
        ]
        
        config_status = []
        working_configs = 0
        
        for config_path in config_files:
            full_path = self.root_path / config_path
            
            if full_path.exists():
                try:
                    if config_path.endswith('.json'):
                        with open(full_path, 'r') as f:
                            config_data = json.load(f)
                        config_status.append({
                            "path": config_path,
                            "status": "OK",
                            "keys": len(config_data) if isinstance(config_data, dict) else 0
                        })
                    else:
                        # .env file
                        with open(full_path, 'r') as f:
                            lines = len(f.readlines())
                        config_status.append({
                            "path": config_path,
                            "status": "OK",
                            "lines": lines
                        })
                    working_configs += 1
                    
                except Exception as e:
                    config_status.append({
                        "path": config_path,
                        "status": "ERROR",
                        "error": str(e)
                    })
            else:
                config_status.append({
                    "path": config_path,
                    "status": "MISSING"
                })
        
        if working_configs == len(config_files):
            status = "OK"
            message = f"All configuration files present"
            recommendations = []
        elif working_configs > 0:
            status = "WARNING"
            message = f"Some configuration files missing"
            recommendations = ["Create missing configuration files"]
        else:
            status = "ERROR"
            message = "Configuration files not found"
            recommendations = ["Initialize system configuration"]
        
        self.results.append(ComponentStatus(
            name="Configuration Files",
            status=status,
            message=message,
            details={"configs": config_status},
            recommendations=recommendations
        ))
    
    def check_system_resources(self):
        """Check system resources (RAM, CPU, disk)"""
        try:
            # Memory check
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            memory_percent = memory.percent
            
            # CPU check
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Disk check
            disk = psutil.disk_usage(str(self.root_path))
            disk_free_gb = disk.free / (1024**3)
            disk_used_percent = (disk.used / disk.total) * 100
            
            # Determine status
            issues = []
            if memory_gb < 4:
                issues.append("Low RAM (recommended: 8GB+)")
            if memory_percent > 90:
                issues.append("High memory usage")
            if cpu_percent > 95:
                issues.append("High CPU usage")
            if disk_free_gb < 5:
                issues.append("Low disk space")
                
            if not issues:
                status = "OK"
                message = "System resources sufficient"
                recommendations = []
            else:
                status = "WARNING"
                message = f"Resource concerns: {', '.join(issues)}"
                recommendations = ["Monitor resource usage", "Consider upgrading hardware if needed"]
            
            details = {
                "memory_total_gb": round(memory_gb, 1),
                "memory_available_gb": round(memory_available_gb, 1),
                "memory_usage_percent": memory_percent,
                "cpu_count": cpu_count,
                "cpu_usage_percent": cpu_percent,
                "disk_free_gb": round(disk_free_gb, 1),
                "disk_usage_percent": round(disk_used_percent, 1)
            }
            
        except Exception as e:
            status = "ERROR"
            message = f"Failed to check system resources: {str(e)}"
            details = {}
            recommendations = ["Install psutil for resource monitoring"]
        
        self.results.append(ComponentStatus(
            name="System Resources",
            status=status,
            message=message,
            details=details,
            recommendations=recommendations
        ))
    
    def check_file_permissions(self):
        """Check file permissions for critical directories"""
        critical_paths = [
            "src",
            "learning_data",
            "optimization_cache",
            "maximum_potential_data"
        ]
        
        permission_issues = []
        
        for path_name in critical_paths:
            full_path = self.root_path / path_name
            
            if full_path.exists():
                if not os.access(str(full_path), os.R_OK):
                    permission_issues.append(f"{path_name}: No read permission")
                if not os.access(str(full_path), os.W_OK):
                    permission_issues.append(f"{path_name}: No write permission")
            else:
                permission_issues.append(f"{path_name}: Path does not exist")
        
        if not permission_issues:
            status = "OK"
            message = "File permissions OK"
            recommendations = []
        else:
            status = "ERROR"
            message = f"Permission issues found"
            recommendations = ["Fix file permissions: chmod -R 755 <directory>"]
        
        self.results.append(ComponentStatus(
            name="File Permissions",
            status=status,
            message=message,
            details={"issues": permission_issues},
            recommendations=recommendations
        ))
    
    def check_network_connectivity(self):
        """Check network connectivity for downloading videos"""
        try:
            import requests
            
            test_urls = [
                "https://www.youtube.com",
                "https://www.google.com",
                "https://pypi.org"
            ]
            
            successful_connections = 0
            
            for url in test_urls:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        successful_connections += 1
                except:
                    pass
            
            if successful_connections == len(test_urls):
                status = "OK"
                message = "Network connectivity OK"
                recommendations = []
            elif successful_connections > 0:
                status = "WARNING"
                message = f"Limited connectivity ({successful_connections}/{len(test_urls)})"
                recommendations = ["Check firewall and proxy settings"]
            else:
                status = "ERROR"
                message = "No network connectivity"
                recommendations = ["Check internet connection"]
                
        except ImportError:
            status = "WARNING"
            message = "Cannot test connectivity (requests not installed)"
            recommendations = ["Install requests library"]
        
        self.results.append(ComponentStatus(
            name="Network Connectivity",
            status=status,
            message=message,
            recommendations=recommendations
        ))
    
    def check_storage_space(self):
        """Check available storage space for video processing"""
        try:
            disk = psutil.disk_usage(str(self.root_path))
            free_gb = disk.free / (1024**3)
            
            if free_gb >= 20:
                status = "OK"
                message = f"Sufficient storage ({free_gb:.1f}GB free)"
                recommendations = []
            elif free_gb >= 5:
                status = "WARNING" 
                message = f"Limited storage ({free_gb:.1f}GB free)"
                recommendations = ["Consider cleaning up old files"]
            else:
                status = "ERROR"
                message = f"Low storage ({free_gb:.1f}GB free)"
                recommendations = ["Free up disk space immediately"]
                
        except Exception as e:
            status = "ERROR"
            message = f"Cannot check storage: {str(e)}"
            recommendations = ["Check disk space manually"]
        
        self.results.append(ComponentStatus(
            name="Storage Space",
            status=status,
            message=message,
            recommendations=recommendations
        ))
    
    def run_component_tests(self):
        """Run basic component functionality tests"""
        test_results = []
        
        # Test video processor import
        try:
            from src.video_processor.downloader import VideoDownloader
            test_results.append("VideoDownloader: OK")
        except Exception as e:
            test_results.append(f"VideoDownloader: FAIL - {str(e)}")
        
        # Test ML engine import
        try:
            from src.ml_engine.content_analyzer import ContentAnalyzer
            test_results.append("ContentAnalyzer: OK")
        except Exception as e:
            test_results.append(f"ContentAnalyzer: FAIL - {str(e)}")
        
        # Test core system import
        try:
            from src.core.config import SystemConfig
            test_results.append("SystemConfig: OK")
        except Exception as e:
            test_results.append(f"SystemConfig: FAIL - {str(e)}")
        
        # Test web interface
        try:
            import flask
            test_results.append("Flask WebUI: OK")
        except Exception as e:
            test_results.append(f"Flask WebUI: FAIL - {str(e)}")
        
        working_components = len([r for r in test_results if "OK" in r])
        total_components = len(test_results)
        
        if working_components == total_components:
            status = "OK"
            message = f"All {total_components} core components importable"
            recommendations = []
        elif working_components > 0:
            status = "WARNING"
            message = f"Some components have issues ({working_components}/{total_components})"
            recommendations = ["Check import errors and missing dependencies"]
        else:
            status = "ERROR"
            message = "Core components not working"
            recommendations = ["Reinstall system dependencies"]
        
        self.results.append(ComponentStatus(
            name="Component Tests",
            status=status,
            message=message,
            details={"test_results": test_results},
            recommendations=recommendations
        ))
    
    def calculate_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        if not self.results:
            return 0.0
        
        score = 0
        for component in self.results:
            if component.status == "OK":
                score += 10
            elif component.status == "WARNING":
                score += 6
            elif component.status == "ERROR":
                score += 2
            # MISSING gets 0 points
        
        max_possible = len(self.results) * 10
        return (score / max_possible) * 100 if max_possible > 0 else 0
    
    def determine_overall_status(self, health_score: float) -> str:
        """Determine overall system status from health score"""
        if health_score >= 90:
            return "EXCELLENT"
        elif health_score >= 75:
            return "GOOD"
        elif health_score >= 60:
            return "WARNING"
        elif health_score >= 40:
            return "CRITICAL"
        else:
            return "FAILING"
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get detailed system information"""
        return {
            "platform": platform.platform(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "hostname": platform.node(),
            "working_directory": str(self.root_path)
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        try:
            memory = psutil.virtual_memory()
            return {
                "memory_usage_percent": memory.percent,
                "cpu_count": psutil.cpu_count(),
                "cpu_usage_percent": psutil.cpu_percent(),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            }
        except:
            return {}
    
    def generate_recommendations(self) -> List[str]:
        """Generate system-wide recommendations"""
        recommendations = []
        
        # Collect all component recommendations
        for component in self.results:
            recommendations.extend(component.recommendations)
        
        # Add general recommendations
        error_count = len([c for c in self.results if c.status == "ERROR"])
        warning_count = len([c for c in self.results if c.status == "WARNING"])
        
        if error_count > 0:
            recommendations.append("Address critical errors before using the system")
        
        if warning_count > 0:
            recommendations.append("Resolve warnings for optimal performance")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def display_results(self, health: SystemHealth, duration: float):
        """Display diagnostic results in a formatted way"""
        print(f"\n🏥 SYSTEM HEALTH REPORT")
        print("=" * 60)
        print(f"Overall Status: {health.overall_status}")
        print(f"Health Score: {health.health_score:.1f}/100")
        print(f"Diagnostic Duration: {duration:.2f} seconds")
        print(f"Components Checked: {len(health.components)}")
        
        # Status summary
        status_counts = {"OK": 0, "WARNING": 0, "ERROR": 0, "MISSING": 0}
        for component in health.components:
            status_counts[component.status] += 1
        
        print(f"\nStatus Summary:")
        for status, count in status_counts.items():
            if count > 0:
                emoji = {"OK": "✅", "WARNING": "⚠️", "ERROR": "❌", "MISSING": "❓"}[status]
                print(f"  {emoji} {status}: {count}")
        
        # Component details
        print(f"\n📊 COMPONENT STATUS:")
        print("-" * 60)
        
        for component in health.components:
            status_emoji = {"OK": "✅", "WARNING": "⚠️", "ERROR": "❌", "MISSING": "❓"}[component.status]
            print(f"{status_emoji} {component.name}: {component.message}")
            
            if component.recommendations:
                for rec in component.recommendations[:2]:  # Show top 2 recommendations
                    print(f"    💡 {rec}")
        
        # System recommendations
        if health.recommendations:
            print(f"\n🎯 TOP RECOMMENDATIONS:")
            print("-" * 60)
            for i, rec in enumerate(health.recommendations[:5], 1):  # Top 5
                print(f"{i}. {rec}")
        
        print(f"\n📋 Detailed report saved to: system_diagnostic_report.json")
    
    def save_diagnostic_report(self, health: SystemHealth):
        """Save detailed diagnostic report to JSON file"""
        report_path = self.root_path / "system_diagnostic_report.json"
        
        try:
            with open(report_path, 'w') as f:
                json.dump(asdict(health), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save diagnostic report: {e}")

def main():
    """Main entry point for diagnostics"""
    try:
        diagnostics = HyperbolicLearnerDiagnostics()
        health_report = diagnostics.run_full_diagnostic()
        
        # Return appropriate exit code
        if health_report.health_score >= 75:
            sys.exit(0)  # Success
        elif health_report.health_score >= 50:
            sys.exit(1)  # Warning
        else:
            sys.exit(2)  # Critical issues
            
    except Exception as e:
        logger.error(f"Diagnostic failed: {e}")
        traceback.print_exc()
        sys.exit(3)  # Diagnostic failure

if __name__ == "__main__":
    main()

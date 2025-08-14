#!/usr/bin/env python3
"""
HyperbolicLearner Quick Status Tool

Provides instant system health overview, component status, and key metrics.
"""

import os
import sys
import json
import time
import sqlite3
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

def get_system_overview() -> Dict[str, Any]:
    """Get comprehensive system overview"""
    root_path = Path(__file__).parent
    
    # Basic system info
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(str(root_path))
        
        system_info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_total_gb": round(memory.total / (1024**3), 1),
            "memory_usage_percent": memory.percent,
            "disk_free_gb": round(disk.free / (1024**3), 1),
            "disk_usage_percent": round((disk.used / disk.total) * 100, 1)
        }
    except Exception:
        system_info = {"error": "Could not gather system info"}
    
    # GPU status
    gpu_status = {"available": False}
    try:
        import torch
        gpu_status = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        if gpu_status["available"]:
            gpu_status["device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        gpu_status["note"] = "PyTorch not installed"
    
    return {
        "system": system_info,
        "gpu": gpu_status
    }

def check_dependencies() -> Dict[str, Any]:
    """Check critical dependencies"""
    dependencies = {
        "core": [],
        "optional": [],
        "missing": []
    }
    
    # Core dependencies
    core_deps = ['numpy', 'opencv-python', 'flask', 'requests', 'pandas', 'psutil']
    for dep in core_deps:
        try:
            __import__(dep.replace('-', '_'))
            dependencies["core"].append(dep)
        except ImportError:
            dependencies["missing"].append(dep)
    
    # Optional dependencies
    optional_deps = ['torch', 'transformers', 'librosa', 'selenium', 'networkx']
    for dep in optional_deps:
        try:
            __import__(dep)
            dependencies["optional"].append(dep)
        except ImportError:
            pass
    
    return dependencies

def check_databases() -> Dict[str, Any]:
    """Check database status"""
    root_path = Path(__file__).parent
    databases = {}
    
    db_paths = [
        ("main", "maximum_potential_data/maximum_potential.db"),
        ("crypto", "UltimateCryptoArbitrageEngine/transcendent_arbitrage.db")
    ]
    
    for name, rel_path in db_paths:
        db_path = root_path / rel_path
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                conn.close()
                
                databases[name] = {
                    "status": "OK",
                    "tables": len(tables),
                    "size_kb": round(db_path.stat().st_size / 1024, 1)
                }
            except Exception as e:
                databases[name] = {"status": "ERROR", "error": str(e)}
        else:
            databases[name] = {"status": "MISSING"}
    
    return databases

def check_configuration() -> Dict[str, Any]:
    """Check configuration files"""
    root_path = Path(__file__).parent
    configs = {}
    
    config_files = [
        ("learning", "learning_data/learning_config.json"),
        ("optimization", "optimization_cache/optimization_config.json"),
        ("system", "system_config.json")
    ]
    
    for name, rel_path in config_files:
        config_path = root_path / rel_path
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                configs[name] = {
                    "status": "OK",
                    "keys": len(config_data) if isinstance(config_data, dict) else 0,
                    "size_kb": round(config_path.stat().st_size / 1024, 1)
                }
            except Exception as e:
                configs[name] = {"status": "ERROR", "error": str(e)}
        else:
            configs[name] = {"status": "MISSING"}
    
    return configs

def check_components() -> Dict[str, Any]:
    """Test component imports"""
    components = {}
    
    # Test core component imports
    test_imports = [
        ("video_processor", "src.video_processor.downloader"),
        ("ml_engine", "src.ml_engine.content_analyzer"),
        ("ui_automation", "src.ui_automation.ui_analyzer"),
        ("knowledge_base", "src.knowledge_base.graph_db"),
        ("web_interface", "flask")
    ]
    
    for name, module_path in test_imports:
        try:
            __import__(module_path)
            components[name] = "OK"
        except ImportError as e:
            components[name] = f"FAIL: {str(e)}"
        except Exception as e:
            components[name] = f"ERROR: {str(e)}"
    
    return components

def get_recent_activity() -> Dict[str, Any]:
    """Check recent system activity"""
    root_path = Path(__file__).parent
    activity = {}
    
    # Check log files
    log_files = list(root_path.glob("*.log"))
    if log_files:
        recent_logs = sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]
        activity["recent_logs"] = [
            {
                "file": log.name,
                "size_kb": round(log.stat().st_size / 1024, 1),
                "modified": datetime.fromtimestamp(log.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            }
            for log in recent_logs
        ]
    
    # Check for recent database activity
    db_path = root_path / "maximum_potential_data" / "maximum_potential.db"
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Try to get recent entries from common tables
            tables_to_check = ['learning_acceleration', 'workflows', 'opportunities']
            for table in tables_to_check:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE created_at > datetime('now', '-7 days')")
                    count = cursor.fetchone()[0]
                    if count > 0:
                        activity[f"recent_{table}"] = count
                except sqlite3.OperationalError:
                    pass  # Table might not exist
            
            conn.close()
        except Exception:
            pass
    
    return activity

def calculate_health_score(status_data: Dict[str, Any]) -> float:
    """Calculate overall system health score"""
    score = 0
    max_score = 0
    
    # Dependencies score (30 points)
    deps = status_data.get("dependencies", {})
    core_count = len(deps.get("core", []))
    missing_count = len(deps.get("missing", []))
    if core_count + missing_count > 0:
        score += (core_count / (core_count + missing_count)) * 30
    max_score += 30
    
    # Database score (25 points)
    dbs = status_data.get("databases", {})
    working_dbs = len([db for db in dbs.values() if db.get("status") == "OK"])
    total_dbs = len(dbs)
    if total_dbs > 0:
        score += (working_dbs / total_dbs) * 25
    max_score += 25
    
    # Components score (25 points)
    comps = status_data.get("components", {})
    working_comps = len([comp for comp in comps.values() if comp == "OK"])
    total_comps = len(comps)
    if total_comps > 0:
        score += (working_comps / total_comps) * 25
    max_score += 25
    
    # Configuration score (20 points)
    configs = status_data.get("configuration", {})
    working_configs = len([cfg for cfg in configs.values() if cfg.get("status") == "OK"])
    total_configs = len(configs)
    if total_configs > 0:
        score += (working_configs / total_configs) * 20
    max_score += 20
    
    return (score / max_score) * 100 if max_score > 0 else 0

def get_status_emoji(score: float) -> str:
    """Get status emoji based on health score"""
    if score >= 90:
        return "🟢 EXCELLENT"
    elif score >= 75:
        return "🟡 GOOD"
    elif score >= 50:
        return "🟠 WARNING"
    else:
        return "🔴 CRITICAL"

def quick_status():
    """Display comprehensive quick status"""
    print("\n🚀 HyperbolicLearner System Status")
    print("=" * 45)
    
    start_time = time.time()
    
    # Gather all status information
    status_data = {
        "timestamp": datetime.now().isoformat(),
        "system_overview": get_system_overview(),
        "dependencies": check_dependencies(),
        "databases": check_databases(),
        "configuration": check_configuration(),
        "components": check_components(),
        "recent_activity": get_recent_activity()
    }
    
    # Calculate health score
    health_score = calculate_health_score(status_data)
    status_emoji = get_status_emoji(health_score)
    
    # Display summary
    print(f"Overall Health: {status_emoji} ({health_score:.1f}%)")
    print(f"System: HyperbolicLearner v2.0")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # System resources
    sys_info = status_data["system_overview"]["system"]
    if "error" not in sys_info:
        print(f"\n💻 System Resources:")
        print(f"  CPU: {sys_info['cpu_count']} cores ({sys_info['cpu_usage']:.1f}% usage)")
        print(f"  Memory: {sys_info['memory_usage_percent']:.1f}% of {sys_info['memory_total_gb']}GB used")
        print(f"  Disk: {sys_info['disk_usage_percent']:.1f}% used ({sys_info['disk_free_gb']}GB free)")
    
    # GPU status
    gpu_info = status_data["system_overview"]["gpu"]
    gpu_icon = "🔥" if gpu_info["available"] else "💻"
    gpu_text = "Available" if gpu_info["available"] else "Not Available"
    if "device_name" in gpu_info:
        gpu_text += f" ({gpu_info['device_name']})"
    print(f"  GPU: {gpu_icon} {gpu_text}")
    
    # Dependencies
    deps = status_data["dependencies"]
    core_count = len(deps["core"])
    optional_count = len(deps["optional"])
    missing_count = len(deps["missing"])
    
    print(f"\n📦 Dependencies:")
    print(f"  Core: {core_count} installed")
    print(f"  Optional: {optional_count} installed")
    if missing_count > 0:
        print(f"  Missing: {missing_count} ⚠️")
    
    # Databases
    dbs = status_data["databases"]
    working_dbs = [name for name, info in dbs.items() if info.get("status") == "OK"]
    print(f"\n🗄️  Databases: {len(working_dbs)}/{len(dbs)} operational")
    for name, info in dbs.items():
        status_icon = "✅" if info.get("status") == "OK" else "❌"
        details = ""
        if info.get("status") == "OK":
            details = f" ({info.get('tables', 0)} tables, {info.get('size_kb', 0)}KB)"
        print(f"  {status_icon} {name.title()}: {info.get('status', 'UNKNOWN')}{details}")
    
    # Components
    comps = status_data["components"]
    working_comps = [name for name, status in comps.items() if status == "OK"]
    print(f"\n🔧 Components: {len(working_comps)}/{len(comps)} working")
    for name, status in comps.items():
        status_icon = "✅" if status == "OK" else "❌"
        print(f"  {status_icon} {name.replace('_', ' ').title()}: {status}")
    
    # Configuration
    configs = status_data["configuration"]
    working_configs = [name for name, info in configs.items() if info.get("status") == "OK"]
    print(f"\n⚙️  Configuration: {len(working_configs)}/{len(configs)} files OK")
    for name, info in configs.items():
        status_icon = "✅" if info.get("status") == "OK" else "❌"
        print(f"  {status_icon} {name.title()}: {info.get('status', 'UNKNOWN')}")
    
    # Recent activity
    activity = status_data["recent_activity"]
    if activity:
        print(f"\n📊 Recent Activity:")
        for key, value in activity.items():
            if key == "recent_logs":
                print(f"  Log Files: {len(value)} recent")
            elif key.startswith("recent_"):
                table_name = key.replace("recent_", "").replace("_", " ").title()
                print(f"  {table_name}: {value} entries (last 7 days)")
    
    # Performance info
    execution_time = time.time() - start_time
    print(f"\n⏱️  Status check completed in {execution_time:.2f}s")
    
    # Quick recommendations
    if health_score < 75:
        print(f"\n💡 Quick Fixes:")
        if missing_count > 0:
            print(f"  • Install missing dependencies: pip install -r requirements.txt")
        
        error_dbs = [name for name, info in dbs.items() if info.get("status") in ["ERROR", "MISSING"]]
        if error_dbs:
            print(f"  • Initialize databases: python system_auto_repair.py")
        
        failing_components = [name for name, status in comps.items() if status != "OK"]
        if failing_components:
            print(f"  • Fix component issues: check import errors")
        
        print(f"  • Run full diagnostic: python system_diagnostics.py")
        print(f"  • Run auto-repair: python system_auto_repair.py")
    
    # JSON output option
    if "--json" in sys.argv:
        print(f"\n{json.dumps(status_data, indent=2)}")
    
    return health_score

if __name__ == "__main__":
    try:
        score = quick_status()
        # Exit with appropriate code
        sys.exit(0 if score >= 75 else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Status check interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Status check failed: {e}")
        sys.exit(1)

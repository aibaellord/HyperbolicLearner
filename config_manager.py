#!/usr/bin/env python3
"""
HyperbolicLearner Configuration Management System

Advanced configuration manager that handles environment detection, performance optimization,
security settings, and dynamic configuration updates.
"""

import os
import sys
import json
import yaml
import logging
import platform
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
import importlib
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class SystemCapabilities:
    """System hardware and software capabilities"""
    cpu_count: int
    memory_gb: float
    gpu_available: bool
    gpu_memory_gb: float
    storage_gb: float
    platform: str
    python_version: str
    cuda_version: Optional[str] = None
    torch_available: bool = False
    tensorflow_available: bool = False
    
    @classmethod
    def detect(cls) -> 'SystemCapabilities':
        """Detect current system capabilities"""
        # CPU and Memory
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Storage
        disk = psutil.disk_usage('.')
        storage_gb = disk.total / (1024**3)
        
        # Platform info
        platform_info = platform.platform()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # GPU detection
        gpu_available = False
        gpu_memory_gb = 0.0
        cuda_version = None
        
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            torch_available = True
            
            if gpu_available:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                cuda_version = torch.version.cuda
        except ImportError:
            torch_available = False
        
        # TensorFlow detection
        tensorflow_available = False
        try:
            import tensorflow as tf
            tensorflow_available = True
        except ImportError:
            pass
        
        return cls(
            cpu_count=cpu_count,
            memory_gb=memory_gb,
            gpu_available=gpu_available,
            gpu_memory_gb=gpu_memory_gb,
            storage_gb=storage_gb,
            platform=platform_info,
            python_version=python_version,
            cuda_version=cuda_version,
            torch_available=torch_available,
            tensorflow_available=tensorflow_available
        )

@dataclass
class PerformanceProfile:
    """Performance optimization profile"""
    name: str
    description: str
    max_concurrent_processes: int
    cache_size_mb: int
    gpu_acceleration: bool
    batch_size: int
    memory_optimization: bool
    parallel_processing: bool
    
    @classmethod
    def create_optimal_profile(cls, capabilities: SystemCapabilities) -> 'PerformanceProfile':
        """Create optimal performance profile based on system capabilities"""
        if capabilities.memory_gb >= 32 and capabilities.gpu_available:
            return cls(
                name="high_performance",
                description="High-end system with GPU acceleration",
                max_concurrent_processes=min(capabilities.cpu_count, 12),
                cache_size_mb=min(int(capabilities.memory_gb * 200), 4096),
                gpu_acceleration=True,
                batch_size=32,
                memory_optimization=True,
                parallel_processing=True
            )
        elif capabilities.memory_gb >= 16:
            return cls(
                name="medium_performance",
                description="Mid-range system optimization",
                max_concurrent_processes=min(capabilities.cpu_count, 8),
                cache_size_mb=min(int(capabilities.memory_gb * 150), 2048),
                gpu_acceleration=capabilities.gpu_available,
                batch_size=16,
                memory_optimization=True,
                parallel_processing=capabilities.cpu_count >= 4
            )
        elif capabilities.memory_gb >= 8:
            return cls(
                name="balanced",
                description="Balanced performance for average systems",
                max_concurrent_processes=min(capabilities.cpu_count, 4),
                cache_size_mb=min(int(capabilities.memory_gb * 100), 1024),
                gpu_acceleration=capabilities.gpu_available,
                batch_size=8,
                memory_optimization=True,
                parallel_processing=capabilities.cpu_count >= 2
            )
        else:
            return cls(
                name="low_resource",
                description="Conservative settings for limited resources",
                max_concurrent_processes=2,
                cache_size_mb=512,
                gpu_acceleration=False,
                batch_size=4,
                memory_optimization=True,
                parallel_processing=False
            )

@dataclass
class SecuritySettings:
    """Security configuration settings"""
    encrypt_sensitive_data: bool = True
    secure_file_permissions: bool = True
    audit_logging: bool = True
    api_rate_limiting: bool = True
    network_timeout_seconds: int = 30
    max_file_size_mb: int = 1024
    allowed_extensions: List[str] = field(default_factory=lambda: ['.mp4', '.avi', '.mov', '.json'])
    
class ConfigurationManager:
    """Advanced configuration management system"""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path.cwd()
        self.config_file = self.config_dir / "system_config.json"
        self.capabilities = SystemCapabilities.detect()
        self.performance_profile = PerformanceProfile.create_optimal_profile(self.capabilities)
        self.security_settings = SecuritySettings()
        
        # Configuration cache
        self._config_cache = {}
        self._config_hash = None
        
        # Initialize configuration
        self.initialize_config()
    
    def initialize_config(self):
        """Initialize or update configuration"""
        config = self.load_config()
        
        if not config or self.needs_update(config):
            logger.info("Creating/updating system configuration")
            config = self.create_default_config()
            self.save_config(config)
        else:
            logger.info("Configuration is current")
        
        self._config_cache = config
        self._config_hash = self.calculate_config_hash(config)
    
    def create_default_config(self) -> Dict[str, Any]:
        """Create default configuration based on system capabilities"""
        config = {
            "system": {
                "version": "2.0.0",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "capabilities": asdict(self.capabilities),
                "performance_profile": asdict(self.performance_profile),
                "security_settings": asdict(self.security_settings)
            },
            
            "video_processing": {
                "default_acceleration_factor": 2.0,
                "max_acceleration_factor": 5.0 if self.capabilities.memory_gb >= 8 else 3.0,
                "quality_threshold": 0.8,
                "frame_sampling_method": "adaptive",
                "audio_processing": {
                    "enabled": True,
                    "pitch_correction": True,
                    "noise_reduction": self.capabilities.memory_gb >= 8,
                    "quality_level": 2 if self.capabilities.memory_gb >= 16 else 1
                },
                "output_formats": ["mp4", "json"],
                "temp_directory": str(self.config_dir / "temp"),
                "cache_directory": str(self.config_dir / "cache")
            },
            
            "machine_learning": {
                "use_gpu": self.capabilities.gpu_available,
                "batch_size": self.performance_profile.batch_size,
                "model_cache_size": self.performance_profile.cache_size_mb,
                "precision": "float32" if self.capabilities.gpu_memory_gb < 8 else "float16",
                "models": {
                    "content_analysis": "microsoft/resnet-50",
                    "text_processing": "sentence-transformers/all-MiniLM-L6-v2",
                    "audio_transcription": "openai/whisper-base",
                    "ui_detection": "facebook/detr-resnet-50"
                },
                "optimization": {
                    "gradient_checkpointing": self.capabilities.memory_gb < 16,
                    "mixed_precision": self.capabilities.gpu_available,
                    "cpu_threads": self.performance_profile.max_concurrent_processes
                }
            },
            
            "ui_automation": {
                "enabled": True,
                "screen_capture_fps": 2,
                "element_detection_threshold": 0.7,
                "action_delay_ms": 500,
                "verification_enabled": True,
                "screenshot_quality": 85,
                "supported_platforms": ["web", "desktop"],
                "safety": {
                    "confirm_destructive_actions": True,
                    "screenshot_before_action": True,
                    "rollback_on_failure": True
                }
            },
            
            "knowledge_base": {
                "type": "sqlite",  # "neo4j" for advanced setups
                "database_path": str(self.config_dir / "maximum_potential_data" / "knowledge.db"),
                "enable_full_text_search": True,
                "backup_enabled": True,
                "backup_interval_hours": 24,
                "max_connections": 5,
                "query_timeout_seconds": 30,
                "indexing": {
                    "concepts": True,
                    "relationships": True,
                    "temporal": True,
                    "similarity_threshold": 0.75
                }
            },
            
            "web_interface": {
                "enabled": True,
                "host": "127.0.0.1",
                "port": 5000,
                "debug": False,
                "auto_reload": False,
                "max_content_length": self.security_settings.max_file_size_mb * 1024 * 1024,
                "session_timeout_minutes": 60,
                "rate_limiting": {
                    "enabled": self.security_settings.api_rate_limiting,
                    "requests_per_minute": 60,
                    "burst_limit": 10
                },
                "cors": {
                    "enabled": False,
                    "origins": ["http://localhost:3000"]
                }
            },
            
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file_path": str(self.config_dir / "hyperbolic_learner.log"),
                "max_file_size_mb": 10,
                "backup_count": 5,
                "components": {
                    "video_processor": "INFO",
                    "ml_engine": "INFO",
                    "ui_automation": "WARNING",
                    "knowledge_base": "INFO",
                    "web_interface": "WARNING"
                },
                "performance_logging": self.capabilities.memory_gb >= 8
            },
            
            "security": asdict(self.security_settings),
            
            "experimental": {
                "features": {
                    "advanced_caching": self.capabilities.memory_gb >= 16,
                    "realtime_optimization": self.capabilities.cpu_count >= 4,
                    "predictive_loading": self.capabilities.gpu_available,
                    "quantum_algorithms": False,  # Placeholder for future features
                    "cloud_integration": False
                },
                "disclaimers": [
                    "Experimental features may impact system stability",
                    "Use in production environments at your own risk"
                ]
            }
        }
        
        return config
    
    def needs_update(self, config: Dict[str, Any]) -> bool:
        """Check if configuration needs updating"""
        if not config.get("system"):
            return True
        
        # Check version compatibility
        config_version = config["system"].get("version", "1.0.0")
        if config_version < "2.0.0":
            return True
        
        # Check if capabilities have changed significantly
        old_capabilities = config["system"].get("capabilities", {})
        current_capabilities = asdict(self.capabilities)
        
        # Key capability changes that require reconfiguration
        key_changes = [
            old_capabilities.get("memory_gb", 0) != current_capabilities["memory_gb"],
            old_capabilities.get("gpu_available", False) != current_capabilities["gpu_available"],
            old_capabilities.get("cpu_count", 1) != current_capabilities["cpu_count"]
        ]
        
        return any(key_changes)
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not self.config_file.exists():
            return {}
        
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            # Ensure config directory exists
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Update timestamp
            config["system"]["updated_at"] = datetime.now().isoformat()
            
            # Save with pretty formatting
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # Set secure permissions
            os.chmod(self.config_file, 0o600)
            
            logger.info(f"Configuration saved to {self.config_file}")
            
        except IOError as e:
            logger.error(f"Failed to save config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self._config_cache
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self._config_cache
        
        # Navigate to parent
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set final value
        config[keys[-1]] = value
        
        # Save updated configuration
        self.save_config(self._config_cache)
        self._config_hash = self.calculate_config_hash(self._config_cache)
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values"""
        for key, value in updates.items():
            self.set(key, value)
    
    def validate_config(self) -> List[str]:
        """Validate current configuration and return list of issues"""
        issues = []
        config = self._config_cache
        
        # Required sections
        required_sections = ["system", "video_processing", "machine_learning"]
        for section in required_sections:
            if section not in config:
                issues.append(f"Missing required section: {section}")
        
        # Video processing validation
        if "video_processing" in config:
            vp = config["video_processing"]
            max_accel = vp.get("max_acceleration_factor", 0)
            if max_accel < 1 or max_accel > 10:
                issues.append("Invalid max_acceleration_factor (must be 1-10)")
        
        # ML validation
        if "machine_learning" in config:
            ml = config["machine_learning"]
            batch_size = ml.get("batch_size", 0)
            if batch_size < 1 or batch_size > 128:
                issues.append("Invalid batch_size (must be 1-128)")
        
        # Path validation
        paths_to_check = [
            "video_processing.temp_directory",
            "video_processing.cache_directory",
            "knowledge_base.database_path"
        ]
        
        for path_key in paths_to_check:
            path_str = self.get(path_key)
            if path_str:
                path = Path(path_str)
                if not path.parent.exists():
                    issues.append(f"Parent directory doesn't exist for {path_key}: {path.parent}")
        
        return issues
    
    def optimize_for_current_system(self):
        """Re-optimize configuration for current system capabilities"""
        logger.info("Re-optimizing configuration for current system")
        
        # Update capabilities
        self.capabilities = SystemCapabilities.detect()
        self.performance_profile = PerformanceProfile.create_optimal_profile(self.capabilities)
        
        # Update relevant configuration sections
        updates = {
            "system.capabilities": asdict(self.capabilities),
            "system.performance_profile": asdict(self.performance_profile),
            "machine_learning.use_gpu": self.capabilities.gpu_available,
            "machine_learning.batch_size": self.performance_profile.batch_size,
            "machine_learning.optimization.cpu_threads": self.performance_profile.max_concurrent_processes
        }
        
        self.update(updates)
    
    def create_backup(self) -> str:
        """Create configuration backup and return backup path"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.config_dir / f"config_backup_{timestamp}.json"
        
        try:
            with open(backup_path, 'w') as f:
                json.dump(self._config_cache, f, indent=2)
            
            logger.info(f"Configuration backup created: {backup_path}")
            return str(backup_path)
            
        except IOError as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def restore_backup(self, backup_path: str):
        """Restore configuration from backup"""
        backup_file = Path(backup_path)
        
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        try:
            with open(backup_file, 'r') as f:
                backup_config = json.load(f)
            
            # Validate backup
            if not backup_config.get("system"):
                raise ValueError("Invalid backup file: missing system section")
            
            self._config_cache = backup_config
            self.save_config(backup_config)
            self._config_hash = self.calculate_config_hash(backup_config)
            
            logger.info(f"Configuration restored from backup: {backup_path}")
            
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to restore backup: {e}")
            raise
    
    def export_config(self, format: str = "json") -> str:
        """Export configuration in specified format"""
        if format.lower() == "yaml":
            import yaml
            return yaml.dump(self._config_cache, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            return json.dumps(self._config_cache, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate hash of configuration for change detection"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def has_changed(self) -> bool:
        """Check if configuration has changed since last load"""
        current_hash = self.calculate_config_hash(self._config_cache)
        return current_hash != self._config_hash
    
    def reset_to_defaults(self):
        """Reset configuration to defaults based on current system"""
        logger.warning("Resetting configuration to defaults")
        
        # Create backup first
        backup_path = self.create_backup()
        logger.info(f"Backup created before reset: {backup_path}")
        
        # Reset to defaults
        default_config = self.create_default_config()
        self._config_cache = default_config
        self.save_config(default_config)
        self._config_hash = self.calculate_config_hash(default_config)
    
    def get_performance_recommendations(self) -> List[str]:
        """Get performance optimization recommendations"""
        recommendations = []
        
        # Memory recommendations
        if self.capabilities.memory_gb < 8:
            recommendations.append("Consider upgrading to 8GB+ RAM for better performance")
        
        # GPU recommendations
        if not self.capabilities.gpu_available:
            recommendations.append("Install CUDA drivers for GPU acceleration")
        elif self.capabilities.gpu_memory_gb < 4:
            recommendations.append("GPU has limited memory - consider reducing batch sizes")
        
        # CPU recommendations
        if self.capabilities.cpu_count < 4:
            recommendations.append("Multi-core CPU recommended for parallel processing")
        
        # Storage recommendations
        if self.capabilities.storage_gb < 50:
            recommendations.append("Low disk space - consider cleaning up or expanding storage")
        
        # Configuration-specific recommendations
        current_batch_size = self.get("machine_learning.batch_size", 8)
        if current_batch_size > 16 and self.capabilities.memory_gb < 16:
            recommendations.append("Reduce batch_size for systems with limited RAM")
        
        return recommendations
    
    def __str__(self) -> str:
        """String representation of configuration manager"""
        return f"ConfigManager(capabilities={self.capabilities}, profile={self.performance_profile.name})"

def main():
    """CLI interface for configuration management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HyperbolicLearner Configuration Manager")
    parser.add_argument("--init", action="store_true", help="Initialize configuration")
    parser.add_argument("--validate", action="store_true", help="Validate configuration")
    parser.add_argument("--optimize", action="store_true", help="Optimize for current system")
    parser.add_argument("--export", choices=["json", "yaml"], help="Export configuration")
    parser.add_argument("--reset", action="store_true", help="Reset to defaults")
    parser.add_argument("--backup", action="store_true", help="Create configuration backup")
    parser.add_argument("--get", help="Get configuration value")
    parser.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"), help="Set configuration value")
    
    args = parser.parse_args()
    
    config_manager = ConfigurationManager()
    
    if args.init:
        print("Configuration initialized successfully")
        
    elif args.validate:
        issues = config_manager.validate_config()
        if issues:
            print("Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Configuration is valid")
    
    elif args.optimize:
        config_manager.optimize_for_current_system()
        print("Configuration optimized for current system")
    
    elif args.export:
        exported = config_manager.export_config(args.export)
        print(exported)
    
    elif args.reset:
        response = input("Reset configuration to defaults? (y/N): ")
        if response.lower() == 'y':
            config_manager.reset_to_defaults()
            print("Configuration reset to defaults")
    
    elif args.backup:
        backup_path = config_manager.create_backup()
        print(f"Backup created: {backup_path}")
    
    elif args.get:
        value = config_manager.get(args.get)
        print(f"{args.get}: {value}")
    
    elif args.set:
        key, value = args.set
        # Try to parse as JSON first
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass  # Keep as string
        
        config_manager.set(key, value)
        print(f"Set {key} = {value}")
    
    else:
        # Show system info and recommendations
        print(f"System Capabilities: {config_manager.capabilities}")
        print(f"Performance Profile: {config_manager.performance_profile.name}")
        
        recommendations = config_manager.get_performance_recommendations()
        if recommendations:
            print("\nPerformance Recommendations:")
            for rec in recommendations:
                print(f"  - {rec}")

if __name__ == "__main__":
    main()

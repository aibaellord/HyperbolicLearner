#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Module for HyperbolicLearner

This module provides comprehensive configuration management for the HyperbolicLearner system,
handling dynamic configuration, environment detection, hardware capability assessment,
and feature toggles. It supports loading from environment variables, configuration files,
and command line arguments with appropriate hierarchy and fallbacks.

The module auto-detects available ML libraries, GPUs, and other system resources to enable
optimal performance across different environments.
"""

import os
import sys
import json
import yaml
import logging
import platform
import argparse
import configparser
import multiprocessing
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum, auto

# Setup logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CONFIG_LOCATIONS = [
    "~/.config/hyperboliclearner/config.yaml",
    "~/.hyperboliclearner.yaml",
    "./config.yaml",
]

ENV_PREFIX = "HYPERBOLIC_"


class EnvironmentType(Enum):
    """Environment types for the system"""
    DEVELOPMENT = auto()
    TESTING = auto()
    PRODUCTION = auto()
    UNKNOWN = auto()


class ResourceType(Enum):
    """Types of resources that can be detected"""


@dataclass
class GPUInfo:
    """GPU resource information"""
    available: bool = False
    count: int = 0
    models: List[str] = field(default_factory=list)
    memory_mb: List[int] = field(default_factory=list)
    cuda_available: bool = False
    cuda_version: str = ""
    cudnn_available: bool = False
    cudnn_version: str = ""
    
    @classmethod
    def detect(cls) -> 'GPUInfo':
        """Detect GPU information"""
        info = cls()
        
        # Check for CUDA availability
        try:
            import torch
            info.cuda_available = torch.cuda.is_available()
            if info.cuda_available:
                info.count = torch.cuda.device_count()
                info.models = [torch.cuda.get_device_name(i) for i in range(info.count)]
                info.memory_mb = []
                
                for i in range(info.count):
                    torch.cuda.set_device(i)
                    free_mem, total_mem = torch.cuda.mem_get_info()
                    info.memory_mb.append(int(total_mem / (1024 * 1024)))
                
                info.cuda_version = torch.version.cuda or ""
                info.available = True
                
                # Check for cuDNN
                if hasattr(torch.backends, 'cudnn'):
                    info.cudnn_available = torch.backends.cudnn.is_available()
                    info.cudnn_version = str(torch.backends.cudnn.version()) if info.cudnn_available else ""
            
        except ImportError:
            logger.info("PyTorch not available for GPU detection")
            
            # Try with tensorflow if torch is not available
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    info.available = True
                    info.count = len(gpus)
                    info.models = [gpu.name for gpu in gpus]
                    
                    # Try to get CUDA version from TF
                    if hasattr(tf, 'version'):
                        if hasattr(tf.version, 'CUDA'):
                            info.cuda_available = True
                            info.cuda_version = tf.version.CUDA or ""
                        if hasattr(tf.version, 'CUDNN'):
                            info.cudnn_available = True
                            info.cudnn_version = tf.version.CUDNN or ""
            
            except ImportError:
                logger.info("TensorFlow not available for GPU detection")
                
                # Try direct nvidia-smi approach
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    
                    if result.returncode == 0 and result.stdout:
                        info.available = True
                        lines = result.stdout.strip().split('\n')
                        info.count = len(lines)
                        
                        for line in lines:
                            parts = line.split(',')
                            if len(parts) >= 2:
                                model = parts[0].strip()
                                mem_str = parts[1].strip()
                                info.models.append(model)
                                
                                # Extract numeric memory value
                                import re
                                mem_match = re.search(r'(\d+)', mem_str)
                                if mem_match:
                                    info.memory_mb.append(int(mem_match.group(1)))
                                else:
                                    info.memory_mb.append(0)
                                    
                        # Try to get CUDA version
                        cuda_result = subprocess.run(['nvcc', '--version'], 
                                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        if cuda_result.returncode == 0:
                            version_match = re.search(r'release (\d+\.\d+)', cuda_result.stdout)
                            if version_match:
                                info.cuda_available = True
                                info.cuda_version = version_match.group(1)
                                
                except Exception as e:
                    logger.debug(f"nvidia-smi GPU detection failed: {e}")
        
        except Exception as e:
            logger.warning(f"Error detecting GPU information: {e}")
        
        return info


@dataclass
class MemoryInfo:
    """System memory information"""
    total_gb: float = 0.0
    available_gb: float = 0.0
    
    @classmethod
    def detect(cls) -> 'MemoryInfo':
        """Detect system memory"""
        info = cls()
        
        try:
            import psutil
            vm = psutil.virtual_memory()
            info.total_gb = vm.total / (1024 ** 3)  # Convert bytes to GB
            info.available_gb = vm.available / (1024 ** 3)
        except ImportError:
            logger.info("psutil not available for memory detection")
            try:
                # Platform-specific fallbacks
                if platform.system() == "Linux":
                    import subprocess
                    # Get total memory
                    cmd = "free -g | grep 'Mem:' | awk '{print $2}'"
                    total = subprocess.check_output(cmd, shell=True).decode().strip()
                    info.total_gb = float(total) if total.replace('.', '', 1).isdigit() else 0.0
                    
                    # Get available memory
                    cmd = "free -g | grep 'Mem:' | awk '{print $7}'"
                    available = subprocess.check_output(cmd, shell=True).decode().strip()
                    info.available_gb = float(available) if available.replace('.', '', 1).isdigit() else 0.0
                
                elif platform.system() == "Darwin":  # macOS
                    import subprocess
                    # Get total memory
                    cmd = "sysctl -n hw.memsize"
                    total_bytes = subprocess.check_output(cmd, shell=True).decode().strip()
                    info.total_gb = int(total_bytes) / (1024 ** 3) if total_bytes.isdigit() else 0.0
                    
                    # Get available memory (approximation)
                    cmd = "vm_stat | grep 'Pages free:' | awk '{print $3}' | tr -d '.'"
                    pages_free = subprocess.check_output(cmd, shell=True).decode().strip()
                    page_size_cmd = "sysctl -n hw.pagesize"
                    page_size = subprocess.check_output(page_size_cmd, shell=True).decode().strip()
                    
                    if pages_free.isdigit() and page_size.isdigit():
                        free_bytes = int(pages_free) * int(page_size)
                        info.available_gb = free_bytes / (1024 ** 3)
                
                elif platform.system() == "Windows":
                    import subprocess
                    # Get memory info from wmic
                    cmd = "wmic ComputerSystem get TotalPhysicalMemory /value"
                    total_output = subprocess.check_output(cmd, shell=True).decode()
                    
                    # Parse total memory
                    for line in total_output.splitlines():
                        if "=" in line:
                            key, value = line.split("=", 1)
                            if key.strip() == "TotalPhysicalMemory" and value.strip().isdigit():
                                info.total_gb = int(value.strip()) / (1024 ** 3)
                    
                    # Get available memory
                    cmd = "wmic OS get FreePhysicalMemory /value"
                    free_output = subprocess.check_output(cmd, shell=True).decode()
                    
                    # Parse free memory
                    for line in free_output.splitlines():
                        if "=" in line:
                            key, value = line.split("=", 1)
                            if key.strip() == "FreePhysicalMemory" and value.strip().isdigit():
                                # FreePhysicalMemory is in KB, convert to GB
                                info.available_gb = int(value.strip()) / (1024 * 1024)
            
            except Exception as e:
                logger.warning(f"Error in fallback memory detection: {e}")
        
        except Exception as e:
            logger.warning(f"Error detecting memory information: {e}")
        
        return info


@dataclass
class DiskInfo:
    """Disk storage information"""
    total_gb: float = 0.0
    available_gb: float = 0.0
    
    @classmethod
    def detect(cls, path: str = ".") -> 'DiskInfo':
        """Detect disk information for the specified path"""
        info = cls()
        
        try:
            import shutil
            usage = shutil.disk_usage(path)
            info.total_gb = usage.total / (1024 ** 3)
            info.available_gb = usage.free / (1024 ** 3)
        except ImportError:
            logger.info("shutil.disk_usage not available for disk detection")
            try:
                # Platform-specific fallbacks
                if platform.system() in ("Linux", "Darwin"):  # Linux or macOS
                    import subprocess
                    import os
                    # Get absolute path
                    abs_path = os.path.abspath(path)
                    # Get disk usage with df
                    cmd = f"df -k '{abs_path}' | tail -1 | awk '{{print $2, $4}}'"
                    output = subprocess.check_output(cmd, shell=True).decode().strip()
                    parts = output.split()
                    if len(parts) >= 2:
                        # Convert KB to GB
                        info.total_gb = int(parts[0]) / (1024 * 1024)
                        info.available_gb = int(parts[1]) / (1024 * 1024)
                elif platform.system() == "Windows":
                    import subprocess
                    import os
                    # Get drive letter
                    drive = os.path.splitdrive(os.path.abspath(path))[0]
                    if drive:
                        # Get disk space from wmic
                        cmd = f"wmic logicaldisk where DeviceID='{drive}' get Size,FreeSpace /value"
                        output = subprocess.check_output(cmd, shell=True).decode()
                        # Parse output
                        size = None
                        free = None
                        # for line in output.splitlines():
                        #     pass  # TODO: Implement disk space parsing for Windows
            except Exception as e:
                logger.warning(f"Disk space detection failed: {e}")
        
        return info


@dataclass
class SystemInfo:
    """Complete system information"""
    cpu_count: int = 0
    platform: str = ""
    gpu_info: GPUInfo = field(default_factory=GPUInfo)
    memory_info: MemoryInfo = field(default_factory=MemoryInfo)
    disk_info: DiskInfo = field(default_factory=DiskInfo)
    python_version: str = ""
    
    @classmethod
    def detect(cls) -> 'SystemInfo':
        """Detect complete system information"""
        info = cls()
        
        try:
            info.cpu_count = multiprocessing.cpu_count()
            info.platform = platform.platform()
            info.python_version = sys.version
            info.gpu_info = GPUInfo.detect()
            info.memory_info = MemoryInfo.detect()
            info.disk_info = DiskInfo.detect()
        except Exception as e:
            logger.warning(f"Error detecting system information: {e}")
        
        return info


class Config:
    """
    Comprehensive configuration manager for HyperbolicLearner
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration"""
        self.config_data = {}
        
        # Load default configuration
        self._load_defaults()
        
        # Load from config file
        self._load_config_file(config_path)
        
        # Load from environment variables
        self._load_environment()
        
        # Detect system capabilities
        self.system_info = SystemInfo.detect()
        
        # Set derived configuration
        self._set_derived_config()
        
        # Create data directory
        self._ensure_data_directory()
    
    def _load_defaults(self):
        """Load default configuration values"""
        self.config_data = {
            # Core settings
            "data_dir": os.path.expanduser("~/.hyperboliclearner/data"),
            "cache_dir": os.path.expanduser("~/.hyperboliclearner/cache"),
            "log_dir": os.path.expanduser("~/.hyperboliclearner/logs"),
            "temp_dir": "/tmp/hyperboliclearner",
            
            # Video processing
            "video_processing": {
                "default_acceleration_factor": 10.0,
                "max_acceleration_factor": 30.0,
                "semantic_compression": True,
                "quality_threshold": 0.8,
                "max_video_duration": 3600,  # 1 hour
                "supported_formats": ["mp4", "avi", "mkv", "webm"]
            },
            
            # ML/AI settings
            "ml_engine": {
                "use_gpu": True,
                "batch_size": 32,
                "num_workers": 4,
                "model_cache_dir": os.path.expanduser("~/.hyperboliclearner/models"),
                "enable_mixed_precision": True
            },
            
            # UI automation
            "ui_automation": {
                "screenshot_on_error": True,
                "max_retries": 3,
                "retry_delay": 1.0,
                "action_delay": 0.5,
                "verification_enabled": True,
                "fallback_strategies": True
            },
            
            # Knowledge base
            "knowledge_base": {
                "max_graph_nodes": 100000,
                "similarity_threshold": 0.7,
                "auto_cleanup": True,
                "backup_interval": 3600  # 1 hour
            },
            
            # Performance
            "performance": {
                "max_parallel_videos": 4,
                "memory_limit_gb": 8.0,
                "disk_cache_gb": 10.0,
                "enable_profiling": False
            },
            
            # Logging
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "max_file_size_mb": 10,
                "backup_count": 5,
                "console_output": True
            },
            
            # N8N integration
            "n8n_integration": {
                "host": "localhost",
                "port": 5678,
                "api_key": None,
                "webhook_enabled": True,
                "auto_start": True,
                "workflow_timeout": 300  # 5 minutes
            },
            
            # Maximum potential settings
            "maximum_potential": {
                "enabled": True,
                "auto_optimization": True,
                "value_amplification": True,
                "success_multiplication": True,
                "opportunity_scanning_interval": 300,  # 5 minutes
                "workflow_optimization_interval": 600,  # 10 minutes
            },
            
            # Security
            "security": {
                "sandbox_mode": False,
                "allowed_domains": [],
                "blocked_domains": [],
                "max_execution_time": 1800  # 30 minutes
            }
        }
    
    def _load_config_file(self, config_path: Optional[str] = None):
        """Load configuration from file"""
        if config_path:
            config_paths = [config_path]
        else:
            config_paths = [os.path.expanduser(path) for path in DEFAULT_CONFIG_LOCATIONS]
        
        for path in config_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        if path.endswith('.yaml') or path.endswith('.yml'):
                            file_config = yaml.safe_load(f)
                        elif path.endswith('.json'):
                            file_config = json.load(f)
                        else:
                            # Try YAML first, then JSON
                            try:
                                file_config = yaml.safe_load(f)
                            except yaml.YAMLError:
                                f.seek(0)
                                file_config = json.load(f)
                    
                    if file_config:
                        self._merge_config(file_config)
                        logger.info(f"Loaded configuration from {path}")
                        break
                
                except Exception as e:
                    logger.warning(f"Error loading config file {path}: {e}")
    
    def _load_environment(self):
        """Load configuration from environment variables"""
        for key, value in os.environ.items():
            if key.startswith(ENV_PREFIX):
                config_key = key[len(ENV_PREFIX):].lower()
                
                # Convert environment variable to config path
                if '__' in config_key:
                    # Handle nested keys like HYPERBOLIC_ML_ENGINE__BATCH_SIZE
                    parts = config_key.split('__')
                    self._set_nested_config(parts, value)
                else:
                    # Handle simple keys
                    self.config_data[config_key] = self._convert_env_value(value)
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """Merge new configuration with existing"""
        def merge_dict(base: Dict[str, Any], new: Dict[str, Any]):
            for key, value in new.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        
        merge_dict(self.config_data, new_config)
    
    def _set_nested_config(self, parts: List[str], value: str):
        """Set nested configuration value"""
        current = self.config_data
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = self._convert_env_value(value)
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type"""
        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Try JSON
        if value.startswith('[') or value.startswith('{'):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # Return as string
        return value
    
    def _set_derived_config(self):
        """Set configuration values derived from system info"""
        # Adjust settings based on available resources
        if self.system_info.gpu_info.available:
            self.config_data["ml_engine"]["use_gpu"] = True
            if self.system_info.gpu_info.count > 1:
                self.config_data["ml_engine"]["num_workers"] = min(self.system_info.gpu_info.count * 2, 8)
        else:
            self.config_data["ml_engine"]["use_gpu"] = False
            self.config_data["ml_engine"]["num_workers"] = min(self.system_info.cpu_count, 8)
        
        # Adjust memory limits based on available memory
        available_memory = self.system_info.memory_info.available_gb
        if available_memory > 0:
            # Use 50% of available memory for processing
            self.config_data["performance"]["memory_limit_gb"] = min(available_memory * 0.5, 16.0)
        
        # Adjust parallel video processing based on CPU and memory
        cpu_factor = max(1, self.system_info.cpu_count // 2)
        memory_factor = max(1, int(available_memory // 4))
        self.config_data["performance"]["max_parallel_videos"] = min(cpu_factor, memory_factor, 8)
        
        # Adjust batch size based on GPU memory
        if self.system_info.gpu_info.available and self.system_info.gpu_info.memory_mb:
            max_gpu_memory = max(self.system_info.gpu_info.memory_mb)
            if max_gpu_memory < 4000:  # Less than 4GB
                self.config_data["ml_engine"]["batch_size"] = 16
            elif max_gpu_memory < 8000:  # Less than 8GB
                self.config_data["ml_engine"]["batch_size"] = 32
            else:  # 8GB or more
                self.config_data["ml_engine"]["batch_size"] = 64
    
    def _ensure_data_directory(self):
        """Ensure data directories exist"""
        directories = [
            self.data_dir,
            self.cache_dir,
            self.log_dir,
            self.config_data["ml_engine"]["model_cache_dir"]
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    # Property accessors for common config values
    @property
    def data_dir(self) -> str:
        """Get data directory"""
        return self.config_data["data_dir"]
    
    @property
    def cache_dir(self) -> str:
        """Get cache directory"""
        return self.config_data["cache_dir"]
    
    @property
    def log_dir(self) -> str:
        """Get log directory"""
        return self.config_data["log_dir"]
    
    @property
    def use_gpu(self) -> bool:
        """Check if GPU should be used"""
        return self.config_data["ml_engine"]["use_gpu"]
    
    @property
    def max_acceleration_factor(self) -> float:
        """Get maximum acceleration factor"""
        return self.config_data["video_processing"]["max_acceleration_factor"]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        if '.' in key:
            # Handle nested keys
            parts = key.split('.')
            current = self.config_data
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            return current
        else:
            return self.config_data.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        if '.' in key:
            # Handle nested keys
            parts = key.split('.')
            current = self.config_data
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            self.config_data[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.config_data.copy()
    
    def save_config(self, path: str):
        """Save current configuration to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                yaml.dump(self.config_data, f, default_flow_style=False)
            else:
                json.dump(self.config_data, f, indent=2)
    
    def get_system_info(self) -> SystemInfo:
        """Get system information"""
        return self.system_info
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available"""
        return self.system_info.gpu_info.available
    
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size for current system"""
        return self.config_data["ml_engine"]["batch_size"]
    
    def get_max_parallel_processes(self) -> int:
        """Get maximum parallel processes for current system"""
        return self.config_data["performance"]["max_parallel_videos"]
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"HyperbolicLearner Config (GPU: {self.is_gpu_available()}, CPUs: {self.system_info.cpu_count}, Memory: {self.system_info.memory_info.total_gb:.1f}GB)"


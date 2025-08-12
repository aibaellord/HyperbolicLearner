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


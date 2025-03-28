"""
Importance Models for Semantic Video Compression.

This module provides sophisticated models for assessing the importance of 
video content across multiple modalities (visual, audio, transcript).
These models are used to guide the semantic compression process, ensuring
the most valuable information is preserved during hyperbolic acceleration.

The module implements GPU-optimized, context-aware models that integrate
with modern machine learning approaches including transformers, attention
mechanisms, and multimodal fusion techniques.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path
import os
import json
import hashlib
import time
import uuid
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from collections import deque, OrderedDict, Counter
import warnings
import sys
import pickle
import io
import traceback
import gc
from datetime import datetime
from itertools import chain

# Optional imports with fallbacks for flexibility
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available, visual importance model will use PyTorch only")

try:
    from transformers import (
        AutoModel, 
        AutoFeatureExtractor, 
        AutoProcessor,
        AutoModelForAudioClassification,
        AutoModelForSequenceClassification,
        AutoModelForImageClassification,
        Wav2Vec2ForCTC,
        CLIPModel, 
        CLIPProcessor,
        BertModel,
        BertTokenizer,
        pipeline
    )
    from transformers.utils import logging as transformers_logging
    transformers_logging.set_verbosity_error()
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoModel = None
    AutoFeatureExtractor = None
    AutoProcessor = None
    AutoModelForAudioClassification = None
    AutoModelForSequenceClassification = None
    AutoModelForImageClassification = None
    Wav2Vec2ForCTC = None
    CLIPModel = None
    CLIPProcessor = None
    BertModel = None
    BertTokenizer = None
    pipeline = None
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available, falling back to basic models")

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    librosa = None
    LIBROSA_AVAILABLE = False
    logging.warning("Librosa not available, audio processing will be limited")

try:
    import sentence_transformers
    from sentence_transformers import SentenceTransformer, util as st_util
    SBERT_AVAILABLE = True
except ImportError:
    sentence_transformers = None
    SentenceTransformer = None
    st_util = None
    SBERT_AVAILABLE = False
    logging.warning("Sentence-Transformers not available, transcript analysis will be limited")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    # Configure ONNX for optimized execution
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.enable_profiling = False
    sess_options.intra_op_num_threads = min(8, os.cpu_count() or 4)
    
    # Set up providers with prioritized execution device
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
    
    # For GPU devices, configure memory settings
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        provider_options = [{'device_id': 0, 'arena_extend_strategy': 'kNextPowerOfTwo', 'gpu_mem_limit': 4 * 1024 * 1024 * 1024, 'cudnn_conv_algo_search': 'EXHAUSTIVE'}, {}]
    else:
        provider_options = [{}]
except ImportError:
    ort = None
    ONNX_AVAILABLE = False
    providers = None
    sess_options = None
    provider_options = None
    logging.warning("ONNX Runtime not available, inference optimization will be limited")

try:
    import torchvision
    import torchvision.transforms as transforms
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    from torchvision.models import resnet50, ResNet50_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    torchvision = None
    transforms = None
    fasterrcnn_resnet50_fpn = None
    FasterRCNN_ResNet50_FPN_Weights = None
    resnet50 = None
    ResNet50_Weights = None
    TORCHVISION_AVAILABLE = False
    logging.warning("TorchVision not available, object detection capabilities will be limited")

try:
    import torch.cuda.amp
    MIXED_PRECISION_AVAILABLE = True
except ImportError:
    MIXED_PRECISION_AVAILABLE = False
    logging.warning("Mixed precision not available, performance might be affected")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    
    # Ensure required NLTK resources are available
    nltk_data_path = os.environ.get('NLTK_DATA', os.path.join(os.path.expanduser('~'), 'nltk_data'))
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.data.path.append(nltk_data_path)
    
    required_resources = ['punkt', 'stopwords', 'wordnet']
    for resource in required_resources:
        try:
            nltk.data.find(f'{resource}')
        except LookupError:
            nltk.download(resource, quiet=True, download_dir=nltk_data_path)
    
    NLTK_AVAILABLE = True
    STOPWORDS = set(stopwords.words('english'))
    LEMMATIZER = WordNetLemmatizer()
except ImportError:
    nltk = None
    stopwords = None
    word_tokenize = None
    sent_tokenize = None
    WordNetLemmatizer = None
    NLTK_AVAILABLE = False
    STOPWORDS = set()
    LEMMATIZER = None
    logging.warning("NLTK not available, text analysis capabilities will be limited")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract not available, OCR capabilities will be limited")

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Constants
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CACHE_DIR = os.environ.get("MODEL_DIR", "./models")
IMPORTANCE_THRESHOLD = float(os.environ.get("IMPORTANCE_THRESHOLD", "0.65"))  # Default threshold for considering content important
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))  # Default batch size for processing
DEFAULT_CACHE_SIZE = int(os.environ.get("CACHE_SIZE", "1000"))  # Default cache size for model predictions
MAX_WORKERS = min(int(os.environ.get("MAX_WORKERS", "8")), os.cpu_count() or 4)  # Maximum number of thread workers
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.75"))  # Threshold for high-confidence predictions
USE_MIXED_PRECISION = os.environ.get("USE_MIXED_PRECISION", "1") == "1" and MIXED_PRECISION_AVAILABLE  # Whether to use mixed precision
ENABLE_PROFILING = os.environ.get("ENABLE_PROFILING", "0") == "1"  # Whether to enable detailed profiling
DEFAULT_OCR_LANG = os.environ.get("OCR_LANG", "eng")  # Default language for OCR
DOMAIN_SPECIFIC_TERMS_PATH = os.environ.get("DOMAIN_TERMS", os.path.join(MODEL_CACHE_DIR, "domain_terms.json"))

# Create model cache directory if it doesn't exist
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Domain-specific terms for importance calculation
DOMAIN_SPECIFIC_TERMS = {}
if os.path.exists(DOMAIN_SPECIFIC_TERMS_PATH):
    try:
        with open(DOMAIN_SPECIFIC_TERMS_PATH, 'r') as f:
            DOMAIN_SPECIFIC_TERMS = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load domain-specific terms: {e}")
else:
    # Create default domain terms file
    default_terms = {
        "programming": ["algorithm", "function", "class", "variable", "loop", "conditionals", "framework"],
        "data_science": ["model", "feature", "training", "validation", "accuracy", "precision", "recall"],
        "design": ["layout", "color", "typography", "spacing", "contrast", "hierarchy", "components"]
    }
    try:
        with open(DOMAIN_SPECIFIC_TERMS_PATH, 'w') as f:
            json.dump(default_terms, f, indent=2)
        DOMAIN_SPECIFIC_TERMS = default_terms
    except Exception as e:
        logger.warning(f"Failed to create default domain-specific terms file: {e}")

# Set up global profiler if enabled
if ENABLE_PROFILING:
    try:
        from torch.profiler import profile, record_function, ProfilerActivity
        PROFILER_ACTIVITIES = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            PROFILER_ACTIVITIES.append(ProfilerActivity.CUDA)
    except ImportError:
        ENABLE_PROFILING = False
        logging.warning("PyTorch profiler not available, disabling profiling")


def profile_function(func):
    """Decorator for profiling functions."""
    if not ENABLE_PROFILING:
        return func
    
    def wrapper(*args, **kwargs):
        function_name = func.__name__
        with profile(activities=PROFILER_ACTIVITIES, profile_memory=True, record_shapes=True) as prof:
            with record_function(function_name):
                result = func(*args, **kwargs)
        
        # Log profiling results
        logger.debug(f"Profiling results for {function_name}:\n{prof.key_averages().table(sort_by='cpu_time_total', row_limit=10)}")
        return result
    
    return wrapper


def convert_to_onnx(model: nn.Module, input_shape: Tuple[int, ...], output_path: str) -> bool:
    """Convert PyTorch model to ONNX format for optimized inference."""
    if not ONNX_AVAILABLE:
        logger.warning("ONNX Runtime not available, skipping conversion")
        return False
    
    try:
        # Create dummy input based on input shape
        dummy_input = torch.randn(*input_shape, device=model.device)
        
        # Set model to evaluation mode
        model.eval()
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        logger.info(f"Successfully converted model to ONNX: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to convert model to ONNX: {e}")
        return False


@dataclass
class ImportanceScore:
    """Data class representing importance scores for a segment of content."""
    value: float                     # Overall importance value (0-1)
    confidence: float                # Confidence in the importance assessment (0-1)
    modality: str                    # Which modality this score is for (visual, audio, transcript)
    features: Dict[str, float]       # Detailed features that contributed to the score
    segment_id: str                  # Identifier for the segment
    timestamp: Tuple[float, float]   # Start and end times for this segment
    context_ids: List[str] = field(default_factory=list)  # IDs of contextually related segments
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "value": float(self.value),
            "confidence": float(self.confidence),
            "modality": self.modality,
            "features": {k: float(v) for k, v in self.features.items()},
            "segment_id": self.segment_id,
            "timestamp": (float(self.timestamp[0]), float(self.timestamp[1])),
            "context_ids": self.context_ids,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImportanceScore':
        """Create instance from dictionary."""
        return cls(
            value=data["value"],
            confidence=data["confidence"],
            modality=data["modality"],
            features=data["features"],
            segment_id=data["segment_id"],
            timestamp=tuple(data["timestamp"]),
            context_ids=data.get("context_ids", []),
            metadata=data.get("metadata", {})
        )
    
    def is_important(self, threshold: float = IMPORTANCE_THRESHOLD) -> bool:
        """Determine if segment is important based on threshold."""
        return self.value >= threshold
    
    def get_key_features(self, top_n: int = 3) -> List[Tuple[str, float]]:
        """Get top N features contributing to importance score."""
        return sorted(self.features.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def merge_with(self, other: 'ImportanceScore', weight: float = 0.5) -> 'ImportanceScore':
        """Merge with another importance score, using weight to balance between them."""
        if self.segment_id != other.segment_id or self.modality != other.modality:
            raise ValueError(f"Cannot merge scores with different segment_id or modality")
        
        # Merge features, taking max values
        merged_features = {**self.features}
        for k, v in other.features.items():
            if k in merged_features


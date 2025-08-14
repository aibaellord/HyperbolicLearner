#!/usr/bin/env python3
"""
HyperbolicLearner Agent Integration Demo

This script demonstrates how to integrate the RealtimeAgent with the main 
HyperbolicLearner system, showing how to set up the agent to learn from
user interactions, take over tasks when the user is away, and handle
token limitations in LLM interactions.

Features demonstrated:
- Learning from tutorial videos at hyperbolic acceleration (up to 30x)
- Extracting multi-dimensional knowledge and UI patterns from videos
- Training an agent on nuanced communication patterns and user behaviors
- Setting up context-aware autonomous operation with safety mechanisms
- Intelligent token limit management with dynamic conversation handling
- Seamless handoff between user and agent control
- Error recovery and system resilience with state preservation
- Multi-modal understanding across visual, audio, and text channels
- Real-time learning and adaptation during execution

Usage:
    python agent_integration_demo.py [options]

Options:
    --video URL           YouTube video URL to learn from
    --period MINUTES      Learning period in minutes (default: 15)
    --tokens NUMBER       Maximum tokens for LLM interactions (default: 4096)
    --verbose             Enable verbose logging
    --gpu                 Enable GPU acceleration
    --autonomous          Enable autonomous mode immediately
    --checkpoint FILE     Load state from checkpoint file
    --export PATH         Export learned knowledge to specified path
    --interactive         Run in interactive mode with user prompting
    --trust-level LEVEL   Set agent trust level (low|medium|high)

Author: HyperbolicLearner Development Team
Version: 1.0.0
License: MIT
"""

import argparse
import logging
import os
import sys
import time
import threading
import json
import pickle
import signal
import traceback
import tempfile
import uuid
from typing import Optional, Dict, Any, List, Tuple, Callable, Union, Set, TypeVar, Generic
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, field

# Add the parent directory to the path so we can import the src package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent_demo.log")
    ]
)
logger = logging.getLogger("AgentDemo")

# Try importing the HyperbolicLearner components
try:
    from src import HyperbolicLearner
    from agents import RealtimeAgent
    from core.config import SystemConfig
    from src.ml_engine.content_analyzer import ContentImportance, ContentAnalyzer
    from src.knowledge_base.graph_db import KnowledgeNode, Relationship, KnowledgeGraph
    from src.video_processor.accelerator import VideoAccelerator
    from src.action_executor.executor import ActionExecutor
    from src.core.exceptions import (
        HyperbolicLearnerError, 
        TokenLimitError, 
        VideoProcessingError,
        KnowledgeExtractionError,
        AgentCommunicationError
    )
except ImportError as e:
    logger.critical(f"Failed to import required modules: {e}")
    logger.critical("Make sure you're running this script from the HyperbolicLearner directory")
    sys.exit(1)


# Type variable for generic functions
T = TypeVar('T')


@dataclass
class ProcessingStatistics:
    """Statistics about processing operations"""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    samples_processed: int = 0
    success_rate: float = 1.0
    error_count: int = 0
    warnings_count: int = 0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    def complete(self) -> 'ProcessingStatistics':
        """Mark processing as complete and calculate duration"""
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        return self
    
    def add_error(self) -> 'ProcessingStatistics':
        """Increment error count and update success rate"""
        self.error_count += 1
        if self.samples_processed > 0:
            self.success_rate = 1.0 - (self.error_count / self.samples_processed)
        return self
    
    def add_warning(self) -> 'ProcessingStatistics':
        """Increment warning count"""
        self.warnings_count += 1
        return self
    
    def add_sample(self) -> 'ProcessingStatistics':
        """Increment sample count and update success rate"""
        self.samples_processed += 1
        if self.samples_processed > 0:
            self.success_rate = 1.0 - (self.error_count / self.samples_processed)
        return self
    
    def update_resource_usage(self, resource: str, value: float) -> 'ProcessingStatistics':
        """Update resource usage statistics"""
        self.resource_usage[resource] = value
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'samples_processed': self.samples_processed,
            'success_rate': self.success_rate,
            'error_count': self.error_count,
            'warnings_count': self.warnings_count,
            'resource_usage': self.resource_usage
        }


class AgentIntegrationDemo:
    """
    Demonstrates the integration between HyperbolicLearner and RealtimeAgent
    
    This class provides a comprehensive example of how these systems can work
    together to create a powerful autonomous learning and execution environment.
    The integration enables several advanced capabilities:
    
    1. Hyperbolic Learning: Extract knowledge from video tutorials at accelerated rates,
       understanding both the content and the interactions demonstrated.
    
    2. Communication Style Replication: Learn and precisely mimic a user's communication 
       patterns, including tone, vocabulary, pacing, emoji usage, formatting preferences,
       technical density, and conversational flow.
    
    3. Autonomous Operation: Take over tasks when the user is away, handling both
       expected workflows and adapting to unexpected situations using the knowledge
       extracted from videos and user behavior.
    
    4. Token Limit Management: Intelligently handle LLM context restrictions through
       techniques like conversation splitting, context compression, and important
       information preservation.
    
    5. Multimodal Understanding: Process and correlate information across visual,
       audio, and textual channels to build a comprehensive understanding of tasks.
    
    6. Adaptive Execution: Apply learned patterns to new contexts, adapting to minor
       UI changes or workflow variations using similarity matching and fallback strategies.
    
    7. Safety-first Operation: Implement multiple safety layers including action
       validation, restricted operations, emergency stops, and verification steps.
    
    The demo is designed to be both educational and practical, showing real-world
    use cases for the HyperbolicLearner and RealtimeAgent integration.
    """
    
    def __init__(
        self, 
        video_url: Optional[str] = None,
        learning_period_minutes: int = 15,
        max_tokens: int = 4096,
        verbose: bool = True,
        checkpoint_dir: str = "./checkpoints",
        use_gpu: bool = False,
        enable_autonomous: bool = False,
        checkpoint_file: Optional[str] = None,
        export_path: Optional[str] = None,
        interactive: bool = False,
        trust_level: str = "medium"
    ):
        """
        Initialize the demonstration with configuration parameters
        
        Args:
            video_url: YouTube URL to learn from (optional)
            learning_period_minutes: How long the agent should learn from user 
                                    interactions before taking over
            max_tokens: Maximum tokens for LLM interactions
            verbose: Whether to print detailed progress information
            checkpoint_dir: Directory to save state checkpoints
            use_gpu: Whether to enable GPU acceleration
            enable_autonomous: Whether to enable autonomous mode immediately
            checkpoint_file: Optional checkpoint file to restore from
            export_path: Optional path to export learned knowledge
            interactive: Whether to run in interactive mode with user prompting
            trust_level: Agent trust level (low|medium|high)
        """
        self.video_url = video_url
        self.learning_period = learning_period_minutes * 60  # Convert to seconds
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir
        self.use_gpu = use_gpu
        self.enable_autonomous = enable_autonomous
        self.checkpoint_file = checkpoint_file
        self.export_path = export_path
        self.interactive = interactive
        self.trust_level = trust_level
        
        # Validate trust level
        if self.trust_level not in ("low", "medium", "high"):
            logger.warning(f"Invalid trust level '{self.trust_level}', defaulting to 'medium'")
            self.trust_level = "medium"
        
        # Internal state
        self._running = True
        self._initialized = False
        self._training_complete = False
        self._last_error = None
        self._active_threads = set()
        self._session_id = str(uuid.uuid4())
        self._shutdown_hooks = []
        
        # Statistics tracking
        self.stats = {
            "video_processing": ProcessingStatistics(),
            "agent_training": ProcessingStatistics(),
            "autonomous_operation": ProcessingStatistics(),
            "token_management": ProcessingStatistics()
        }
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.export_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.export_path)), exist_ok=True)
        
        # Handle signals for clean shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        # Print welcome message
        self._print_welcome()
        
        try:
            # Initialize the systems
            logger.info("Initializing HyperbolicLearner system...")
            self.config = self._initialize_config()
            self.learner = self._initialize_learner()
            self.agent = self._initialize_agent()
            self.executor = self._initialize_executor()
            
            # State tracking
            self.knowledge_graph_id = None
            self.is_agent_active = False
            self.user_communication_style = {}
            self.token_usage_history = []
            self.last_checkpoint_time = time.time()
            self.training_stats = {}
            self.video_processing_stats = {}
            self.session_start_time = datetime.now()
            
            # Event handlers and callbacks
            self._register_event_handlers()
            
            # Recovery or checkpoint loading
            if self.checkpoint_file:
                self._load_checkpoint(self.checkpoint_file)
                
            self._initialized = True
            logger.info("AgentIntegrationDemo initialized successfully")
            
            # Register shutdown hook for cleanup
            self._add_shutdown_hook(self._save_final_checkpoint)
            
        except Exception as e:
            logger.critical(f"Failed to initialize AgentIntegrationDemo: {e}")
            traceback.print_exc()
            raise
    
    def _add_shutdown_hook(self, hook_func: Callable[[], None]):
        """Add a function to be called during shutdown"""
        self._shutdown_hooks.append(hook_func)
            
    def _print_welcome(self):
        """Display welcome message and system information"""
        logger.info("=" * 80)
        logger.info("HyperbolicLearner Agent Integration Demo")
        logger.info("A demonstration of advanced video learning and autonomous agent capabilities")
        logger.info("-" * 80)
        logger.info(f"Session ID: {self._session_id}")
        logger.info(f"Video URL: {self.video_url or 'Not specified'}")
        logger.info(f"Learning period: {self.learning_period // 60} minutes")
        logger.info(f"GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
        logger.info(f"Autonomous mode: {'Enabled' if self.enable_autonomous else 'Disabled'}")
        logger.info(f"Token limit: {self.max_tokens}")
        logger.info(f"Trust level: {self.trust_level}")
        logger.info(f"Interactive mode: {'Enabled' if self.interactive else 'Disabled'}")
        logger.info(f"Export path: {self.export_path or 'Not specified'}")
        logger.info("=" * 80)
            
    def _initialize_config(self) -> SystemConfig:
        """
        Set up system configuration with optimal settings for integration
        
        This method configures all aspects of the system including resource allocation,
        model settings, data storage locations, and feature enablement. It optimizes
        the configuration for the agent-learner integration scenario.
        
        The configuration settings include:
        - Hardware resource allocation (CPU, GPU, memory)
        - LLM settings (provider, model, token limits)
        - Storage paths for different types of data
        - Processing parameters (video acceleration, UI detection thresholds)
        - Advanced feature toggles for optimal integration
        - Security and privacy settings
        - Performance optimizations
        
        Returns:
            SystemConfig: The initialized configuration object
        """
        try:
            config = SystemConfig()
            
            # Hardware resource configuration
            config.set_gpu_enabled(self.use_gpu)
            if self.use_gpu:
                config.set_gpu_utilization(0.7)  # Use 70% of GPU resources
                config.set_gpu_memory_limit(4096)  # 4GB GPU memory limit
                config.enable_feature("tensor_optimization")
                config.set_cuda_device_index(0)  # Use first CUDA device
                config.set_half_precision(True)  # Use FP16 for faster computation
            
            config.set_cpu_threads("auto")   # Auto-determine optimal thread count
            config.set_memory_limit(4096)    # Limit memory usage to 4GB
            config.set_disk_cache_limit(10240)  # 10GB disk cache limit
            config.set_network_concurrency(5)  # 5 concurrent network requests
            
            # LLM settings
            config.set_llm_provider("openai")
            config.set_default_model("gpt-4")
            config.set_token_buffer(0.2)  # Keep 20% token buffer for safety
            config.set_request_timeout(30)  # 30 second timeout for API requests
            config.set_retry_strategy("exponential_backoff")
            config.set_max_retries(3)
            config.set_fallback_model("gpt


"""
Advanced Action Executor Module for HyperbolicLearner

This module implements a state-of-the-art system for translating learned UI interactions 
into executable actions across any operating system. It features:

- Neural-enhanced UI element recognition
- Adaptive execution timing
- Intelligent error recovery
- Self-learning execution patterns
- Context-aware action adaptation
- Cross-platform compatibility
- Verification and validation systems

The ActionExecutor serves as the execution engine for the HyperbolicLearner system,
enabling learned knowledge to be accurately applied to real-world applications.
"""

import os
import time
import json
import uuid
import base64
import logging
import platform
import tempfile
import subprocess
import threading
import concurrent.futures
from enum import Enum, auto
from typing import Dict, List, Tuple, Union, Optional, Callable, Any, Generator, TypeVar, Generic
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from datetime import datetime
import traceback
import numpy as np
import cv2
from pathlib import Path

# Import our internal modules
from ..core.config import SystemConfig
from ..knowledge_base.graph_db import KnowledgeNode, ActionSequence
from ..ml_engine.content_analyzer import UIElementDetector
from ..core.exceptions import ExecutionError, ElementNotFoundError, ValidationError, TimeoutError

# Conditionally import platform-specific modules
try:
    import pyautogui
    import pynput
    import mss
    import pytesseract
    from PIL import Image
    
    if platform.system() == "Windows":
        import win32gui
        import win32con
        import win32api
        import win32process
        import ctypes
        from ctypes import wintypes
    elif platform.system() == "Darwin":  # macOS
        import Quartz
        import AppKit
        import objc
    else:  # Linux
        import Xlib
        import Xlib.display
        from Xlib import X, XK
        import wnck
except ImportError as e:
    logging.warning(f"Some platform-specific modules couldn't be imported: {e}")
    logging.warning("Functionality might be limited. Install required packages.")

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for generics
T = TypeVar('T')
R = TypeVar('R')


class ActionType(Enum):
    """Comprehensive enumeration of supported action types with detailed descriptions."""
    # Basic mouse actions
    CLICK = auto()                # Simple mouse click
    RIGHT_CLICK = auto()          # Right mouse button click
    DOUBLE_CLICK = auto()         # Double click
    MIDDLE_CLICK = auto()         # Middle mouse button click
    DRAG = auto()                 # Click and drag
    SCROLL = auto()               # Scroll wheel action
    HOVER = auto()                # Move and hover over location
    
    # Keyboard actions
    TYPE = auto()                 # Type text
    HOTKEY = auto()               # Press multiple keys simultaneously
    KEY_DOWN = auto()             # Press and hold a key
    KEY_UP = auto()               # Release a held key
    
    # Advanced actions
    FIND_UI_ELEMENT = auto()      # Find UI element using vision
    FIND_AND_CLICK = auto()       # Find element and click
    FIND_AND_TYPE = auto()        # Find input field and type
    WAIT_FOR_ELEMENT = auto()     # Wait for element to appear
    EXECUTE_UNTIL = auto()        # Execute actions until condition
    
    # System actions
    LAUNCH_APP = auto()           # Launch an application
    CLOSE_APP = auto()            # Close an application
    EXECUTE_COMMAND = auto()      # Execute a shell command
    SWITCH_WINDOW = auto()        # Switch to different window
    SET_CLIPBOARD = auto()        # Set clipboard content
    GET_CLIPBOARD = auto()        # Get clipboard content
    
    # Workflow actions
    WAIT = auto()                 # Wait for specified time
    SEQUENCE = auto()             # Execute a sequence of actions
    CONDITIONAL = auto()          # Conditional execution
    REPEAT = auto()               # Repeat actions N times
    PARALLEL = auto()             # Execute actions in parallel
    
    # Visual verification
    VERIFY_ELEMENT = auto()       # Verify UI element exists
    VERIFY_IMAGE = auto()         # Verify image on screen
    VERIFY_TEXT = auto()          # Verify text on screen
    CAPTURE_SCREEN = auto()       # Capture screenshot
    COMPARE_IMAGES = auto()       # Compare two images
    
    def __str__(self):
        return self.name.lower()


@dataclass
class UIElement:
    """Represents a UI element with comprehensive properties."""
    element_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    element_type: str = ""  # button, text_field, checkbox, etc.
    text: Optional[str] = None
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height
    confidence: float = 0.0
    image_hash: Optional[str] = None  # Perceptual hash of element image
    ocr_text: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    parent_element: Optional[str] = None
    children_elements: List[str] = field(default_factory=list)
    
    @property
    def center(self) -> Optional[Tuple[int, int]]:
        """Get the center coordinates of the element."""
        if self.bounding_box:
            x, y, width, height = self.bounding_box
            return (x + width // 2, y + height // 2)
        return None


@dataclass
class ExecutionContext:
    """Rich context for action execution with state tracking and adaptability."""
    app_name: Optional[str] = None
    window_title: Optional[str] = None
    active_elements: Dict[str, UIElement] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    last_action_time: float = field(default_factory=time.time)
    execution_speed: float = 1.0  # Multiplier for timing (0.5 = slower, 2.0 = faster)
    retry_strategy: str = "exponential_backoff"  # or "fixed", "linear", "none"
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_context: Optional['ExecutionContext'] = None
    start_time: float = field(default_factory=time.time)
    screenshots: List[Tuple[float, str]] = field(default_factory=list)  # (timestamp, path)
    error_count: int = 0
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a variable with inheritance from parent contexts."""
        if name in self.variables:
            return self.variables[name]
        elif self.parent_context:
            return self.parent_context.get_variable(name, default)
        return default
    
    def add_screenshot(self, image_path: str) -> None:
        """Add a screenshot to the context for later analysis."""
        self.screenshots.append((time.time(), image_path))
    
    def clone(self) -> 'ExecutionContext':
        """Create a new context inheriting from this one."""
        return ExecutionContext(
            app_name=self.app_name,
            window_title=self.window_title,
            variables=dict(self.variables),
            execution_speed=self.execution_speed,
            retry_strategy=self.retry_strategy,
            parent_context=self
        )


@dataclass
class ActionResult:
    """Comprehensive result of an executed action with detailed analytics."""
    success: bool
    action_type: ActionType
    start_time: float
    end_time: float = field(default_factory=time.time)
    params: Dict[str, Any] = field(default_factory=dict)
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    error_traceback: Optional[str] = None
    screenshot_before: Optional[str] = None
    screenshot_after: Optional[str] = None
    execution_context: Optional[ExecutionContext] = None
    retry_count: int = 0
    confidence: float = 1.0
    
    @property
    def duration(self) -> float:
        """Get the duration of the action execution in seconds."""
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling complex nested objects."""
        result = asdict(self)
        # Handle any complex conversions here
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string, handling complex nested objects."""
        return json.dumps(self.to_dict())


@dataclass
class Action:
    """Represents a sophisticated action to be executed with extensive configuration options."""
    action_type: ActionType
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Execution control
    timeout: float = 30.0
    retry_count: int = 3
    retry_delay: float = 1.0
    retry_backoff_factor: float = 1.5
    
    # Validation & verification
    pre_conditions: List[Callable[[ExecutionContext], bool]] = field(default_factory=list)
    post_conditions: List[Callable[[ActionResult, ExecutionContext], bool]] = field(default_factory=list)
    verification_actions: List['Action'] = field(default_factory=list)
    
    # Recovery strategies
    recovery_actions: List['Action'] = field(default_factory=list)
    on_failure: Optional[Callable[['Action', ActionResult, ExecutionContext], None]] = None
    
    # Adaptability
    adaptive_timing: bool = True
    context_aware: bool = True
    
    # Metadata
    name: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    source_node: Optional[str] = None  # Reference to knowledge graph node
    
    def with_param(self, key: str, value: Any) -> 'Action':
        """Return a new action with an updated parameter."""
        new_params = dict(self.params)
        new_params[key] = value
        return Action(
            action_type=self.action_type,
            params=new_params,
            timeout=self.timeout,
            retry_count=self.retry_count,
            retry_delay=self.retry_delay,
            retry_backoff_factor=self.retry_backoff_factor,
            pre_conditions=self.pre_conditions,
            post_conditions=self.post_conditions,
            verification_actions=self.verification_actions,
            recovery_actions=self.recovery_actions,
            on_failure=self.on_failure,
            adaptive_timing=self.adaptive_timing,
            context_aware=self.context_aware,
            name=self.name,
            description=self.description,
            tags=self.tags,
            source_node=self.source_node
        )


class RetryStrategy:
    """Encapsulates different retry strategies for failed actions."""
    
    @staticmethod
    def no_retry(base_delay: float, attempt: int) -> float:
        """No retry, return 0 delay."""
        return 0
    
    @staticmethod
    def fixed_delay(base_delay: float, attempt: int) -> float:
        """Fixed delay between retries."""
        return base_delay
    
    @staticmethod
    def linear_backoff(base_delay: float, attempt: int) -> float:
        """Linear increase in delay between retries."""
        return base_delay * attempt
    
    @staticmethod
    def exponential_backoff(base_delay: float, attempt: int, factor: float = 2.0) -> float:
        """Exponential increase in delay between retries."""
        return base_delay * (factor ** (attempt - 1))
    
    @staticmethod
    def get_strategy(name: str) -> Callable[[float, int], float]:
        """Get a retry strategy by name."""
        strategies = {
            "none": RetryStrategy.no_retry,
            "fixed": RetryStrategy.fixed_delay,
            "linear": RetryStrategy.linear_backoff,
            "exponential_backoff": RetryStrategy.exponential_backoff
        }
        return strategies.get(name, RetryStrategy.exponential_backoff)


class UIElementFinder:
    """
    Advanced UI element detection engine using multiple strategies:
    - Image recognition
    - OCR (text recognition)
    - Accessibility API integration
    - Machine learning-based object detection
    """
    
    def __init__(self, 
                 ml_detector: Optional[UIElementDetector] = None,
                 confidence_threshold: float = 0.7,
                 use_cached_results: bool = True,
                 cache_timeout: float = 5.0):
        """
        Initialize the UI element finder.
        
        Args:
            ml_detector: Optional ML-based UI element detector
            confidence_threshold: Minimum confidence for element detection
            use_cached_results: Whether to cache and reuse detection results
            cache_timeout: How long cached results remain valid (seconds)
        """
        self.ml_detector = ml_detector
        self.confidence_threshold = confidence_threshold
        self.use_cached_results = use_cached_results
        self.cache_timeout = cache_timeout
        self._cache = {}  # Cache of recently found elements
        self._cache_timestamps = {}  # When elements were cached
        
        # Initialize OCR engine
        try:
            self.ocr_available = pytesseract.get_tesseract_version() is not None
            logger.info(f"OCR engine initialized, version: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            logger.warning(f"OCR engine initialization failed: {e}")
            self.ocr_available = False
    
    def find_element_by_image(self, template_path: str, confidence: float = 0.8) -> Optional[UIElement]:
        """
        Find UI element matching the given template image.
        
        Args:
            template_path: Path to template image
            confidence: Minimum match confidence
            
        Returns:
            UIElement if found, None otherwise
        """
        # Check cache first
        cache_key = f"image:{template_path}:{confidence}"
        if self.use_cached_results and cache_key in self._cache:
            if time.time() - self._cache_timestamps[cache_key] < self.cache_timeout:
                return self._cache[cache_key]
        
        # Take screenshot
        with mss.mss() as sct:
            monitor = sct.monitors[0]  # Primary monitor
            screenshot = np.array(sct.grab(monitor))
        
        # Load template
        template = cv2.imread(template_path)
        
        # Handle different color formats (BGR vs RGB)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        
        # Perform template matching
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.min


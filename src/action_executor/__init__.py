"""
Action Executor Module
======================

This module provides robust functionality for executing learned UI interactions
on the user's system across different operating systems. It translates abstract
action representations from the knowledge graph into concrete system interactions.

Key capabilities:
- Cross-platform UI automation (Windows, macOS, Linux)
- Multiple interaction types (clicks, typing, drag-drop, etc.)
- Verification of action outcomes
- Error recovery strategies
- Adaptive execution based on UI state

This module serves as the bridge between learned knowledge and real-world execution.
"""

__version__ = "0.1.0"
__author__ = "HyperbolicLearner Team"
__license__ = "MIT"

from typing import Dict, List, Optional, Union, Any

from .executor import (
    ActionExecutor,
    UIAction,
    ActionResult,
    ExecutionContext,
    ActionSequence,
    VerificationStrategy,
    ExecutionError,
    ActionType,
    UIElement
)

# Constants
DEFAULT_TIMEOUT = 10.0  # seconds
MAX_RETRY_ATTEMPTS = 3

# Common exceptions
class ActionExecutorError(Exception):
    """Base exception for all action executor errors."""
    pass

class VerificationError(ActionExecutorError):
    """Raised when action verification fails."""
    pass

class UnsupportedPlatformError(ActionExecutorError):
    """Raised when an action is not supported on the current platform."""
    pass

# Convenience function for getting an executor instance
def get_executor(
    config: Optional[Dict[str, Any]] = None, 
    verification_enabled: bool = True,
    platform: Optional[str] = None
) -> ActionExecutor:
    """
    Get a configured ActionExecutor instance.
    
    Args:
        config: Optional configuration dictionary
        verification_enabled: Whether to verify action outcomes
        platform: Override platform detection with specific platform
        
    Returns:
        Configured ActionExecutor instance
    """
    return ActionExecutor(
        config=config or {},
        verification_strategy=VerificationStrategy.VISUAL if verification_enabled else VerificationStrategy.NONE,
        platform=platform
    )

__all__ = [
    # Classes
    "ActionExecutor",
    "UIAction",
    "ActionResult",
    "ExecutionContext", 
    "ActionSequence",
    "VerificationStrategy",
    "UIElement",
    
    # Exceptions
    "ExecutionError",
    "ActionExecutorError",
    "VerificationError",
    "UnsupportedPlatformError",
    
    # Enums
    "ActionType",
    
    # Functions
    "get_executor",
    
    # Constants
    "DEFAULT_TIMEOUT",
    "MAX_RETRY_ATTEMPTS"
]


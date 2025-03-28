"""
HyperbolicLearner Agents Module

This module provides intelligent agent implementations for the HyperbolicLearner system
that can monitor, learn from, and autonomously replicate user behavior.

The primary offering is the RealtimeAgent class, which can:
- Learn communication styles through natural language analysis
- Monitor and replicate UI interactions
- Autonomously operate when users are away
- Manage LLM token limitations through conversation management
- Dynamically adapt to changing contexts and requirements

Usage:
    from hyperbolic_learner.agents import create_agent
    
    # Create a default agent
    agent = create_agent()
    
    # Start monitoring and learning
    agent.start()
    
    # Configure for takeover in 2 hours
    agent.schedule_takeover(hours=2)
"""

from typing import Optional, Dict, Any, List, Callable, Union, Tuple
import logging
import os
import sys
from pathlib import Path

# Import the RealtimeAgent class and related components
from .realtime_agent import (
    RealtimeAgent,
    UserInteraction,
    InteractionType,
    AgentConfig,
    AgentState,
    CommunicationStyle,
    TakeoverMode,
    TokenManagementStrategy,
    TerminalMonitor,
    InteractionMonitor
)

__version__ = "0.1.0"
__author__ = "HyperbolicLearner Team"

# Configure module-level logging
logger = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_handler.setFormatter(_formatter)
logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# Default configuration paths
_DEFAULT_CONFIG_DIR = Path.home() / ".hyperbolic_learner"
_DEFAULT_CONFIG_FILE = _DEFAULT_CONFIG_DIR / "agent_config.json"

def create_agent(
    config: Optional[Dict[str, Any]] = None,
    style_learning_rate: float = 0.15,
    auto_takeover: bool = False,
    takeover_duration_hours: Optional[float] = None,
    ui_interaction_enabled: bool = True,
    communication_style: Optional[CommunicationStyle] = None,
    token_management: Union[bool, TokenManagementStrategy] = True,
    monitor_terminals: Optional[List[str]] = None,
    response_delay_seconds: float = 0.5,
    fallback_handler: Optional[Callable[[Exception], Any]] = None,
    persistence_enabled: bool = True,
    log_level: int = logging.INFO,
    config_file: Optional[Union[str, Path]] = None,
) -> RealtimeAgent:
    """
    Create and configure a RealtimeAgent instance with the specified settings.
    
    Args:
        config: Optional dictionary with configuration parameters that override other arguments
        style_learning_rate: How quickly the agent adapts to user communication style (0.0-1.0)
        auto_takeover: Whether the agent should automatically take over when user is inactive
        takeover_duration_hours: Maximum duration for auto-takeover mode
        ui_interaction_enabled: Whether UI interaction capabilities are enabled
        communication_style: Pre-configured communication style or None to learn from user
        token_management: Enable automatic token limit handling or specify a strategy
        monitor_terminals: List of terminal names/paths to monitor (None for auto-detection)
        response_delay_seconds: Delay between detecting an event and responding
        fallback_handler: Function to call when agent encounters an error
        persistence_enabled: Whether to save learned behavior to disk
        log_level: Logging level for the agent module
        config_file: Path to a JSON configuration file (overrides other arguments)
        
    Returns:
        A configured RealtimeAgent instance ready to use
    
    Raises:
        ValueError: If configuration parameters are invalid
        FileNotFoundError: If specified config_file does not exist
    """
    # Set module logging level
    logger.setLevel(log_level)
    
    # Load from config file if specified
    if config_file:
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        import json
        with open(config_path, 'r') as f:
            file_config = json.load(f)
        
        # File config takes precedence over passed config
        if config:
            file_config.update(config)
        config = file_config
    
    # Convert token_management to strategy if it's a boolean
    if isinstance(token_management, bool):
        token_management = (
            TokenManagementStrategy.AUTOMATIC if token_management 
            else TokenManagementStrategy.DISABLED
        )
    
    # Create base configuration
    agent_config = AgentConfig(
        style_learning_rate=style_learning_rate,
        auto_takeover=auto_takeover,
        takeover_duration_hours=takeover_duration_hours or 4.0,
        ui_interaction_enabled=ui_interaction_enabled,
        token_management_strategy=token_management,
        response_delay_seconds=response_delay_seconds,
        monitor_terminals=monitor_terminals or [],
        persistence_enabled=persistence_enabled,
    )
    
    # Override with any provided config dict
    if config:
        for key, value in config.items():
            if hasattr(agent_config, key):
                setattr(agent_config, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
    
    # Validate configuration
    if agent_config.style_learning_rate < 0.0 or agent_config.style_learning_rate > 1.0:
        raise ValueError("style_learning_rate must be between 0.0 and 1.0")
    
    if agent_config.takeover_duration_hours <= 0:
        raise ValueError("takeover_duration_hours must be positive")
    
    # Create the agent
    agent = RealtimeAgent(config=agent_config)
    
    # Set communication style if provided
    if communication_style:
        agent.set_communication_style(communication_style)
    
    # Set fallback handler if provided
    if fallback_handler:
        agent.set_fallback_handler(fallback_handler)
    
    return agent

def create_communication_agent(
    learning_rate: float = 0.25,
    token_management: bool = True,
    persistence_path: Optional[str] = None,
    style_model: Optional[str] = None,
) -> RealtimeAgent:
    """
    Create a RealtimeAgent focused on learning and mimicking communication style.
    
    This specialized agent prioritizes natural language understanding and 
    communication style adaptation while minimizing UI automation features.
    
    Args:
        learning_rate: How quickly the agent adapts to user communication style (0.0-1.0)
        token_management: Enable automatic token limit handling
        persistence_path: Where to save the learned communication style
        style_model: Optional pre-trained model to use as starting point
        
    Returns:
        A RealtimeAgent configured for communication style learning
    """
    config = {
        "style_learning_rate": learning_rate,
        "ui_interaction_enabled": False,
        "token_management": token_management,
        "communication_priority": True,
    }
    
    if persistence_path:
        config["persistence_path"] = persistence_path
    
    agent = create_agent(**config)
    
    if style_model:
        agent.load_style_model(style_model)
        
    # Optimize agent settings for communication focus
    agent.configure_nlp_pipeline(extended=True)
    
    return agent

def create_automation_agent(
    takeover_duration_hours: float = 4.0,
    auto_takeover: bool = True,
    ui_detection_sensitivity: float = 0.8,
    terminal_apps: List[str] = None,
) -> RealtimeAgent:
    """
    Create a RealtimeAgent focused on UI automation and takeover capabilities.
    
    This specialized agent prioritizes UI interaction detection and replication,
    with capabilities to completely take over terminal operation while the user is away.
    
    Args:
        takeover_duration_hours: Maximum duration for auto-takeover mode
        auto_takeover: Whether the agent should automatically take over when user is inactive
        ui_detection_sensitivity: How sensitive UI element detection should be (0.0-1.0)
        terminal_apps: List of terminal applications to monitor and control
        
    Returns:
        A RealtimeAgent configured for UI automation
    """
    if terminal_apps is None:
        terminal_apps = ["warp", "terminal", "iterm", "cmd", "powershell", "gnome-terminal"]
        
    config = {
        "style_learning_rate": 0.05,  # Lower learning rate, focusing more on actions
        "auto_takeover": auto_takeover,
        "takeover_duration_hours": takeover_duration_hours,
        "ui_interaction_enabled": True,
        "ui_detection_sensitivity": ui_detection_sensitivity,
        "monitor_terminals": terminal_apps,
        "takeover_mode": TakeoverMode.FULL,
    }
    
    agent = create_agent(**config)
    
    # Configure the interaction monitor for higher accuracy
    agent.configure_interaction_monitor(
        capture_interval=0.1,
        motion_detection=True,
        ocr_enabled=True
    )
    
    return agent

def get_agent_version() -> str:
    """Return the current version of the agents module."""
    return __version__

def list_available_terminals() -> List[Tuple[str, str]]:
    """
    Returns a list of available terminals that can be monitored by the agent.
    
    Returns:
        List of (terminal_name, terminal_path) tuples
    """
    return TerminalMonitor.detect_terminals()

def validate_agent_environment() -> Dict[str, bool]:
    """
    Validates the current environment for agent compatibility.
    
    Returns:
        Dictionary of feature names and their availability status
    """
    return {
        "screen_capture": InteractionMonitor.check_screen_capture_available(),
        "keyboard_control": InteractionMonitor.check_keyboard_control_available(),
        "mouse_control": InteractionMonitor.check_mouse_control_available(),
        "ocr_available": InteractionMonitor.check_ocr_available(),
        "nlp_available": RealtimeAgent.check_nlp_available(),
    }

# Clean namespace exports
__all__ = [
    # Main classes
    'RealtimeAgent',
    'UserInteraction',
    'InteractionType',
    'AgentConfig',
    'AgentState',
    'CommunicationStyle',
    'TakeoverMode',
    'TokenManagementStrategy',
    'TerminalMonitor',
    'InteractionMonitor',
    
    # Factory functions
    'create_agent',
    'create_communication_agent',
    'create_automation_agent',
    
    # Utility functions
    'get_agent_version',
    'list_available_terminals',
    'validate_agent_environment',
]


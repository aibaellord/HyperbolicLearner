"""
HyperbolicLearner - An advanced system for accelerated learning from videos and UI automation.

This package provides a comprehensive framework for downloading and processing
YouTube videos at accelerated speeds, extracting knowledge, detecting UI interactions,
and automating workflows based on the learned information.
"""

__version__ = '0.1.0'
__author__ = 'HyperbolicLearner Team'

# Import core components for easy access
from .core.config import ConfigManager, SystemCapabilities
from .video_processor.youtube_learner import YouTubeLearner
from .video_processor.downloader import VideoDownloader
from .video_processor.accelerator import VideoAccelerator
from .ui_automation.ui_analyzer import UIAnalyzer
from .knowledge_base.graph_db import KnowledgeGraph
from .ml_engine import MLEngine
from .web_crawler import WebCrawler

# Main application entry point
from .main import HyperbolicLearnerApp

# Define what's available when using `from hyperbolic_learner import *`
__all__ = [
    'HyperbolicLearnerApp',
    'ConfigManager', 
    'SystemCapabilities',
    'YouTubeLearner',
    'VideoDownloader',
    'VideoAccelerator',
    'UIAnalyzer',
    'KnowledgeGraph',
    'MLEngine',
    'WebCrawler',
]


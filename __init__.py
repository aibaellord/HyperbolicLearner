"""
HyperbolicLearner - Root package initialization

This file enables importing directly from the root package when installed.
It imports and exposes all the main components from the src package.
"""

from src import *

# Re-export version information
from src import __version__, __author__


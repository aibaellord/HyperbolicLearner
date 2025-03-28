"""
HyperbolicLearner UI Automation Components

This package handles detecting UI elements in videos, understanding interactions,
and creating automation scripts that can replicate those interactions.
"""

from .ui_analyzer import UIAnalyzer, UIElement, UIInteraction

# These classes are assumed to be implemented in future modules
class AutomationEngine:
    """Executes learned UI interactions on the local system."""
    pass

class VerificationSystem:
    """Verifies that automated actions produce the expected results."""
    pass

__all__ = [
    'UIAnalyzer',
    'UIElement',
    'UIInteraction',
    'AutomationEngine',
    'VerificationSystem',
]


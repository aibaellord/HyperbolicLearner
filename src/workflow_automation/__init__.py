"""
Workflow Automation Integration Module for HyperbolicLearner

This module provides integration with various workflow automation platforms
including n8n, Zapier, and custom automation systems.
"""

from .n8n_integration import N8NIntegrationManager, HyperbolicN8NBridge, WorkflowExecution

__all__ = [
    'N8NIntegrationManager',
    'HyperbolicN8NBridge', 
    'WorkflowExecution'
]

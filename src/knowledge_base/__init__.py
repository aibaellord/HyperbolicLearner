"""
HyperbolicLearner Knowledge Base Components

This package provides components for storing, retrieving, and reasoning about
knowledge extracted from videos and other sources.
"""

from .graph_db import KnowledgeGraph, KnowledgeNode, Relationship

# These classes are assumed to be implemented in future modules
class QueryEngine:
    """Provides sophisticated querying capabilities for the knowledge graph."""
    pass

class InferenceEngine:
    """Draws inferences and generates new knowledge based on existing data."""
    pass

__all__ = [
    'KnowledgeGraph',
    'KnowledgeNode',
    'Relationship',
    'QueryEngine',
    'InferenceEngine',
]


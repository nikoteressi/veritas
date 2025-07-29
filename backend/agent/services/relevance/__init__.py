"""
Relevance Services Module

This module contains all relevance-related services and components.
"""

from .relevance_orchestrator import RelevanceOrchestrator
from .relevance_embeddings_coordinator import RelevanceEmbeddingsCoordinator
from .cached_hybrid_relevance_scorer import CachedHybridRelevanceScorer
from .explainable_relevance_scorer import ExplainableRelevanceScorer

# Convenience functions


def get_relevance_manager():
    """Get a configured relevance orchestrator instance."""
    return RelevanceOrchestrator()


def close_relevance_manager(orchestrator):
    """Close and cleanup a relevance orchestrator instance."""
    if orchestrator:
        orchestrator.close()


__all__ = [
    'RelevanceOrchestrator',
    'RelevanceEmbeddingsCoordinator',
    'CachedHybridRelevanceScorer',
    'ExplainableRelevanceScorer',
    'get_relevance_manager',
    'close_relevance_manager'
]

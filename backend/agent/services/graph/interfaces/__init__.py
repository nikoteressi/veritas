"""
Abstract interfaces for the graph services.

This module defines the abstract interfaces that implement the Strategy,
Repository, and Factory patterns for the graph-based fact verification system.
"""

from .clustering_strategy import ClusteringStrategy
from .graph_repository import GraphRepository
from .storage_strategy import StorageStrategy
from .verification_repository import VerificationRepository
from .verification_strategy import VerificationStrategy

__all__ = [
    "ClusteringStrategy",
    "VerificationStrategy",
    "StorageStrategy",
    "GraphRepository",
    "VerificationRepository",
]

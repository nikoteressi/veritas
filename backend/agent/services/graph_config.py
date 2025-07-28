"""
Configuration classes for graph-based verification system.

This module contains configuration dataclasses used across the graph verification system
to avoid circular import issues.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class VerificationConfig:
    """Configuration for graph verification."""

    max_search_results: int = 10
    confidence_threshold: float = 0.7
    max_concurrent_verifications: int = 3
    max_concurrent_scrapes: int = 3  # Limit concurrent web scraping operations
    enable_cross_verification: bool = True
    enable_contradiction_detection: bool = True
    batch_size: int = 5


@dataclass
class ClusteringConfig:
    """Configuration for graph clustering."""

    similarity_threshold: float = 0.8
    min_cluster_size: int = 2
    max_cluster_size: int = 10
    clustering_algorithm: str = "hierarchical"  # "hierarchical", "kmeans", "dbscan"
    embedding_model: str = "nomic-embed-text"
    enable_optimization: bool = True
    merge_threshold: float = 0.9
    overlap_threshold: float = 0.7

    # DBSCAN specific parameters
    # Maximum distance between two samples for one to be considered as in the neighborhood of the other
    dbscan_eps: float = 0.5
    # Number of samples in a neighborhood for a point to be considered as a core point
    dbscan_min_samples: int = 2

    # Clustering type enablers
    enable_domain_clustering: bool = True
    enable_temporal_clustering: bool = True
    enable_causal_clustering: bool = True


@dataclass
class ClusterVerificationResult:
    """Result of verifying a cluster of facts."""

    cluster_id: str
    overall_verdict: str  # "TRUE", "FALSE", "MIXED", "INSUFFICIENT_EVIDENCE"
    confidence: float
    # node_id -> verification result
    individual_results: dict[str, dict[str, Any]]
    cross_verification_results: list[dict[str, Any]]
    contradictions_found: list[dict[str, Any]]
    supporting_evidence: list[dict[str, Any]]
    verification_time: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "overall_verdict": self.overall_verdict,
            "confidence": self.confidence,
            "individual_results": self.individual_results,
            "cross_verification_results": self.cross_verification_results,
            "contradictions_found": self.contradictions_found,
            "supporting_evidence": self.supporting_evidence,
            "verification_time": self.verification_time,
            "metadata": self.metadata,
        }

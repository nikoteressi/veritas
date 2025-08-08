"""Uncertainty analysis service for graph-based fact checking.

This service handles Bayesian uncertainty analysis for verification results,
providing confidence scores and uncertainty metrics.
"""

import logging
from typing import Any

from agent.services.analysis.bayesian_uncertainty import (
    BayesianUncertaintyHandler,
    UncertaintyConfig,
)

logger = logging.getLogger(__name__)


class UncertaintyAnalyzer:
    """Analyzes uncertainty in graph-based fact verification results.

    Uses Bayesian uncertainty analysis to provide confidence scores
    and uncertainty metrics for verification results.
    """

    def __init__(self):
        """Initialize the uncertainty analyzer."""
        self.uncertainty_handler = None
        self._uncertainty_handler_initialized = False
        self.logger = logging.getLogger(__name__)

    async def ensure_uncertainty_handler(self) -> None:
        """Ensure Bayesian uncertainty handler is initialized (lazy initialization)."""
        if not self._uncertainty_handler_initialized:
            try:
                self.logger.info("Initializing Bayesian uncertainty handler (lazy initialization)")
                uncertainty_config = UncertaintyConfig()
                self.uncertainty_handler = BayesianUncertaintyHandler(uncertainty_config)
                self._uncertainty_handler_initialized = True
                self.logger.info("Bayesian uncertainty handler initialized successfully")
            except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
                self.logger.error("Uncertainty handler unavailable: %s", e)
                self.uncertainty_handler = None
                self._uncertainty_handler_initialized = True  # Mark as attempted

    async def analyze_verification_uncertainty(
        self, verification_results: dict[str, Any], graph_nodes: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Analyze uncertainty in verification results.

        Args:
            verification_results: Results from graph verification
            graph_nodes: Graph nodes with confidence scores

        Returns:
            Dict[str, Any] | None: Uncertainty analysis results or None if unavailable
        """
        await self.ensure_uncertainty_handler()

        if not self.uncertainty_handler:
            self.logger.warning("Uncertainty handler not available")
            return None

        try:
            # Prepare verification data for uncertainty analysis
            verification_data = {
                "evidence": [
                    {
                        "score": node.get("confidence", 0.0),
                        "age_days": 0,
                        "source_reliability": 0.7,
                    }
                    for node in graph_nodes.values()
                ],
                "cluster_results": verification_results.get("cluster_results", []),
            }

            uncertainty_analysis = await self.uncertainty_handler.analyze_verification_uncertainty(verification_data)

            self.logger.info(
                "Applied Bayesian uncertainty analysis: %s", uncertainty_analysis.get("uncertainty_level", "unknown")
            )

            return uncertainty_analysis

        except (ValueError, RuntimeError, ConnectionError, TimeoutError) as e:
            self.logger.warning("Bayesian uncertainty analysis failed: %s", e)
            return None

    async def calculate_confidence_score(
        self, evidence_scores: list[float], source_reliabilities: list[float] | None = None
    ) -> float:
        """Calculate overall confidence score from evidence.

        Args:
            evidence_scores: List of evidence confidence scores
            source_reliabilities: Optional list of source reliability scores

        Returns:
            float: Overall confidence score (0.0 to 1.0)
        """
        if not evidence_scores:
            return 0.0

        await self.ensure_uncertainty_handler()

        if not self.uncertainty_handler:
            # Fallback to simple average if uncertainty handler unavailable
            return sum(evidence_scores) / len(evidence_scores)

        try:
            # Use uncertainty handler for more sophisticated calculation
            verification_data = {
                "evidence": [
                    {
                        "score": score,
                        "age_days": 0,
                        "source_reliability": (
                            source_reliabilities[i] if source_reliabilities and i < len(source_reliabilities) else 0.7
                        ),
                    }
                    for i, score in enumerate(evidence_scores)
                ]
            }

            analysis = await self.uncertainty_handler.analyze_verification_uncertainty(verification_data)

            # Extract confidence from uncertainty analysis
            return analysis.get("confidence_score", sum(evidence_scores) / len(evidence_scores))

        except Exception as e:
            self.logger.warning("Error calculating confidence score: %s", e)
            # Fallback to simple average
            return sum(evidence_scores) / len(evidence_scores)

    async def assess_uncertainty_level(self, verification_results: dict[str, Any]) -> str:
        """Assess the uncertainty level of verification results.

        Args:
            verification_results: Results from graph verification

        Returns:
            str: Uncertainty level ('low', 'medium', 'high', 'unknown')
        """
        await self.ensure_uncertainty_handler()

        if not self.uncertainty_handler:
            return "unknown"

        try:
            # Extract confidence scores from results
            confidence_scores = []

            for cluster_result in verification_results.get("cluster_results", []):
                for individual_result in cluster_result.get("individual_results", {}).values():
                    confidence = individual_result.get("confidence", 0.0)
                    confidence_scores.append(confidence)

            for individual_result in verification_results.get("individual_results", []):
                confidence = individual_result.get("confidence", 0.0)
                confidence_scores.append(confidence)

            if not confidence_scores:
                return "unknown"

            # Calculate uncertainty metrics
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            confidence_variance = sum((c - avg_confidence) ** 2 for c in confidence_scores) / len(confidence_scores)

            # Determine uncertainty level based on confidence and variance
            if avg_confidence > 0.8 and confidence_variance < 0.1:
                return "low"
            elif avg_confidence > 0.6 and confidence_variance < 0.2:
                return "medium"
            else:
                return "high"

        except Exception as e:
            self.logger.warning("Error assessing uncertainty level: %s", e)
            return "unknown"

    def is_uncertainty_handler_available(self) -> bool:
        """Check if uncertainty handler is available.

        Returns:
            bool: True if uncertainty handler is available, False otherwise
        """
        return self._uncertainty_handler_initialized and self.uncertainty_handler is not None

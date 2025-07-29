"""
Verification Processor for the Enhanced Fact-Checking System.

This module handles core fact verification processing, source reputation analysis,
and verification result processing, extracted from the original FactChecker class
to follow the Single Responsibility Principle.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from app.exceptions import AgentError

from ..core.component_manager import ComponentManager


class VerificationProcessor:
    """
    Handles core fact verification processing and analysis.

    This class is responsible for:
    - Processing fact verification requests
    - Managing verification statistics and metadata
    - Interfacing with GraphFactCheckingService
    - Analyzing source reputation
    - Processing fact relationships
    - Handling uncertainty analysis
    """

    def __init__(self, component_manager: ComponentManager):
        """
        Initialize the verification processor.

        Args:
            component_manager: The component manager providing access to services
        """
        self.component_manager = component_manager
        self.logger = logging.getLogger(__name__)

        # Verification tracking
        self.verification_count = 0

    async def verify_facts(self, facts: list[dict[str, Any]], context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Verify facts using the enhanced system.

        Args:
            facts: List of facts to verify
            context: Additional context for verification

        Returns:
            Comprehensive verification results with metadata
        """
        if not self.component_manager.initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")

        if not self.component_manager.graph_service:
            raise RuntimeError("Graph service not available")

        try:
            self.verification_count += 1
            start_time = datetime.now()

            # Perform verification using the enhanced graph service
            result = await self.component_manager.graph_service.verify_facts(context)

            # Add system metadata
            result["system_metadata"] = {
                "verification_id": self.verification_count,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "system_version": "enhanced_v1.0",
                "components_used": self.component_manager.get_active_components(),
                "timestamp": datetime.now().isoformat(),
                "facts_processed": len(facts) if facts else 0,
            }

            self.logger.info("Verification completed: ID %d", self.verification_count)
            return result

        except Exception as e:
            self.logger.error("Verification failed: %s", e)
            raise AgentError(f"Fact verification failed: {str(e)}") from e

    async def analyze_source_reputation(self, source_url: str) -> dict[str, Any]:
        """
        Analyze reputation of a specific source.

        Args:
            source_url: URL of the source to analyze

        Returns:
            dict: Source reputation analysis results
        """
        if not self.component_manager.reputation_system:
            return {"error": "Source reputation system not available"}

        try:
            # Extract domain from URL for analysis
            domain = urlparse(source_url).netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]

            # Get source analysis
            analysis = self.component_manager.reputation_system.get_source_analysis(domain)
            if "error" not in analysis:
                return {
                    "source_url": source_url,
                    "domain": domain,
                    "overall_reliability": analysis.get("overall_reliability", 0.0),
                    "metrics": analysis.get("metrics", {}),
                    "verification_stats": analysis.get("verification_stats", {}),
                    "source_type": analysis.get("source_type", "unknown"),
                    "last_updated": analysis.get("last_updated"),
                    "analysis_timestamp": datetime.now().isoformat(),
                }
            else:
                # Try to evaluate the source if no profile exists
                profile = self.component_manager.reputation_system.evaluate_source(source_url)
                return {
                    "source_url": source_url,
                    "domain": domain,
                    "overall_reliability": profile.metrics.reliability_score,
                    "metrics": {
                        "accuracy": profile.metrics.accuracy_score,
                        "bias_score": profile.metrics.bias_score,
                        "transparency": profile.metrics.transparency_score,
                        "expertise": profile.metrics.expertise_score,
                        "recency": profile.metrics.recency_score,
                    },
                    "source_type": profile.source_type.value,
                    "last_updated": profile.updated_at.isoformat(),
                    "analysis_timestamp": datetime.now().isoformat(),
                }
        except Exception as e:
            self.logger.error("Source reputation analysis failed for %s: %s", source_url, e)
            raise AgentError(f"Source reputation analysis failed for {source_url}: {str(e)}") from e

    async def analyze_fact_relationships(self, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Analyze relationships between facts.

        Args:
            facts: List of facts to analyze for relationships

        Returns:
            list: Relationship analysis results
        """
        if not self.component_manager.graph_service:
            self.logger.warning("Graph service not available for relationship analysis")
            return []

        try:
            relationships = await self.component_manager.graph_service.analyze_fact_relationships_standalone(facts)

            # Add metadata to each relationship
            for relationship in relationships:
                relationship["analysis_timestamp"] = datetime.now().isoformat()
                relationship["analyzer_version"] = "enhanced_v1.0"

            self.logger.info(
                "Analyzed relationships for %d facts, found %d relationships",
                len(facts),
                len(relationships),
            )
            return relationships

        except Exception as e:
            self.logger.error("Fact relationship analysis failed: %s", e)
            raise AgentError(f"Fact relationship analysis failed: {str(e)}") from e

    async def get_uncertainty_analysis(self, verification_data: dict[str, Any]) -> dict[str, Any]:
        """
        Get uncertainty analysis for verification data.

        Args:
            verification_data: Data to analyze for uncertainty

        Returns:
            dict: Uncertainty analysis results
        """
        if not self.component_manager.graph_service:
            return {"error": "Graph service not available"}

        try:
            uncertainty_result = await self.component_manager.graph_service.get_uncertainty_analysis(verification_data)

            # Add metadata
            uncertainty_result["analysis_timestamp"] = datetime.now().isoformat()
            uncertainty_result["analyzer_version"] = "enhanced_v1.0"

            return uncertainty_result

        except Exception as e:
            self.logger.error("Uncertainty analysis failed: %s", e)
            raise AgentError(f"Uncertainty analysis failed: {str(e)}") from e

    async def process_batch_verification(self, batch_requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Process multiple verification requests in batch.

        Args:
            batch_requests: List of verification requests

        Returns:
            list: Batch verification results
        """
        results = []
        batch_start_time = datetime.now()

        for i, request in enumerate(batch_requests):
            try:
                facts = request.get("facts", [])
                context = request.get("context", {})

                result = await self.verify_facts(facts, context)
                result["batch_index"] = i
                results.append(result)

            except Exception as e:
                self.logger.error("Batch verification failed for request %d: %s", i, e)
                raise AgentError(f"Batch verification failed for request {i}: {str(e)}") from e

        # Add batch metadata
        batch_metadata = {
            "batch_size": len(batch_requests),
            "successful_verifications": sum(1 for r in results if r.get("success", True)),
            "failed_verifications": sum(1 for r in results if not r.get("success", True)),
            "total_processing_time": (datetime.now() - batch_start_time).total_seconds(),
            "batch_timestamp": datetime.now().isoformat(),
        }

        return {
            "batch_metadata": batch_metadata,
            "results": results,
        }

    def get_verification_statistics(self) -> dict[str, Any]:
        """
        Get verification processing statistics.

        Returns:
            dict: Verification statistics
        """
        return {
            "total_verifications": self.verification_count,
            "processor_version": "enhanced_v1.0",
            "statistics_timestamp": datetime.now().isoformat(),
        }

    async def validate_verification_request(
        self, facts: list[dict[str, Any]], context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Validate a verification request before processing.

        Args:
            facts: Facts to validate
            context: Context to validate

        Returns:
            dict: Validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "validation_timestamp": datetime.now().isoformat(),
        }

        # Validate facts structure
        if not facts or not isinstance(facts, list):
            validation_result["valid"] = False
            validation_result["errors"].append("Facts must be a non-empty list")

        # Validate individual facts
        for i, fact in enumerate(facts):
            if not isinstance(fact, dict):
                validation_result["valid"] = False
                validation_result["errors"].append(f"Fact {i} must be a dictionary")
            elif not fact.get("statement"):
                validation_result["warnings"].append(f"Fact {i} missing statement field")

        # Validate context
        if context is not None and not isinstance(context, dict):
            validation_result["valid"] = False
            validation_result["errors"].append("Context must be a dictionary")

        # Check system readiness
        if not self.component_manager.initialized:
            validation_result["valid"] = False
            validation_result["errors"].append("System not initialized")

        return validation_result

    def reset_verification_count(self) -> None:
        """Reset the verification counter (for testing purposes)."""
        self.verification_count = 0
        self.logger.info("Verification count reset to 0")

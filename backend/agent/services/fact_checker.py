"""
from __future__ import annotations

Enhanced fact-checking system integration module.
This module provides a unified interface for all enhanced components.
"""

import logging
from datetime import datetime
from typing import Any

from .advanced_clustering import AdvancedClusteringSystem, ClusteringConfig
from .bayesian_uncertainty import BayesianVerificationModel, UncertaintyConfig
from .graph_fact_checking import GraphFactCheckingService
from .graph_storage import Neo4jGraphStorage
from .intelligent_cache import IntelligentCache
from .relationship_analysis import RelationshipAnalysisEngine, RelationshipConfig
from .source_reputation import SourceReputationSystem
from .system_config import (
    SystemConfig,
    get_default_config,
    get_development_config,
    get_production_config,
)


class FactChecker:
    """
    Unified enhanced fact-checking system that integrates all components.

    This class provides a high-level interface for the enhanced fact-checking
    system with intelligent caching, graph persistence, source reputation,
    advanced clustering, Bayesian uncertainty, and relationship analysis.
    """

    def __init__(self, config: SystemConfig | None = None):
        """Initialize the enhanced fact-checking system."""
        self.config = config or get_default_config()
        self.logger = logging.getLogger(__name__)

        # Core components
        self.graph_service: GraphFactCheckingService | None = None
        self.cache_system: IntelligentCache | None = None
        self.graph_storage: Neo4jGraphStorage | None = None
        self.reputation_system: SourceReputationSystem | None = None
        self.clustering_system: AdvancedClusteringSystem | None = None
        self.uncertainty_handler: BayesianVerificationModel | None = None
        self.relationship_analyzer: RelationshipAnalysisEngine | None = None

        # System state
        self.is_initialized = False
        self.initialization_time: datetime | None = None
        self.verification_count = 0

    async def initialize(self) -> bool:
        """Initialize all system components."""
        try:
            self.logger.info("Initializing enhanced fact-checking system...")

            # Initialize cache system
            self.cache_system = IntelligentCache(
                max_memory_size=self.config.cache_config.memory_cache_size
            )
            await self.cache_system.initialize()
            self.logger.info("Cache system initialized")

            # Initialize graph storage (if Neo4j config is provided)
            if self.config.neo4j_config:
                self.graph_storage = Neo4jGraphStorage(
                    uri=self.config.neo4j_config.uri,
                    auth=(
                        self.config.neo4j_config.username,
                        self.config.neo4j_config.password,
                    ),
                    database=self.config.neo4j_config.database,
                )
                self.logger.info("Graph storage initialized")

            # Initialize source reputation system
            self.reputation_system = SourceReputationSystem()
            self.logger.info("Source reputation system initialized")

            # Initialize advanced clustering
            clustering_config = ClusteringConfig(
                use_gnn=True,  # Enable GNN by default
                gnn_hidden_dim=self.config.clustering_config.hidden_dim,
                gnn_num_layers=self.config.clustering_config.num_layers,
                attention_heads=4,  # Default value
            )
            self.clustering_system = AdvancedClusteringSystem(clustering_config)
            self.logger.info("Advanced clustering system initialized")

            # Initialize uncertainty handler
            uncertainty_config = UncertaintyConfig(
                use_bayesian_inference=True,  # Enable by default
                mcmc_samples=self.config.uncertainty_config.num_samples,
                mcmc_tune=self.config.uncertainty_config.num_warmup,
                mcmc_chains=self.config.uncertainty_config.num_chains,
            )
            self.uncertainty_handler = BayesianVerificationModel(uncertainty_config)
            self.logger.info("Bayesian uncertainty handler initialized")

            # Initialize relationship analyzer
            relationship_config = RelationshipConfig(
                enable_causal_inference=self.config.relationship_config.enable_causal_analysis,
                enable_temporal_analysis=self.config.relationship_config.enable_temporal_analysis,
                enable_semantic_analysis=self.config.relationship_config.enable_semantic_analysis,
                use_transformer_embeddings=True,
            )
            self.relationship_analyzer = RelationshipAnalysisEngine(relationship_config)
            self.logger.info("Relationship analysis engine initialized")

            # Initialize main graph service with simplified interface
            self.graph_service = GraphFactCheckingService(
                search_tool=None  # This should be set by the calling code
            )

            self.is_initialized = True
            self.initialization_time = datetime.now()
            self.logger.info("Enhanced fact-checking system fully initialized")

            return True

        except (ConnectionError, ValueError, RuntimeError, OSError) as e:
            self.logger.error("Failed to initialize enhanced system: %s", e)
            return False

    def set_search_tool(self, search_tool):
        """Set the search tool for the graph service."""
        if self.graph_service:
            self.graph_service.search_tool = search_tool
            self.graph_service.verification_engine.search_tool = search_tool
            self.logger.info("Search tool set for graph service")
        else:
            self.logger.warning("Graph service not initialized, cannot set search tool")

    async def verify_facts(
        self, facts: list[dict[str, Any]], context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Verify facts using the enhanced system.

        Args:
            facts: List of facts to verify
            context: Additional context for verification

        Returns:
            Comprehensive verification results
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")

        if not self.graph_service:
            raise RuntimeError("Graph service not available")

        try:
            self.verification_count += 1
            start_time = datetime.now()

            # Perform verification using the enhanced graph service
            result = await self.graph_service.verify_facts(context)

            # Add system metadata
            result["system_metadata"] = {
                "verification_id": self.verification_count,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "system_version": "enhanced_v1.0",
                "components_used": self._get_active_components(),
                "timestamp": datetime.now().isoformat(),
            }

            return result

        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def analyze_source_reputation(self, source_url: str) -> dict[str, Any]:
        """Analyze reputation of a specific source."""
        if not self.reputation_system:
            return {"error": "Source reputation system not available"}

        try:
            # Extract domain from URL for analysis
            from urllib.parse import urlparse

            domain = urlparse(source_url).netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]

            # Get source analysis
            analysis = self.reputation_system.get_source_analysis(domain)
            if "error" not in analysis:
                return {
                    "source_url": source_url,
                    "domain": domain,
                    "overall_reliability": analysis.get("overall_reliability", 0.0),
                    "metrics": analysis.get("metrics", {}),
                    "verification_stats": analysis.get("verification_stats", {}),
                    "source_type": analysis.get("source_type", "unknown"),
                    "last_updated": analysis.get("last_updated"),
                }
            else:
                # Try to evaluate the source if no profile exists
                profile = self.reputation_system.evaluate_source(source_url)
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
                }
        except Exception as e:
            return {"error": str(e)}

    async def analyze_fact_relationships(
        self, facts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Analyze relationships between facts."""
        if not self.graph_service:
            return []

        return await self.graph_service.analyze_fact_relationships_standalone(facts)

    async def get_uncertainty_analysis(
        self, verification_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Get uncertainty analysis for verification data."""
        if not self.graph_service:
            return {"error": "Graph service not available"}

        return await self.graph_service.get_uncertainty_analysis(verification_data)

    async def get_system_statistics(self) -> dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            "system_info": {
                "initialized": self.is_initialized,
                "initialization_time": (
                    self.initialization_time.isoformat()
                    if self.initialization_time
                    else None
                ),
                "verification_count": self.verification_count,
                "active_components": self._get_active_components(),
            }
        }

        # Add component-specific stats
        if self.graph_service:
            if hasattr(self.graph_service, "get_detailed_stats"):
                stats["graph_service"] = self.graph_service.get_detailed_stats()

        if self.cache_system:
            if hasattr(self.cache_system, "get_cache_stats"):
                stats["cache_system"] = await self.cache_system.get_cache_stats()

        if self.graph_storage:
            if hasattr(self.graph_storage, "get_storage_stats"):
                stats["graph_storage"] = await self.graph_storage.get_storage_stats()

        if self.reputation_system:
            if hasattr(self.reputation_system, "get_reputation_stats"):
                stats["reputation_system"] = (
                    await self.reputation_system.get_reputation_stats()
                )

        return stats

    async def clear_all_caches(self) -> bool:
        """Clear all system caches."""
        try:
            if self.cache_system:
                if hasattr(self.cache_system, "clear_all"):
                    await self.cache_system.clear_all()

            if self.graph_service:
                if hasattr(self.graph_service, "clear_caches"):
                    await self.graph_service.clear_caches()

            self.logger.info("All caches cleared")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear caches: {e}")
            return False

    async def shutdown(self) -> bool:
        """Shutdown the enhanced system gracefully."""
        try:
            self.logger.info("Shutting down enhanced fact-checking system...")

            # Shutdown components in reverse order
            if self.graph_service:
                # Close graph service (which will close verification engine and source manager)
                if hasattr(self.graph_service, "close"):
                    await self.graph_service.close()
                else:
                    # Fallback: clear caches if close method doesn't exist
                    if hasattr(self.graph_service, "clear_caches"):
                        await self.graph_service.clear_caches()

            # Most components don't have shutdown methods, so we just clean up references
            self.relationship_analyzer = None
            self.uncertainty_handler = None
            self.clustering_system = None
            self.reputation_system = None

            if self.graph_storage:
                # Note: Neo4jGraphStorage.close() is synchronous, not async
                self.graph_storage.close()

            if self.cache_system:
                # Clear cache before cleanup
                if hasattr(self.cache_system, "clear"):
                    await self.cache_system.clear()
                self.cache_system = None

            self.is_initialized = False
            self.logger.info("Enhanced system shutdown complete")
            return True

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            return False

    async def close(self):
        """Close all resources and cleanup (alias for shutdown)."""
        await self.shutdown()

    async def __aenter__(self):
        """Async context manager entry."""
        if not self.is_initialized:
            await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def _get_active_components(self) -> list[str]:
        """Get list of active components."""
        components = []

        if self.cache_system:
            components.append("intelligent_cache")
        if self.graph_storage:
            components.append("neo4j_storage")
        if self.reputation_system:
            components.append("source_reputation")
        if self.clustering_system:
            components.append("advanced_clustering")
        if self.uncertainty_handler:
            components.append("bayesian_uncertainty")
        if self.relationship_analyzer:
            components.append("relationship_analysis")
        if self.graph_service:
            components.append("graph_fact_checking")

        return components

    async def health_check(self) -> dict[str, Any]:
        """Perform system health check."""
        health_status = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
        }

        try:
            # Check each component
            if self.cache_system:
                health_status["components"]["cache"] = await self._check_cache_health()

            if self.graph_storage:
                health_status["components"][
                    "storage"
                ] = await self._check_storage_health()

            if self.reputation_system:
                health_status["components"][
                    "reputation"
                ] = await self._check_reputation_health()

            # Determine overall status
            component_statuses = [
                comp.get("status", "unknown")
                for comp in health_status["components"].values()
            ]
            if "error" in component_statuses:
                health_status["overall_status"] = "degraded"
            elif "warning" in component_statuses:
                health_status["overall_status"] = "warning"

        except Exception as e:
            health_status["overall_status"] = "error"
            health_status["error"] = str(e)

        return health_status

    async def _check_cache_health(self) -> dict[str, Any]:
        """Check cache system health."""
        try:
            if not self.cache_system:
                return {"status": "disabled"}

            if hasattr(self.cache_system, "get_cache_stats"):
                stats = await self.cache_system.get_cache_stats()
                return {
                    "status": "healthy",
                    "hit_rate": stats.get("hit_rate", 0),
                    "total_operations": stats.get("total_operations", 0),
                }
            else:
                return {"status": "healthy", "note": "basic health check passed"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _check_storage_health(self) -> dict[str, Any]:
        """Check graph storage health."""
        try:
            if not self.graph_storage:
                return {"status": "disabled"}

            if hasattr(self.graph_storage, "get_storage_stats"):
                stats = await self.graph_storage.get_storage_stats()
                return {
                    "status": "healthy",
                    "total_graphs": stats.get("total_graphs", 0),
                    "connection_status": "connected",
                }
            else:
                return {"status": "healthy", "note": "basic health check passed"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _check_reputation_health(self) -> dict[str, Any]:
        """Check reputation system health."""
        try:
            if not self.reputation_system:
                return {"status": "disabled"}

            if hasattr(self.reputation_system, "get_reputation_stats"):
                stats = await self.reputation_system.get_reputation_stats()
                return {
                    "status": "healthy",
                    "total_sources": stats.get("total_sources", 0),
                }
            else:
                return {"status": "healthy", "note": "basic health check passed"}
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Factory functions for common configurations


async def create_fact_checker(config_type: str = "default") -> FactChecker:
    """
    Create and initialize a fact-checking system.

    Args:
        config_type: Type of configuration ("default", "production", "development")

    Returns:
        Initialized fact-checking system
    """

    if config_type == "production":
        config = get_production_config()
    elif config_type == "development":
        config = get_development_config()
    else:
        config = get_default_config()

    system = FactChecker(config)
    await system.initialize()
    return system


async def create_minimal_system() -> FactChecker:
    """Create a minimal system with only essential components."""
    config = get_default_config()

    system = FactChecker(config)
    await system.initialize()
    return system

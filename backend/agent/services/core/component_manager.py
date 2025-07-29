"""
Component Manager for the Enhanced Fact-Checking System.

This module handles the initialization, lifecycle management, and configuration
of all system components, extracted from the original FactChecker class to
follow the Single Responsibility Principle.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from ..analysis.advanced_clustering import AdvancedClusteringSystem, ClusteringConfig
from ..analysis.bayesian_uncertainty import BayesianVerificationModel, UncertaintyConfig
from ..graph.graph_fact_checking import GraphFactCheckingService
from ..graph.graph_storage import Neo4jGraphStorage
from ..cache.intelligent_cache import IntelligentCache
from ..analysis.relationship_analysis import RelationshipAnalysisEngine, RelationshipConfig
from ..reputation.source_reputation import SourceReputationSystem
from .system_config import SystemConfig


class ComponentManager:
    """
    Manages the lifecycle and configuration of all system components.

    This class is responsible for:
    - Initializing all system components with proper configuration
    - Managing component dependencies and injection
    - Handling component lifecycle (startup/shutdown)
    - Providing clean component access interfaces
    - Tracking active components
    """

    def __init__(self, config: SystemConfig):
        """Initialize the component manager with system configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Core components
        self.graph_service: GraphFactCheckingService | None = None
        self.cache_system: IntelligentCache | None = None
        self.graph_storage: Neo4jGraphStorage | None = None
        self.reputation_system: SourceReputationSystem | None = None
        self.clustering_system: AdvancedClusteringSystem | None = None
        self.uncertainty_handler: BayesianVerificationModel | None = None
        self.relationship_analyzer: RelationshipAnalysisEngine | None = None

        # Component state
        self.is_initialized = False
        self.initialization_time: datetime | None = None

    async def initialize_components(self) -> bool:
        """
        Initialize all system components with proper configuration.

        Returns:
            bool: True if all components initialized successfully, False otherwise
        """
        try:
            self.logger.info(
                "Initializing enhanced fact-checking system components...")

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
            self.clustering_system = AdvancedClusteringSystem(
                clustering_config)
            self.logger.info("Advanced clustering system initialized")

            # Initialize uncertainty handler
            uncertainty_config = UncertaintyConfig(
                use_bayesian_inference=True,  # Enable by default
                mcmc_samples=self.config.uncertainty_config.num_samples,
                mcmc_tune=self.config.uncertainty_config.num_warmup,
                mcmc_chains=self.config.uncertainty_config.num_chains,
            )
            self.uncertainty_handler = BayesianVerificationModel(
                uncertainty_config)
            self.logger.info("Bayesian uncertainty handler initialized")

            # Initialize relationship analyzer
            relationship_config = RelationshipConfig(
                enable_causal_inference=self.config.relationship_config.enable_causal_analysis,
                enable_temporal_analysis=self.config.relationship_config.enable_temporal_analysis,
                enable_semantic_analysis=self.config.relationship_config.enable_semantic_analysis,
                use_transformer_embeddings=True,
            )
            self.relationship_analyzer = RelationshipAnalysisEngine(
                relationship_config)
            self.logger.info("Relationship analysis engine initialized")

            # Initialize main graph service with simplified interface
            self.graph_service = GraphFactCheckingService(
                search_tool=None  # This should be set by the calling code
            )
            self.logger.info("Graph fact-checking service initialized")

            self.is_initialized = True
            self.initialization_time = datetime.now()
            self.logger.info("All system components initialized successfully")

            return True

        except (ConnectionError, ValueError, RuntimeError, OSError) as e:
            self.logger.error("Failed to initialize system components: %s", e)
            return False

    async def shutdown_components(self) -> bool:
        """
        Shutdown all system components gracefully.

        Returns:
            bool: True if shutdown successful, False otherwise
        """
        try:
            self.logger.info("Shutting down system components...")

            # Shutdown components in reverse order of initialization
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
            self.logger.info("System components shutdown complete")
            return True

        except Exception as e:
            self.logger.error("Error during component shutdown: %s", e)
            return False

    def get_component(self, component_name: str) -> Any:
        """
        Get a specific component by name.

        Args:
            component_name: Name of the component to retrieve

        Returns:
            The requested component or None if not found
        """
        component_map = {
            "cache_system": self.cache_system,
            "graph_storage": self.graph_storage,
            "reputation_system": self.reputation_system,
            "clustering_system": self.clustering_system,
            "uncertainty_handler": self.uncertainty_handler,
            "relationship_analyzer": self.relationship_analyzer,
            "graph_service": self.graph_service,
        }

        return component_map.get(component_name)

    def set_search_tool(self, search_tool) -> None:
        """
        Set the search tool for components that require it.

        Args:
            search_tool: The search tool to set
        """
        if self.graph_service:
            self.graph_service.search_tool = search_tool
            self.graph_service.verification_engine.search_tool = search_tool
            self.logger.info("Search tool set for graph service")
        else:
            self.logger.warning(
                "Graph service not initialized, cannot set search tool")

    def get_active_components(self) -> list[str]:
        """
        Get list of currently active components.

        Returns:
            List of active component names
        """
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

    async def clear_all_caches(self) -> bool:
        """
        Clear all system caches.

        Returns:
            bool: True if caches cleared successfully, False otherwise
        """
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
            self.logger.error("Failed to clear caches: %s", e)
            return False

    @property
    def initialized(self) -> bool:
        """Check if components are initialized."""
        return self.is_initialized

    @property
    def initialization_timestamp(self) -> datetime | None:
        """Get the initialization timestamp."""
        return self.initialization_time

    async def validate_configuration(self) -> dict[str, Any]:
        """
        Validate the current system configuration.

        Returns:
            dict: Configuration validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "validation_timestamp": datetime.now().isoformat(),
        }

        # Validate configuration object
        if not self.config:
            validation_result["valid"] = False
            validation_result["errors"].append(
                "Configuration object is missing")
            return validation_result

        # Check for required configuration sections
        required_sections = ["cache_config",
                             "neo4j_config", "source_reputation_config"]
        for section in required_sections:
            if not hasattr(self.config, section):
                validation_result["warnings"].append(
                    f"Configuration section '{section}' is missing")

        # Validate cache configuration
        if hasattr(self.config, "cache_config") and self.config.cache_config:
            if self.config.cache_config.memory_cache_size <= 0:
                validation_result["errors"].append(
                    "Cache memory size must be positive")
                validation_result["valid"] = False

        # Validate Neo4j configuration
        if hasattr(self.config, "neo4j_config") and self.config.neo4j_config:
            if not self.config.neo4j_config.uri:
                validation_result["warnings"].append(
                    "Neo4j URI is not configured")

        return validation_result

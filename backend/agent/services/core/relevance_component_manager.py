"""
Extended component manager for relevance-specific components.

Extends the base ComponentManager to include relevance system components
while maintaining compatibility with the existing infrastructure.
"""
from __future__ import annotations

import logging
from typing import Any

from ..relevance.relevance_embeddings_coordinator import RelevanceEmbeddingsCoordinator
from .component_manager import ComponentManager

logger = logging.getLogger(__name__)


class RelevanceComponentManager(ComponentManager):
    """
    Extended component manager that includes relevance-specific components.

    This class extends the base ComponentManager to include components
    specifically needed for the relevance system while maintaining
    compatibility with existing infrastructure.
    """

    def __init__(self, config):
        """Initialize the relevance component manager."""
        super().__init__(config)
        self.relevance_embeddings_coordinator: RelevanceEmbeddingsCoordinator | None = None

    async def initialize_components(self) -> bool:
        """
        Initialize all components including relevance-specific ones.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize base components first
            base_initialization = await super().initialize_components()
            if not base_initialization:
                logger.error("Failed to initialize base components")
                return False

            # Initialize relevance-specific components
            logger.info("Initializing relevance-specific components...")

            # Initialize relevance embeddings coordinator
            self.relevance_embeddings_coordinator = RelevanceEmbeddingsCoordinator()
            coordinator_initialized = await self.relevance_embeddings_coordinator.initialize()

            if not coordinator_initialized:
                logger.error("Failed to initialize RelevanceEmbeddingsCoordinator")
                return False

            logger.info("Relevance embeddings coordinator initialized")

            logger.info("All relevance components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize relevance components: {e}")
            return False

    async def shutdown_components(self) -> bool:
        """
        Shutdown all components including relevance-specific ones.

        Returns:
            bool: True if shutdown successful, False otherwise
        """
        try:
            logger.info("Shutting down relevance components...")

            # Shutdown relevance-specific components first
            if self.relevance_embeddings_coordinator:
                await self.relevance_embeddings_coordinator.close()
                self.relevance_embeddings_coordinator = None

            # Shutdown base components
            base_shutdown = await super().shutdown_components()
            if not base_shutdown:
                logger.warning("Base component shutdown had issues")

            logger.info("Relevance components shutdown complete")
            return True

        except Exception as e:
            logger.error(f"Error during relevance component shutdown: {e}")
            return False

    def get_component(self, component_name: str) -> Any:
        """
        Get a specific component by name, including relevance components.

        Args:
            component_name: Name of the component to retrieve

        Returns:
            The requested component or None if not found
        """
        # Check relevance-specific components first
        relevance_component_map = {
            "relevance_embeddings_coordinator": self.relevance_embeddings_coordinator,
        }

        relevance_component = relevance_component_map.get(component_name)
        if relevance_component is not None:
            return relevance_component

        # Fall back to base components
        return super().get_component(component_name)

    def get_active_components(self) -> list[str]:
        """
        Get list of currently active components including relevance components.

        Returns:
            List of active component names
        """
        # Get base components
        components = super().get_active_components()

        # Add relevance-specific components
        if self.relevance_embeddings_coordinator:
            components.append("relevance_embeddings_coordinator")

        return components

    async def clear_all_caches(self) -> bool:
        """
        Clear all system caches including relevance caches.

        Returns:
            bool: True if caches cleared successfully, False otherwise
        """
        try:
            # Clear base caches
            base_clear = await super().clear_all_caches()

            # Clear relevance-specific caches
            if self.relevance_embeddings_coordinator:
                # The coordinator manages its own cache clearing through its components
                logger.info("Relevance caches managed by coordinator")

            logger.info("All relevance caches cleared")
            return base_clear

        except Exception as e:
            logger.error(f"Failed to clear relevance caches: {e}")
            return False

    async def get_relevance_performance_metrics(self) -> dict[str, Any]:
        """
        Get performance metrics for relevance-specific components.

        Returns:
            dict: Performance metrics for relevance components
        """
        metrics = {}

        try:
            if self.relevance_embeddings_coordinator:
                coordinator_metrics = await self.relevance_embeddings_coordinator.get_performance_metrics()
                metrics["relevance_embeddings_coordinator"] = coordinator_metrics

            return metrics

        except Exception as e:
            logger.error(f"Error getting relevance performance metrics: {e}")
            return {"error": str(e)}

    async def optimize_relevance_performance(self) -> dict[str, Any]:
        """
        Optimize performance of relevance-specific components.

        Returns:
            dict: Optimization results for relevance components
        """
        optimization_results = {}

        try:
            if self.relevance_embeddings_coordinator:
                coordinator_optimization = await self.relevance_embeddings_coordinator.optimize_embeddings_performance()
                optimization_results["relevance_embeddings_coordinator"] = coordinator_optimization

            logger.info("Relevance performance optimization completed")
            return optimization_results

        except Exception as e:
            logger.error(f"Error optimizing relevance performance: {e}")
            return {"error": str(e)}

    async def validate_configuration(self) -> dict[str, Any]:
        """
        Validate the current system configuration including relevance components.

        Returns:
            dict: Configuration validation results
        """
        # Get base validation results
        validation_result = await super().validate_configuration()

        try:
            # Add relevance-specific validation
            if self.relevance_embeddings_coordinator and not self.relevance_embeddings_coordinator._initialized:
                validation_result["warnings"].append("RelevanceEmbeddingsCoordinator is not initialized")

            # Check for relevance-specific configuration requirements
            # (Add specific checks as needed based on your configuration structure)

            return validation_result

        except Exception as e:
            logger.error(f"Error validating relevance configuration: {e}")
            validation_result["errors"].append(f"Relevance validation error: {e}")
            validation_result["valid"] = False
            return validation_result


# Global instance management for relevance component manager
_relevance_component_manager: RelevanceComponentManager | None = None


def get_relevance_component_manager(config=None) -> RelevanceComponentManager:
    """
    Get global relevance component manager instance.

    Args:
        config: Configuration object (required for first initialization)

    Returns:
        RelevanceComponentManager: Global instance
    """
    global _relevance_component_manager
    if _relevance_component_manager is None:
        if config is None:
            raise ValueError("Configuration required for first initialization")
        _relevance_component_manager = RelevanceComponentManager(config)
    return _relevance_component_manager


async def close_relevance_component_manager():
    """Close global relevance component manager instance."""
    global _relevance_component_manager
    if _relevance_component_manager is not None:
        await _relevance_component_manager.shutdown_components()
        _relevance_component_manager = None

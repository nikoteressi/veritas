"""
Fact Checking Orchestrator for the Enhanced Fact-Checking System.

This module provides the main orchestration layer that coordinates all components
and maintains the public interface for backward compatibility with the original
FactChecker class.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from .component_manager import ComponentManager
from .system_health_monitor import SystemHealthMonitor
from ..graph.verification.verification_processor import VerificationProcessor


class FactCheckingOrchestrator:
    """
    Main orchestrator for the fact-checking system.

    This class coordinates all components and provides the public interface
    that maintains backward compatibility with the original FactChecker class.

    Responsibilities:
    - Coordinating component interactions
    - Providing the main public interface
    - Managing system lifecycle
    - Maintaining backward compatibility
    - Handling high-level operations
    """

    def __init__(self, config: Any):
        """
        Initialize the fact-checking orchestrator.

        Args:
            config: System configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize component managers
        self.component_manager = ComponentManager(config)
        self.health_monitor = SystemHealthMonitor(self.component_manager)
        self.verification_processor = VerificationProcessor(
            self.component_manager)

        # Orchestrator state
        self._initialized = False
        self._initialization_timestamp = None

    async def initialize(self) -> None:
        """
        Initialize the entire fact-checking system.

        This method coordinates the initialization of all components
        and ensures the system is ready for operation.
        """
        if self._initialized:
            self.logger.warning("System already initialized")
            return

        try:
            self.logger.info("Starting fact-checking system initialization")
            start_time = datetime.now()

            # Initialize components through component manager
            await self.component_manager.initialize_components()

            # Mark as initialized
            self._initialized = True
            self._initialization_timestamp = datetime.now()

            initialization_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                "Fact-checking system initialized successfully in %.2fs", initialization_time)

        except Exception as e:
            self.logger.error("System initialization failed: %s", e)
            self._initialized = False
            raise

    async def verify_facts(
        self, facts: list[dict[str, Any]], context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Verify facts using the enhanced system.

        This is the main public interface for fact verification.

        Args:
            facts: List of facts to verify
            context: Additional context for verification

        Returns:
            Comprehensive verification results with metadata
        """
        if not self._initialized:
            raise RuntimeError(
                "System not initialized. Call initialize() first.")

        # Validate request
        validation_result = await self.verification_processor.validate_verification_request(facts, context)
        if not validation_result["valid"]:
            return {
                "success": False,
                "error": "Validation failed",
                "validation_errors": validation_result["errors"],
                "timestamp": datetime.now().isoformat(),
            }

        # Process verification
        return await self.verification_processor.verify_facts(facts, context)

    async def analyze_source_reputation(self, source_url: str) -> dict[str, Any]:
        """
        Analyze reputation of a specific source.

        Args:
            source_url: URL of the source to analyze

        Returns:
            dict: Source reputation analysis results
        """
        if not self._initialized:
            raise RuntimeError(
                "System not initialized. Call initialize() first.")

        return await self.verification_processor.analyze_source_reputation(source_url)

    async def analyze_fact_relationships(
        self, facts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Analyze relationships between facts.

        Args:
            facts: List of facts to analyze for relationships

        Returns:
            list: Relationship analysis results
        """
        if not self._initialized:
            raise RuntimeError(
                "System not initialized. Call initialize() first.")

        return await self.verification_processor.analyze_fact_relationships(facts)

    async def get_uncertainty_analysis(
        self, verification_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Get uncertainty analysis for verification data.

        Args:
            verification_data: Data to analyze for uncertainty

        Returns:
            dict: Uncertainty analysis results
        """
        if not self._initialized:
            raise RuntimeError(
                "System not initialized. Call initialize() first.")

        return await self.verification_processor.get_uncertainty_analysis(verification_data)

    def set_search_tool(self, search_tool: Any) -> None:
        """
        Set the search tool for the system.

        Args:
            search_tool: Search tool instance to use
        """
        self.component_manager.set_search_tool(search_tool)

    async def get_system_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive system statistics.

        Returns:
            dict: System statistics from all components
        """
        if not self._initialized:
            return {
                "error": "System not initialized",
                "timestamp": datetime.now().isoformat(),
            }

        # Get statistics from health monitor
        health_stats = await self.health_monitor.get_system_statistics()

        # Get verification statistics
        verification_stats = self.verification_processor.get_verification_statistics()

        # Add orchestrator statistics
        orchestrator_stats = {
            "orchestrator": {
                "initialized": self._initialized,
                "initialization_timestamp": self._initialization_timestamp.isoformat() if self._initialization_timestamp else None,
                "uptime_seconds": (datetime.now() - self._initialization_timestamp).total_seconds() if self._initialization_timestamp else 0,
                "version": "enhanced_v1.0",
            }
        }

        # Combine all statistics
        return {
            **health_stats,
            **verification_stats,
            **orchestrator_stats,
            "statistics_timestamp": datetime.now().isoformat(),
        }

    async def clear_all_caches(self) -> dict[str, Any]:
        """
        Clear all system caches.

        Returns:
            dict: Cache clearing results
        """
        if not self._initialized:
            return {
                "error": "System not initialized",
                "timestamp": datetime.now().isoformat(),
            }

        return await self.component_manager.clear_all_caches()

    async def health_check(self) -> dict[str, Any]:
        """
        Perform comprehensive system health check.

        Returns:
            dict: Health check results
        """
        if not self._initialized:
            return {
                "overall_health": "unhealthy",
                "error": "System not initialized",
                "timestamp": datetime.now().isoformat(),
            }

        return await self.health_monitor.health_check()

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the fact-checking system.

        This method ensures all components are properly closed
        and resources are cleaned up.
        """
        if not self._initialized:
            self.logger.warning("System not initialized, nothing to shutdown")
            return

        try:
            self.logger.info("Starting system shutdown")

            # Shutdown components through component manager
            await self.component_manager.shutdown_components()

            # Mark as not initialized
            self._initialized = False
            self._initialization_timestamp = None

            self.logger.info("System shutdown completed successfully")

        except Exception as e:
            self.logger.error("Error during shutdown: %s", e)
            raise

    async def close(self) -> None:
        """
        Close the system (alias for shutdown for backward compatibility).
        """
        await self.shutdown()

    # Context manager support for backward compatibility
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()

    # Properties for backward compatibility
    @property
    def initialized(self) -> bool:
        """Check if the system is initialized."""
        return self._initialized

    @property
    def initialization_timestamp(self) -> datetime | None:
        """Get the initialization timestamp."""
        return self._initialization_timestamp

    def _get_active_components(self) -> list[str]:
        """
        Get list of active component names.

        Returns:
            list: Names of active components
        """
        return self.component_manager.get_active_components()

    # Advanced orchestration methods
    async def process_batch_verification(
        self, batch_requests: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Process multiple verification requests in batch.

        Args:
            batch_requests: List of verification requests

        Returns:
            dict: Batch verification results
        """
        if not self._initialized:
            raise RuntimeError(
                "System not initialized. Call initialize() first.")

        return await self.verification_processor.process_batch_verification(batch_requests)

    async def get_system_status(self) -> dict[str, Any]:
        """
        Get comprehensive system status including health and statistics.

        Returns:
            dict: Complete system status
        """
        if not self._initialized:
            return {
                "status": "not_initialized",
                "timestamp": datetime.now().isoformat(),
            }

        # Get health check
        health_result = await self.health_check()

        # Get statistics
        stats_result = await self.get_system_statistics()

        # Combine results
        return {
            "status": "operational" if health_result.get("overall_health") == "healthy" else "degraded",
            "health": health_result,
            "statistics": stats_result,
            "active_components": self._get_active_components(),
            "timestamp": datetime.now().isoformat(),
        }

    async def validate_system_configuration(self) -> dict[str, Any]:
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
        required_sections = ["cache", "neo4j", "source_reputation"]
        for section in required_sections:
            if not hasattr(self.config, section):
                validation_result["warnings"].append(
                    f"Configuration section '{section}' is missing")

        # Validate component manager configuration
        if self.component_manager:
            component_validation = await self.component_manager.validate_configuration()
            if not component_validation.get("valid", True):
                validation_result["valid"] = False
                validation_result["errors"].extend(
                    component_validation.get("errors", []))
                validation_result["warnings"].extend(
                    component_validation.get("warnings", []))

        return validation_result

    def get_orchestrator_info(self) -> dict[str, Any]:
        """
        Get information about the orchestrator itself.

        Returns:
            dict: Orchestrator information
        """
        return {
            "class_name": self.__class__.__name__,
            "version": "enhanced_v1.0",
            "initialized": self._initialized,
            "initialization_timestamp": self._initialization_timestamp.isoformat() if self._initialization_timestamp else None,
            "component_managers": {
                "component_manager": self.component_manager.__class__.__name__,
                "health_monitor": self.health_monitor.__class__.__name__,
                "verification_processor": self.verification_processor.__class__.__name__,
            },
            "info_timestamp": datetime.now().isoformat(),
        }

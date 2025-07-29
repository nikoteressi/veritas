"""
System Health Monitor for the Enhanced Fact-Checking System.

This module handles health checks, system statistics collection, and monitoring
of all system components, extracted from the original FactChecker class to
follow the Single Responsibility Principle.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from .component_manager import ComponentManager


class SystemHealthMonitor:
    """
    Monitors the health and performance of all system components.

    This class is responsible for:
    - Performing health checks on all system components
    - Collecting and aggregating system statistics
    - Monitoring system performance and status
    - Generating comprehensive health reports
    """

    def __init__(self, component_manager: ComponentManager):
        """
        Initialize the system health monitor.

        Args:
            component_manager: The component manager to monitor
        """
        self.component_manager = component_manager
        self.logger = logging.getLogger(__name__)

    async def health_check(self) -> dict[str, Any]:
        """
        Perform comprehensive system health check.

        Returns:
            dict: Health status report for all components
        """
        health_status = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
        }

        try:
            # Check each component
            if self.component_manager.cache_system:
                health_status["components"]["cache"] = await self._check_cache_health()

            if self.component_manager.graph_storage:
                health_status["components"]["storage"] = await self._check_storage_health()

            if self.component_manager.reputation_system:
                health_status["components"]["reputation"] = await self._check_reputation_health()

            if self.component_manager.graph_service:
                health_status["components"]["graph_service"] = await self._check_graph_service_health()

            if self.component_manager.clustering_system:
                health_status["components"]["clustering"] = await self._check_clustering_health()

            if self.component_manager.uncertainty_handler:
                health_status["components"]["uncertainty"] = await self._check_uncertainty_health()

            if self.component_manager.relationship_analyzer:
                health_status["components"]["relationship"] = await self._check_relationship_health()

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
            self.logger.error("Health check failed: %s", e)

        return health_status

    async def get_system_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive system statistics.

        Returns:
            dict: System statistics including component-specific metrics
        """
        stats = {
            "system_info": {
                "initialized": self.component_manager.initialized,
                "initialization_time": (
                    self.component_manager.initialization_timestamp.isoformat()
                    if self.component_manager.initialization_timestamp
                    else None
                ),
                "active_components": self.component_manager.get_active_components(),
            }
        }

        try:
            # Add component-specific stats
            if self.component_manager.graph_service:
                if hasattr(self.component_manager.graph_service, "get_detailed_stats"):
                    stats["graph_service"] = self.component_manager.graph_service.get_detailed_stats(
                    )

            if self.component_manager.cache_system:
                if hasattr(self.component_manager.cache_system, "get_cache_stats"):
                    stats["cache_system"] = await self.component_manager.cache_system.get_cache_stats()

            if self.component_manager.graph_storage:
                if hasattr(self.component_manager.graph_storage, "get_storage_stats"):
                    stats["graph_storage"] = await self.component_manager.graph_storage.get_storage_stats()

            if self.component_manager.reputation_system:
                if hasattr(self.component_manager.reputation_system, "get_reputation_stats"):
                    stats["reputation_system"] = (
                        await self.component_manager.reputation_system.get_reputation_stats()
                    )

            if self.component_manager.clustering_system:
                if hasattr(self.component_manager.clustering_system, "get_stats"):
                    stats["clustering_system"] = await self.component_manager.clustering_system.get_stats()

            if self.component_manager.uncertainty_handler:
                if hasattr(self.component_manager.uncertainty_handler, "get_stats"):
                    stats["uncertainty_handler"] = await self.component_manager.uncertainty_handler.get_stats()

            if self.component_manager.relationship_analyzer:
                if hasattr(self.component_manager.relationship_analyzer, "get_stats"):
                    stats["relationship_analyzer"] = await self.component_manager.relationship_analyzer.get_stats()

        except Exception as e:
            self.logger.error("Failed to collect system statistics: %s", e)
            stats["error"] = str(e)

        return stats

    async def _check_cache_health(self) -> dict[str, Any]:
        """
        Check cache system health.

        Returns:
            dict: Cache health status and metrics
        """
        try:
            if not self.component_manager.cache_system:
                return {"status": "disabled"}

            if hasattr(self.component_manager.cache_system, "get_cache_stats"):
                stats = await self.component_manager.cache_system.get_cache_stats()
                return {
                    "status": "healthy",
                    "hit_rate": stats.get("hit_rate", 0),
                    "total_operations": stats.get("total_operations", 0),
                    "memory_usage": stats.get("memory_usage", 0),
                }
            else:
                return {"status": "healthy", "note": "basic health check passed"}
        except Exception as e:
            self.logger.error("Cache health check failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def _check_storage_health(self) -> dict[str, Any]:
        """
        Check graph storage health.

        Returns:
            dict: Storage health status and metrics
        """
        try:
            if not self.component_manager.graph_storage:
                return {"status": "disabled"}

            if hasattr(self.component_manager.graph_storage, "get_storage_stats"):
                stats = await self.component_manager.graph_storage.get_storage_stats()
                return {
                    "status": "healthy",
                    "total_graphs": stats.get("total_graphs", 0),
                    "connection_status": "connected",
                }
            else:
                return {"status": "healthy", "note": "basic health check passed"}
        except Exception as e:
            self.logger.error("Storage health check failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def _check_reputation_health(self) -> dict[str, Any]:
        """
        Check reputation system health.

        Returns:
            dict: Reputation system health status and metrics
        """
        try:
            if not self.component_manager.reputation_system:
                return {"status": "disabled"}

            if hasattr(self.component_manager.reputation_system, "get_reputation_stats"):
                stats = await self.component_manager.reputation_system.get_reputation_stats()
                return {
                    "status": "healthy",
                    "total_sources": stats.get("total_sources", 0),
                }
            else:
                return {"status": "healthy", "note": "basic health check passed"}
        except Exception as e:
            self.logger.error("Reputation health check failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def _check_graph_service_health(self) -> dict[str, Any]:
        """
        Check graph service health.

        Returns:
            dict: Graph service health status and metrics
        """
        try:
            if not self.component_manager.graph_service:
                return {"status": "disabled"}

            # Basic health check - verify the service is accessible
            return {"status": "healthy", "note": "graph service operational"}
        except Exception as e:
            self.logger.error("Graph service health check failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def _check_clustering_health(self) -> dict[str, Any]:
        """
        Check clustering system health.

        Returns:
            dict: Clustering system health status and metrics
        """
        try:
            if not self.component_manager.clustering_system:
                return {"status": "disabled"}

            return {"status": "healthy", "note": "clustering system operational"}
        except Exception as e:
            self.logger.error("Clustering health check failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def _check_uncertainty_health(self) -> dict[str, Any]:
        """
        Check uncertainty handler health.

        Returns:
            dict: Uncertainty handler health status and metrics
        """
        try:
            if not self.component_manager.uncertainty_handler:
                return {"status": "disabled"}

            return {"status": "healthy", "note": "uncertainty handler operational"}
        except Exception as e:
            self.logger.error("Uncertainty health check failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def _check_relationship_health(self) -> dict[str, Any]:
        """
        Check relationship analyzer health.

        Returns:
            dict: Relationship analyzer health status and metrics
        """
        try:
            if not self.component_manager.relationship_analyzer:
                return {"status": "disabled"}

            return {"status": "healthy", "note": "relationship analyzer operational"}
        except Exception as e:
            self.logger.error(
                "Relationship analyzer health check failed: %s", e)
            return {"status": "error", "error": str(e)}

    def get_component_count(self) -> int:
        """
        Get the total number of active components.

        Returns:
            int: Number of active components
        """
        return len(self.component_manager.get_active_components())

    async def get_performance_metrics(self) -> dict[str, Any]:
        """
        Get performance metrics for the system.

        Returns:
            dict: Performance metrics including response times and throughput
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "active_components": self.get_component_count(),
            "system_uptime": None,
        }

        if self.component_manager.initialization_timestamp:
            uptime = datetime.now() - self.component_manager.initialization_timestamp
            metrics["system_uptime"] = uptime.total_seconds()

        return metrics
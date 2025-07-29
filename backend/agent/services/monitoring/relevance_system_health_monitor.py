"""
Extended system health monitor for relevance-specific health checks.

Extends the base SystemHealthMonitor to include health checks for
relevance system components while maintaining compatibility.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from ..core.system_health_monitor import SystemHealthMonitor
from ..core.relevance_component_manager import RelevanceComponentManager

logger = logging.getLogger(__name__)


class RelevanceSystemHealthMonitor(SystemHealthMonitor):
    """
    Extended system health monitor that includes relevance-specific health checks.

    This class extends the base SystemHealthMonitor to include health checks
    specifically for relevance system components.
    """

    def __init__(self, component_manager: RelevanceComponentManager):
        """
        Initialize the relevance system health monitor.

        Args:
            component_manager: RelevanceComponentManager instance
        """
        super().__init__(component_manager)
        self.relevance_component_manager = component_manager

    async def check_system_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive system health check including relevance components.

        Returns:
            dict: Comprehensive health status
        """
        try:
            # Get base health check results
            health_status = await super().health_check()

            # Add relevance-specific health checks
            relevance_health = await self._check_relevance_components_health()
            health_status["relevance_components"] = relevance_health

            # Update overall status based on relevance components
            if relevance_health["status"] != "healthy":
                if health_status["overall_status"] == "healthy":
                    health_status["overall_status"] = "degraded"
                elif health_status["overall_status"] == "degraded" and relevance_health["status"] == "critical":
                    health_status["overall_status"] = "critical"

            return health_status

        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return {
                "overall_status": "critical",
                "error": str(e),
                "timestamp": self._get_current_timestamp()
            }

    async def _check_relevance_components_health(self) -> Dict[str, Any]:
        """
        Check health of relevance-specific components.

        Returns:
            dict: Health status of relevance components
        """
        health_status = {
            "status": "healthy",
            "components": {},
            "issues": [],
            "timestamp": self._get_current_timestamp()
        }

        try:
            # Check RelevanceEmbeddingsCoordinator
            coordinator_health = await self._check_embeddings_coordinator_health()
            health_status["components"]["embeddings_coordinator"] = coordinator_health

            if coordinator_health["status"] != "healthy":
                health_status["status"] = "degraded"
                health_status["issues"].extend(
                    coordinator_health.get("issues", []))

            # Check relevance performance metrics
            performance_health = await self._check_relevance_performance_health()
            health_status["components"]["performance"] = performance_health

            if performance_health["status"] != "healthy":
                if health_status["status"] == "healthy":
                    health_status["status"] = "degraded"
                health_status["issues"].extend(
                    performance_health.get("issues", []))

            return health_status

        except Exception as e:
            logger.error(f"Error checking relevance components health: {e}")
            return {
                "status": "critical",
                "error": str(e),
                "timestamp": self._get_current_timestamp()
            }

    async def _check_embeddings_coordinator_health(self) -> Dict[str, Any]:
        """
        Check health of the RelevanceEmbeddingsCoordinator.

        Returns:
            dict: Health status of embeddings coordinator
        """
        health_status = {
            "status": "healthy",
            "details": {},
            "issues": []
        }

        try:
            coordinator = self.relevance_component_manager.get_component(
                "relevance_embeddings_coordinator")

            if not coordinator:
                health_status["status"] = "critical"
                health_status["issues"].append(
                    "RelevanceEmbeddingsCoordinator not found")
                return health_status

            if not coordinator._initialized:
                health_status["status"] = "critical"
                health_status["issues"].append(
                    "RelevanceEmbeddingsCoordinator not initialized")
                return health_status

            # Check individual components within coordinator
            components_to_check = [
                ("embeddings", coordinator.embeddings),
                ("hybrid_scorer", coordinator.hybrid_scorer),
                ("temporal_cache", coordinator.temporal_cache),
                ("explainable_scorer", coordinator.explainable_scorer),
                ("intelligent_cache", coordinator.intelligent_cache),
                ("adaptive_thresholds", coordinator.adaptive_thresholds)
            ]

            for component_name, component in components_to_check:
                if component is None:
                    health_status["status"] = "degraded"
                    health_status["issues"].append(
                        f"{component_name} is not initialized")
                else:
                    health_status["details"][component_name] = "initialized"

            # Test basic functionality
            try:
                test_embeddings = await coordinator.generate_embeddings("test query")
                if test_embeddings is None:
                    health_status["status"] = "degraded"
                    health_status["issues"].append(
                        "Embeddings generation test failed")
                else:
                    health_status["details"]["embeddings_test"] = "passed"
            except Exception as e:
                health_status["status"] = "degraded"
                health_status["issues"].append(f"Embeddings test error: {e}")

            return health_status

        except Exception as e:
            logger.error(f"Error checking embeddings coordinator health: {e}")
            return {
                "status": "critical",
                "error": str(e)
            }

    async def _check_relevance_performance_health(self) -> Dict[str, Any]:
        """
        Check performance health of relevance components.

        Returns:
            dict: Performance health status
        """
        health_status = {
            "status": "healthy",
            "metrics": {},
            "issues": []
        }

        try:
            # Get performance metrics
            metrics = await self.relevance_component_manager.get_relevance_performance_metrics()

            if "error" in metrics:
                health_status["status"] = "degraded"
                health_status["issues"].append(
                    f"Failed to get performance metrics: {metrics['error']}")
                return health_status

            health_status["metrics"] = metrics

            # Analyze metrics for health issues
            coordinator_metrics = metrics.get(
                "relevance_embeddings_coordinator", {})

            # Check cache performance
            cache_metrics = coordinator_metrics.get("cache", {})
            if cache_metrics:
                hit_rate = cache_metrics.get("hit_rate", 0)
                if hit_rate < 0.5:
                    health_status["status"] = "degraded"
                    health_status["issues"].append(
                        f"Low cache hit rate: {hit_rate:.2%}")

            # Check embeddings performance
            embeddings_metrics = coordinator_metrics.get("embeddings", {})
            if embeddings_metrics:
                avg_response_time = embeddings_metrics.get(
                    "avg_response_time", 0)
                if avg_response_time > 5.0:  # 5 seconds threshold
                    health_status["status"] = "degraded"
                    health_status["issues"].append(
                        f"High embeddings response time: {avg_response_time:.2f}s")

            return health_status

        except Exception as e:
            logger.error(f"Error checking relevance performance health: {e}")
            return {
                "status": "critical",
                "error": str(e)
            }

    async def get_comprehensive_health_report(self) -> str:
        """
        Generate comprehensive health report including relevance components.

        Returns:
            str: Detailed health report
        """
        try:
            # Get base health status using the parent's health_check method
            base_health = await super().health_check()
            
            # Convert base health to report format
            base_report = "=== SYSTEM HEALTH REPORT ===\n"
            base_report += f"Overall Status: {base_health.get('overall_status', 'unknown').upper()}\n"
            base_report += f"Timestamp: {base_health.get('timestamp', 'unknown')}\n\n"
            
            # Add base component details
            for component_name, component_health in base_health.get("components", {}).items():
                base_report += f"--- {component_name.upper()} ---\n"
                base_report += f"Status: {component_health.get('status', 'unknown').upper()}\n"
                
                # Add any additional details from the component health
                for key, value in component_health.items():
                    if key != 'status':
                        base_report += f"  {key}: {value}\n"
                base_report += "\n"

            # Add relevance-specific health information
            relevance_health = await self._check_relevance_components_health()

            relevance_report = "=== RELEVANCE SYSTEM HEALTH ===\n"
            relevance_report += f"Overall Status: {relevance_health['status'].upper()}\n"
            relevance_report += f"Timestamp: {relevance_health['timestamp']}\n\n"

            # Component details
            for component_name, component_health in relevance_health.get("components", {}).items():
                relevance_report += f"--- {component_name.upper()} ---\n"
                relevance_report += f"Status: {component_health['status'].upper()}\n"

                if "details" in component_health:
                    for detail_key, detail_value in component_health["details"].items():
                        relevance_report += f"  {detail_key}: {detail_value}\n"

                if "metrics" in component_health:
                    relevance_report += "  Metrics:\n"
                    for metric_key, metric_value in component_health["metrics"].items():
                        relevance_report += f"    {metric_key}: {metric_value}\n"

                relevance_report += "\n"

            # Issues summary
            if relevance_health.get("issues"):
                relevance_report += "--- RELEVANCE ISSUES ---\n"
                for issue in relevance_health["issues"]:
                    relevance_report += f"  - {issue}\n"
                relevance_report += "\n"

            return base_report + relevance_report

        except Exception as e:
            logger.error(f"Error generating comprehensive health report: {e}")
            return f"Error generating health report: {e}"

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat()


# Global instance management
_relevance_health_monitor: Optional[RelevanceSystemHealthMonitor] = None


def get_relevance_health_monitor(component_manager: RelevanceComponentManager = None) -> RelevanceSystemHealthMonitor:
    """
    Get global relevance health monitor instance.

    Args:
        component_manager: RelevanceComponentManager instance (required for first initialization)

    Returns:
        RelevanceSystemHealthMonitor: Global instance
    """
    global _relevance_health_monitor
    if _relevance_health_monitor is None:
        if component_manager is None:
            raise ValueError(
                "RelevanceComponentManager required for first initialization")
        _relevance_health_monitor = RelevanceSystemHealthMonitor(
            component_manager)
    return _relevance_health_monitor


def close_relevance_health_monitor():
    """Close global relevance health monitor instance."""
    global _relevance_health_monitor
    _relevance_health_monitor = None

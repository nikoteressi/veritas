"""
Temporal clustering strategy implementation.

This module implements clustering based on temporal relationships between facts.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

from agent.models.graph import ClusterType, FactCluster, FactNode
from agent.services.graph.interfaces.clustering_strategy import ClusteringStrategy

logger = logging.getLogger(__name__)


class TemporalClusteringStrategy(ClusteringStrategy):
    """
    Clustering strategy based on temporal relationships.

    Groups facts that are temporally related or occurred within
    similar time periods.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize temporal clustering strategy."""
        self._config = self._get_default_config()
        if config:
            self._config.update(config)

        self._validate_config(self._config)
        logger.info(f"Initialized {self.get_strategy_name()} with config: {self._config}")

    def create_clusters(self, nodes: list[FactNode]) -> list[FactCluster]:
        """
        Create temporal-based clusters from fact nodes.

        Args:
            nodes: List of fact nodes to cluster

        Returns:
            List of fact clusters based on temporal relationships
        """
        if not nodes:
            logger.warning("No nodes provided for clustering")
            return []

        # Extract temporal information and group nodes
        temporal_groups = self._group_by_temporal_windows(nodes)

        # Create clusters from temporal groups
        clusters = []
        for window_key, window_nodes in temporal_groups.items():
            cluster = FactCluster(
                id=f"temporal_cluster_{window_key}",
                nodes=window_nodes,
                cluster_type=ClusterType.TEMPORAL_CLUSTER,
                verification_strategy=self._get_verification_strategy(len(window_nodes)),
            )
            clusters.append(cluster)

        logger.info(f"Created {len(clusters)} temporal clusters from {len(nodes)} nodes")
        return clusters

    def _group_by_temporal_windows(self, nodes: list[FactNode]) -> dict[str, list[FactNode]]:
        """Group nodes by temporal windows."""
        temporal_groups = defaultdict(list)

        for node in nodes:
            timestamp = self._extract_timestamp(node)
            window_key = self._get_temporal_window(timestamp)
            temporal_groups[window_key].append(node)

        return temporal_groups

    def _extract_timestamp(self, node: FactNode) -> datetime | None:
        """Extract timestamp from node metadata."""
        if not node.metadata:
            return None

        # Try different timestamp fields
        timestamp_fields = ["timestamp", "created_at", "date", "time"]

        for field in timestamp_fields:
            if field in node.metadata:
                timestamp_value = node.metadata[field]

                if isinstance(timestamp_value, datetime):
                    return timestamp_value
                elif isinstance(timestamp_value, str):
                    try:
                        return datetime.fromisoformat(timestamp_value.replace("Z", "+00:00"))
                    except ValueError:
                        continue
                elif isinstance(timestamp_value, int | float):
                    try:
                        return datetime.fromtimestamp(timestamp_value)
                    except (ValueError, OSError):
                        continue

        return None

    def _get_temporal_window(self, timestamp: datetime | None) -> str:
        """Get temporal window key for a timestamp."""
        if timestamp is None:
            return "no_timestamp"

        window_type = self._config["window_type"]

        if window_type == "hour":
            return f"{timestamp.year}-{timestamp.month:02d}-{timestamp.day:02d}-{timestamp.hour:02d}"
        elif window_type == "day":
            return f"{timestamp.year}-{timestamp.month:02d}-{timestamp.day:02d}"
        elif window_type == "week":
            # Get week number
            year, week, _ = timestamp.isocalendar()
            return f"{year}-W{week:02d}"
        elif window_type == "month":
            return f"{timestamp.year}-{timestamp.month:02d}"
        elif window_type == "custom":
            # Custom window size in hours
            window_hours = self._config["custom_window_hours"]
            window_start = timestamp.replace(minute=0, second=0, microsecond=0)
            window_start = window_start.replace(hour=(window_start.hour // window_hours) * window_hours)
            return f"custom_{window_start.isoformat()}"
        else:
            raise ValueError(f"Unknown window type: {window_type}")

    def _get_verification_strategy(self, cluster_size: int) -> str:
        """Determine verification strategy based on cluster size."""
        if cluster_size == 1:
            return "individual"
        elif cluster_size <= self._config["batch_threshold"]:
            return "batch"
        else:
            return "cross_verification"

    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "temporal_clustering"

    def validate_config(self, config: dict[str, Any]) -> bool:
        """
        Validate configuration parameters.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        return self._validate_config(config)

    def _validate_config(self, config: dict[str, Any]) -> bool:
        """Internal config validation."""
        if "window_type" not in config:
            raise ValueError("Missing required config key: window_type")

        valid_window_types = ["hour", "day", "week", "month", "custom"]
        if config["window_type"] not in valid_window_types:
            raise ValueError(f"window_type must be one of: {valid_window_types}")

        if config["window_type"] == "custom":
            if "custom_window_hours" not in config:
                raise ValueError("custom_window_hours required when window_type is 'custom'")
            if not isinstance(config["custom_window_hours"], int) or config["custom_window_hours"] < 1:
                raise ValueError("custom_window_hours must be a positive integer")

        if "batch_threshold" in config:
            if not isinstance(config["batch_threshold"], int) or config["batch_threshold"] < 1:
                raise ValueError("batch_threshold must be a positive integer")

        return True

    def get_config(self) -> dict[str, Any]:
        """Get current configuration."""
        return self._config.copy()

    def update_config(self, config: dict[str, Any]) -> None:
        """
        Update configuration.

        Args:
            config: New configuration parameters
        """
        new_config = self._config.copy()
        new_config.update(config)
        self._validate_config(new_config)
        self._config = new_config
        logger.info(f"Updated config for {self.get_strategy_name()}: {self._config}")

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration."""
        return {
            "window_type": "day",  # hour, day, week, month, custom
            "custom_window_hours": 24,  # Used when window_type is 'custom'
            "batch_threshold": 5,  # Switch to cross-verification above this size
        }

"""
Abstract interface for verification strategies.

Defines the contract for different verification approaches used in fact checking.
"""

from abc import ABC, abstractmethod
from typing import Any

from agent.models.graph import FactCluster, FactGraph, FactNode
from agent.models.graph_verification_models import VerificationResponse


class VerificationStrategy(ABC):
    """
    Abstract base class for verification strategies.

    Implements the Strategy pattern for different verification approaches
    such as individual verification, batch verification, cross-verification, etc.
    """

    def __init__(self, config: dict[str, Any] = None):
        """
        Initialize verification strategy with configuration.

        Args:
            config: Strategy-specific configuration parameters
        """
        self.config = config or {}

    @abstractmethod
    async def verify_node(self, node: FactNode, context: dict[str, Any] | None = None) -> VerificationResponse:
        """
        Verify a single fact node.

        Args:
            node: Fact node to verify
            context: Additional context for verification

        Returns:
            Verification response with verdict and confidence

        Raises:
            VerificationError: If verification fails
        """

    @abstractmethod
    async def verify_cluster(
        self, cluster: FactCluster, context: dict[str, Any] | None = None
    ) -> dict[str, VerificationResponse]:
        """
        Verify all nodes in a cluster.

        Args:
            cluster: Fact cluster to verify
            context: Additional context for verification

        Returns:
            Dictionary mapping node IDs to verification responses

        Raises:
            VerificationError: If verification fails
        """

    @abstractmethod
    async def cross_verify(self, nodes: list[FactNode], context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Perform cross-verification between multiple nodes.

        Args:
            nodes: List of nodes to cross-verify
            context: Additional context for verification

        Returns:
            Cross-verification results including relationships and contradictions

        Raises:
            VerificationError: If cross-verification fails
        """

    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        Get the name of this verification strategy.

        Returns:
            Strategy name identifier
        """

    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate the strategy configuration.

        Returns:
            True if configuration is valid, False otherwise
        """

    def get_config(self) -> dict[str, Any]:
        """
        Get current strategy configuration.

        Returns:
            Configuration dictionary
        """
        return self.config.copy()

    def update_config(self, new_config: dict[str, Any]) -> None:
        """
        Update strategy configuration.

        Args:
            new_config: New configuration parameters
        """
        self.config.update(new_config)
        if not self.validate_config():
            raise ValueError(f"Invalid configuration for {self.get_strategy_name()}")

    async def prepare_verification_context(self, graph: FactGraph, target_nodes: list[FactNode]) -> dict[str, Any]:
        """
        Prepare context for verification based on graph structure.

        Args:
            graph: Complete fact graph
            target_nodes: Nodes to be verified

        Returns:
            Verification context dictionary
        """
        return {
            "graph_stats": graph.get_stats(),
            "target_node_count": len(target_nodes),
            "strategy": self.get_strategy_name(),
            "config": self.get_config(),
        }

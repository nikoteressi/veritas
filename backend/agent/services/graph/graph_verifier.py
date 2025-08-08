"""
Graph verifier wrapper for the enhanced verification engine.
"""

from typing import Any

from agent.models.graph import FactGraph
from agent.models.verification_context import VerificationContext
from agent.tools import SearxNGSearchTool

from .graph_config import VerificationConfig
from .verification.engine import EnhancedGraphVerificationEngine


class GraphVerifier:
    """
    Wrapper class for the enhanced graph verification engine.

    Provides a simplified interface for graph verification operations.
    """

    def __init__(self, config: VerificationConfig):
        """Initialize the graph verifier with configuration."""
        self.config = config
        self._engine = None

    def set_search_tool(self, search_tool: SearxNGSearchTool) -> None:
        """Set the search tool and initialize the engine."""
        self._engine = EnhancedGraphVerificationEngine(search_tool, self.config)

    async def verify_graph(self, graph: FactGraph, context: VerificationContext) -> dict[str, Any]:
        """
        Verify a fact graph.

        Args:
            graph: The fact graph to verify
            context: Verification context

        Returns:
            Verification results
        """
        if not self._engine:
            raise RuntimeError("Search tool not set. Call set_search_tool() first.")

        return await self._engine.verify_graph(graph, context)

    def set_progress_callback(self, callback) -> None:
        """Set progress callback for the verification engine."""
        if self._engine:
            self._engine.set_progress_callback(callback)

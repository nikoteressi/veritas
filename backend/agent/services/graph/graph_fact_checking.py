"""Graph-based fact checking service with new component architecture.

This service uses graph-based verification with modular components for better
maintainability and separation of concerns.

New architecture components:
- FactVerificationOrchestrator: Main coordination
- GraphCacheManager: Caching management
- VerificationResultCompiler: Result compilation
- UncertaintyAnalyzer: Uncertainty analysis
"""

import logging
from typing import Any

from agent.models import FactCheckResult
from agent.models.verification_context import VerificationContext
from agent.tools.search import SearxNGSearchTool
from app.exceptions import AgentError
from app.models.progress_callback import NoOpProgressCallback, ProgressCallback

from .cache_manager import GraphCacheManager
from .graph_config import VerificationConfig
from .result_compiler import VerificationResultCompiler
from .uncertainty_analyzer import UncertaintyAnalyzer
from .verification_orchestrator import FactVerificationOrchestrator
from .graph_builder import GraphBuilder
from .graph_verifier import GraphVerifier

logger = logging.getLogger(__name__)


class GraphFactCheckingService:
    """Graph-based fact checking service with enhanced verification capabilities.

    Refactored to use new component architecture for better modularity and maintainability.
    """

    def __init__(
        self,
        search_tool: SearxNGSearchTool,
        config: VerificationConfig | None = None,
        progress_callback: ProgressCallback | None = None,
    ):
        """Initialize the graph fact checking service.

        Args:
            search_tool: Tool for searching information
            config: Configuration for verification
            progress_callback: Callback for progress updates
        """
        self.search_tool = search_tool
        self.config = config or VerificationConfig()
        self.progress_callback = progress_callback or NoOpProgressCallback()

        # New component architecture - lazy initialization
        self._orchestrator: FactVerificationOrchestrator | None = None
        self._cache_manager: GraphCacheManager | None = None
        self._result_compiler: VerificationResultCompiler | None = None
        self._uncertainty_analyzer: UncertaintyAnalyzer | None = None

        logger.info(
            "GraphFactCheckingService initialized with new architecture")

    def set_progress_callback(self, callback: ProgressCallback | None) -> None:
        """Set the progress callback for detailed progress reporting."""
        self.progress_callback = callback or NoOpProgressCallback()

    def update_progress(self, message: str, percentage: float = 0.0) -> None:
        """Update progress with a message and percentage."""
        self.progress_callback.update_progress(message, percentage)

    def _ensure_orchestrator(self) -> FactVerificationOrchestrator:
        """Ensure orchestrator is initialized."""
        if self._orchestrator is None:
            # Import here to avoid circular imports
            graph_builder = GraphBuilder(self.config)
            graph_verifier = GraphVerifier(self.config)

            # Set search tool for graph verifier
            graph_verifier.set_search_tool(self.search_tool)

            self._orchestrator = FactVerificationOrchestrator(
                graph_builder=graph_builder,
                graph_verifier=graph_verifier,
                cache_manager=self._ensure_cache_manager(),
                uncertainty_analyzer=self._ensure_uncertainty_analyzer(),
                result_compiler=self._ensure_result_compiler(),
            )
            logger.info("FactVerificationOrchestrator initialized")
        return self._orchestrator

    def _ensure_cache_manager(self) -> GraphCacheManager:
        """Ensure cache manager is initialized."""
        if self._cache_manager is None:
            self._cache_manager = GraphCacheManager()
            logger.info("GraphCacheManager initialized")
        return self._cache_manager

    def _ensure_result_compiler(self) -> VerificationResultCompiler:
        """Ensure result compiler is initialized."""
        if self._result_compiler is None:
            self._result_compiler = VerificationResultCompiler(
                uncertainty_analyzer=self._ensure_uncertainty_analyzer()
            )
            logger.info("VerificationResultCompiler initialized")
        return self._result_compiler

    def _ensure_uncertainty_analyzer(self) -> UncertaintyAnalyzer:
        """Ensure uncertainty analyzer is initialized."""
        if self._uncertainty_analyzer is None:
            self._uncertainty_analyzer = UncertaintyAnalyzer()
            logger.info("UncertaintyAnalyzer initialized")
        return self._uncertainty_analyzer

    async def verify_facts(self, context: VerificationContext) -> FactCheckResult:
        """Verify facts using graph-based analysis with new component architecture.

        Args:
            context: Verification context containing facts to verify

        Returns:
            FactCheckResult: Comprehensive verification result

        Raises:
            AgentError: If verification fails
        """
        try:
            self.update_progress("Starting graph-based fact verification", 0.0)

            # Validate input
            fact_hierarchy = context.fact_hierarchy
            if not fact_hierarchy or not fact_hierarchy.facts:
                raise AgentError("No facts provided for verification")

            # Initialize components
            orchestrator = self._ensure_orchestrator()
            cache_manager = self._ensure_cache_manager()
            result_compiler = self._ensure_result_compiler()

            # Check cache first
            self.update_progress("Checking cache", 5.0)
            cache_key = cache_manager.generate_cache_key(fact_hierarchy)
            cached_result = await cache_manager.get_cached_result(cache_key)
            if cached_result:
                self.update_progress("Retrieved result from cache", 100.0)
                return cached_result

            # Orchestrate verification process
            self.update_progress("Orchestrating verification", 10.0)
            verification_data = await orchestrator.verify_multiple_claims(
                [fact.claim for fact in fact_hierarchy.facts], context
            )

            # Compile final result
            self.update_progress("Compiling verification result", 90.0)

            # Extract necessary data for result compilation
            graph_nodes = verification_data.get('graph_nodes', {})
            claim = fact_hierarchy.facts[0].claim if fact_hierarchy.facts else "Unknown claim"
            sources = verification_data.get('sources', [])
            queries = verification_data.get('queries', [])

            result = await result_compiler.convert_to_fact_check_result(
                verification_data, graph_nodes, claim, sources, queries
            )

            # Cache the result
            await cache_manager.cache_result(cache_key, result)

            self.update_progress("Fact verification completed", 100.0)
            return result

        except Exception as e:
            logger.error("Graph fact verification failed: %s", e)
            raise AgentError(f"Graph fact verification failed: {e}") from e

    async def get_uncertainty_analysis(
        self, verification_data: dict[str, Any], graph_nodes: list[Any] | None = None
    ) -> dict[str, Any]:
        """
        Get uncertainty analysis for verification results.

        Args:
            verification_data: Verification data to analyze
            graph_nodes: Optional graph nodes for analysis

        Returns:
            Dict containing uncertainty analysis
        """
        try:
            uncertainty_analyzer = self._ensure_uncertainty_analyzer()

            # Extract verification results from verification_data
            verification_results = verification_data.get(
                'results', verification_data)
            nodes = graph_nodes or verification_data.get('graph_nodes', [])

            analysis = await uncertainty_analyzer.analyze_verification_uncertainty(
                verification_results, nodes
            )
            return analysis
        except Exception as e:
            logger.error("Error in uncertainty analysis: %s", e)
            return {
                "error": str(e),
                "uncertainty_level": "high",
                "confidence_intervals": {}
            }

    async def clear_cache(self) -> None:
        """Clear all caches."""
        try:
            if self._cache_manager:
                await self._cache_manager.clear_all_cache()
            logger.info("Graph fact checking cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            raise

    async def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        stats = {
            "service": "GraphFactCheckingService",
            "initialized_components": {
                "cache_manager": self._cache_manager is not None,
                "orchestrator": self._orchestrator is not None,
                "uncertainty_analyzer": self._uncertainty_analyzer is not None,
                "result_compiler": self._result_compiler is not None,
            }
        }

        # Note: Component stats methods are not implemented in the current components
        # This is a simplified version that only shows initialization status

        return stats

    async def close(self) -> None:
        """Close all components and clean up resources."""
        try:
            # Clean up component references
            # Note: Components don't have close() methods in current implementation
            self._result_compiler = None
            self._uncertainty_analyzer = None
            self._orchestrator = None

            # Cache manager also doesn't have close method - just clear reference
            self._cache_manager = None

            logger.info("GraphFactCheckingService closed successfully")

        except Exception as e:
            logger.error(f"Error closing GraphFactCheckingService: {e}")
            raise

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

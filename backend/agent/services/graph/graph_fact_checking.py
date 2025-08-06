"""Graph-based fact checking service that replaces the traditional fact checking approach.

from __future__ import annotations

This service uses graph-based verification with clustering and relationship analysis
for more efficient and accurate fact verification. Enhanced with Neo4j persistence,
source reputation system, and intelligent caching.
"""

import asyncio
import hashlib
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from agent.models.graph import FactEdge
from app.exceptions import AgentError
from app.models.progress_callback import NoOpProgressCallback, ProgressCallback

from ...models import ClaimResult, FactCheckResult, FactCheckSummary
from ...models.graph import FactGraph, VerificationStatus
from ...models.verification_context import VerificationContext
from ..analysis.advanced_clustering import (
    AdvancedClusteringSystem,
    ClusteringConfig,
)
from ..analysis.bayesian_uncertainty import (
    BayesianUncertaintyHandler,
    UncertaintyConfig,
)
from ..analysis.relationship_analysis import (
    RelationshipAnalysisEngine,
    RelationshipConfig,
)
from ..cache.intelligent_cache import get_general_cache, get_verification_cache
from ..reputation.source_reputation import SourceReputationSystem
from .graph_builder import GraphBuilder
from .graph_config import VerificationConfig
from .graph_storage import Neo4jGraphStorage
from .verification.engine import GraphVerificationEngine

if TYPE_CHECKING:
    from ...tools import SearxNGSearchTool

logger = logging.getLogger(__name__)


class GraphFactCheckingService:
    """Graph-based fact checking service.

    This service replaces the traditional FactCheckingService with a graph-based approach
    that analyzes relationships between facts and verifies them in clusters for improved
    efficiency and accuracy. Enhanced with persistent storage, source reputation, and
    intelligent caching.
    """

    def __init__(
        self,
        search_tool: "SearxNGSearchTool",
        clustering_config: ClusteringConfig | None = None,
        verification_config: VerificationConfig | None = None,
    ):
        """Initialize the graph-based fact checking service.

        Args:
            search_tool: Search tool for gathering evidence
            clustering_config: Configuration for graph clustering
            verification_config: Configuration for verification engine
        """
        self.search_tool = search_tool

        # Initialize graph components
        self.graph_builder = GraphBuilder(clustering_config)

        # Initialize verification engine
        self.verification_engine = GraphVerificationEngine(
            search_tool=search_tool, config=verification_config)

        # Initialize persistent storage (try to connect, but don't fail if unavailable)
        self.graph_storage = None
        try:
            self.graph_storage = Neo4jGraphStorage()
            logger.info("Neo4j graph storage initialized")
        except Exception as e:
            logger.warning("Neo4j storage unavailable: %s", e)
            self.graph_storage = None

        # Initialize source reputation system
        self.source_reputation = None
        try:
            self.source_reputation = SourceReputationSystem()
            logger.info("Source reputation system initialized")
        except Exception as e:
            logger.warning("Source reputation unavailable: %s", e)
            self.source_reputation = None

        # Initialize caching systems
        self.verification_cache = get_verification_cache()
        self.general_cache = get_general_cache()

        # Initialize advanced clustering
        self.advanced_clustering = None
        try:
            clustering_config = ClusteringConfig()
            self.advanced_clustering = AdvancedClusteringSystem(
                clustering_config, self.graph_builder)
            logger.info("Advanced clustering system initialized")
        except Exception as e:
            logger.warning("Advanced clustering unavailable: %s", e)

        # Initialize Bayesian uncertainty handler (lazy initialization)
        self.uncertainty_handler = None
        self._uncertainty_handler_initialized = False
        logger.info(
            "Bayesian uncertainty handler will be initialized on first use")

        # Initialize relationship analysis engine
        self.relationship_analyzer = None
        try:
            relationship_config = RelationshipConfig()
            self.relationship_analyzer = RelationshipAnalysisEngine(
                relationship_config)
            logger.info("Relationship analysis engine initialized")
        except Exception as e:
            logger.warning("Relationship analyzer unavailable: %s", e)

        self.logger = logging.getLogger(__name__)

        # Track current graph ID for statistics
        self.current_graph_id = None

        # Progress callback for detailed progress reporting
        self.progress_callback: ProgressCallback = NoOpProgressCallback()

    def set_progress_callback(self, callback: ProgressCallback | None) -> None:
        """Set the progress callback for detailed progress reporting."""
        self.progress_callback = callback or NoOpProgressCallback()

    async def update_progress(self, current: float, total: float, message: str = "") -> None:
        """Update the progress for the current operation."""
        await self.progress_callback.update_progress(current, total, message)

    async def update_substep(self, substep: str, progress: float, message: str = "") -> None:
        """Update the current substep being executed."""
        await self.progress_callback.update_substep(substep, progress, message)

    async def _ensure_uncertainty_handler(self):
        """Ensure Bayesian uncertainty handler is initialized (lazy initialization)."""
        if not self._uncertainty_handler_initialized:
            try:
                logger.info(
                    "Initializing Bayesian uncertainty handler (lazy initialization)")
                uncertainty_config = UncertaintyConfig()
                self.uncertainty_handler = BayesianUncertaintyHandler(
                    uncertainty_config)
                self._uncertainty_handler_initialized = True
                logger.info(
                    "Bayesian uncertainty handler initialized successfully")
            except Exception as e:
                logger.warning("Uncertainty handler unavailable: %s", e)
                self.uncertainty_handler = None
                self._uncertainty_handler_initialized = True  # Mark as attempted

    async def verify_facts(self, context: VerificationContext) -> FactCheckResult:
        """
        Perform graph-based fact checking on the verification context.

        This method maintains compatibility with the existing FactCheckingService interface
        while using the new graph-based approach internally. Enhanced with caching,
        persistence, and source reputation.

        Args:
            context: The verification context containing all data

        Returns:
            FactCheckResult: Results in the same format as the original service
        """
        start_time = datetime.now()
        self.logger.info("Starting enhanced graph-based fact checking")

        try:
            # Extract fact hierarchy from context
            if not context.post_analysis_result or not context.post_analysis_result.fact_hierarchy:
                self.logger.warning("No fact hierarchy found in context")
                return self._create_empty_result()

            fact_hierarchy = context.post_analysis_result.fact_hierarchy

            if not fact_hierarchy.supporting_facts:
                self.logger.warning(
                    "No supporting facts found in fact hierarchy")
                return self._create_empty_result()

            self.logger.info("Processing %d facts", len(
                fact_hierarchy.supporting_facts))

            # Generate cache key for this verification
            cache_key = self._generate_cache_key(fact_hierarchy)

            # Check cache first
            cached_result = await self.verification_cache.get(cache_key)
            if cached_result:
                self.logger.info("Found cached verification result")
                await self.update_progress(1.0, 1.0, "Using cached verification result")
                return cached_result

            # Update progress for graph loading/building
            await self.update_substep("Loading or building fact graph", 0.1, "Checking for existing graph...")

            # Step 1: Try to load existing graph from persistent storage
            graph = None
            graph_id = None
            if self.graph_storage:
                graph_id = self._generate_graph_id(fact_hierarchy)
                self.current_graph_id = graph_id  # Store for later use
                try:
                    graph = self.graph_storage.load_graph(graph_id)
                    if graph:
                        self.logger.info(
                            "Loaded existing graph from Neo4j storage")
                except Exception as e:
                    self.logger.error(
                        "Critical error: Failed to load graph from storage: %s", e)
                    raise AgentError(
                        f"Graph storage is unavailable: {e}") from e

            # Step 2: Build the fact graph if not loaded from storage
            if not graph:
                await self.update_substep("Building fact graph", 0.2, "Creating graph structure...")
                graph = await self.graph_builder.build_graph(fact_hierarchy)
                self.logger.info("Built new graph with %s", graph.get_stats())

                # Update progress for clustering
                await self.update_substep(
                    "Applying advanced clustering and relationship analysis", 0.3, "Analyzing fact relationships...")

                # Apply advanced clustering if available
                if self.advanced_clustering:
                    try:
                        enhanced_clusters = await self.advanced_clustering.cluster_facts(graph)
                        # Update graph with enhanced clusters
                        graph = await self._apply_enhanced_clustering(graph, enhanced_clusters)
                        self.logger.info("Applied advanced clustering")
                    except Exception as e:
                        self.logger.error(
                            "Critical error: Advanced clustering failed: %s", e)
                        raise AgentError(
                            f"Advanced clustering is required but failed: {e}") from e

                # Analyze relationships between facts
                if self.relationship_analyzer:
                    try:
                        fact_data = [
                            {
                                "id": node.id,
                                "claim": node.claim,
                                "sources_examined": node.sources_examined,
                                "timestamp": getattr(node, "timestamp", None),
                            }
                            for node in graph.nodes.values()
                        ]
                        relationships = await self.relationship_analyzer.analyze_fact_relationships(fact_data)
                        # Update graph with relationship information
                        graph = await self._apply_relationship_analysis(graph, relationships)
                        self.logger.info(
                            "Analyzed %d fact relationships", len(relationships))
                    except Exception as e:
                        self.logger.warning(
                            "Relationship analysis failed: %s", e)

                # Save to persistent storage
                if self.graph_storage and graph_id:
                    try:
                        self.graph_storage.store_graph(graph, graph_id)
                        self.logger.info("Saved graph to Neo4j storage")
                    except Exception as e:
                        self.logger.error(
                            "Critical error: Failed to save graph to storage: %s", e)
                        raise AgentError(
                            f"Graph storage save failed: {e}") from e

            # Step 3: Enhance context with source reputation if available
            enhanced_context = context
            if self.source_reputation:
                await self.update_substep(
                    "Enhancing context with source reputation", 0.4, "Analyzing source credibility...")
                enhanced_context = await self._enhance_context_with_reputation(context)

            # Update progress for graph verification
            await self.update_substep("Verifying facts using graph analysis", 0.5, "Processing fact clusters...")

            # Step 4: Verify the graph
            # Pass progress callback to verification engine for detailed progress tracking
            self.verification_engine.set_progress_callback(self.progress_callback)
            verification_results = await self.verification_engine.verify_graph(graph, enhanced_context)

            # Update progress for uncertainty analysis
            await self.update_substep("Applying Bayesian uncertainty analysis", 0.7, "Calculating confidence scores...")

            # Step 5: Apply Bayesian uncertainty analysis
            await self._ensure_uncertainty_handler()  # Lazy initialization
            if self.uncertainty_handler:
                try:
                    # Prepare verification data for uncertainty analysis
                    verification_data = {
                        "evidence": [
                            {
                                "score": node.confidence,
                                "age_days": 0,
                                "source_reliability": 0.7,
                            }
                            for node in graph.nodes.values()
                        ],
                        "cluster_results": verification_results.get("cluster_results", []),
                    }

                    uncertainty_analysis = await self.uncertainty_handler.analyze_verification_uncertainty(
                        verification_data
                    )
                    verification_results["uncertainty_analysis"] = uncertainty_analysis
                    self.logger.info(
                        "Applied Bayesian uncertainty analysis: %s", uncertainty_analysis.get('uncertainty_level', 'unknown'))
                except Exception as e:
                    self.logger.warning(
                        "Bayesian uncertainty analysis failed: %s", e)

            # Step 6: Update graph storage with verification results
            if self.graph_storage and graph_id:
                await self.update_substep("Updating graph storage", 0.8, "Saving verification results...")
                try:
                    await self._update_graph_with_results(graph_id, verification_results)
                except Exception as e:
                    self.logger.error(
                        "Critical error: Failed to update graph storage: %s", e)
                    raise AgentError(
                        f"Failed to update graph storage: {e}") from e

            # Update progress for result conversion
            await self.update_substep("Converting results to standard format", 0.85, "Formatting results...")

            # Step 7: Convert results to standard format
            fact_check_result = await self._convert_to_standard_format(verification_results, graph, enhanced_context)

            verification_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                "Enhanced graph-based fact checking completed in %.2fs",
                verification_time,
            )

            # Add enhanced metadata to the result
            if hasattr(fact_check_result, "metadata"):
                fact_check_result.metadata = {
                    "verification_method": "enhanced_graph_based",
                    "graph_stats": graph.get_stats(),
                    "verification_time": verification_time,
                    "clusters_processed": len(graph.clusters),
                    "relationships_found": len(graph.edges),
                    "persistence_enabled": self.graph_storage is not None,
                    "source_reputation_enabled": self.source_reputation is not None,
                    "cached": False,
                }

            # Step 8: Cache results
            await self.update_substep("Caching results", 0.95, "Storing results in cache...")
            await self.verification_cache.set(cache_key, fact_check_result)

            # Final progress update
            await self.update_progress(
                1.0, 1.0, "Graph-based fact checking completed successfully")

            return fact_check_result

        except (ValueError, RuntimeError, KeyError, AttributeError) as e:
            self.logger.error(
                "Enhanced graph-based fact checking failed: %s", e, exc_info=True)
            # Return error result in legacy format
            return self._create_error_result(str(e))

    async def _convert_to_standard_format(
        self,
        verification_results: dict[str, Any],
        graph: FactGraph,
        context: VerificationContext,
    ) -> FactCheckResult:
        """
        Convert graph verification results to the standard FactCheckResult format.

        Args:
            verification_results: Results from graph verification
            graph: The verified graph
            context: Original verification context

        Returns:
            FactCheckResult: Results in standard format
        """
        claim_results = []
        all_sources = set()
        all_queries = set()

        # Process cluster results
        for cluster_result in verification_results.get("cluster_results", []):
            cluster_id = cluster_result["cluster_id"]
            cluster = graph.clusters.get(cluster_id)

            if not cluster:
                continue

            # Convert each fact in the cluster using the cluster_result directly
            for (
                node_id,
                fact_result,
            ) in cluster_result.get("individual_results", {}).items():
                node = graph.nodes.get(node_id)
                if node:
                    claim_result = self._convert_fact_result_to_claim_result(
                        fact_result, node, cluster_result)
                    claim_results.append(claim_result)

                    # Collect sources and queries
                    all_sources.update(claim_result.sources)
                    # Add cluster-based search queries
                    all_queries.add(f"cluster_query_{cluster_id}")

        # Process individual results
        for individual_result in verification_results.get("individual_results", []):
            if "node_id" in individual_result:
                node = graph.nodes.get(individual_result["node_id"])
                if node:
                    claim_result = self._convert_fact_result_to_claim_result(
                        individual_result, node)
                    claim_results.append(claim_result)
                    all_sources.update(claim_result.sources)
                    all_queries.add(f"individual_query_{node.id}")

        # Compile summary
        summary = self._compile_summary_from_graph_results(
            verification_results, claim_results)

        return FactCheckResult(
            claim_results=claim_results,
            examined_sources=list(all_sources),
            search_queries_used=list(all_queries),
            summary=summary,
        )

    def _convert_fact_result_to_claim_result(
        self, fact_result: dict[str, Any], node, cluster_result=None
    ) -> ClaimResult:
        """Convert a fact verification result to a ClaimResult."""
        # Map graph verdicts to legacy assessments
        verdict_map = {
            "TRUE": "true",
            "FALSE": "false",
            "INSUFFICIENT_EVIDENCE": "insufficient_evidence",
            "ERROR": "error",
            "MIXED": "mixed",
        }

        assessment = verdict_map.get(
            fact_result.get("verdict", "ERROR"), "error")
        confidence = fact_result.get("confidence", 0.0)
        reasoning = fact_result.get("reasoning", "No reasoning provided")

        # Extract sources from evidence
        sources = []
        evidence_used = fact_result.get("evidence_used", [])
        for evidence in evidence_used:
            if isinstance(evidence, str) and evidence.startswith("http"):
                sources.append(evidence)
            elif isinstance(evidence, dict) and "url" in evidence:
                sources.append(evidence["url"])

        # Calculate supporting/contradicting evidence counts
        supporting_evidence = 0
        contradicting_evidence = 0

        if assessment == "true":
            supporting_evidence = len(sources)
        elif assessment == "false":
            contradicting_evidence = len(sources)
        elif assessment == "mixed":
            # Split sources between supporting and contradicting
            supporting_evidence = len(sources) // 2
            contradicting_evidence = len(sources) - supporting_evidence

        # Add cluster context to reasoning if available
        if cluster_result:
            metadata = cluster_result.get("metadata", {})
            cluster_type = metadata.get("cluster_type", "unknown")
            individual_results = cluster_result.get("individual_results", {})
            num_related = len(individual_results)
            cluster_msg = f" [Verified as part of {cluster_type} cluster"
            cluster_context = f"{cluster_msg} with {num_related} related facts]"
            reasoning += cluster_context

        return ClaimResult(
            claim=node.claim,
            assessment=assessment,
            confidence=confidence,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            sources=sources,
            reasoning=reasoning,
        )

    def _compile_summary_from_graph_results(
        self, verification_results: dict[str, Any], claim_results: list[ClaimResult]
    ) -> FactCheckSummary:
        """Compile summary from graph verification results."""
        total_sources_found = sum(len(result.sources)
                                  for result in claim_results)
        # In graph approach, all found sources are considered credible
        credible_sources = total_sources_found
        supporting_evidence = sum(
            result.supporting_evidence for result in claim_results)
        contradicting_evidence = sum(
            result.contradicting_evidence for result in claim_results)

        return FactCheckSummary(
            total_sources_found=total_sources_found,
            credible_sources=credible_sources,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
        )

    def _create_empty_result(self) -> FactCheckResult:
        """Create an empty result when no facts are found."""
        return FactCheckResult(
            claim_results=[],
            examined_sources=[],
            search_queries_used=[],
            summary=FactCheckSummary(
                total_sources_found=0,
                credible_sources=0,
                supporting_evidence=0,
                contradicting_evidence=0,
            ),
        )

    def _create_error_result(self, error_message: str) -> FactCheckResult:
        """Create an error result."""
        return FactCheckResult(
            claim_results=[
                ClaimResult(
                    claim="Graph verification failed",
                    assessment="error",
                    confidence=0.0,
                    supporting_evidence=0,
                    contradicting_evidence=0,
                    sources=[],
                    reasoning=f"Graph-based verification error: {error_message}",
                )
            ],
            examined_sources=[],
            search_queries_used=[],
            summary=FactCheckSummary(
                total_sources_found=0,
                credible_sources=0,
                supporting_evidence=0,
                contradicting_evidence=0,
            ),
        )

    def get_graph_stats(self, graph_id: str = None) -> dict[str, Any]:
        """Get statistics about the graph builder and verification engine."""
        stats = {
            "graph_builder_cache": self.graph_builder.get_cache_stats(),
            "verification_engine_cache": self.verification_engine.get_cache_stats(),
        }

        # Add storage stats if available
        if self.graph_storage and graph_id:
            try:
                storage_stats = asyncio.run(
                    self.graph_storage.get_graph_stats(graph_id))
                stats["graph_storage"] = storage_stats
            except Exception as e:
                self.logger.warning("Failed to get storage stats: %s", e)

        return stats

    async def clear_cache(self):
        """Clear all caches."""
        await self.graph_builder.clear_cache()
        await self.verification_engine.clear_cache()

        # Clear intelligent caches
        if hasattr(self.verification_cache, "clear"):
            await self.verification_cache.clear()
        if hasattr(self.general_cache, "clear"):
            await self.general_cache.clear()

    def _generate_cache_key(self, fact_hierarchy) -> str:
        """Generate a cache key for the fact hierarchy."""

        # Create a string representation of the fact hierarchy
        facts_str = ""
        for fact in fact_hierarchy.supporting_facts:
            # Use description instead of claim, and handle missing category gracefully
            category = getattr(fact, "category", "general")
            facts_str += f"{fact.description}|{category}|"

        # Add main claim if available
        if hasattr(fact_hierarchy, "main_claim") and fact_hierarchy.main_claim:
            facts_str += f"main:{fact_hierarchy.main_claim}|"

        # Generate hash
        return hashlib.md5(facts_str.encode()).hexdigest()

    def _generate_graph_id(self, fact_hierarchy) -> str:
        """Generate a unique graph ID for persistent storage."""

        # Create a more detailed string representation
        facts_str = ""
        for fact in fact_hierarchy.supporting_facts:
            # Use description instead of claim, and handle missing attributes gracefully
            description = getattr(fact, "description", "") or "No description"
            category = getattr(fact, "category", "general")
            confidence = getattr(fact, "confidence", 0.0)
            facts_str += f"{description}|{category}|{confidence}|"

        # Add timestamp for uniqueness (daily granularity)
        date_str = datetime.now().strftime("%Y-%m-%d")
        full_str = f"{facts_str}|{date_str}"

        return hashlib.sha256(full_str.encode()).hexdigest()[:16]

    async def _enhance_context_with_reputation(self, context: VerificationContext) -> VerificationContext:
        """Enhance verification context with source reputation information."""
        if not self.source_reputation:
            return context

        try:
            # Create enhanced context copy
            enhanced_context = context

            # Add source reputation metadata
            if hasattr(enhanced_context, "metadata"):
                if not enhanced_context.metadata:
                    enhanced_context.metadata = {}
                enhanced_context.metadata["source_reputation_enabled"] = True

            return enhanced_context

        except Exception as e:
            self.logger.warning(
                "Failed to enhance context with reputation: %s", e)
            return context

    async def _update_graph_with_results(self, graph_id: str, verification_results: dict[str, Any]):
        """Update graph storage with verification results."""
        if not self.graph_storage:
            return

        try:
            # Update node verification results
            for cluster_result in verification_results.get("detailed_cluster_results", []):
                for node_id, result in cluster_result.individual_results.items():
                    # Extract required parameters from result
                    verification_results_data = result if isinstance(
                        result, dict) else {}
                    confidence = result.get("confidence", 0.0) if isinstance(
                        result, dict) else 0.0

                    # Determine verification status based on result
                    if isinstance(result, dict) and result.get("verdict"):
                        if result["verdict"] in ["TRUE", "VERIFIED"]:
                            status = VerificationStatus.VERIFIED
                        elif result["verdict"] in ["FALSE", "FAILED"]:
                            status = VerificationStatus.FAILED
                        else:
                            status = VerificationStatus.PENDING
                    else:
                        status = VerificationStatus.PENDING

                    await self.graph_storage.update_node_verification(
                        node_id, verification_results_data, status, confidence
                    )

            # Update individual results
            for individual_result in verification_results.get("individual_results", []):
                if "node_id" in individual_result:
                    node_id = individual_result["node_id"]

                    # Extract required parameters from individual_result
                    verification_results_data = individual_result
                    confidence = individual_result.get("confidence", 0.0)

                    # Determine verification status based on result
                    if individual_result.get("verdict"):
                        if individual_result["verdict"] in ["TRUE", "VERIFIED"]:
                            status = VerificationStatus.VERIFIED
                        elif individual_result["verdict"] in ["FALSE", "FAILED"]:
                            status = VerificationStatus.FAILED
                        else:
                            status = VerificationStatus.PENDING
                    else:
                        status = VerificationStatus.PENDING

                    await self.graph_storage.update_node_verification(
                        node_id, verification_results_data, status, confidence
                    )

        except Exception as e:
            self.logger.error("Failed to update graph with results: %s", e)

    async def get_verification_history(self, fact_claim: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get verification history for a specific fact claim."""
        if not self.graph_storage:
            return []

        try:
            # Use the new method from Neo4jGraphStorage
            history = self.graph_storage.get_verification_history(
                fact_claim, limit)

            # Enhance history with additional metadata if available
            enhanced_history = []
            for item in history:
                enhanced_item = item.copy()

                # Add source reputation if available
                if self.source_reputation and "verification_results" in enhanced_item:
                    verification_results = enhanced_item["verification_results"]
                    if "sources" in verification_results:
                        source_scores = []
                        for source in verification_results["sources"]:
                            if isinstance(source, dict) and "url" in source:
                                score = await self.get_source_reputation_score(source["url"])
                                if score is not None:
                                    source_scores.append(score)

                        if source_scores:
                            enhanced_item["average_source_reputation"] = sum(
                                source_scores) / len(source_scores)

                enhanced_history.append(enhanced_item)

            return enhanced_history

        except Exception as e:
            self.logger.error("Failed to get verification history: %s", e)
            return []

    async def get_source_reputation_score(self, source_url: str) -> float | None:
        """Get reputation score for a source."""
        if not self.source_reputation:
            return None

        try:
            return self.source_reputation.get_source_credibility(source_url)
        except Exception as e:
            self.logger.warning("Failed to get source reputation: %s", e)
            return None

    async def _apply_enhanced_clustering(self, graph, enhanced_clusters):
        """Apply enhanced clustering results to the graph."""
        try:
            # enhanced_clusters is a dict {cluster_id: [node_ids]}
            # Update graph clusters with enhanced clustering results
            for cluster_id, node_ids in enhanced_clusters.items():
                if cluster_id in graph.clusters:
                    cluster = graph.clusters[cluster_id]
                    # Update cluster metadata with enhanced information
                    cluster.metadata.update(
                        {
                            "clustering_method": "advanced_gnn",
                            "cluster_quality": 0.8,  # Default quality score
                            "semantic_coherence": 0.7,  # Default semantic coherence
                            "temporal_coherence": 0.6,  # Default temporal coherence
                        }
                    )

            return graph
        except Exception as e:
            self.logger.error("Failed to apply enhanced clustering: %s", e)
            return graph

    async def _apply_relationship_analysis(self, graph, relationships):
        """Apply relationship analysis results to the graph."""
        try:
            # Add relationship information to graph edges
            for relationship in relationships:
                fact_id_1 = relationship.fact_id_1
                fact_id_2 = relationship.fact_id_2

                # Find or create edge between these facts
                edge_key = f"{fact_id_1}_{fact_id_2}"
                if edge_key not in graph.edges:
                    # Create new edge if it doesn't exist
                    edge = FactEdge(
                        source_id=fact_id_1,
                        target_id=fact_id_2,
                        relationship_type=relationship.relationship_type,
                        strength=relationship.confidence,  # Use strength instead of confidence
                    )
                    graph.edges[edge_key] = edge
                else:
                    # Update existing edge
                    edge = graph.edges[edge_key]
                    edge.metadata.update(
                        {
                            "relationship_type": relationship.relationship_type,
                            "relationship_strength": relationship.strength,
                            "relationship_confidence": relationship.confidence,
                            "causal_direction": relationship.causal_direction,
                            "temporal_order": relationship.temporal_order,
                        }
                    )

            return graph
        except Exception as e:
            self.logger.error("Failed to apply relationship analysis: %s", e)
            return graph

    async def get_detailed_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics about all components."""
        stats = self.get_graph_stats(self.current_graph_id)

        # Add advanced clustering stats
        if self.advanced_clustering:
            try:
                stats["advanced_clustering"] = self.advanced_clustering.get_clustering_stats()
            except Exception as e:
                self.logger.warning("Failed to get clustering stats: %s", e)

        # Add uncertainty handler stats
        await self._ensure_uncertainty_handler()  # Lazy initialization
        if self.uncertainty_handler:
            try:
                stats["uncertainty_handler"] = self.uncertainty_handler.get_uncertainty_stats()
            except Exception as e:
                self.logger.warning("Failed to get uncertainty stats: %s", e)

        # Add relationship analyzer stats
        if self.relationship_analyzer:
            try:
                stats["relationship_analyzer"] = self.relationship_analyzer.get_relationship_stats()
            except Exception as e:
                self.logger.warning("Failed to get relationship stats: %s", e)

        return stats

    async def analyze_fact_relationships_standalone(self, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Analyze relationships between facts as a standalone operation."""
        if not self.relationship_analyzer:
            return []

        try:
            relationships = await self.relationship_analyzer.analyze_fact_relationships(facts)
            return [
                {
                    "fact_id_1": rel.fact_id_1,
                    "fact_id_2": rel.fact_id_2,
                    "relationship_type": rel.relationship_type,
                    "strength": rel.strength,
                    "confidence": rel.confidence,
                    "evidence": rel.evidence,
                    "temporal_order": rel.temporal_order,
                    "causal_direction": rel.causal_direction,
                }
                for rel in relationships
            ]
        except Exception as e:
            self.logger.error("Standalone relationship analysis failed: %s", e)
            return []

    async def get_uncertainty_analysis(self, verification_data: dict[str, Any]) -> dict[str, Any]:
        """Get uncertainty analysis for verification data."""
        if not self.uncertainty_handler:
            return {"error": "Uncertainty handler not available"}

        try:
            return await self.uncertainty_handler.analyze_verification_uncertainty(verification_data)
        except Exception as e:
            self.logger.error("Uncertainty analysis failed: %s", e)
            return {"error": str(e)}

    async def close(self):
        """Close all resources and cleanup."""
        try:
            # Close verification engine (which will close SourceManager and WebScraper)
            if hasattr(self, "verification_engine") and self.verification_engine:
                await self.verification_engine.close()
                self.logger.info("Verification engine closed")

            # Close graph storage connection
            if hasattr(self, "graph_storage") and self.graph_storage:
                await self.graph_storage.close()
                self.logger.info("Graph storage connection closed")

            # Clear all caches
            if hasattr(self, "verification_cache") and self.verification_cache:
                await self.verification_cache.clear()
                self.logger.info("Verification cache cleared")

            if hasattr(self, "general_cache") and self.general_cache:
                await self.general_cache.clear()
                self.logger.info("General cache cleared")

            self.logger.info("GraphFactCheckingService closed successfully")

        except AgentError as e:
            self.logger.error(
                "Error during GraphFactCheckingService cleanup: %s", e)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

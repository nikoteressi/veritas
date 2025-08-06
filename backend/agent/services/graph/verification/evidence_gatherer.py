"""
Evidence gatherer for fact verification.

Handles search query generation and evidence collection for fact clusters.
Enhanced with intelligent caching and adaptive relevance scoring.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.output_parsers import PydanticOutputParser

from agent.llm.manager import llm_manager
from agent.models.fact_checking_models import QueryGenerationOutput
from agent.models.graph import ClusterType, FactCluster
from agent.prompts.manager import PromptManager
from agent.services.analysis.adaptive_thresholds import get_adaptive_thresholds
from app.cache.factory import get_verification_cache

from .response_parser import ResponseParser
from .source_manager import EnhancedSourceManager

if TYPE_CHECKING:
    from ....tools import SearxNGSearchTool

logger = logging.getLogger(__name__)


class EnhancedEvidenceGatherer:
    """Enhanced evidence gatherer with intelligent caching and adaptive relevance scoring."""

    def __init__(self, search_tool: SearxNGSearchTool, config=None):
        self._search_tool = search_tool
        self.config = config
        self.prompt_manager = PromptManager()
        self.response_parser = ResponseParser()

        # Use unified cache system
        self.cache = None
        self._cache_initialized = False

        # Initialize enhanced source manager
        max_concurrent_scrapes = 3  # Default value
        if self.config and hasattr(self.config, "max_concurrent_scrapes"):
            max_concurrent_scrapes = self.config.max_concurrent_scrapes
        self.source_manager = EnhancedSourceManager(
            max_concurrent_scrapes=max_concurrent_scrapes)

        # Get adaptive thresholds instance
        self.adaptive_thresholds = get_adaptive_thresholds()

        # Performance tracking
        self.performance_metrics = {
            "queries_executed": 0,
            "cache_hits": 0,
            "sources_evaluated": 0,
            "relevance_scores": [],
        }

    async def _ensure_cache_initialized(self):
        """Ensure the cache is initialized."""
        if not self._cache_initialized:
            self.cache = await get_verification_cache()
            self._cache_initialized = True
            logger.info("Evidence gatherer cache initialized")

    async def create_cluster_search_queries(self, cluster: FactCluster) -> list[str]:
        """Create optimized search queries for a cluster using LLM."""
        # Prepare cluster information for LLM
        # Limit to first 3 claims
        claims = [node.claim for node in cluster.nodes[:3]]
        primary_claim = claims[0] if claims else ""

        # Create context based on cluster type
        cluster_context = self._build_cluster_context(cluster)

        # Use LLM to generate search queries
        parser = PydanticOutputParser(pydantic_object=QueryGenerationOutput)

        # Get domain-specific role description
        role_description = self.prompt_manager.get_domain_role_description(
            "general")

        prompt_template = self.prompt_manager.get_prompt_template(
            "query_generation")
        prompt = await prompt_template.aformat(
            role_description=role_description,
            claim=primary_claim,
            primary_thesis=cluster_context,
            temporal_context="No specific temporal context provided.",
            format_instructions=parser.get_format_instructions(),
        )

        logger.info("QUERY GENERATION PROMPT: \n %s", prompt)

        response = await llm_manager.invoke_text_only(prompt)
        parsed_response = parser.parse(response)
        logger.info("QUERY GENERATION PARSED RESULT: \n %s", parsed_response)

        # Extract queries
        queries = [query.query for query in parsed_response.queries]
        logger.info(
            "Generated %d LLM queries for cluster %s: %s", len(queries), cluster.id, queries)
        return queries

    def _build_cluster_context(self, cluster: FactCluster) -> str:
        """Build context string based on cluster type and metadata."""
        if cluster.cluster_type == ClusterType.DOMAIN_CLUSTER:
            domain = cluster.metadata.get("domain", "")
            return f"Domain: {domain}"
        elif cluster.cluster_type == ClusterType.TEMPORAL_CLUSTER:
            return "Temporal context: chronological events"
        elif cluster.cluster_type == ClusterType.CAUSAL_CLUSTER:
            return "Causal context: cause-effect relationships"
        else:
            # For similarity clusters, use cleaned shared context
            shared_context = cluster.shared_context or ""
            if shared_context.startswith("Combined:"):
                shared_context = shared_context.replace(
                    "Combined:", "").strip()
            if shared_context.startswith("Common themes:") or shared_context.startswith("Themes:"):
                shared_context = shared_context.replace(
                    "Common themes:", "").replace("Themes:", "").strip()
            return f"Context: {shared_context}" if shared_context else "General context"

    async def execute_searches_batch(
        self, queries: list[str], query_context: str | None = None
    ) -> list[dict[str, Any]]:
        """Execute multiple search queries in parallel with intelligent caching."""
        if not queries:
            return []

        await self._ensure_cache_initialized()

        logger.info(
            "Executing %d search queries with context: %s", len(queries), query_context)

        all_results = []
        cache_hits = 0

        for query in queries:
            try:
                # Generate cache key
                cache_key = f"search_result:{hash(query)}"

                # Check intelligent cache first
                cached_result = await self.cache.get(cache_key)
                if cached_result:
                    logger.debug("Cache hit for query: %s", query)
                    all_results.append(cached_result)
                    cache_hits += 1
                    self.performance_metrics["cache_hits"] += 1
                    continue

                # Execute search if not in cache
                logger.debug("Executing search for query: %s", query)

                # Get max_results from config, default to 10
                max_results = 10
                if self.config and hasattr(self.config, "max_search_results"):
                    max_results = self.config.max_search_results

                search_result = await self._search_tool._arun(query=query, max_results=max_results)

                result_data = {
                    "query": query,
                    "results": search_result,
                    "context": query_context,
                    "timestamp": "now",
                }

                # Store in intelligent cache
                cache_metadata = {
                    "query": query,
                    "context": query_context,
                    "result_length": len(str(search_result)),
                }

                await self.cache.set(
                    cache_key,
                    result_data,
                    ttl=1800,  # 30 minutes TTL
                )

                all_results.append(result_data)
                self.performance_metrics["queries_executed"] += 1

            except Exception as e:
                logger.error(
                    "Search query '%s' failed: %s", query, e)
                error_result = {
                    "query": query,
                    "results": f"Search failed: {e}",
                    "error": True,
                }
                all_results.append(error_result)

        cache_hit_rate = cache_hits / len(queries) if queries else 0
        logger.info(
            f"Search batch completed. Cache hit rate: {cache_hit_rate:.2%}")

        return all_results

    async def select_credible_sources_batch(
        self,
        combined_claim: str,
        search_results_data: list[dict[str, Any]],
        source_type: str = "web",
        query_type: str = "factual",
    ) -> list[str]:
        """Select credible sources with enhanced relevance scoring and adaptive thresholds."""
        if not search_results_data:
            return []

        # Extract all sources from search results
        all_sources = set()
        for result_data in search_results_data:
            sources = self.response_parser.extract_sources_from_result(
                result_data.get("results", ""))
            all_sources.update(sources)

        all_sources_list = list(all_sources)

        if not all_sources_list:
            logger.warning("No sources found in search results")
            return []

        # Limit sources for processing
        sources_to_evaluate = all_sources_list[:20]
        self.performance_metrics["sources_evaluated"] += len(
            sources_to_evaluate)

        # Get adaptive threshold for source selection
        threshold = await self.adaptive_thresholds.get_adaptive_threshold(
            query_type=query_type,
            source_type=source_type,
            context={"claim_length": len(combined_claim)},
        )

        # Scrape and evaluate sources with relevance scoring
        logger.info(
            f"Scraping and evaluating {len(sources_to_evaluate)} sources")

        source_contents = await self.source_manager.scrape_sources_batch(
            sources_to_evaluate, query_context=combined_claim
        )

        # Calculate relevance scores for each source
        scored_sources = []
        for url, scraped_info in source_contents.items():
            content = scraped_info.get("content", "")
            publication_date = scraped_info.get("publication_date")

            if content and not content.startswith("Failed to scrape"):
                relevance_score = await self.source_manager.calculate_relevance(
                    content, combined_claim, source_type, query_type
                )

                scored_sources.append({
                    "url": url,
                    "content": content,
                    "relevance_score": relevance_score,
                    "publication_date": publication_date
                })

                self.performance_metrics["relevance_scores"].append(
                    relevance_score)

        # Filter sources by adaptive threshold
        credible_sources = [
            source for source in scored_sources if source["relevance_score"] >= threshold]

        # Sort by relevance score and limit to top sources
        credible_sources.sort(key=lambda x: x["relevance_score"], reverse=True)
        top_sources = credible_sources[:10]  # Limit to top 10

        credible_urls = [source["url"] for source in top_sources]

        # Record performance metrics
        avg_relevance = sum(s["relevance_score"]
                            for s in top_sources) / len(top_sources) if top_sources else 0

        # Calculate performance metrics for adaptive thresholds
        precision = len(credible_urls) / \
            len(scored_sources) if scored_sources else 0.0
        # Assume ideal is 10 sources
        recall = min(1.0, len(credible_urls) / 10)
        f1_score = 2 * (precision * recall) / (precision +
                                               recall) if (precision + recall) > 0 else 0.0
        source_retention_rate = len(
            credible_urls) / len(sources_to_evaluate) if sources_to_evaluate else 0.0

        await self.adaptive_thresholds.record_performance_metrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            source_retention_rate=source_retention_rate,
            query_type=query_type,
            source_type=source_type,
        )

        logger.info(
            "Selected {} credible sources from {} (threshold: {:.3f}, avg_relevance: {:.3f})".format(
                len(credible_urls), len(
                    sources_to_evaluate), threshold, avg_relevance
            )
        )

        return credible_urls

    async def clear_search_cache(self):
        """Clear component-specific search cache (shared cache managed by factory)."""
        # Shared cache is managed by CacheFactory and should not be cleared here
        # This method is kept for interface compatibility
        logger.info("Component-specific search cache cleared (shared cache managed by factory)")

    async def gather_cluster_evidence(
        self,
        cluster: FactCluster,
        context,
        source_type: str = "web",
        query_type: str = "factual",
    ) -> list[dict[str, Any]]:
        """
        Gather evidence for a cluster with enhanced relevance scoring and adaptive thresholds.

        Args:
            cluster: The fact cluster to gather evidence for
            context: Verification context
            source_type: Type of sources to prioritize (web, academic, news, etc.)
            query_type: Type of query (factual, opinion, analysis, etc.)

        Returns:
            List of evidence dictionaries containing search results and sources
        """
        try:
            # Step 1: Generate search queries for the cluster
            search_queries = await self.create_cluster_search_queries(cluster)

            if not search_queries:
                logger.warning(
                    f"No search queries generated for cluster {cluster.id}")
                return []

            # Step 2: Execute searches with context
            combined_claims = "; ".join(
                [node.claim for node in cluster.nodes[:3]])
            search_results = await self.execute_searches_batch(search_queries, query_context=combined_claims)

            # Step 3: Select credible sources with enhanced scoring
            credible_sources = await self.select_credible_sources_batch(
                combined_claims, search_results, source_type, query_type
            )

            # Step 4: Compile enhanced evidence
            evidence = []
            for result_data in search_results:
                # Calculate query-specific relevance if not an error
                query_relevance = 0.0
                if not result_data.get("error", False):
                    query_text = result_data.get("query", "")
                    search_content = str(result_data.get("results", ""))
                    if query_text and search_content:
                        query_relevance = await self.source_manager.calculate_relevance(
                            search_content, query_text, source_type, query_type
                        )

                evidence_item = {
                    "query": result_data.get("query", ""),
                    "search_results": result_data.get("results", ""),
                    "credible_sources": credible_sources,
                    "cluster_id": cluster.id,
                    "query_relevance": query_relevance,
                    "source_type": source_type,
                    "query_type": query_type,
                    "context": result_data.get("context", ""),
                    "timestamp": result_data.get("timestamp", ""),
                }
                evidence.append(evidence_item)

            # Log comprehensive metrics
            avg_query_relevance = sum(
                e["query_relevance"] for e in evidence) / len(evidence) if evidence else 0
            logger.info(
                "Gathered evidence for cluster {}: {} queries, {} credible sources, avg_query_relevance: {:.3f}".format(
                    cluster.id, len(search_queries), len(
                        credible_sources), avg_query_relevance
                )
            )

            return evidence

        except Exception as e:
            logger.error(
                f"Failed to gather evidence for cluster {cluster.id}: {e}")
            return []

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache and performance statistics."""
        await self._ensure_cache_initialized()
        cache_stats = await self.cache.get_stats()

        # Calculate performance metrics
        avg_relevance = (
            (sum(self.performance_metrics["relevance_scores"]) /
             len(self.performance_metrics["relevance_scores"]))
            if self.performance_metrics["relevance_scores"]
            else 0.0
        )

        return {
            "cache_stats": cache_stats,
            "performance_metrics": {
                "queries_executed": self.performance_metrics["queries_executed"],
                "cache_hits": self.performance_metrics["cache_hits"],
                "sources_evaluated": self.performance_metrics["sources_evaluated"],
                "average_relevance_score": avg_relevance,
                "total_relevance_calculations": len(self.performance_metrics["relevance_scores"]),
            },
            "cache_hit_rate": (
                self.performance_metrics["cache_hits"]
                / max(
                    1,
                    self.performance_metrics["queries_executed"] +
                    self.performance_metrics["cache_hits"],
                )
            ),
        }

    async def optimize_performance(self):
        """Optimize cache and performance based on usage patterns."""
        # Optimize intelligent cache
        if self.cache:
            await self.cache.optimize()

        # Optimize source manager cache
        await self.source_manager.optimize_cache()

        # Get optimization recommendations from adaptive thresholds
        recommendations = await self.adaptive_thresholds.get_threshold_recommendations()

        logger.info(
            f"Performance optimization completed. Recommendations: {recommendations}")
        return recommendations

    async def cleanup(self):
        """Clean up resources (cache is managed by factory)."""
        # Cache is managed by CacheFactory, no need to close here

        if self.source_manager:
            await self.source_manager.close()
            logger.info("EvidenceGatherer: SourceManager closed")

        logger.info("EvidenceGatherer: Resources cleaned up")

    async def clear_cache(self):
        """Clear component-specific cache (shared cache managed by factory)."""
        # Shared cache is managed by CacheFactory and should not be cleared here
        # This method is kept for interface compatibility
        logger.info("Component-specific evidence cache cleared (shared cache managed by factory)")


# Backward compatibility alias
EvidenceGatherer = EnhancedEvidenceGatherer

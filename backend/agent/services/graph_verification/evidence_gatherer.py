"""
Evidence gatherer for fact verification.

Handles search query generation and evidence collection for fact clusters.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from langchain.output_parsers import PydanticOutputParser

from agent.llm import llm_manager
from agent.models.fact_checking_models import QueryGenerationOutput
from agent.models.graph import ClusterType, FactCluster
from agent.prompt_manager import PromptManager

from .response_parser import ResponseParser

if TYPE_CHECKING:
    from agent.tools import SearxNGSearchTool

logger = logging.getLogger(__name__)


class EvidenceGatherer:
    """Gathers evidence for fact verification through search and analysis."""

    def __init__(self, search_tool: "SearxNGSearchTool"):
        self._search_tool = search_tool
        self.prompt_manager = PromptManager()
        self.response_parser = ResponseParser()
        self._search_cache: dict[str, list[dict[str, Any]]] = {}

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
        role_description = self.prompt_manager.get_domain_role_description("general")

        prompt_template = self.prompt_manager.get_prompt_template("query_generation")
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
            f"Generated {len(queries)} LLM queries for cluster {cluster.id}: {queries}"
        )
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
                shared_context = shared_context.replace("Combined:", "").strip()
            if shared_context.startswith("Common themes:") or shared_context.startswith(
                "Themes:"
            ):
                shared_context = (
                    shared_context.replace("Common themes:", "")
                    .replace("Themes:", "")
                    .strip()
                )
            return f"Context: {shared_context}" if shared_context else "General context"

    async def execute_searches_batch(self, queries: list[str]) -> list[dict[str, Any]]:
        """Execute multiple search queries in parallel."""
        if not queries:
            return []

        # Filter out cached queries
        queries_to_execute = [q for q in queries if q not in self._search_cache]

        if not queries_to_execute:
            logger.info("All queries already cached")
            return [self._search_cache[q] for q in queries if q in self._search_cache]

        logger.info(f"Executing {len(queries_to_execute)} search queries")

        # Execute searches in parallel
        search_tasks = [
            self._search_tool._arun(query=query) for query in queries_to_execute
        ]

        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Process results and update cache
        all_results = []

        for query, result in zip(queries_to_execute, search_results, strict=False):
            if isinstance(result, Exception):
                logger.error(f"Search query '{query}' failed: {result}")
                result_data = {"query": query, "results": f"Search failed: {result}"}
            else:
                result_data = {"query": query, "results": result}
                self._search_cache[query] = result_data

            all_results.append(result_data)

        # Add cached results for all requested queries
        for query in queries:
            if query in self._search_cache and query not in queries_to_execute:
                all_results.append(self._search_cache[query])

        return all_results

    async def select_credible_sources_batch(
        self, combined_claim: str, search_results_data: list[dict[str, Any]]
    ) -> list[str]:
        """Select credible sources from all search results using LLM."""
        if not search_results_data:
            return []

        # Extract all sources from search results
        all_sources = set()
        for result_data in search_results_data:
            sources = self.response_parser.extract_sources_from_result(
                result_data.get("results", "")
            )
            all_sources.update(sources)

        all_sources_list = list(all_sources)

        if not all_sources_list:
            logger.warning("No sources found in search results")
            return []

        # Limit sources for LLM processing
        sources_to_evaluate = all_sources_list[:20]  # Limit to top 20 sources

        # Create prompt for source selection
        sources_text = "\n".join(
            [f"{i+1}. {url}" for i, url in enumerate(sources_to_evaluate)]
        )

        format_unstructions = (
            self.response_parser.get_source_selection_format_instructions()
        )

        prompt_template = self.prompt_manager.get_prompt_template(
            "select_credible_sources"
        )
        prompt = await prompt_template.aformat(
            sources=sources_text,
            claims=combined_claim,
            max_sources=min(10, len(sources_to_evaluate)),
            format_instructions=format_unstructions,
        )

        try:
            llm_response = await llm_manager.invoke_text_only(prompt)
            selected_indices = self.response_parser.parse_source_selection_response(
                llm_response
            )

            # Return selected sources
            credible_urls = []
            for idx in selected_indices:
                if 0 <= idx < len(sources_to_evaluate):
                    credible_urls.append(sources_to_evaluate[idx])

            logger.info(
                f"Selected {len(credible_urls)} credible sources from {len(sources_to_evaluate)}"
            )
            return credible_urls

        except Exception as e:
            logger.error(f"Failed to select credible sources: {e}")
            raise RuntimeError(f"Failed to select credible sources: {e}")

    async def clear_search_cache(self):
        """Clear the search cache."""
        self._search_cache.clear()
        logger.info("Search cache cleared")

    async def gather_cluster_evidence(
        self, cluster: FactCluster, context
    ) -> list[dict[str, Any]]:
        """
        Gather evidence for a cluster by executing search queries and collecting sources.

        Args:
            cluster: The fact cluster to gather evidence for
            context: Verification context (currently unused but kept for compatibility)

        Returns:
            List of evidence dictionaries containing search results and sources
        """
        try:
            # Step 1: Generate search queries for the cluster
            search_queries = await self.create_cluster_search_queries(cluster)

            if not search_queries:
                logger.warning(f"No search queries generated for cluster {cluster.id}")
                return []

            # Step 2: Execute searches
            search_results = await self.execute_searches_batch(search_queries)

            # Step 3: Extract combined claims for source selection
            combined_claims = "; ".join([node.claim for node in cluster.nodes[:3]])

            # Step 4: Select credible sources
            credible_sources = await self.select_credible_sources_batch(
                combined_claims, search_results
            )

            # Step 5: Compile evidence
            evidence = []
            for result_data in search_results:
                evidence_item = {
                    "query": result_data.get("query", ""),
                    "search_results": result_data.get("results", ""),
                    "credible_sources": credible_sources,
                    "cluster_id": cluster.id,
                }
                evidence.append(evidence_item)

            logger.info(
                f"Gathered evidence for cluster {cluster.id}: "
                f"{len(search_queries)} queries, {len(credible_sources)} credible sources"
            )

            return evidence

        except Exception as e:
            logger.error(f"Failed to gather evidence for cluster {cluster.id}: {e}")
            return []

    def get_cache_stats(self) -> dict[str, int]:
        """Get search cache statistics."""
        return {
            "cached_queries": len(self._search_cache),
            "total_results": sum(
                len(str(data.get("results", "")))
                for data in self._search_cache.values()
            ),
        }

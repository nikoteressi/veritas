"""
Verification processor for individual fact verification.

Handles the verification of individual facts within cluster context.
Enhanced with intelligent caching and adaptive confidence scoring.
"""

import logging
from typing import Any

from agent.llm import llm_manager
from agent.models.graph import FactCluster, FactNode
from agent.models.verification_context import VerificationContext
from agent.prompt_manager import PromptManager
from ...analysis.adaptive_thresholds import get_adaptive_thresholds
from ...cache.intelligent_cache import IntelligentCache

from .response_parser import ResponseParser

logger = logging.getLogger(__name__)


class EnhancedVerificationProcessor:
    """Enhanced processor for verification of individual facts within clusters."""

    def __init__(self):
        self.prompt_manager = PromptManager()
        self.response_parser = ResponseParser()

        # Initialize intelligent cache for verification results
        self.cache = IntelligentCache(max_memory_size=1000)

        # Initialize adaptive thresholds for confidence scoring
        self.adaptive_thresholds = get_adaptive_thresholds()

        # Performance metrics
        self.performance_metrics = {
            "verifications_performed": 0,
            "cache_hits": 0,
            "confidence_scores": [],
            "verification_times": [],
        }

    async def verify_cluster_facts(
        self,
        cluster: FactCluster,
        evidence: list[dict[str, Any]],
        context: VerificationContext,
    ) -> dict[str, dict[str, Any]]:
        """Verify individual facts within the cluster context with enhanced caching and scoring."""
        results = {}

        # Create cluster context for LLM
        cluster_context = self._create_cluster_context(cluster, evidence)

        for node in cluster.nodes:
            # Create cache key for this verification
            cache_key = (
                f"cluster_verification:{cluster.id}:{node.id}:{hash(str(evidence))}"
            )

            # Check cache first
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                self.performance_metrics["cache_hits"] += 1
                results[node.id] = cached_result
                logger.debug(f"Cache hit for node {node.id} verification")
                continue

            # Verify this fact with cluster context
            verification_prompt = self._create_cluster_verification_prompt(
                node, cluster_context, evidence
            )

            try:
                import time

                start_time = time.time()

                llm_response = await llm_manager.invoke_text_only(verification_prompt)

                verification_time = time.time() - start_time
                self.performance_metrics["verification_times"].append(verification_time)

                # Parse LLM response
                fact_result = self.response_parser.parse_verification_response(
                    llm_response
                )

                # Ensure fact_result is not None and is a dictionary
                if fact_result is None or not isinstance(fact_result, dict):
                    fact_result = {
                        "verdict": "ERROR",
                        "confidence": 0.0,
                        "reasoning": "Failed to parse LLM response",
                        "evidence_used": [],
                    }

                # Enhance confidence scoring with adaptive thresholds
                original_confidence = fact_result.get("confidence", 0.0)
                enhanced_confidence = await self._enhance_confidence_score(
                    original_confidence,
                    fact_result.get("verdict", "UNKNOWN"),
                    len(evidence),
                    cluster.cluster_type.value,
                )

                fact_result["confidence"] = enhanced_confidence
                fact_result["original_confidence"] = original_confidence
                fact_result["node_id"] = node.id
                fact_result["claim"] = node.claim
                fact_result["verification_time"] = verification_time
                fact_result["evidence_count"] = len(evidence)

                # Cache the result
                await self.cache.set(cache_key, fact_result, ttl_seconds=3600)

                results[node.id] = fact_result
                self.performance_metrics["verifications_performed"] += 1
                self.performance_metrics["confidence_scores"].append(
                    enhanced_confidence
                )

            except Exception as e:
                logger.error(f"Failed to verify fact {node.id}: {e}")
                error_result = {
                    "node_id": node.id,
                    "claim": node.claim,
                    "verdict": "ERROR",
                    "confidence": 0.0,
                    "reasoning": f"Verification failed: {str(e)}",
                    "evidence_used": [],
                    "verification_time": 0.0,
                    "evidence_count": len(evidence),
                }
                results[node.id] = error_result

        # Record metrics in adaptive thresholds
        if results:
            avg_confidence = sum(
                r.get("confidence", 0) for r in results.values()
            ) / len(results)
            await self.adaptive_thresholds.record_performance_metrics(
                precision=0.8,  # Placeholder precision
                recall=0.7,  # Placeholder recall
                f1_score=0.75,  # Placeholder F1 score
                source_retention_rate=0.9,  # Placeholder retention rate
                query_type="verification",
                source_type="cluster",
            )

        return results

    def _create_cluster_context(
        self, cluster: FactCluster, evidence: list[dict[str, Any]]
    ) -> str:
        """Create enhanced context string for cluster verification."""
        context_parts = []

        # Add cluster information
        context_parts.append(f"Cluster Type: {cluster.cluster_type.value}")
        context_parts.append(f"Cluster ID: {cluster.id}")

        if cluster.shared_context:
            context_parts.append(f"Shared Context: {cluster.shared_context}")

        # Add cluster metadata if available
        if hasattr(cluster, "metadata") and cluster.metadata:
            context_parts.append(f"Cluster Metadata: {cluster.metadata}")

        # Add related claims in cluster
        other_claims = [f"- {node.claim}" for node in cluster.nodes]
        if other_claims:
            context_parts.append("Related Claims in Cluster:")
            context_parts.extend(other_claims[:5])  # Limit to 5 claims

        # Add enhanced evidence summary
        if evidence:
            context_parts.append("Available Evidence:")
            for i, ev in enumerate(evidence[:3]):  # Limit to 3 evidence pieces
                # Handle different evidence formats
                if isinstance(ev, dict):
                    source = ev.get("url", ev.get("source", "Unknown source"))
                    content = ev.get("content", ev.get("search_results", ""))
                    relevance = ev.get("query_relevance", ev.get("relevance_score", 0))

                    content_preview = str(content)[:150] if content else "No content"
                    context_parts.append(f"- Source {i+1}: {source}")
                    context_parts.append(f"  Preview: {content_preview}...")
                    if relevance:
                        context_parts.append(f"  Relevance: {relevance:.3f}")
                else:
                    # Handle string evidence
                    content_preview = str(ev)[:150]
                    context_parts.append(f"- Evidence {i+1}: {content_preview}...")

        return "\n".join(context_parts)

    def _create_cluster_verification_prompt(
        self, node: FactNode, cluster_context: str, evidence: list[dict[str, Any]]
    ) -> str:
        """Create enhanced verification prompt for a fact within cluster context."""
        # Prepare enhanced evidence text
        evidence_text = ""
        if evidence:
            evidence_pieces = []
            for i, ev in enumerate(evidence[:5]):  # Limit to 5 evidence pieces
                if isinstance(ev, dict):
                    source = ev.get("url", ev.get("source", "Unknown source"))
                    content = ev.get("content", ev.get("search_results", ""))
                    relevance = ev.get("query_relevance", ev.get("relevance_score", 0))

                    # Limit content length but preserve important information
                    content_text = (
                        str(content)[:800] if content else "No content available"
                    )

                    evidence_piece = f"Evidence {i+1} (Source: {source}"
                    if relevance:
                        evidence_piece += f", Relevance: {relevance:.3f}"
                    evidence_piece += f"):\n{content_text}"

                    evidence_pieces.append(evidence_piece)
                else:
                    # Handle string evidence
                    content_text = str(ev)[:800]
                    evidence_pieces.append(f"Evidence {i+1}:\n{content_text}")

            evidence_text = "\n\n".join(evidence_pieces)

        # Get verification prompt template
        prompt_template = self.prompt_manager.get_prompt_template(
            "cluster_fact_verification"
        )

        # Get format instructions for structured output
        format_instructions = (
            self.response_parser.get_verification_format_instructions()
        )

        # Prepare node metadata
        node_metadata = ""
        if hasattr(node, "metadata") and node.metadata:
            node_metadata = str(node.metadata)
        else:
            node_metadata = "No metadata available"

        return prompt_template.format(
            claim=node.claim,
            cluster_context=cluster_context,
            evidence=evidence_text or "No evidence available",
            node_metadata=node_metadata,
            format_instructions=format_instructions,
        )

    async def verify_individual_node(
        self, node: FactNode, context: VerificationContext
    ) -> dict[str, Any]:
        """Verify a single node that's not part of any cluster with enhanced caching."""
        # Create cache key for individual verification
        cache_key = (
            f"individual_verification:{node.id}:{hash(str(context.additional_context))}"
        )

        # Check cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            self.performance_metrics["cache_hits"] += 1
            logger.debug(f"Cache hit for individual node {node.id} verification")
            return cached_result

        try:
            import time

            start_time = time.time()

            # Create simple verification prompt for individual node
            prompt_template = self.prompt_manager.get_prompt_template(
                "individual_fact_verification"
            )

            # Get format instructions for structured output
            format_instructions = (
                self.response_parser.get_verification_format_instructions()
            )

            # Ensure all format parameters are not None
            additional_context = (
                context.additional_context
                if context.additional_context is not None
                else "No additional context"
            )

            # Prepare source info
            source_info = ""
            if hasattr(node, "metadata") and node.metadata:
                source_info = str(node.metadata)
            else:
                source_info = "No source information available"

            prompt = prompt_template.format(
                claim=node.claim,
                evidence="No specific evidence available for individual verification",
                source_info=source_info,
                context=additional_context,
                format_instructions=format_instructions,
            )

            llm_response = await llm_manager.invoke_text_only(prompt)

            verification_time = time.time() - start_time
            self.performance_metrics["verification_times"].append(verification_time)

            # Parse response
            result = self.response_parser.parse_verification_response(llm_response)

            # Ensure result is not None and is a dictionary
            if result is None or not isinstance(result, dict):
                result = {
                    "verdict": "ERROR",
                    "confidence": 0.0,
                    "reasoning": "Failed to parse LLM response",
                    "evidence_used": [],
                }

            # Enhance confidence scoring for individual verification
            original_confidence = result.get("confidence", 0.0)
            enhanced_confidence = await self._enhance_confidence_score(
                original_confidence,
                result.get("verdict", "UNKNOWN"),
                0,
                "individual",  # No evidence for individual verification
            )

            result["confidence"] = enhanced_confidence
            result["original_confidence"] = original_confidence
            result["node_id"] = node.id
            result["claim"] = node.claim
            result["verification_time"] = verification_time
            result["evidence_count"] = 0

            # Cache the result
            await self.cache.set(cache_key, result, ttl_seconds=3600)

            self.performance_metrics["verifications_performed"] += 1
            self.performance_metrics["confidence_scores"].append(enhanced_confidence)

            return result

        except Exception as e:
            logger.error(f"Failed to verify individual node {node.id}: {e}")
            return {
                "node_id": node.id,
                "claim": node.claim,
                "verdict": "ERROR",
                "confidence": 0.0,
                "reasoning": f"Individual verification failed: {str(e)}",
                "evidence_used": [],
                "verification_time": 0.0,
                "evidence_count": 0,
            }

    def create_cross_verification_prompt(
        self, fact1: FactNode, fact2: FactNode, evidence: list[dict[str, Any]]
    ) -> str:
        """Create enhanced prompt for cross-verification between two facts."""
        evidence_text = ""
        if evidence:
            evidence_pieces = []
            for i, ev in enumerate(evidence[:3]):
                if isinstance(ev, dict):
                    source = ev.get("url", ev.get("source", "Unknown source"))
                    content = ev.get("content", ev.get("search_results", ""))
                    relevance = ev.get("query_relevance", ev.get("relevance_score", 0))

                    content_text = str(content)[:400] if content else "No content"
                    evidence_piece = f"Evidence {i+1} (Source: {source}"
                    if relevance:
                        evidence_piece += f", Relevance: {relevance:.3f}"
                    evidence_piece += f"): {content_text}"

                    evidence_pieces.append(evidence_piece)
                else:
                    content_text = str(ev)[:400]
                    evidence_pieces.append(f"Evidence {i+1}: {content_text}")

            evidence_text = "\n".join(evidence_pieces)

        prompt_template = self.prompt_manager.get_prompt_template("cross_verification")

        return prompt_template.format(
            primary_claims=f"1. {fact1.claim}",
            related_claims=f"2. {fact2.claim}",
            evidence_pool=evidence_text or "No evidence available",
            relationship_context="Cross-verification between two individual claims",
        )

    async def _enhance_confidence_score(
        self,
        original_confidence: float,
        verdict: str,
        evidence_count: int,
        context_type: str,
    ) -> float:
        """Enhance confidence score using adaptive thresholds and context."""
        try:
            # Get adaptive threshold for this context
            threshold = await self.adaptive_thresholds.get_adaptive_threshold(
                query_type=context_type,
                source_type="verification",
                context={"evidence_count": evidence_count},
            )

            # Apply confidence enhancement based on evidence and verdict
            enhancement_factor = 1.0

            # Boost confidence for strong verdicts with good evidence
            if verdict in ["TRUE", "SUPPORTED"] and evidence_count > 2:
                enhancement_factor = 1.1
            elif verdict in ["FALSE", "REFUTED"] and evidence_count > 2:
                enhancement_factor = 1.05
            elif verdict in ["UNCERTAIN", "INSUFFICIENT"] and evidence_count < 2:
                enhancement_factor = 0.9

            # Apply threshold-based adjustment
            if original_confidence > threshold:
                enhancement_factor *= 1.05
            elif original_confidence < threshold * 0.7:
                enhancement_factor *= 0.95

            enhanced_confidence = min(1.0, original_confidence * enhancement_factor)

            return enhanced_confidence

        except Exception as e:
            logger.warning(f"Failed to enhance confidence score: {e}")
            return original_confidence

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics."""
        cache_stats = await self.cache.get_stats()

        avg_confidence = (
            (
                sum(self.performance_metrics["confidence_scores"])
                / len(self.performance_metrics["confidence_scores"])
            )
            if self.performance_metrics["confidence_scores"]
            else 0.0
        )

        avg_verification_time = (
            (
                sum(self.performance_metrics["verification_times"])
                / len(self.performance_metrics["verification_times"])
            )
            if self.performance_metrics["verification_times"]
            else 0.0
        )

        return {
            "cache_stats": cache_stats,
            "performance_metrics": {
                "verifications_performed": self.performance_metrics[
                    "verifications_performed"
                ],
                "cache_hits": self.performance_metrics["cache_hits"],
                "average_confidence_score": avg_confidence,
                "average_verification_time": avg_verification_time,
                "total_confidence_calculations": len(
                    self.performance_metrics["confidence_scores"]
                ),
            },
            "cache_hit_rate": (
                self.performance_metrics["cache_hits"]
                / max(
                    1,
                    self.performance_metrics["verifications_performed"]
                    + self.performance_metrics["cache_hits"],
                )
            ),
        }

    async def clear_cache(self):
        """Clear the verification cache."""
        await self.cache.clear()
        logger.info("Verification cache cleared")

    async def optimize_performance(self):
        """Optimize cache and performance based on usage patterns."""
        await self.cache.optimize()

        # Get optimization recommendations from adaptive thresholds
        recommendations = await self.adaptive_thresholds.get_threshold_recommendations()

        logger.info(
            f"Verification processor optimization completed. Recommendations: {recommendations}"
        )
        return recommendations

    async def close(self):
        """Close all resources and clean up."""
        if self.cache:
            await self.cache.close()
            logger.info("VerificationProcessor: IntelligentCache closed")

        logger.info("VerificationProcessor: Resources cleaned up")


# Backward compatibility alias
VerificationProcessor = EnhancedVerificationProcessor

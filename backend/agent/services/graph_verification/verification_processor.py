"""
Verification processor for individual fact verification.

Handles the verification of individual facts within cluster context.
"""

import logging
from typing import Any

from agent.llm import llm_manager
from agent.models.graph import FactCluster, FactNode
from agent.models.verification_context import VerificationContext
from agent.prompt_manager import PromptManager

from .response_parser import ResponseParser

logger = logging.getLogger(__name__)


class VerificationProcessor:
    """Processes verification of individual facts within clusters."""

    def __init__(self):
        self.prompt_manager = PromptManager()
        self.response_parser = ResponseParser()

    async def verify_cluster_facts(
        self,
        cluster: FactCluster,
        evidence: list[dict[str, Any]],
        context: VerificationContext,
    ) -> dict[str, dict[str, Any]]:
        """Verify individual facts within the cluster context."""
        results = {}

        # Create cluster context for LLM
        cluster_context = self._create_cluster_context(cluster, evidence)

        for node in cluster.nodes:
            # Verify this fact with cluster context
            verification_prompt = self._create_cluster_verification_prompt(
                node, cluster_context, evidence
            )

            try:
                llm_response = await llm_manager.invoke_text_only(verification_prompt)

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

                fact_result["node_id"] = node.id
                fact_result["claim"] = node.claim

                results[node.id] = fact_result

            except Exception as e:
                logger.error(f"Failed to verify fact {node.id}: {e}")
                results[node.id] = {
                    "node_id": node.id,
                    "claim": node.claim,
                    "verdict": "ERROR",
                    "confidence": 0.0,
                    "reasoning": f"Verification failed: {str(e)}",
                    "evidence_used": [],
                }

        return results

    def _create_cluster_context(
        self, cluster: FactCluster, evidence: list[dict[str, Any]]
    ) -> str:
        """Create context string for cluster verification."""
        context_parts = []

        # Add cluster information
        context_parts.append(f"Cluster Type: {cluster.cluster_type.value}")

        if cluster.shared_context:
            context_parts.append(f"Shared Context: {cluster.shared_context}")

        # Add related claims in cluster
        other_claims = [f"- {node.claim}" for node in cluster.nodes]
        if other_claims:
            context_parts.append("Related Claims in Cluster:")
            context_parts.extend(other_claims[:5])  # Limit to 5 claims

        # Add evidence summary
        if evidence:
            context_parts.append("Available Evidence:")
            for i, ev in enumerate(evidence[:3]):  # Limit to 3 evidence pieces
                source = ev.get("url", "Unknown source")
                content_preview = ev.get("content", "")[:100]
                context_parts.append(f"- Source {i+1}: {source}")
                context_parts.append(f"  Preview: {content_preview}...")

        return "\n".join(context_parts)

    def _create_cluster_verification_prompt(
        self, node: FactNode, cluster_context: str, evidence: list[dict[str, Any]]
    ) -> str:
        """Create verification prompt for a fact within cluster context."""
        # Prepare evidence text
        evidence_text = ""
        if evidence:
            evidence_pieces = []
            for i, ev in enumerate(evidence[:5]):  # Limit to 5 evidence pieces
                source = ev.get("url", "Unknown source")
                content = ev.get("content", "")[:500]  # Limit content length
                evidence_pieces.append(f"Evidence {i+1} (Source: {source}):\n{content}")
            evidence_text = "\n\n".join(evidence_pieces)

        # Get verification prompt template
        prompt_template = self.prompt_manager.get_prompt_template(
            "cluster_fact_verification"
        )

        # Get format instructions for structured output
        format_instructions = (
            self.response_parser.get_verification_format_instructions()
        )

        return prompt_template.format(
            claim=node.claim,
            cluster_context=cluster_context,
            evidence=evidence_text or "No evidence available",
            node_metadata=str(node.metadata) if node.metadata else "No metadata",
            format_instructions=format_instructions,
        )

    async def verify_individual_node(
        self, node: FactNode, context: VerificationContext
    ) -> dict[str, Any]:
        """Verify a single node that's not part of any cluster."""
        try:
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
            source_info = (
                str(node.metadata)
                if node.metadata
                else "No source information available"
            )

            prompt = prompt_template.format(
                claim=node.claim,
                evidence="No specific evidence available for individual verification",
                source_info=source_info,
                context=additional_context,
                format_instructions=format_instructions,
            )

            llm_response = await llm_manager.invoke_text_only(prompt)

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

            result["node_id"] = node.id
            result["claim"] = node.claim

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
            }

    def create_cross_verification_prompt(
        self, fact1: FactNode, fact2: FactNode, evidence: list[dict[str, Any]]
    ) -> str:
        """Create prompt for cross-verification between two facts."""
        evidence_text = ""
        if evidence:
            evidence_pieces = []
            for i, ev in enumerate(evidence[:3]):
                source = ev.get("url", "Unknown source")
                content = ev.get("content", "")[:300]
                evidence_pieces.append(f"Evidence {i+1}: {content}")
            evidence_text = "\n".join(evidence_pieces)

        prompt_template = self.prompt_manager.get_prompt_template("cross_verification")

        return prompt_template.format(
            primary_claims=f"1. {fact1.claim}",
            related_claims=f"2. {fact2.claim}",
            evidence_pool=evidence_text or "No evidence available",
            relationship_context="Cross-verification between two individual claims",
        )

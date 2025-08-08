"""
Cluster analyzer for cross-verification and contradiction detection.

Handles analysis of fact clusters for contradictions and cross-verification.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agent.llm.manager import llm_manager
from agent.models.graph import FactCluster, FactGraph, FactNode
from agent.prompts.manager import PromptManager

from .response_parser import ResponseParser
from .verification_processor import VerificationProcessor

logger = logging.getLogger(__name__)


class ClusterAnalyzer:
    """Analyzes clusters for contradictions and performs cross-verification."""

    def __init__(self):
        self.prompt_manager = PromptManager()
        self.response_parser = ResponseParser()
        self.verification_processor = VerificationProcessor()

    async def cross_verify_cluster_facts(
        self,
        cluster: FactCluster,
        individual_results: dict[str, dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Perform cross-verification between facts in a cluster."""
        if len(cluster.nodes) < 2:
            return []

        cross_verification_results = []

        # Create pairs of facts for cross-verification
        nodes = list(cluster.nodes)
        pairs_to_verify = []

        for i, _ in enumerate(nodes):
            for j in range(i + 1, min(i + 3, len(nodes))):  # Limit pairs to avoid explosion
                pairs_to_verify.append((nodes[i], nodes[j]))

        # Limit total pairs
        pairs_to_verify = pairs_to_verify[:5]

        # Perform cross-verification for each pair
        for fact1, fact2 in pairs_to_verify:
            try:
                cross_result = await self._cross_verify_pair(fact1, fact2, evidence)
                cross_verification_results.append(cross_result)
            except (ValueError, KeyError, TypeError, RuntimeError) as e:
                logger.error("Cross-verification failed for %s vs %s: %s", fact1.id, fact2.id, e)
                cross_verification_results.append(
                    {
                        "fact1_id": fact1.id,
                        "fact2_id": fact2.id,
                        "relationship": "ERROR",
                        "confidence": 0.0,
                        "reasoning": f"Cross-verification failed: {str(e)}",
                    }
                )

        return cross_verification_results

    async def _cross_verify_pair(
        self, fact1: FactNode, fact2: FactNode, evidence: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Cross-verify a pair of facts."""
        prompt = self.verification_processor.create_cross_verification_prompt(fact1, fact2, evidence)

        llm_response = await llm_manager.invoke_text_only(prompt)

        # Parse cross-verification response
        result = self._parse_cross_verification_response(llm_response)
        result["fact1_id"] = fact1.id
        result["fact2_id"] = fact2.id

        return result

    def _parse_cross_verification_response(self, response: str) -> dict[str, Any]:
        """Parse LLM response for cross-verification."""
        try:
            # Try to extract structured information
            relationship = "UNKNOWN"
            confidence = 0.0
            reasoning = response[:200]

            # Look for relationship indicators
            response_lower = response.lower()
            if "support" in response_lower or "consistent" in response_lower:
                relationship = "SUPPORTING"
            elif "contradict" in response_lower or "conflict" in response_lower:
                relationship = "CONTRADICTING"
            elif "independent" in response_lower or "unrelated" in response_lower:
                relationship = "INDEPENDENT"

            # Extract confidence if present
            confidence_match = re.search(r'confidence["\']?\s*:\s*([0-9.]+)', response, re.IGNORECASE)
            if confidence_match:
                confidence = float(confidence_match.group(1))

            return {
                "relationship": relationship,
                "confidence": confidence,
                "reasoning": reasoning,
            }

        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.error("Failed to parse cross-verification response: %s", e)
            return {
                "relationship": "ERROR",
                "confidence": 0.0,
                "reasoning": f"Parse error: {str(e)}",
            }

    async def detect_cluster_contradictions(
        self,
        cluster: FactCluster,
        individual_results: dict[str, dict[str, Any]],
        graph: FactGraph,
    ) -> list[dict[str, Any]]:
        """Detect contradictions within a cluster."""
        contradictions = []

        # Check for verdict contradictions
        verdict_contradictions = self._detect_verdict_contradictions(individual_results)
        contradictions.extend(verdict_contradictions)

        # Check for semantic contradictions using LLM
        semantic_contradictions = await self._detect_semantic_contradictions(cluster, individual_results)
        contradictions.extend(semantic_contradictions)

        return contradictions

    def _detect_verdict_contradictions(self, individual_results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        """Detect contradictions based on verification verdicts."""
        contradictions = []

        verdicts = {}
        for node_id, result in individual_results.items():
            verdict = result.get("verdict", "UNKNOWN")
            if verdict not in verdicts:
                verdicts[verdict] = []
            verdicts[verdict].append(node_id)

        # Check if we have both TRUE and FALSE verdicts
        if "TRUE" in verdicts and "FALSE" in verdicts:
            contradictions.append(
                {
                    "type": "verdict_contradiction",
                    "description": "Cluster contains both TRUE and FALSE facts",
                    "true_facts": verdicts["TRUE"],
                    "false_facts": verdicts["FALSE"],
                    "severity": "high",
                }
            )

        return contradictions

    async def _detect_semantic_contradictions(
        self, cluster: FactCluster, individual_results: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Detect semantic contradictions using LLM analysis."""
        if len(cluster.nodes) < 2:
            return []

        # Prepare claims and their verdicts for LLM analysis
        claims_with_verdicts = []
        for node in cluster.nodes:
            result = individual_results.get(node.id, {})
            verdict = result.get("verdict", "UNKNOWN")
            claims_with_verdicts.append(f"Claim: {node.claim} | Verdict: {verdict}")

        claims_text = "\n".join(claims_with_verdicts)

        # Create contradiction detection prompt
        prompt_template = self.prompt_manager.get_prompt_template("detect_contradictions")

        # Get format instructions for structured output
        format_instructions = self.response_parser.get_contradiction_format_instructions()

        # Ensure temporal_context is not None
        temporal_context = (
            cluster.shared_context if cluster.shared_context is not None else "No temporal context available"
        )

        prompt = prompt_template.format(
            cluster_claims=claims_text,
            evidence_context="Evidence from individual verification results",
            source_information="Multiple sources from fact verification",
            temporal_context=temporal_context,
            format_instructions=format_instructions,
        )

        try:
            llm_response = await llm_manager.invoke_text_only(prompt)
            contradictions = self.response_parser.parse_contradiction_response(llm_response)

            # Add cluster context to contradictions
            for contradiction in contradictions:
                contradiction["cluster_id"] = cluster.id

            return contradictions

        except (ValueError, TypeError, RuntimeError, KeyError) as e:
            logger.error("Failed to detect semantic contradictions: %s", e)
            return []

    def compile_cluster_verdict(
        self,
        individual_results: dict[str, dict[str, Any]],
        cross_verification_results: list[dict[str, Any]],
        contradictions: list[dict[str, Any]],
    ) -> tuple[str, float]:
        """Compile overall verdict and confidence for a cluster."""
        if not individual_results:
            return "UNKNOWN", 0.0

        # Count verdicts
        verdict_counts = {}
        total_confidence = 0.0
        valid_results = 0

        for result in individual_results.values():
            verdict = result.get("verdict", "UNKNOWN")
            confidence = result.get("confidence", 0.0)

            if verdict != "ERROR":
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
                total_confidence += confidence
                valid_results += 1

        if valid_results == 0:
            return "ERROR", 0.0

        # Determine overall verdict
        most_common_verdict = max(verdict_counts, key=verdict_counts.get)
        avg_confidence = total_confidence / valid_results

        # Adjust confidence based on contradictions
        if contradictions:
            high_severity_contradictions = [c for c in contradictions if c.get("severity") == "high"]
            if high_severity_contradictions:
                avg_confidence *= 0.5  # Reduce confidence significantly
            else:
                avg_confidence *= 0.8  # Reduce confidence moderately

        # Adjust confidence based on cross-verification
        if cross_verification_results:
            supporting_count = sum(1 for cv in cross_verification_results if cv.get("relationship") == "SUPPORTING")
            contradicting_count = sum(
                1 for cv in cross_verification_results if cv.get("relationship") == "CONTRADICTING"
            )

            if supporting_count > contradicting_count:
                avg_confidence *= 1.1  # Boost confidence
            elif contradicting_count > supporting_count:
                avg_confidence *= 0.9  # Reduce confidence

        # Ensure confidence is within bounds
        avg_confidence = max(0.0, min(1.0, avg_confidence))

        return most_common_verdict, avg_confidence

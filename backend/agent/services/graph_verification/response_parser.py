"""
Response parser for LLM outputs in graph verification.

Handles parsing of various LLM response formats used in fact verification.
"""

import json
import logging
import re
from typing import Any

from langchain.output_parsers import PydanticOutputParser
from pydantic import ValidationError

from agent.models.graph_verification_models import (
    ClusterVerificationResponse,
    ContradictionDetectionResponse,
    CrossVerificationResponse,
    EvidenceAnalysisResponse,
    SourceSelectionResponse,
    VerificationResponse,
)
from agent.models.search_models import SearchResultWrapper

logger = logging.getLogger(__name__)


class ResponseParser:
    """Parser for various LLM response formats using Pydantic models."""

    def __init__(self):
        """Initialize parsers for different response types."""
        self.verification_parser = PydanticOutputParser(
            pydantic_object=VerificationResponse
        )
        self.source_selection_parser = PydanticOutputParser(
            pydantic_object=SourceSelectionResponse
        )
        self.contradiction_parser = PydanticOutputParser(
            pydantic_object=ContradictionDetectionResponse
        )
        self.cross_verification_parser = PydanticOutputParser(
            pydantic_object=CrossVerificationResponse
        )
        self.cluster_verification_parser = PydanticOutputParser(
            pydantic_object=ClusterVerificationResponse
        )
        self.evidence_analysis_parser = PydanticOutputParser(
            pydantic_object=EvidenceAnalysisResponse
        )
        self.source_selection_parser = PydanticOutputParser(
            pydantic_object=SourceSelectionResponse
        )

    def get_verification_format_instructions(self) -> str:
        """Get format instructions for verification response."""
        return self.verification_parser.get_format_instructions()

    def get_cluster_verification_format_instructions(self) -> str:
        """Get format instructions for cluster verification response."""
        return self.cluster_verification_parser.get_format_instructions()

    def get_source_selection_format_instructions(self) -> str:
        """Get format instructions for source selection response."""
        return self.source_selection_parser.get_format_instructions()

    def get_contradiction_format_instructions(self) -> str:
        """Get format instructions for contradiction detection response."""
        return self.contradiction_parser.get_format_instructions()

    def get_cross_verification_format_instructions(self) -> str:
        """Get format instructions for cross verification response."""
        return self.cross_verification_parser.get_format_instructions()

    def get_evidence_analysis_format_instructions(self) -> str:
        """Get format instructions for evidence analysis response."""
        return self.evidence_analysis_parser.get_format_instructions()

    def parse_verification_response(self, response: str) -> dict[str, Any]:
        """Parse LLM verification response into structured format using Pydantic."""
        try:
            # Use Pydantic parser for structured output
            parsed_response = self.verification_parser.parse(response)
            return {
                "verdict": parsed_response.verdict,
                "confidence": parsed_response.confidence,
                "reasoning": parsed_response.reasoning,
                "evidence_used": parsed_response.evidence_used,
            }
        except (ValidationError, ValueError, TypeError, AttributeError, json.JSONDecodeError) as e:
            logger.error(
                "Pydantic validation error in verification response: %s", e)
            return {
                "verdict": "ERROR",
                "confidence": 0.0,
                "reasoning": f"Failed to parse response: {str(e)}",
                "evidence_used": [],
            }

    def parse_source_selection_response(self, response: str) -> list[int]:
        """Parse LLM response to extract selected source indices using Pydantic."""
        try:
            # Use Pydantic parser for structured output
            parsed_response = self.source_selection_parser.parse(response)
            # Convert 1-based indices to 0-based if needed
            return [
                idx - 1 if idx > 0 else idx for idx in parsed_response.selected_sources
            ]
        except ValidationError as e:
            logger.error("Failed to parse source selection response: %s", e)
            raise RuntimeError(
                f"Failed to parse source selection response: {e}") from e

    def parse_contradiction_response(self, response: str) -> list[dict[str, Any]]:
        """Parse LLM response for contradiction detection using Pydantic."""
        try:
            # Use Pydantic parser for structured output
            parsed_response = self.contradiction_parser.parse(response)
            logger.info("PARSED CONTRATICTION RESULT: %s", parsed_response)

            # Convert Contradiction objects to dictionaries
            contradictions = []
            for contradiction in parsed_response.contradictions:
                if hasattr(contradiction, "model_dump"):
                    # It's a Pydantic model, convert to dict
                    contradiction_dict = contradiction.model_dump()
                elif isinstance(contradiction, dict):
                    # It's already a dictionary
                    contradiction_dict = contradiction
                else:
                    # Fallback: convert object attributes to dict
                    contradiction_dict = {
                        "claim": getattr(contradiction, "claim", ""),
                        "verdict": getattr(contradiction, "verdict", ""),
                        "reasoning": getattr(contradiction, "reasoning", ""),
                        "type": getattr(contradiction, "type", "unknown"),
                        "confidence": getattr(contradiction, "confidence", 0.0),
                    }

                # Normalize the format for backward compatibility
                normalized_contradiction = {
                    "type": contradiction_dict.get("type", "unknown"),
                    "description": contradiction_dict.get(
                        "reasoning", contradiction_dict.get("description", "")
                    ),
                    "confidence": contradiction_dict.get("confidence", 0.0),
                    "claims_involved": [contradiction_dict.get("claim", "")],
                    "resolution_strategy": "manual_review",
                    # Keep original fields for backward compatibility
                    "claim": contradiction_dict.get("claim", ""),
                    "verdict": contradiction_dict.get("verdict", ""),
                    "reasoning": contradiction_dict.get("reasoning", ""),
                }
                contradictions.append(normalized_contradiction)

            return contradictions

        except (ValidationError, ValueError, TypeError, AttributeError, json.JSONDecodeError) as e:
            logger.error("Failed to parse contradiction response: %s", e)
            return []

    def parse_cluster_verification_response(self, response: str) -> dict[str, Any]:
        """Parse cluster verification response using Pydantic."""
        try:
            parsed_response = self.cluster_verification_parser.parse(response)
            return {
                "overall_verdict": parsed_response.overall_verdict,
                "confidence_score": parsed_response.confidence_score,
                "individual_verdicts": [
                    {
                        "claim_id": verdict.claim_id,
                        "verdict": verdict.verdict,
                        "confidence": verdict.confidence,
                        "reasoning": verdict.reasoning,
                    }
                    for verdict in parsed_response.individual_verdicts
                ],
                "cluster_analysis": parsed_response.cluster_analysis,
                "contradictions_found": parsed_response.contradictions_found,
            }
        except (ValidationError, ValueError, TypeError, AttributeError, json.JSONDecodeError) as e:
            logger.error(
                "Failed to parse cluster verification response: %s", e)
            return {
                "overall_verdict": "ERROR",
                "confidence_score": 0.0,
                "individual_verdicts": [],
                "cluster_analysis": f"Failed to parse response: {str(e)}",
                "contradictions_found": [],
            }

    def extract_sources_from_result(self, result: str) -> list[str]:
        """
        Extract source URLs from search result using Pydantic models.

        Args:
            result: JSON string containing search results from SearxNG

        Returns:
            List of source URLs

        Raises:
            ValueError: If result format is invalid
        """
        try:
            # Parse the result using Pydantic models
            search_wrapper = SearchResultWrapper.model_validate_json(result)

            if not search_wrapper.success:
                # Handle error case
                error_msg = "No search results available"
                if search_wrapper.error:
                    error_msg = f"Search error: {search_wrapper.error.message}"
                logger.warning(error_msg)
                return []

            if not search_wrapper.data:
                logger.warning("Search wrapper has no data")
                return []

            # Extract URLs from search results
            urls = []
            for search_result in search_wrapper.data.results:
                if search_result.url:
                    urls.append(search_result.url)

            logger.debug(
                "Extracted %d source URLs from search results", len(urls))
            return urls

        except Exception as e:
            raise ValueError(
                f"Failed to extract sources from result: {e}") from e

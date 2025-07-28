"""
Pydantic models for graph verification LLM responses.

These models ensure structured parsing of LLM outputs in graph-based fact checking.
"""

from __future__ import annotations

from typing import Any, Union

from pydantic import BaseModel, Field


class VerificationResponse(BaseModel):
    """Response model for individual fact verification."""

    verdict: str = Field(
        description="Verification verdict: TRUE, FALSE, PARTIALLY_TRUE, or INSUFFICIENT_EVIDENCE"
    )
    confidence: float = Field(
        description="Confidence score from 0.0 to 1.0", ge=0.0, le=1.0
    )
    reasoning: str = Field(description="Detailed reasoning for the verdict")
    evidence_used: list[Union[str, dict]] = Field(
        default_factory=list, description="List of evidence pieces used in verification (can be strings or objects with source/reasoning)"
    )
    supporting_evidence: int = Field(
        default=0, description="Number of supporting evidence pieces"
    )
    contradicting_evidence: int = Field(
        default=0, description="Number of contradicting evidence pieces"
    )


class SourceSelectionResponse(BaseModel):
    """Response model for source selection from search results."""

    selected_sources: list[int] = Field(
        description="List of selected source indices (1-based)"
    )
    reasoning: str = Field(
        description="Brief explanation of why these sources were selected"
    )


class Contradiction(BaseModel):
    """Model for individual contradiction."""

    claim: str = Field(description="The claim that contains contradiction")
    verdict: str = Field(
        description="Verdict for the claim: TRUE, FALSE, PARTIALLY_TRUE, etc."
    )
    reasoning: str = Field(
        description="Reasoning for the contradiction detection")
    type: str = Field(
        description="Type of contradiction: direct, indirect, temporal, etc."
    )
    confidence: float = Field(
        description="Confidence in contradiction detection", ge=0.0, le=1.0
    )


class ContradictionDetectionResponse(BaseModel):
    """Response model for contradiction detection."""

    contradictions: list[Contradiction] = Field(
        default_factory=list, description="List of detected contradictions with details"
    )
    overall_consistency: str = Field(
        description="Overall consistency assessment: CONSISTENT, INCONSISTENT, or MIXED"
    )
    confidence: float = Field(
        description="Confidence in contradiction detection", ge=0.0, le=1.0
    )


class CrossVerificationResponse(BaseModel):
    """Response model for cross-verification between facts."""

    relationship: str = Field(
        description="Relationship type: SUPPORTING, CONTRADICTING, or INDEPENDENT"
    )
    confidence: float = Field(
        description="Confidence in relationship assessment", ge=0.0, le=1.0
    )
    reasoning: str = Field(description="Explanation of the relationship")
    evidence_overlap: list[str] = Field(
        default_factory=list, description="Overlapping evidence between facts"
    )


class ClusterVerificationResponse(BaseModel):
    """Response model for cluster-level verification."""

    overall_verdict: str = Field(
        description="Overall cluster verdict: TRUE, FALSE, MIXED, or INSUFFICIENT_EVIDENCE"
    )
    confidence: float = Field(
        description="Overall confidence score", ge=0.0, le=1.0)
    individual_verdicts: dict[str, str] = Field(
        default_factory=dict, description="Individual verdicts for each fact in cluster"
    )
    contradictions_found: list[dict[str, Any]] = Field(
        default_factory=list, description="List of contradictions found within cluster"
    )
    supporting_relationships: list[dict[str, Any]] = Field(
        default_factory=list, description="List of supporting relationships found"
    )
    reasoning: str = Field(
        description="Overall reasoning for cluster verification")


class EvidenceAnalysisResponse(BaseModel):
    """Response model for evidence analysis."""

    relevance_score: float = Field(
        description="Relevance score of evidence to claim", ge=0.0, le=1.0
    )
    credibility_score: float = Field(
        description="Credibility score of evidence source", ge=0.0, le=1.0
    )
    key_points: list[str] = Field(
        default_factory=list, description="Key points extracted from evidence"
    )
    supports_claim: bool = Field(
        description="Whether evidence supports the claim")
    contradicts_claim: bool = Field(
        description="Whether evidence contradicts the claim"
    )
    reasoning: str = Field(description="Reasoning for evidence assessment")

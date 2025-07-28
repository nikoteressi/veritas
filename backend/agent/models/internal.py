"""
Pydantic models for internal agent data structures.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClaimResult(BaseModel):
    """Result of checking a single claim."""

    claim: str = Field(description="The claim that was checked")
    assessment: str = Field(
        description='One of "true", "likely_true", "unverified", "likely_false", "false"'
    )
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    supporting_evidence: int = Field(description="Number of supporting evidence pieces")
    contradicting_evidence: int = Field(
        description="Number of contradicting evidence pieces"
    )
    sources: list[str] = Field(
        default_factory=list, description="Sources used for this claim"
    )
    reasoning: str = Field(description="Reasoning for the assessment")


class FactCheckSummary(BaseModel):
    """Summary of the fact-checking process."""

    total_sources_found: int
    credible_sources: int
    supporting_evidence: int
    contradicting_evidence: int


class FactCheckResult(BaseModel):
    """Detailed results from the fact-checking process."""

    claim_results: list[ClaimResult] = Field(description="Typed results for each claim")
    examined_sources: list[str]
    search_queries_used: list[str]
    summary: FactCheckSummary


class VerdictResult(BaseModel):
    """Internal model for the final verdict and its components."""

    verdict: str
    confidence_score: float
    reasoning: str
    sources: list[str] | None = []

    # Typed motives analysis
    motives_analysis_typed: MotivesAnalysisResult | None = Field(
        None, description="Typed motives analysis"
    )

    def set_motives_analysis(self, analysis: MotivesAnalysisResult) -> None:
        """Set motives analysis using the typed field."""
        self.motives_analysis_typed = analysis

    def get_motives_analysis(self) -> MotivesAnalysisResult | None:
        """Get motives analysis using the typed field."""
        return self.motives_analysis_typed


class CredibleSource(BaseModel):
    """Represents a single credible source with its name and URL."""

    source: str = Field(
        description="The name of the source (e.g., 'Reuters', 'Associated Press')."
    )
    url: str = Field(description="The direct URL to the article or source.")


class SourceSelectionResponse(BaseModel):
    """Response model for source selection from search results."""

    credible_urls: list[str] = Field(
        description="List of selected credible URLs from the provided search results. "
        "Each URL must be exactly as provided in the search results, without modification."
    )
    reasoning: str = Field(
        description="Brief explanation of why these sources were selected as most credible and relevant."
    )


# Import for forward reference
from .motives_analysis import MotivesAnalysisResult

VerdictResult.model_rebuild()

"""
Pydantic models for internal agent data structures.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ClaimResult(BaseModel):
    """Result of checking a single claim."""
    claim: str = Field(description="The claim that was checked")
    assessment: str = Field(description='One of "true", "likely_true", "unverified", "likely_false", "false"')
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    supporting_evidence: int = Field(description="Number of supporting evidence pieces")
    contradicting_evidence: int = Field(description="Number of contradicting evidence pieces")
    sources: List[str] = Field(default_factory=list, description="Sources used for this claim")
    reasoning: str = Field(description="Reasoning for the assessment")


class FactCheckSummary(BaseModel):
    """Summary of the fact-checking process."""
    total_sources_found: int
    credible_sources: int
    supporting_evidence: int
    contradicting_evidence: int


class FactCheckResult(BaseModel):
    """Detailed results from the fact-checking process."""
    claim_results: List[ClaimResult] = Field(description="Typed results for each claim")
    examined_sources: List[str]
    search_queries_used: List[str]
    summary: FactCheckSummary
    
    # DEPRECATED: Legacy field for backward compatibility
    claim_results_legacy: Optional[List[Dict[str, Any]]] = Field(None, description="[DEPRECATED] Use claim_results instead")


class VerdictResult(BaseModel):
    """Internal model for the final verdict and its components."""
    verdict: str
    confidence_score: float
    reasoning: str
    sources: Optional[List[str]] = []
    
    # Typed motives analysis
    motives_analysis_typed: Optional['MotivesAnalysisResult'] = Field(None, description="Typed motives analysis")
    
    def set_motives_analysis(self, analysis: 'MotivesAnalysisResult') -> None:
        """Set motives analysis using the typed field."""
        self.motives_analysis_typed = analysis
    
    def get_motives_analysis(self) -> Optional['MotivesAnalysisResult']:
        """Get motives analysis using the typed field."""
        return self.motives_analysis_typed


class CredibleSource(BaseModel):
    """Represents a single credible source with its name and URL."""
    source: str = Field(description="The name of the source (e.g., 'Reuters', 'Associated Press').")
    url: str = Field(description="The direct URL to the article or source.")


class SourceSelectionResponse(BaseModel):
    """Response model for source selection from search results."""
    credible_urls: List[str] = Field(
        description="List of selected credible URLs from the provided search results. "
                   "Each URL must be exactly as provided in the search results, without modification."
    )
    reasoning: str = Field(
        description="Brief explanation of why these sources were selected as most credible and relevant."
    )


class FactCheckerResponse(BaseModel):
    """Defines the expected JSON structure from the fact-checker LLM."""
    assessment: str = Field(description='One of "true", "likely_true", "unverified", "likely_false", "false"')
    summary: str = Field(description="A concise summary of findings, explaining the reasoning for the assessment.")
    confidence: float = Field(description="A score from 0.0 to 1.0 representing confidence in the assessment.")
    supporting_evidence: int = Field(description="The number of pieces of evidence that support the claim.")
    contradicting_evidence: int = Field(description="The number of pieces of evidence that contradict the claim.")
    credible_sources: List[CredibleSource] = Field(description="A list of credible sources with their names and URLs.")


# Import for forward reference
from .motives_analysis import MotivesAnalysisResult
VerdictResult.model_rebuild()
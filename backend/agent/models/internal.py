"""
Pydantic models for internal agent data structures.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class FactCheckSummary(BaseModel):
    """Summary of the fact-checking process."""
    total_sources_found: int
    credible_sources: int
    supporting_evidence: int
    contradicting_evidence: int

class FactCheckResult(BaseModel):
    """Detailed results from the fact-checking process."""
    claim_results: List[Dict[str, Any]]
    examined_sources: List[str]
    search_queries_used: List[str]
    summary: FactCheckSummary

class VerdictResult(BaseModel):
    """Internal model for the final verdict and its components."""
    verdict: str
    confidence_score: float
    reasoning: str
    sources: Optional[List[str]] = []
    motives_analysis: Optional[Dict[str, Any]] = None


class CredibleSource(BaseModel):
    """Represents a single credible source with its name and URL."""
    source: str = Field(description="The name of the source (e.g., 'Reuters', 'Associated Press').")
    url: str = Field(description="The direct URL to the article or source.")


class FactCheckerResponse(BaseModel):
    """Defines the expected JSON structure from the fact-checker LLM."""
    assessment: str = Field(description='One of "true", "likely_true", "unverified", "likely_false", "false"')
    summary: str = Field(description="A concise summary of findings, explaining the reasoning for the assessment.")
    confidence: float = Field(description="A score from 0.0 to 1.0 representing confidence in the assessment.")
    supporting_evidence: int = Field(description="The number of pieces of evidence that support the claim.")
    contradicting_evidence: int = Field(description="The number of pieces of evidence that contradict the claim.")
    credible_sources: List[CredibleSource] = Field(description="A list of credible sources with their names and URLs.") 
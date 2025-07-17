"""
Pydantic models for verification results.
"""
from typing import List, Optional
from pydantic import BaseModel, Field

from .temporal_analysis import TemporalAnalysisResult
from .motives_analysis import MotivesAnalysisResult
from .internal import FactCheckSummary


class FactCheckResults(BaseModel):
    """Fact-checking results."""
    examined_sources: int = Field(..., description="Number of examined sources")
    search_queries_used: List[str] = Field(default_factory=list, description="Search queries used")
    summary: FactCheckSummary = Field(..., description="Fact-checking summary")


class UserReputation(BaseModel):
    """User reputation model."""
    nickname: str
    true_count: int
    partially_true_count: int
    false_count: int
    ironic_count: int
    total_posts_checked: int
    warning_issued: bool
    notification_issued: bool


class VerificationResult(BaseModel):
    """Final verification result."""
    status: str = Field(..., description="Verification status")
    message: str = Field(..., description="Result message")
    verification_id: Optional[str] = Field(None, description="Verification ID")


class VerificationData(BaseModel):
    nickname: str = Field(..., description="User nickname")
    extracted_text: str = Field(..., description="Extracted text")
    primary_topic: Optional[str] = Field(None, description="Primary topic")


class VerificationOutput(BaseModel):
    identified_claims: List[str] = Field(default_factory=list, description="Identified claims")
    verdict: str = Field(..., description="Verdict")
    justification: str = Field(..., description="Justification")
    confidence_score: float = Field(..., description="Confidence level")


class VerificationAnalysis(BaseModel):
    temporal_analysis: TemporalAnalysisResult = Field(..., description="Temporal analysis")
    motives_analysis: MotivesAnalysisResult = Field(..., description="Motives analysis")
    fact_check_results: FactCheckResults = Field(..., description="Fact-checking results")


class VerificationMetadata(BaseModel):
    processing_time_seconds: int = Field(..., description="Processing time")
    sources: List[str] = Field(default_factory=list, description="Sources")
    user_reputation: UserReputation = Field(..., description="User reputation")
    warnings: List[str] = Field(default_factory=list, description="Warnings")


class VerificationRequest(BaseModel):
    prompt: str = Field(..., description="Original request")
    filename: str = Field(..., description="Filename")
    file_size: int = Field(..., description="File size")
    summary: Optional[str] = Field(None, description="Summary")
"""
Pydantic models for motives analysis results.
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class MotivesAnalysisResult(BaseModel):
    """Motives analysis result."""
    primary_motive: Optional[str] = Field(None, description="Primary identified motive")
    confidence_score: Optional[float] = Field(None, description="Confidence score (0.0-1.0)")
    credibility_assessment: Optional[str] = Field(None, description="Credibility assessment (high/moderate/low)")
    risk_level: Optional[str] = Field(None, description="Risk level (high/moderate/low)")
    manipulation_indicators: List[str] = Field(default_factory=list, description="Manipulation indicators")
    analysis_summary: Optional[str] = Field(None, description="Analysis summary")
    
    # Legacy fields for backward compatibility
    potential_motives: List[str] = Field(default_factory=list, description="Potential motives (legacy)")
    bias_indicators: List[str] = Field(default_factory=list, description="Bias indicators (legacy)")
    credibility_score: Optional[float] = Field(None, description="Credibility score (legacy)")
    confidence: Optional[float] = Field(None, description="Confidence level (legacy)")
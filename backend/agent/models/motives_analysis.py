"""
Pydantic models for motives analysis results.
"""

from __future__ import annotations


from typing import Optional
from pydantic import BaseModel, Field


class MotivesAnalysisResult(BaseModel):
    """


Motives analysis result.
"""

    primary_motive: Optional[str] = Field(None, description="Primary identified motive")
    confidence_score: Optional[float] = Field(
        None, description="Confidence score (0.0-1.0)"
    )
    credibility_assessment: Optional[str] = Field(
        None, description="Credibility assessment (high/moderate/low)"
    )
    risk_level: Optional[str] = Field(None, description="Risk level (high/moderate/low)")
    manipulation_indicators: list[str] = Field(
        default_factory=list, description="Manipulation indicators"
    )
    analysis_summary: Optional[str] = Field(None, description="Analysis summary")

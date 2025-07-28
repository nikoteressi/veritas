"""
Pydantic models for motives analysis results.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MotivesAnalysisResult(BaseModel):
    """


    Motives analysis result.
    """

    primary_motive: str | None = Field(None, description="Primary identified motive")
    confidence_score: float | None = Field(
        None, description="Confidence score (0.0-1.0)"
    )
    credibility_assessment: str | None = Field(
        None, description="Credibility assessment (high/moderate/low)"
    )
    risk_level: str | None = Field(None, description="Risk level (high/moderate/low)")
    manipulation_indicators: list[str] = Field(
        default_factory=list, description="Manipulation indicators"
    )
    analysis_summary: str | None = Field(None, description="Analysis summary")

"""
Pydantic models for summarization results.
"""

from __future__ import annotations


from typing import Optional


from pydantic import BaseModel, Field


class SummarizationResult(BaseModel):
    """Summarization result."""

    summary: str = Field(description="Summary text")
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence score (0.0-1.0)"
    )
    sources_used: list[str] = Field(
        default_factory=list, description="Sources used for creating summary"
    )
    key_points: list[str] = Field(
        default_factory=list, description="Key points highlighted in summary"
    )
    temporal_context_included: bool = Field(
        default=False, description="Whether temporal context is included in summary"
    )
    fact_check_summary: str = Field(
        description="Brief summary of fact-checking results"
    )
    claims_analyzed: int = Field(ge=0, description="Number of analyzed claims")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {float: lambda v: round(v, 3) if v is not None else None}

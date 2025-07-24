"""
Pydantic models for temporal analysis results.
"""

from __future__ import annotations


from typing import Optional


from pydantic import BaseModel, Field


class TemporalAnalysisResult(BaseModel):
    """Temporal analysis result."""

    post_date: Optional[str] = Field(None, description="Post date")
    mentioned_dates: list[str] = Field(
        default_factory=list, description="Mentioned dates"
    )
    recency_score: Optional[float] = Field(None, description="Recency score")
    temporal_context: Optional[str] = Field(None, description="Temporal context")
    date_relevance: Optional[str] = Field(None, description="Date relevance")

"""
Pydantic models for temporal analysis results.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TemporalAnalysisResult(BaseModel):
    """Temporal analysis result."""

    post_date: str | None = Field(None, description="Post date")
    mentioned_dates: list[str] = Field(
        default_factory=list, description="Mentioned dates"
    )
    recency_score: float | None = Field(None, description="Recency score")
    temporal_context: str | None = Field(None, description="Temporal context")
    date_relevance: str | None = Field(None, description="Date relevance")

"""
Pydantic models for verification results.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .internal import FactCheckSummary


class FactCheckResults(BaseModel):
    """Fact-checking results."""

    examined_sources: int = Field(..., description="Number of examined sources")
    search_queries_used: list[str] = Field(
        default_factory=list, description="Search queries used"
    )
    summary: FactCheckSummary = Field(..., description="Fact-checking summary")


class VerificationResult(BaseModel):
    """Final verification result."""

    status: str = Field(..., description="Verification status")
    message: str = Field(..., description="Result message")
    verification_id: str | None = Field(None, description="Verification ID")

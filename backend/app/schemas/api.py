"""
Pydantic schemas for API request/response models.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class UserReputation(BaseModel):
    """Response model for user reputation data."""

    nickname: str
    true_count: int
    partially_true_count: int
    false_count: int
    ironic_count: int
    total_posts_checked: int
    warning_issued: bool
    notification_issued: bool
    created_at: datetime | None
    last_checked_date: datetime | None

    class Config:
        from_attributes = True


class VerificationRequest(BaseModel):
    """Request model for post verification."""

    prompt: str = Field(
        ..., min_length=1, max_length=1000, description="User's question or prompt"
    )
    message: str
    verification_id: str | None = None
    session_id: str | None = None


class VerificationResponse(BaseModel):
    """Response model for verification results."""

    status: str
    message: str
    verification_id: str | None = None
    nickname: str | None = None
    extracted_text: str | None = None
    primary_topic: str | None = None
    identified_claims: list[str] | None = None
    verdict: str | None = None
    justification: str | None = None
    confidence_score: float | None = None
    processing_time_seconds: int | None = None
    temporal_analysis: dict | None = None
    motives_analysis: dict | None = None
    fact_check_results: dict | None = None
    sources: list[dict] | None = None
    user_reputation: UserReputation | None = None
    warnings: list[str] | None = None
    prompt: str | None = None
    filename: str | None = None
    file_size: int | None = None
    summary: str | None = None


class VerificationStatusResponse(BaseModel):
    """Response model for verification status."""

    verification_id: str
    status: str  # pending, processing, completed, failed
    progress: int | None = None  # 0-100
    current_step: str | None = None
    estimated_time_remaining: int | None = None


class ReputationStatsResponse(BaseModel):
    """Response model for overall reputation statistics."""

    total_users: int
    total_posts_checked: int
    accuracy_rate: float
    true_posts_percentage: float
    false_posts_percentage: float
    partially_true_percentage: float
    ironic_posts_percentage: float
    users_with_warnings: int
    users_with_notifications: int


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now())

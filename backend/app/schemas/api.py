"""
Pydantic schemas for API request/response models.
"""
from datetime import datetime
from typing import Optional, List
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
    created_at: Optional[datetime]
    last_checked_date: Optional[datetime]
    
    class Config:
        from_attributes = True


class VerificationRequest(BaseModel):
    """Request model for post verification."""
    prompt: str = Field(..., min_length=1, max_length=1000, description="User's question or prompt")
    message: str
    verification_id: Optional[str] = None
    session_id: Optional[str] = None


class VerificationResponse(BaseModel):
    """Response model for verification results."""
    status: str
    message: str
    verification_id: Optional[str] = None
    nickname: Optional[str] = None
    extracted_text: Optional[str] = None
    primary_topic: Optional[str] = None
    identified_claims: Optional[List[str]] = None
    verdict: Optional[str] = None
    justification: Optional[str] = None
    confidence_score: Optional[float] = None
    processing_time_seconds: Optional[int] = None
    temporal_analysis: Optional[dict] = None
    motives_analysis: Optional[dict] = None
    fact_check_results: Optional[dict] = None
    sources: Optional[List[dict]] = None
    user_reputation: Optional[UserReputation] = None
    warnings: Optional[List[str]] = None
    prompt: Optional[str] = None
    filename: Optional[str] = None
    file_size: Optional[int] = None
    summary: Optional[str] = None


class VerificationStatusResponse(BaseModel):
    """Response model for verification status."""
    verification_id: str
    status: str  # pending, processing, completed, failed
    progress: Optional[int] = None  # 0-100
    current_step: Optional[str] = None
    estimated_time_remaining: Optional[int] = None


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
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now())
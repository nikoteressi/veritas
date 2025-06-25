"""
Pydantic schemas for API request/response models.
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class UserReputationResponse(BaseModel):
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


class VerificationResponse(BaseModel):
    """Response model for verification results."""
    status: str
    message: str
    verification_id: Optional[str] = None  # Changed from int to str to support UUIDs
    user_nickname: Optional[str] = None
    extracted_text: Optional[str] = None
    primary_topic: Optional[str] = None
    identified_claims: Optional[List[str]] = None
    verdict: Optional[str] = None  # true, partially_true, false, ironic
    justification: Optional[str] = None
    confidence_score: Optional[int] = None
    processing_time_seconds: Optional[int] = None
    user_reputation: Optional[UserReputationResponse] = None
    warnings: Optional[List[str]] = None


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


class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str  # status_update, progress, result, error
    data: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

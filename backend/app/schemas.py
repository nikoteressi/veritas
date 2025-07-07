"""
Pydantic schemas for API request/response models.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID


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
    verification_id: Optional[str] = None  # Changed from int to str to support UUIDs
    user_nickname: Optional[str] = None
    extracted_text: Optional[str] = None
    primary_topic: Optional[str] = None
    identified_claims: Optional[List[str]] = None
    verdict: Optional[str] = None  # true, partially_true, false, ironic
    justification: Optional[str] = None
    confidence_score: Optional[float] = None
    processing_time_seconds: Optional[int] = None
    user_reputation: Optional[UserReputation] = None
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


# ==============================================================================
# Hierarchical Fact Structure Models
# ==============================================================================

class Fact(BaseModel):
    """Represents a single, atomic, verifiable fact that supports the primary thesis."""
    description: str = Field(description="A clear, concise statement of the fact for verification.")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured data extracted for this fact (e.g., amounts, dates, entities) to aid in targeted verification."
    )


class FactHierarchy(BaseModel):
    """Represents the structured, hierarchical understanding of the claims made in the source."""
    primary_thesis: str = Field(description="The single, overarching claim or main point of the source. This summarizes the entire message.")
    supporting_facts: List[Fact] = Field(description="A list of atomic, verifiable facts that support the primary thesis.")


class ImageAnalysisResult(BaseModel):
    """Pydantic model for structured image analysis results."""
    username: Optional[str] = Field(
        default="unknown",
        description="The COMPLETE username of the person who posted the content. Extract the full username without truncation. Should be 'unknown' if not identifiable."
    )
    post_date: Optional[str] = Field(
        default=None,
        description="The timestamp of the post, e.g., 'May 21, 2024' or '15 hours ago' or '2 days ago'. Extract exactly as seen in the image."
    )
    mentioned_dates: List[str] = Field(
        default_factory=list,
        description="A list of any other dates or time references found in the text."
    )
    extracted_text: str = Field(
        description="A verbatim transcription of all visible text from the image."
    )
    fact_hierarchy: FactHierarchy = Field(
        description="A structured representation of the claims and their relationships."
    )
    primary_topic: str = Field(
        default="general",
        description="The primary topic, chosen from: financial, medical, political, scientific, technology, entertainment, general, humorous/ironic."
    )
    irony_assessment: str = Field(
        default="not_ironic",
        description="Assessment of irony, chosen from: not_ironic, potentially_ironic, clearly_ironic."
    )
    visual_elements_summary: str = Field(
        description="A summary of key visual elements like charts, graphs, or UI components. If no visual elements are found, return 'No visual elements found'."
    )
    contextual_information: Dict[str, str] = Field(
        default_factory=dict,
        description="Contextual info, including the current_date (take current date from the prompt) and the user_prompt."
    )


# ==============================================================================
# Internal Agent Data Models
# ==============================================================================

class FactCheckSummary(BaseModel):
    """Summary of the fact-checking process."""
    total_sources_found: int
    credible_sources: int
    supporting_evidence: int
    contradicting_evidence: int

class FactCheckResult(BaseModel):
    """Detailed results from the fact-checking process."""
    claim_results: List[Dict[str, Any]]
    examined_sources: List[str]
    search_queries_used: List[str]
    summary: FactCheckSummary

class VerdictResult(BaseModel):
    """Internal model for the final verdict and its components."""
    verdict: str
    confidence_score: float
    reasoning: str
    sources: Optional[List[str]] = []
    motives_analysis: Optional[Dict[str, Any]] = None


class FactCheckerResponse(BaseModel):
    """Defines the expected JSON structure from the fact-checker LLM."""
    assessment: str = Field(description='One of "true", "likely_true", "unverified", "likely_false", "false"')
    summary: str = Field(description="A concise summary of findings, explaining the reasoning for the assessment.")
    confidence: float = Field(description="A score from 0.0 to 100.0 representing confidence in the assessment.")
    supporting_evidence: int = Field(description="The number of pieces of evidence that support the claim.")
    contradicting_evidence: int = Field(description="The number of pieces of evidence that contradict the claim.")
    credible_sources: int = Field(description="The number of sources deemed credible.")


# --- API Response Models ---

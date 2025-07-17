"""
This package defines the internal data schemas for the agent.
"""
from .fact import Fact, FactHierarchy
from .image_analysis import ImageAnalysisResult
from .internal import (
    FactCheckSummary,
    FactCheckResult,
    VerdictResult,
    FactCheckerResponse,
    ClaimResult,
)
from .verification_context import VerificationContext
from .temporal_analysis import TemporalAnalysisResult
from .motives_analysis import MotivesAnalysisResult
from .extracted_info import ExtractedInfo
from .verification_result import VerificationResult, FactCheckResults, UserReputation
from .websocket_models import ProgressEventPayload, WebSocketMessage

__all__ = [
    "Fact",
    "FactHierarchy",
    "ImageAnalysisResult",
    "FactCheckSummary",
    "FactCheckResult",
    "VerdictResult",
    "FactCheckerResponse",
    "ClaimResult",
    "VerificationContext",
    "TemporalAnalysisResult",
    "MotivesAnalysisResult",
    "ExtractedInfo",
    "VerificationResult",
    "FactCheckResults",
    "UserReputation",
    "ProgressEventPayload",
    "WebSocketMessage",
]
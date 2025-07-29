"""
This package defines the internal data schemas for the agent.
"""

from __future__ import annotations

from .extracted_info import ExtractedInfo
from .fact import Fact, FactHierarchy
from .image_analysis import ImageAnalysisResult
from .internal import ClaimResult, FactCheckResult, FactCheckSummary, VerdictResult
from .motives_analysis import MotivesAnalysisResult
from .temporal_analysis import TemporalAnalysisResult
from .verification_context import VerificationContext
from .verification_result import FactCheckResults, VerificationResult
from .websocket_models import WebSocketMessage

__all__ = [
    "Fact",
    "FactHierarchy",
    "ImageAnalysisResult",
    "FactCheckSummary",
    "FactCheckResult",
    "VerdictResult",
    "ClaimResult",
    "VerificationContext",
    "TemporalAnalysisResult",
    "MotivesAnalysisResult",
    "ExtractedInfo",
    "VerificationResult",
    "FactCheckResults",
    "WebSocketMessage",
]

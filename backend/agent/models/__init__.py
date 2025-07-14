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
)
from .verification_context import VerificationContext

__all__ = [
    "Fact",
    "FactHierarchy",
    "ImageAnalysisResult",
    "FactCheckSummary",
    "FactCheckResult",
    "VerdictResult",
    "FactCheckerResponse",
    "VerificationContext",
] 
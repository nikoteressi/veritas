"""
from __future__ import annotations

Custom exceptions for the Veritas application.
"""

from typing import Any


class VeritasException(Exception):
    """Base exception for Veritas application."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ImageProcessingError(VeritasException):
    """Exception raised when image processing fails."""


class LLMError(VeritasException):
    """Exception raised when LLM operations fail."""


class DatabaseError(VeritasException):
    """Exception raised when database operations fail."""


class WebSocketError(VeritasException):
    """Exception raised when WebSocket operations fail."""


class ValidationError(VeritasException):
    """Exception raised when input validation fails."""


class ServiceUnavailableError(VeritasException):
    """Exception raised when external services are unavailable."""


class AgentError(VeritasException):
    """Exception raised when agent operations fail."""


class VectorStoreError(VeritasException):
    """Exception raised for vector store operations."""


class ToolError(VeritasException):
    """Exception raised when tool operations fail."""


class StorageError(VeritasException):
    """Exception raised for storage operations."""


class TemporalAnalysisError(VeritasException):
    """Exception raised for temporal analysis operations."""


class MotivesAnalysisError(VeritasException):
    """Exception raised for motives analysis operations."""


class ScreenshotParsingError(VeritasException):
    """Exception raised for screenshot parsing operations."""


class CacheError(VeritasException):
    """Exception raised for cache operations."""


class EmbeddingError(VeritasException):
    """Exception raised for embedding operations."""


class GraphError(VeritasException):
    """Exception raised for graph operations."""


class RelevanceError(VeritasException):
    """Exception raised for relevance scoring operations."""


class PipelineError(VeritasException):
    """Exception raised for pipeline execution errors."""


class AnalysisError(VeritasException):
    """Exception raised for general analysis operations."""

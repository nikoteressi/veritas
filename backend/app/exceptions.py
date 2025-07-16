"""
Custom exceptions for the Veritas application.
"""
from typing import Optional, Dict, Any


class VeritasException(Exception):
    """Base exception for Veritas application."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ImageProcessingError(VeritasException):
    """Exception raised when image processing fails."""
    pass


class LLMError(VeritasException):
    """Exception raised when LLM operations fail."""
    pass


class DatabaseError(VeritasException):
    """Exception raised when database operations fail."""
    pass


class WebSocketError(VeritasException):
    """Exception raised when WebSocket operations fail."""
    pass


class ValidationError(VeritasException):
    """Exception raised when input validation fails."""
    pass


class ServiceUnavailableError(VeritasException):
    """Exception raised when external services are unavailable."""
    pass


class AgentError(VeritasException):
    """Exception raised when agent operations fail."""
    pass


class VectorStoreError(VeritasException):
    """Exception raised for vector store operations."""
    pass


class ToolError(VeritasException):
    """Exception raised when tool operations fail."""
    pass


class StorageError(VeritasException):
    """Exception raised for storage operations."""
    pass

class TemporalAnalysisError(VeritasException):
    """Exception raised for temporal analysis operations."""
    pass

class MotivesAnalysisError(VeritasException):
    """Exception raised for motives analysis operations."""
    pass

class ScreenshotParsingError(VeritasException):
    """Exception raised for screenshot parsing operations."""
    pass

"""
Error handlers for the FastAPI application.
"""

import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.exceptions import (
    AgentError,
    DatabaseError,
    ImageProcessingError,
    LLMError,
    ServiceUnavailableError,
    ToolError,
    ValidationError,
    VeritasException,
    WebSocketError,
)

logger = logging.getLogger(__name__)


async def veritas_exception_handler(
    request: Request, exc: VeritasException
) -> JSONResponse:
    """Handle custom Veritas exceptions."""
    logger.error(
        f"Veritas exception: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
        },
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": exc.message,
            "error_code": exc.error_code,
            "details": exc.details,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path),
        },
    )


async def image_processing_error_handler(
    request: Request, exc: ImageProcessingError
) -> JSONResponse:
    """Handle image processing errors."""
    logger.error(f"Image processing error: {exc.message}")

    return JSONResponse(
        status_code=400,
        content={
            "error": "Image processing failed",
            "message": exc.message,
            "error_code": exc.error_code or "IMAGE_PROCESSING_ERROR",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    """Handle LLM errors."""
    logger.error(f"LLM error: {exc.message}")

    return JSONResponse(
        status_code=503,
        content={
            "error": "AI service temporarily unavailable",
            "message": "The AI analysis service is currently experiencing issues. Please try again later.",
            "error_code": exc.error_code or "LLM_SERVICE_ERROR",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


async def database_error_handler(request: Request, exc: DatabaseError) -> JSONResponse:
    """Handle database errors."""
    logger.error(f"Database error: {exc.message}")

    return JSONResponse(
        status_code=503,
        content={
            "error": "Database service unavailable",
            "message": "The database service is currently experiencing issues. Please try again later.",
            "error_code": exc.error_code or "DATABASE_ERROR",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


async def websocket_error_handler(
    request: Request, exc: WebSocketError
) -> JSONResponse:
    """Handle WebSocket errors."""
    logger.error(f"WebSocket error: {exc.message}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "WebSocket communication error",
            "message": exc.message,
            "error_code": exc.error_code or "WEBSOCKET_ERROR",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


async def validation_error_handler(
    request: Request, exc: ValidationError
) -> JSONResponse:
    """Handle validation errors."""
    logger.warning(f"Validation error: {exc.message}")

    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation failed",
            "message": exc.message,
            "error_code": exc.error_code or "VALIDATION_ERROR",
            "details": exc.details,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


async def service_unavailable_error_handler(
    request: Request, exc: ServiceUnavailableError
) -> JSONResponse:
    """Handle service unavailable errors."""
    logger.error(f"Service unavailable: {exc.message}")

    return JSONResponse(
        status_code=503,
        content={
            "error": "External service unavailable",
            "message": exc.message,
            "error_code": exc.error_code or "SERVICE_UNAVAILABLE",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


async def agent_error_handler(request: Request, exc: AgentError) -> JSONResponse:
    """Handle agent errors."""
    logger.error(f"Agent error: {exc.message}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "AI agent error",
            "message": "The AI verification agent encountered an error. Please try again.",
            "error_code": exc.error_code or "AGENT_ERROR",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


async def tool_error_handler(request: Request, exc: ToolError) -> JSONResponse:
    """Handle tool errors."""
    logger.error(f"Tool error: {exc.message}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "Tool execution error",
            "message": "A verification tool encountered an error. The analysis may be incomplete.",
            "error_code": exc.error_code or "TOOL_ERROR",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP error",
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "error_code": "INTERNAL_SERVER_ERROR",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# Exception handler mapping
EXCEPTION_HANDLERS = {
    VeritasException: veritas_exception_handler,
    ImageProcessingError: image_processing_error_handler,
    LLMError: llm_error_handler,
    DatabaseError: database_error_handler,
    WebSocketError: websocket_error_handler,
    ValidationError: validation_error_handler,
    ServiceUnavailableError: service_unavailable_error_handler,
    AgentError: agent_error_handler,
    ToolError: tool_error_handler,
    HTTPException: http_exception_handler,
    StarletteHTTPException: http_exception_handler,
    Exception: general_exception_handler,
}


def setup_error_handlers(app: FastAPI):
    """
    Register all exception handlers for the FastAPI application.
    """
    for exc, handler in EXCEPTION_HANDLERS.items():
        app.add_exception_handler(exc, handler)
    logger.info("Custom exception handlers registered.")

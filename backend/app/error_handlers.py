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
    AnalysisError,
    CacheError,
    DatabaseError,
    EmbeddingError,
    GraphError,
    ImageProcessingError,
    LLMError,
    PipelineError,
    RelevanceError,
    ServiceUnavailableError,
    ToolError,
    ValidationError,
    VeritasException,
    WebSocketError,
)

logger = logging.getLogger(__name__)


async def veritas_exception_handler(request: Request, exc: VeritasException) -> JSONResponse:
    """Handle custom Veritas exceptions."""
    logger.error(
        "Veritas exception: %s",
        exc.message,
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
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path),
        },
    )


async def image_processing_error_handler(exc: ImageProcessingError) -> JSONResponse:
    """Handle image processing errors."""
    logger.error("Image processing error: %s", exc.message)

    return JSONResponse(
        status_code=400,
        content={
            "error": "Image processing failed",
            "message": exc.message,
            "error_code": exc.error_code or "IMAGE_PROCESSING_ERROR",
            "timestamp": datetime.now().isoformat(),
        },
    )


async def llm_error_handler(exc: LLMError) -> JSONResponse:
    """Handle LLM errors."""
    logger.error("LLM error: %s", exc.message)

    return JSONResponse(
        status_code=503,
        content={
            "error": "AI service temporarily unavailable",
            "message": "The AI analysis service is currently experiencing issues. Please try again later.",
            "error_code": exc.error_code or "LLM_SERVICE_ERROR",
            "timestamp": datetime.now().isoformat(),
        },
    )


async def database_error_handler(exc: DatabaseError) -> JSONResponse:
    """Handle database errors."""
    logger.error("Database error: %s", exc.message)

    return JSONResponse(
        status_code=503,
        content={
            "error": "Database service unavailable",
            "message": "The database service is currently experiencing issues. Please try again later.",
            "error_code": exc.error_code or "DATABASE_ERROR",
            "timestamp": datetime.now().isoformat(),
        },
    )


async def websocket_error_handler(exc: WebSocketError) -> JSONResponse:
    """Handle WebSocket errors."""
    logger.error("WebSocket error: %s", exc.message)

    return JSONResponse(
        status_code=500,
        content={
            "error": "WebSocket communication error",
            "message": exc.message,
            "error_code": exc.error_code or "WEBSOCKET_ERROR",
            "timestamp": datetime.now().isoformat(),
        },
    )


async def validation_error_handler(exc: ValidationError) -> JSONResponse:
    """Handle validation errors."""
    logger.warning("Validation error: %s", exc.message)

    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation failed",
            "message": exc.message,
            "error_code": exc.error_code or "VALIDATION_ERROR",
            "details": exc.details,
            "timestamp": datetime.now().isoformat(),
        },
    )


async def service_unavailable_error_handler(exc: ServiceUnavailableError) -> JSONResponse:
    """Handle service unavailable errors."""
    logger.error("Service unavailable: %s", exc.message)

    return JSONResponse(
        status_code=503,
        content={
            "error": "External service unavailable",
            "message": exc.message,
            "error_code": exc.error_code or "SERVICE_UNAVAILABLE",
            "timestamp": datetime.now().isoformat(),
        },
    )


async def agent_error_handler(exc: AgentError) -> JSONResponse:
    """Handle agent errors."""
    logger.error("Agent error: %s", exc.message)

    return JSONResponse(
        status_code=500,
        content={
            "error": "AI agent error",
            "message": "The AI verification agent encountered an error. Please try again.",
            "error_code": exc.error_code or "AGENT_ERROR",
            "timestamp": datetime.now().isoformat(),
        },
    )


async def tool_error_handler(exc: ToolError) -> JSONResponse:
    """Handle tool errors."""
    logger.error("Tool error: %s", exc.message)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Tool execution error",
            "message": "A verification tool encountered an error. The analysis may be incomplete.",
            "error_code": exc.error_code or "TOOL_ERROR",
            "timestamp": datetime.now().isoformat(),
        },
    )


async def http_exception_handler(exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    logger.warning("HTTP exception: %s - %s", exc.status_code, exc.detail)

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP error",
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
        },
    )


async def general_exception_handler(exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    logger.error("Unhandled exception: %s", str(exc), exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "error_code": "INTERNAL_SERVER_ERROR",
            "timestamp": datetime.now().isoformat(),
        },
    )


async def cache_error_handler(exc: CacheError) -> JSONResponse:
    """Handle cache errors."""
    logger.error("Cache error: %s", exc.message)
    return JSONResponse(
        status_code=503,
        content={
            "error": "Cache service error",
            "message": "The caching service encountered an issue. Some operations may be slower.",
            "error_code": exc.error_code or "CACHE_ERROR",
            "timestamp": datetime.now().isoformat(),
        },
    )


async def embedding_error_handler(exc: EmbeddingError) -> JSONResponse:
    """Handle embedding errors."""
    logger.error("Embedding error: %s", exc.message)
    return JSONResponse(
        status_code=503,
        content={
            "error": "Embedding service error",
            "message": "The embedding service is temporarily unavailable.",
            "error_code": exc.error_code or "EMBEDDING_ERROR",
            "timestamp": datetime.now().isoformat(),
        },
    )


async def graph_error_handler(exc: GraphError) -> JSONResponse:
    """Handle graph errors."""
    logger.error("Graph error: %s", exc.message)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Graph operation error",
            "message": "Graph verification encountered an error.",
            "error_code": exc.error_code or "GRAPH_ERROR",
            "timestamp": datetime.now().isoformat(),
        },
    )


async def relevance_error_handler(exc: RelevanceError) -> JSONResponse:
    """Handle relevance errors."""
    logger.error("Relevance error: %s", exc.message)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Relevance scoring error",
            "message": "Relevance analysis encountered an error.",
            "error_code": exc.error_code or "RELEVANCE_ERROR",
            "timestamp": datetime.now().isoformat(),
        },
    )


async def pipeline_error_handler(exc: PipelineError) -> JSONResponse:
    """Handle pipeline errors."""
    logger.error("Pipeline error: %s", exc.message)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Pipeline execution error",
            "message": "The verification pipeline encountered an error.",
            "error_code": exc.error_code or "PIPELINE_ERROR",
            "timestamp": datetime.now().isoformat(),
        },
    )


async def analysis_error_handler(exc: AnalysisError) -> JSONResponse:
    """Handle analysis errors."""
    logger.error("Analysis error: %s", exc.message)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Analysis error",
            "message": "Content analysis encountered an error.",
            "error_code": exc.error_code or "ANALYSIS_ERROR",
            "timestamp": datetime.now().isoformat(),
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
    CacheError: cache_error_handler,
    EmbeddingError: embedding_error_handler,
    GraphError: graph_error_handler,
    RelevanceError: relevance_error_handler,
    PipelineError: pipeline_error_handler,
    AnalysisError: analysis_error_handler,
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

"""
Main FastAPI application for Veritas with unified basic/enhanced support.
"""

import logging.config
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from agent.services.core.agent_manager import create_agent_manager
from app.config import settings
from app.database import create_db_and_tables
from app.error_handlers import EXCEPTION_HANDLERS
from app.handlers.websocket_handler import websocket_handler
from app.cache import cache_factory
from app.matplotlib_config import configure_matplotlib
from app.routers import reputation, verification
from app.services.progress_manager import initialize_progress_manager
from app.services.verification_service import verification_service

# --- Telemetry Patch ---
# This is a workaround for a bug in chromadb 0.5.3. It prevents telemetry errors
# by replacing the problematic module with a mock before it's ever imported.
patch.dict("sys.modules", {
           "chromadb.telemetry.product.posthog": MagicMock()}).start()
# --- End Telemetry Patch ---

# --- Matplotlib Configuration ---
# Configure matplotlib early to avoid font warnings and improve startup performance
configure_matplotlib()
# --- End Matplotlib Configuration ---


# Configure logging
logging.config.fileConfig(settings.logging_config_file,
                          disable_existing_loggers=False)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Veritas application...")

    # Initialize Cache
    await cache_factory.initialize()
    logger.info("Cache connection pool initialized.")

    # Initialize database
    create_db_and_tables()

    # Initialize vector store (with lazy loading)
    # The actual initialization will happen on the first request
    logger.info("Vector store scheduled for lazy initialization.")

    # Initialize progress manager with websocket manager
    initialize_progress_manager()
    logger.info("Progress manager initialized with WebSocket manager.")

    # Initialize AgentManager
    try:
        agent_manager = await create_agent_manager()
        app_instance.state.agent_manager = agent_manager
        logger.info("AgentManager initialized and attached to app state.")
    except Exception as e:
        logger.error("Failed to initialize AgentManager: %s", e)
        raise

    yield

    logger.info("Shutting down Veritas application...")

    # Close verification service and all its resources
    try:
        await verification_service.close()
        logger.info("Verification service closed successfully.")
    except (ConnectionError, TimeoutError, RuntimeError) as e:
        logger.error("Error closing verification service: %s", e)

    # Disconnect from Cache
    await cache_factory.close()
    logger.info("Cache connection pool closed.")


# Create FastAPI application
app = FastAPI(
    title="Veritas API",
    description="API for Veritas fact-checking agent",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add exception handlers
for exception_type, handler in EXCEPTION_HANDLERS.items():
    app.add_exception_handler(exception_type, handler)

# Include routers
app.include_router(verification.router, prefix="/api/v1",
                   tags=["Verification"])
app.include_router(reputation.router, prefix="/api/v1", tags=["Reputation"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Veritas API is running", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "veritas-api"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket_handler.handle_connection(websocket)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level=str(settings.log_level).lower(),
    )

"""
Main FastAPI application for Veritas.
"""
# --- Telemetry Patch ---
# This is a workaround for a bug in chromadb 0.5.3. It prevents telemetry errors
# by replacing the problematic module with a mock before it's ever imported.
from unittest.mock import patch, MagicMock
patch.dict('sys.modules', {'chromadb.telemetry.product.posthog': MagicMock()}).start()
# --- End Telemetry Patch ---

import logging.config
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from unittest.mock import MagicMock, patch

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import Base, async_engine, create_db_and_tables
from app.error_handlers import EXCEPTION_HANDLERS, setup_error_handlers
from agent.vector_store import vector_store
from app.redis_client import redis_manager
from app.routers import verification, reputation
from app.handlers.websocket_handler import websocket_handler
from agent.services.agent_manager import create_agent_manager


# Configure logging
logging.config.fileConfig(settings.logging_config_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Veritas application...")
    
    # Initialize Redis
    await redis_manager.init_redis()
    logger.info("Redis connection pool initialized.")
    
    # Initialize database
    create_db_and_tables()
    
    # Initialize vector store (with lazy loading)
    # The actual initialization will happen on the first request
    logger.info("Vector store scheduled for lazy initialization.")
    
    # Initialize AgentManager
    try:
        agent_manager = await create_agent_manager()
        app.state.agent_manager = agent_manager
        logger.info("AgentManager initialized and attached to app state.")
    except Exception as e:
        logger.error(f"Failed to initialize AgentManager: {e}")
        raise
    
    yield
    
    logger.info("Shutting down Veritas application...")
    # Disconnect from Redis
    await redis_manager.close()
    logger.info("Redis connection pool closed.")
    
    # Clean up the telemetry patch
    if 'posthog_patcher' in globals():
        posthog_patcher.stop()
        logger.info("Removed ChromaDB telemetry monkey-patch.")


# Create FastAPI application
app = FastAPI(
    title="Veritas API",
    description="API for Veritas fact-checking agent",
    version="0.1.0",
    lifespan=lifespan
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
app.include_router(verification.router, prefix="/api/v1", tags=["Verification"])
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
        log_level=settings.log_level.lower()
    )

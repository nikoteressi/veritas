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
from datetime import datetime
from typing import AsyncGenerator
from unittest.mock import MagicMock, patch

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import Base, async_engine, create_db_and_tables
from app.error_handlers import EXCEPTION_HANDLERS, setup_error_handlers
from agent.llm import llm_manager
from agent.vector_store import vector_store
from app.redis_client import redis_manager
from app.routers import verification, reputation
from app.websocket_manager import connection_manager


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
    
    yield
    
    logger.info("Shutting down Veritas application...")
    # Disconnect from Redis
    await redis_manager.close()
    logger.info("Redis connection pool closed.")
    
    # Clean up the telemetry patch
    if 'posthog_patcher' in globals():
        posthog_patcher.stop()
        print("Removed ChromaDB telemetry monkey-patch.")


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
    session_id = None
    try:
        # Accept connection and get session ID
        session_id = await connection_manager.connect(websocket)

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                logger.info(f"Received WebSocket message from {session_id}: {data}")

                # Parse message (expecting JSON)
                import json
                try:
                    message = json.loads(data)
                    message_type = message.get("type", "unknown")

                    if message_type == "ping":
                        # Respond to ping with pong
                        await connection_manager.send_message(session_id, {
                            "type": "pong",
                            "data": {"timestamp": message.get("timestamp")},
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    elif message_type == "status_request":
                        # Send session status
                        status = connection_manager.get_session_status(session_id)
                        await connection_manager.send_message(session_id, {
                            "type": "status_response",
                            "data": status or {"status": "connected"},
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    else:
                        # Echo unknown messages
                        await connection_manager.send_message(session_id, {
                            "type": "echo",
                            "data": message,
                            "timestamp": datetime.utcnow().isoformat()
                        })

                except json.JSONDecodeError:
                    # Handle non-JSON messages
                    await connection_manager.send_message(session_id, {
                        "type": "error",
                        "data": {"message": "Invalid JSON format"},
                        "timestamp": datetime.utcnow().isoformat()
                    })

            except WebSocketDisconnect:
                logger.info(f"WebSocket client disconnected: {session_id}")
                break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if session_id:
            await connection_manager.disconnect(session_id)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )

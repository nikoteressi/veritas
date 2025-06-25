"""
Main FastAPI application for Veritas.
"""
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.routers import verification, reputation
from app.error_handlers import EXCEPTION_HANDLERS


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("Starting Veritas application...")
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    yield
    
    logger.info("Shutting down Veritas application...")


# Create FastAPI application
app = FastAPI(
    title="Veritas API",
    description="AI-Powered Social Post Verifier",
    version="1.0.0",
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
app.include_router(verification.router, prefix="/api/v1", tags=["verification"])
app.include_router(reputation.router, prefix="/api/v1", tags=["reputation"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Veritas API is running", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "veritas-api"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    from app.websocket_manager import connection_manager

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
            connection_manager.disconnect(session_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )

"""
Database configuration and models for Veritas.
"""
import logging
from datetime import datetime
from typing import AsyncGenerator, Optional

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.config import settings

logger = logging.getLogger(__name__)

# Create database base
Base = declarative_base()


class User(Base):
    """User model for reputation tracking."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    nickname = Column(String(255), unique=True, index=True, nullable=False)
    
    # Reputation counters
    true_count = Column(Integer, default=0)
    partially_true_count = Column(Integer, default=0)
    false_count = Column(Integer, default=0)
    ironic_count = Column(Integer, default=0)
    total_posts_checked = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_checked_date = Column(DateTime, default=datetime.utcnow)
    
    # Warning flags
    warning_issued = Column(Boolean, default=False)
    notification_issued = Column(Boolean, default=False)


class VerificationResult(Base):
    """Model to store verification results."""
    __tablename__ = "verification_results"
    
    id = Column(Integer, primary_key=True, index=True)
    user_nickname = Column(String(255), index=True, nullable=False)
    
    # Post content
    image_hash = Column(String(64), index=True)  # SHA-256 hash of the image
    extracted_text = Column(Text)
    user_prompt = Column(Text)
    
    # Analysis results
    primary_topic = Column(String(100))
    identified_claims = Column(Text)  # JSON string of claims
    verdict = Column(String(50))  # true, partially_true, false, ironic
    justification = Column(Text)
    confidence_score = Column(Integer)  # 0-100
    
    # Processing metadata
    processing_time_seconds = Column(Integer)
    model_used = Column(String(100))
    tools_used = Column(Text)  # JSON string of tools used
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)


# Database engine and session setup
engine = None
async_engine = None
SessionLocal = None
AsyncSessionLocal = None


def init_sync_db():
    """Initialize synchronous database connection."""
    global engine, SessionLocal
    
    # Use synchronous PostgreSQL URL
    sync_db_url = settings.database_url
    
    engine = create_engine(sync_db_url, echo=settings.debug)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    logger.info("Synchronous database initialized")


async def init_db():
    """Initialize asynchronous database connection."""
    global async_engine, AsyncSessionLocal
    
    # Use asynchronous PostgreSQL URL
    async_db_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")
    
    async_engine = create_async_engine(async_db_url, echo=settings.debug)
    AsyncSessionLocal = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    # Create tables
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Asynchronous database initialized")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


def get_sync_db():
    """Get synchronous database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

"""
Database configuration and models for Veritas.
"""

import logging
from collections.abc import AsyncGenerator
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base

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
    verdict = Column(Text)  # true, partially_true, false, ironic
    justification = Column(Text)
    confidence_score = Column(Integer)  # 0-100

    # Processing metadata
    processing_time_seconds = Column(Integer)
    vision_model_used = Column(String(100))
    reasoning_model_used = Column(String(100))
    tools_used = Column(Text)  # JSON string of tools used

    # Reputation data
    reputation_data = Column(Text)  # JSON string of reputation information

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)


# Asynchronous database setup
async_db_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")
sync_db_url = settings.database_url

async_engine = create_async_engine(async_db_url)
sync_engine = create_engine(sync_db_url)

async_session_factory = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


def create_db_and_tables():
    """Create all database tables."""
    try:
        logger.info("Creating database tables if they don't exist...")
        Base.metadata.create_all(bind=sync_engine)
        logger.info("Database tables created successfully.")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get an async database session."""
    async with async_session_factory() as session:
        yield session

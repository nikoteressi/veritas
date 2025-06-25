"""
CRUD operations for database models.
"""
import logging
from datetime import datetime
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from app.database import User, VerificationResult
from app.config import settings

logger = logging.getLogger(__name__)


class UserCRUD:
    """CRUD operations for User model."""
    
    @staticmethod
    async def get_user_by_nickname(db: AsyncSession, nickname: str) -> Optional[User]:
        """Get user by nickname."""
        result = await db.execute(select(User).where(User.nickname == nickname))
        return result.scalar_one_or_none()
    
    @staticmethod
    async def create_user(db: AsyncSession, nickname: str) -> User:
        """Create a new user."""
        user = User(nickname=nickname)
        db.add(user)
        try:
            await db.commit()
            await db.refresh(user)
            logger.info(f"Created new user: {nickname}")
            return user
        except IntegrityError:
            await db.rollback()
            # User might have been created by another request
            return await UserCRUD.get_user_by_nickname(db, nickname)
    
    @staticmethod
    async def get_or_create_user(db: AsyncSession, nickname: str) -> User:
        """Get existing user or create new one."""
        user = await UserCRUD.get_user_by_nickname(db, nickname)
        if not user:
            user = await UserCRUD.create_user(db, nickname)
        return user
    
    @staticmethod
    async def update_user_reputation(
        db: AsyncSession, 
        nickname: str, 
        verdict: str
    ) -> User:
        """Update user reputation based on verification verdict."""
        user = await UserCRUD.get_or_create_user(db, nickname)
        
        # Update counters based on verdict
        if verdict == "true":
            user.true_count += 1
        elif verdict == "partially_true":
            user.partially_true_count += 1
        elif verdict == "false":
            user.false_count += 1
        elif verdict == "ironic":
            user.ironic_count += 1
        
        user.total_posts_checked += 1
        user.last_checked_date = datetime.utcnow()
        
        # Check for warning/notification thresholds
        if user.false_count >= settings.warning_threshold and not user.warning_issued:
            user.warning_issued = True
            logger.warning(f"Warning threshold reached for user: {nickname}")
        
        if user.false_count >= settings.notification_threshold and not user.notification_issued:
            user.notification_issued = True
            logger.warning(f"Notification threshold reached for user: {nickname}")
        
        await db.commit()
        await db.refresh(user)
        
        logger.info(f"Updated reputation for {nickname}: {verdict}")
        return user
    
    @staticmethod
    async def get_users_with_warnings(db: AsyncSession) -> List[User]:
        """Get all users who have received warnings."""
        result = await db.execute(
            select(User).where(User.warning_issued == True)
        )
        return result.scalars().all()


class VerificationResultCRUD:
    """CRUD operations for VerificationResult model."""
    
    @staticmethod
    async def create_verification_result(
        db: AsyncSession,
        user_nickname: str,
        image_hash: str,
        extracted_text: str,
        user_prompt: str,
        primary_topic: str,
        identified_claims: str,
        verdict: str,
        justification: str,
        confidence_score: int,
        processing_time_seconds: int,
        model_used: str,
        tools_used: str
    ) -> VerificationResult:
        """Create a new verification result."""
        result = VerificationResult(
            user_nickname=user_nickname,
            image_hash=image_hash,
            extracted_text=extracted_text,
            user_prompt=user_prompt,
            primary_topic=primary_topic,
            identified_claims=identified_claims,
            verdict=verdict,
            justification=justification,
            confidence_score=confidence_score,
            processing_time_seconds=processing_time_seconds,
            model_used=model_used,
            tools_used=tools_used
        )
        
        db.add(result)
        await db.commit()
        await db.refresh(result)
        
        logger.info(f"Created verification result for {user_nickname}: {verdict}")
        return result
    
    @staticmethod
    async def get_verification_results_by_user(
        db: AsyncSession, 
        user_nickname: str,
        limit: int = 10
    ) -> List[VerificationResult]:
        """Get recent verification results for a user."""
        result = await db.execute(
            select(VerificationResult)
            .where(VerificationResult.user_nickname == user_nickname)
            .order_by(VerificationResult.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()
    
    @staticmethod
    async def get_verification_result_by_id(
        db: AsyncSession, 
        result_id: int
    ) -> Optional[VerificationResult]:
        """Get verification result by ID."""
        result = await db.execute(
            select(VerificationResult).where(VerificationResult.id == result_id)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_recent_results(
        db: AsyncSession, 
        limit: int = 50
    ) -> List[VerificationResult]:
        """Get recent verification results across all users."""
        result = await db.execute(
            select(VerificationResult)
            .order_by(VerificationResult.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

"""
CRUD operations for database models.
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import User, VerificationResult

logger = logging.getLogger(__name__)


class UserCRUD:
    """CRUD operations for User model."""

    @staticmethod
    async def get_user_by_nickname(db: AsyncSession, nickname: str) -> User | None:
        """Get user by nickname."""
        result = await db.execute(select(User).where(User.nickname == nickname))
        return result.scalar_one_or_none()

    @staticmethod
    async def create_user(db: AsyncSession, nickname: str) -> User:
        """Create a new user."""
        user = User(nickname=nickname)
        db.add(user)
        await db.flush()
        await db.refresh(user)
        logger.info(f"Created new user: {nickname}")
        return user

    @staticmethod
    async def get_or_create_user(db: AsyncSession, nickname: str) -> User:
        """Get a user by nickname, or create it if it doesn't exist."""
        # Use PostgreSQL's INSERT ... ON CONFLICT DO NOTHING for atomic operation
        stmt = insert(User).values(nickname=nickname)
        stmt = stmt.on_conflict_do_nothing(index_elements=["nickname"])
        await db.execute(stmt)
        await db.flush()

        # Now fetch the user (either existing or newly created)
        result = await db.execute(select(User).filter(User.nickname == nickname))
        user = result.scalars().first()

        if not user:
            # This should not happen, but handle it gracefully
            logger.error(f"Failed to get or create user: {nickname}")
            raise RuntimeError(f"Could not get or create user: {nickname}")

        logger.debug(f"Retrieved user: {nickname}")
        return user

    @staticmethod
    async def update_user_reputation(db: AsyncSession, nickname: str, verdict: str) -> User | None:
        """Update a user's reputation based on the verification verdict."""
        user = await UserCRUD.get_or_create_user(db, nickname)
        if not user:
            return None

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
        if user.false_count >= 10 and not user.warning_issued:
            user.warning_issued = True
            logger.warning(f"Warning threshold reached for user: {nickname}")

        if user.false_count >= 20 and not user.notification_issued:
            user.notification_issued = True
            logger.warning(
                f"Notification threshold reached for user: {nickname}")

        await db.flush()
        await db.refresh(user)

        logger.info(f"Updated reputation for {user.nickname}: {verdict}")
        return user

    @staticmethod
    async def get_users_with_warnings(db: AsyncSession) -> list[User]:
        """Get all users who have received warnings."""
        result = await db.execute(select(User).where(User.warning_issued))
        return result.scalars().all()


class VerificationResultCRUD:
    """CRUD operations for VerificationResult model."""

    @staticmethod
    async def create_verification_result(db: AsyncSession, result_data: dict) -> VerificationResult:
        """Create a new verification result."""
        result = VerificationResult(**result_data)
        db.add(result)
        await db.flush()
        await db.refresh(result)

        logger.info(
            f"Created verification result for {result.user_nickname}: {result.verdict}")
        return result

    @staticmethod
    async def get_verification_results_by_user(
        db: AsyncSession, user_nickname: str, limit: int = 10
    ) -> list[VerificationResult]:
        """Get recent verification results for a user."""
        result = await db.execute(
            select(VerificationResult)
            .where(VerificationResult.user_nickname == user_nickname)
            .order_by(VerificationResult.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

    @staticmethod
    async def get_verification_result_by_id(db: AsyncSession, result_id: int) -> VerificationResult | None:
        """Get a verification result by its ID."""
        result = await db.execute(select(VerificationResult).filter(VerificationResult.id == result_id))
        return result.scalars().first()

    @staticmethod
    async def get_recent_results(db: AsyncSession, limit: int = 50) -> list[VerificationResult]:
        """Get recent verification results across all users."""
        result = await db.execute(
            select(VerificationResult).order_by(
                VerificationResult.created_at.desc()).limit(limit)
        )
        return result.scalars().all()

    @staticmethod
    async def get_all_verification_results(
        db: AsyncSession,
    ) -> list[VerificationResult]:
        """Get all verification results."""
        result = await db.execute(select(VerificationResult).order_by(VerificationResult.created_at.desc()))
        return result.scalars().all()

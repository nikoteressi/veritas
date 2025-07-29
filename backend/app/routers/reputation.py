"""
User reputation endpoints.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import User, get_db

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/user-reputation/{nickname}")
async def get_user_reputation(nickname: str, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """
    Get reputation information for a user.

    Args:
        nickname: User's nickname/username
        db: Database session

    Returns:
        User reputation data
    """
    try:
        # Query user from database
        result = await db.execute(select(User).where(User.nickname == nickname))
        user = result.scalar_one_or_none()

        if not user:
            # Return default reputation for new user
            return {
                "nickname": nickname,
                "true_count": 0,
                "partially_true_count": 0,
                "false_count": 0,
                "ironic_count": 0,
                "total_posts_checked": 0,
                "warning_issued": False,
                "notification_issued": False,
                "created_at": None,
                "last_checked_date": None,
            }

        return {
            "nickname": user.nickname,
            "true_count": user.true_count,
            "partially_true_count": user.partially_true_count,
            "false_count": user.false_count,
            "ironic_count": user.ironic_count,
            "total_posts_checked": user.total_posts_checked,
            "warning_issued": user.warning_issued,
            "notification_issued": user.notification_issued,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_checked_date": (user.last_checked_date.isoformat() if user.last_checked_date else None),
        }

    except Exception as e:
        logger.error(f"Error getting user reputation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reputation-stats")
async def get_reputation_stats(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """
    Get overall reputation statistics.

    Args:
        db: Database session

    Returns:
        Overall statistics
    """
    try:
        from sqlalchemy import func

        # Single optimized query to get all statistics at once
        stats_result = await db.execute(
            select(
                func.count(User.id).label("total_users"),
                func.sum(User.total_posts_checked).label("total_posts"),
                func.sum(User.true_count).label("true_posts"),
                func.sum(User.false_count).label("false_posts"),
                func.sum(User.partially_true_count).label("partially_true"),
                func.sum(User.ironic_count).label("ironic_posts"),
                func.sum(func.case((User.warning_issued == True, 1), else_=0)).label("users_with_warnings"),
                func.sum(func.case((User.notification_issued == True, 1), else_=0)).label("users_with_notifications"),
            )
        )

        stats = stats_result.first()

        # Extract values with defaults
        total_users = stats.total_users or 0
        total_posts = stats.total_posts or 0
        true_posts = stats.true_posts or 0
        false_posts = stats.false_posts or 0
        partially_true = stats.partially_true or 0
        ironic_posts = stats.ironic_posts or 0
        users_with_warnings = stats.users_with_warnings or 0
        users_with_notifications = stats.users_with_notifications or 0

        # Calculate percentages
        accuracy_rate = (true_posts + partially_true) / total_posts * 100 if total_posts > 0 else 0
        true_percentage = true_posts / total_posts * 100 if total_posts > 0 else 0
        false_percentage = false_posts / total_posts * 100 if total_posts > 0 else 0
        partially_true_percentage = partially_true / total_posts * 100 if total_posts > 0 else 0
        ironic_percentage = ironic_posts / total_posts * 100 if total_posts > 0 else 0

        return {
            "total_users": total_users,
            "total_posts_checked": total_posts,
            "accuracy_rate": round(accuracy_rate, 2),
            "true_posts_percentage": round(true_percentage, 2),
            "false_posts_percentage": round(false_percentage, 2),
            "partially_true_percentage": round(partially_true_percentage, 2),
            "ironic_posts_percentage": round(ironic_percentage, 2),
            "users_with_warnings": users_with_warnings,
            "users_with_notifications": users_with_notifications,
        }

    except Exception as e:
        logger.error(f"Error getting reputation stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users-with-warnings")
async def get_users_with_warnings(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """
    Get list of users who have received warnings.

    Args:
        db: Database session

    Returns:
        List of users with warnings
    """
    try:
        from app.crud import UserCRUD

        users = await UserCRUD.get_users_with_warnings(db)

        user_list = []
        for user in users:
            user_list.append(
                {
                    "nickname": user.nickname,
                    "false_count": user.false_count,
                    "total_posts_checked": user.total_posts_checked,
                    "false_rate": (
                        round(user.false_count / user.total_posts_checked * 100, 2)
                        if user.total_posts_checked > 0
                        else 0
                    ),
                    "warning_issued": user.warning_issued,
                    "notification_issued": user.notification_issued,
                    "last_checked_date": (user.last_checked_date.isoformat() if user.last_checked_date else None),
                }
            )

        return {"users": user_list, "total_count": len(user_list)}

    except Exception as e:
        logger.error(f"Error getting users with warnings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/leaderboard")
async def get_reputation_leaderboard(limit: int = 10, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """
    Get reputation leaderboard showing top users by accuracy.

    Args:
        limit: Number of users to return (default 10)
        db: Database session

    Returns:
        Leaderboard data
    """
    try:
        # Get users with at least 5 posts checked, ordered by accuracy
        result = await db.execute(
            select(User)
            .where(User.total_posts_checked >= 5)
            .order_by(
                ((User.true_count + User.partially_true_count) / User.total_posts_checked).desc(),
                User.total_posts_checked.desc(),
            )
            .limit(limit)
        )
        users = result.scalars().all()

        leaderboard = []
        for i, user in enumerate(users, 1):
            accuracy = (user.true_count + user.partially_true_count) / user.total_posts_checked * 100
            leaderboard.append(
                {
                    "rank": i,
                    "nickname": user.nickname,
                    "accuracy_rate": round(accuracy, 2),
                    "total_posts_checked": user.total_posts_checked,
                    "true_count": user.true_count,
                    "partially_true_count": user.partially_true_count,
                    "false_count": user.false_count,
                    "ironic_count": user.ironic_count,
                }
            )

        return {"leaderboard": leaderboard, "total_users": len(leaderboard)}

    except Exception as e:
        logger.error(f"Error getting leaderboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

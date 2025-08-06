"""
Service for managing user reputation.
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.crud import UserCRUD

logger = logging.getLogger(__name__)


class ReputationService:
    """Service to manage user reputation scores."""

    async def get_or_create(self, db: AsyncSession, nickname: str) -> Any:
        """Get or create a user's reputation entry."""
        return await UserCRUD.get_or_create_user(db, nickname)

    async def update(self, db: AsyncSession, nickname: str, verdict: str) -> Any:
        """Update a user's reputation based on a new verdict."""
        return await UserCRUD.update_user_reputation(db, nickname, verdict)

    def generate_warnings(self, user_reputation: Any) -> list[str]:
        """Generate warnings based on the user's reputation."""
        warnings = []
        if user_reputation.warning_issued:
            warnings.append("A warning has been issued for repeatedly submitting content that was found to be false.")
        if user_reputation.notification_issued:
            warnings.append("A notification has been sent regarding a high volume of false content submissions.")
        return warnings


# Singleton instance
reputation_service = ReputationService()

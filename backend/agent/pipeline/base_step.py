"""
Base pipeline step for the verification workflow.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from agent.models.verification_context import VerificationContext
from app.exceptions import AgentError


class BasePipelineStep(ABC):
    """Base class for verification pipeline steps."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    async def execute(self, context: VerificationContext) -> VerificationContext:
        """
        Execute the verification step.

        Args:
            context: Verification context containing all necessary data

        Returns:
            Updated verification context
        """

    async def safe_execute(self, context: VerificationContext) -> VerificationContext:
        """
        Safely execute the step with error handling and logging.

        Args:
            context: Verification context containing all necessary data

        Returns:
            Updated verification context
        """
        self.logger.info("Starting %s step", self.name)
        try:
            result_context = await self.execute(context)
            self.logger.info("Completed %s step successfully", self.name)
            return result_context
        except Exception as e:
            self.logger.error("Failed %s step: %s",
                              self.name, e, exc_info=True)
            raise AgentError(f"Failed {self.name} step: {e}") from e

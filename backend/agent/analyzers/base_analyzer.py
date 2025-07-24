"""
Base analyzer class for verification pipeline analyzers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from agent.models.verification_context import VerificationContext


class BaseAnalyzer(ABC):
    """
    Base class for verification pipeline analyzers.

    All analyzers should inherit from this class and implement the analyze method.
    """

    def __init__(self, analyzer_name: str):
        """
        Initialize the analyzer.

        Args:
            analyzer_name: Name of the analyzer for logging purposes
        """
        self.analyzer_name = analyzer_name
        self.logger = logging.getLogger(f"{__name__}.{analyzer_name}")

    @abstractmethod
    async def analyze(self, context: VerificationContext) -> dict[str, Any]:
        """
        Perform analysis on the verification context.

        Args:
            context: Verification context containing all necessary data

        Returns:
            Analysis results dictionary
        """
        pass

    def _log_analysis_start(self) -> None:
        """Log the start of analysis."""
        self.logger.info(f"Starting {self.analyzer_name} analysis")

    def _log_analysis_complete(self, result: dict[str, Any]) -> None:
        """Log the completion of analysis."""
        self.logger.info(f"Completed {self.analyzer_name} analysis")
        self.logger.debug(f"{self.analyzer_name} analysis result: {result}")

    def _log_analysis_error(self, error: Exception) -> None:
        """Log analysis error."""
        self.logger.error(
            f"Failed {self.analyzer_name} analysis: {error}", exc_info=True
        )

    async def safe_analyze(self, context: VerificationContext) -> dict[str, Any]:
        """
        Safely perform analysis with error handling and logging.

        Args:
            context: Verification context containing all necessary data

        Returns:
            Analysis results dictionary, or error result if analysis fails
        """
        try:
            self._log_analysis_start()
            result = await self.analyze(context)
            self._log_analysis_complete(result)
            return result
        except Exception as e:
            self._log_analysis_error(e)
            return self._get_error_result(e)

    def _get_error_result(self, error: Exception) -> dict[str, Any]:
        """
        Get error result for failed analysis.

        Args:
            error: The exception that occurred

        Returns:
            Error result dictionary
        """
        return {
            "error": True,
            "error_message": str(error),
            "analyzer": self.analyzer_name,
        }

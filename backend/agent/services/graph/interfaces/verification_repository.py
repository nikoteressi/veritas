"""
Abstract repository interface for verification results.

Defines the contract for verification data access layer using Repository pattern.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from agent.models.graph_verification_models import VerificationResponse


class VerificationRepository(ABC):
    """
    Abstract repository for verification results operations.

    Implements the Repository pattern to abstract verification data access
    and provide a clean interface for verification persistence operations.
    """

    @abstractmethod
    async def save_verification_result(
        self, node_id: str, graph_id: str, result: VerificationResponse, metadata: dict[str, Any] | None = None
    ) -> str:
        """
        Save a verification result for a specific node.

        Args:
            node_id: Unique node identifier
            graph_id: Unique graph identifier
            result: Verification response
            metadata: Optional metadata to associate with the result

        Returns:
            Unique verification result identifier

        Raises:
            RepositoryError: If save operation fails
        """

    @abstractmethod
    async def load_verification_result(self, result_id: str) -> VerificationResponse | None:
        """
        Load a verification result by identifier.

        Args:
            result_id: Unique verification result identifier

        Returns:
            Verification response or None if not found

        Raises:
            RepositoryError: If load operation fails
        """

    @abstractmethod
    async def get_node_verification_history(self, node_id: str) -> list[dict[str, Any]]:
        """
        Get verification history for a specific node.

        Args:
            node_id: Unique node identifier

        Returns:
            List of verification result summaries ordered by timestamp

        Raises:
            RepositoryError: If retrieval operation fails
        """

    @abstractmethod
    async def get_graph_verification_summary(self, graph_id: str) -> dict[str, Any]:
        """
        Get verification summary for an entire graph.

        Args:
            graph_id: Unique graph identifier

        Returns:
            Graph verification summary with statistics

        Raises:
            RepositoryError: If summary generation fails
        """

    @abstractmethod
    async def find_verification_results_by_verdict(
        self, verdict: str, graph_id: str | None = None, confidence_threshold: float | None = None
    ) -> list[str]:
        """
        Find verification results by verdict criteria.

        Args:
            verdict: Verification verdict to search for
            graph_id: Optional graph identifier to limit search
            confidence_threshold: Optional minimum confidence threshold

        Returns:
            List of verification result identifiers matching criteria

        Raises:
            RepositoryError: If search operation fails
        """

    @abstractmethod
    async def get_verification_statistics(
        self, graph_id: str | None = None, time_range: tuple[datetime, datetime] | None = None
    ) -> dict[str, Any]:
        """
        Get verification statistics with optional filtering.

        Args:
            graph_id: Optional graph identifier to limit statistics
            time_range: Optional tuple of (start_time, end_time) for filtering

        Returns:
            Verification statistics dictionary

        Raises:
            RepositoryError: If statistics generation fails
        """

    @abstractmethod
    async def save_cross_verification_result(
        self,
        node_ids: list[str],
        graph_id: str,
        cross_verification_data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save cross-verification results between multiple nodes.

        Args:
            node_ids: List of node identifiers involved in cross-verification
            graph_id: Unique graph identifier
            cross_verification_data: Cross-verification analysis results
            metadata: Optional metadata to associate with the result

        Returns:
            Unique cross-verification result identifier

        Raises:
            RepositoryError: If save operation fails
        """

    @abstractmethod
    async def get_cross_verification_results(self, node_id: str) -> list[dict[str, Any]]:
        """
        Get all cross-verification results involving a specific node.

        Args:
            node_id: Unique node identifier

        Returns:
            List of cross-verification result summaries

        Raises:
            RepositoryError: If retrieval operation fails
        """

    @abstractmethod
    async def save_cluster_verification_result(
        self,
        cluster_id: str,
        graph_id: str,
        cluster_results: dict[str, VerificationResponse],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save verification results for an entire cluster.

        Args:
            cluster_id: Unique cluster identifier
            graph_id: Unique graph identifier
            cluster_results: Dictionary mapping node IDs to verification responses
            metadata: Optional metadata to associate with the result

        Returns:
            Unique cluster verification result identifier

        Raises:
            RepositoryError: If save operation fails
        """

    @abstractmethod
    async def get_cluster_verification_results(self, cluster_id: str) -> dict[str, VerificationResponse] | None:
        """
        Get verification results for a specific cluster.

        Args:
            cluster_id: Unique cluster identifier

        Returns:
            Dictionary mapping node IDs to verification responses or None if not found

        Raises:
            RepositoryError: If retrieval operation fails
        """

    @abstractmethod
    async def delete_verification_results(self, graph_id: str) -> bool:
        """
        Delete all verification results for a specific graph.

        Args:
            graph_id: Unique graph identifier

        Returns:
            True if deletion was successful, False otherwise

        Raises:
            RepositoryError: If delete operation fails
        """

    @abstractmethod
    async def archive_old_verification_results(self, cutoff_date: datetime, archive_location: str) -> int:
        """
        Archive verification results older than cutoff date.

        Args:
            cutoff_date: Date before which results should be archived
            archive_location: Location to store archived results

        Returns:
            Number of results archived

        Raises:
            RepositoryError: If archive operation fails
        """

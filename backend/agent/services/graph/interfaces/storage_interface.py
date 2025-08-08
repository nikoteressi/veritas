"""
Storage interface for dependency injection.

Defines the contract for storage services used in graph persistence.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class StorageInterface(ABC):
    """
    Abstract interface for storage services.

    This interface abstracts the storage functionality to enable
    dependency injection and easier testing.
    """

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the storage backend.

        Raises:
            StorageError: If connection fails
        """

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close connection to the storage backend.

        Raises:
            StorageError: If disconnection fails
        """

    @abstractmethod
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute a query against the storage backend.

        Args:
            query: Query string to execute
            parameters: Optional query parameters

        Returns:
            Query result

        Raises:
            StorageError: If query execution fails
        """

    @abstractmethod
    async def execute_transaction(self, queries: list[tuple[str, Optional[Dict[str, Any]]]]) -> Any:
        """
        Execute multiple queries in a transaction.

        Args:
            queries: List of (query, parameters) tuples

        Returns:
            Transaction result

        Raises:
            StorageError: If transaction fails
        """

    @abstractmethod
    def get_driver(self) -> Any:
        """
        Get the underlying storage driver.

        Returns:
            Storage driver instance
        """

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if storage is connected.

        Returns:
            True if connected, False otherwise
        """

"""
Configuration interface for dependency injection.

Defines the contract for configuration services used throughout the graph system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class ConfigInterface(ABC):
    """
    Abstract interface for configuration services.

    This interface abstracts the configuration functionality to enable
    dependency injection and easier testing.
    """

    @abstractmethod
    def get_config(self, key: str) -> Any:
        """
        Get a configuration value by key.

        Args:
            key: Configuration key (supports dot notation for nested keys)

        Returns:
            Configuration value

        Raises:
            ConfigError: If key is not found
        """

    @abstractmethod
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.

        Args:
            section: Section name

        Returns:
            Dictionary containing all keys in the section

        Raises:
            ConfigError: If section is not found
        """

    @abstractmethod
    def get_config_with_default(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with a default fallback.

        Args:
            key: Configuration key
            default: Default value if key is not found

        Returns:
            Configuration value or default
        """

    @abstractmethod
    def has_config(self, key: str) -> bool:
        """
        Check if a configuration key exists.

        Args:
            key: Configuration key to check

        Returns:
            True if key exists, False otherwise
        """

    @abstractmethod
    def get_embedding_config(self) -> Dict[str, Any]:
        """
        Get embedding-specific configuration.

        Returns:
            Dictionary containing embedding configuration
        """

    @abstractmethod
    def get_storage_config(self) -> Dict[str, Any]:
        """
        Get storage-specific configuration.

        Returns:
            Dictionary containing storage configuration
        """

    @abstractmethod
    def get_verification_config(self) -> Dict[str, Any]:
        """
        Get verification-specific configuration.

        Returns:
            Dictionary containing verification configuration
        """

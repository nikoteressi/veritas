"""
Configuration Provider.

Implements ConfigInterface using application settings and configuration files.
"""

import logging
from typing import Any, Dict
import os

from agent.services.graph.interfaces.config_interface import ConfigInterface
from agent.services.graph.dependency_injection import singleton

logger = logging.getLogger(__name__)


@singleton()
class ConfigProvider(ConfigInterface):
    """
    Configuration provider.

    Provides access to application configuration from various sources
    including environment variables, settings files, and default values.
    """

    def __init__(self):
        """Initialize configuration provider."""
        self._config_cache: Dict[str, Any] = {}
        self._logger = logging.getLogger(__name__)

        # Load configuration
        self._load_configuration()

    def _load_configuration(self) -> None:
        """Load configuration from various sources."""
        try:
            # Load from environment variables and defaults
            self._config_cache = {
                'embedding': {
                    'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
                    'model_name': os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text'),
                    'timeout': int(os.getenv('OLLAMA_TIMEOUT', '30')),
                    'max_retries': int(os.getenv('OLLAMA_MAX_RETRIES', '3'))
                },
                'storage': {
                    'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                    'username': os.getenv('NEO4J_USERNAME', 'neo4j'),
                    'password': os.getenv('NEO4J_PASSWORD', 'password'),
                    'database': os.getenv('NEO4J_DATABASE', 'neo4j'),
                    'max_connection_lifetime': int(os.getenv('NEO4J_MAX_CONNECTION_LIFETIME', '3600')),
                    'max_connection_pool_size': int(os.getenv('NEO4J_MAX_CONNECTION_POOL_SIZE', '50')),
                    'connection_acquisition_timeout': int(os.getenv('NEO4J_CONNECTION_TIMEOUT', '60'))
                },
                'verification': {
                    'max_concurrent_verifications': int(os.getenv('MAX_CONCURRENT_VERIFICATIONS', '5')),
                    'verification_timeout': int(os.getenv('VERIFICATION_TIMEOUT', '300')),
                    'evidence_gathering_timeout': int(os.getenv('EVIDENCE_GATHERING_TIMEOUT', '120')),
                    'max_evidence_sources': int(os.getenv('MAX_EVIDENCE_SOURCES', '10')),
                    'similarity_threshold': float(os.getenv('SIMILARITY_THRESHOLD', '0.8')),
                    'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))
                },
                'clustering': {
                    'min_cluster_size': int(os.getenv('MIN_CLUSTER_SIZE', '2')),
                    'max_cluster_size': int(os.getenv('MAX_CLUSTER_SIZE', '20')),
                    'similarity_threshold': float(os.getenv('CLUSTERING_SIMILARITY_THRESHOLD', '0.75')),
                    'algorithm': os.getenv('CLUSTERING_ALGORITHM', 'hierarchical')
                },
                'logging': {
                    'level': os.getenv('LOG_LEVEL', 'INFO'),
                    'format': os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
                    'file_path': os.getenv('LOG_FILE_PATH', 'logs/veritas.log')
                }
            }

            # Try to load from settings file if available
            self._load_from_settings_file()

            self._logger.info("Configuration loaded successfully")

        except Exception as e:
            self._logger.error("Failed to load configuration: %s", e)
            raise

    def _load_from_settings_file(self) -> None:
        """Load configuration from settings file if available."""
        try:
            # Try to import settings from the application
            try:
                from app.core.config import settings

                # Update with settings values if available
                if hasattr(settings, 'ollama_base_url'):
                    self._config_cache['embedding']['base_url'] = settings.ollama_base_url

                if hasattr(settings, 'neo4j_uri'):
                    self._config_cache['storage']['uri'] = settings.neo4j_uri

                if hasattr(settings, 'neo4j_username'):
                    self._config_cache['storage']['username'] = settings.neo4j_username

                if hasattr(settings, 'neo4j_password'):
                    self._config_cache['storage']['password'] = settings.neo4j_password

                if hasattr(settings, 'neo4j_database'):
                    self._config_cache['storage']['database'] = settings.neo4j_database

                self._logger.debug("Loaded configuration from settings file")

            except ImportError:
                self._logger.debug(
                    "Settings file not available, using environment variables and defaults")

        except Exception as e:
            self._logger.warning("Failed to load from settings file: %s", e)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation, e.g., 'embedding.base_url')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        try:
            keys = key.split('.')
            value = self._config_cache

            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default

            return value

        except Exception as e:
            self._logger.warning(
                "Failed to get configuration key '%s': %s", key, e)
            return default

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.

        Args:
            section: Section name

        Returns:
            Configuration section as dictionary
        """
        return self._config_cache.get(section, {})

    def has_key(self, key: str) -> bool:
        """
        Check if configuration key exists.

        Args:
            key: Configuration key (supports dot notation)

        Returns:
            True if key exists, False otherwise
        """
        try:
            keys = key.split('.')
            value = self._config_cache

            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return False

            return True

        except Exception:
            return False

    def get_embedding_config(self) -> Dict[str, Any]:
        """
        Get embedding-specific configuration.

        Returns:
            Embedding configuration dictionary
        """
        return self.get_section('embedding')

    def get_storage_config(self) -> Dict[str, Any]:
        """
        Get storage-specific configuration.

        Returns:
            Storage configuration dictionary
        """
        return self.get_section('storage')

    def get_verification_config(self) -> Dict[str, Any]:
        """
        Get verification-specific configuration.

        Returns:
            Verification configuration dictionary
        """
        return self.get_section('verification')

    def get_clustering_config(self) -> Dict[str, Any]:
        """
        Get clustering-specific configuration.

        Returns:
            Clustering configuration dictionary
        """
        return self.get_section('clustering')

    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging-specific configuration.

        Returns:
            Logging configuration dictionary
        """
        return self.get_section('logging')

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        try:
            keys = key.split('.')
            config = self._config_cache

            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]

            # Set the value
            config[keys[-1]] = value

            self._logger.debug(
                "Set configuration key '%s' to '%s'", key, value)

        except Exception as e:
            self._logger.error(
                "Failed to set configuration key '%s': %s", key, e)
            raise

    def update_section(self, section: str, config: Dict[str, Any]) -> None:
        """
        Update entire configuration section.

        Args:
            section: Section name
            config: Configuration dictionary to merge
        """
        try:
            if section not in self._config_cache:
                self._config_cache[section] = {}

            self._config_cache[section].update(config)

            self._logger.debug("Updated configuration section '%s'", section)

        except Exception as e:
            self._logger.error(
                "Failed to update configuration section '%s': %s", section, e)
            raise

    def reload(self) -> None:
        """Reload configuration from sources."""
        try:
            self._config_cache.clear()
            self._load_configuration()
            self._logger.info("Configuration reloaded")

        except Exception as e:
            self._logger.error("Failed to reload configuration: %s", e)
            raise

    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration.

        Returns:
            Complete configuration dictionary
        """
        return self._config_cache.copy()

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check required sections
            required_sections = ['embedding', 'storage', 'verification']
            for section in required_sections:
                if section not in self._config_cache:
                    self._logger.error(
                        "Missing required configuration section: %s", section)
                    return False

            # Check required embedding config
            embedding_config = self.get_embedding_config()
            if not embedding_config.get('base_url'):
                self._logger.error("Missing embedding base_url configuration")
                return False

            # Check required storage config
            storage_config = self.get_storage_config()
            required_storage_keys = ['uri', 'username', 'password', 'database']
            for key in required_storage_keys:
                if not storage_config.get(key):
                    self._logger.error(
                        "Missing storage configuration: %s", key)
                    return False

            self._logger.info("Configuration validation passed")
            return True

        except Exception as e:
            self._logger.error("Configuration validation failed: %s", e)
            return False

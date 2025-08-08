"""
Bootstrap Module for Dependency Injection.

Initializes and configures the DI container with all necessary services
and their dependencies for the graph services module.
"""

import logging
from typing import Optional

from .dependency_injection import DIContainer, ServiceLifetime, set_container
from .interfaces.embedding_interface import EmbeddingInterface
from .interfaces.storage_interface import StorageInterface
from .interfaces.config_interface import ConfigInterface
from .providers.config_provider import ConfigProvider
from .providers.ollama_embedding_provider import OllamaEmbeddingProvider
from .providers.neo4j_storage_provider import Neo4jStorageProvider

logger = logging.getLogger(__name__)

# Global container instance
_container: Optional[DIContainer] = None


def create_container() -> DIContainer:
    """
    Create and configure the DI container.

    Returns:
        Configured DI container
    """
    container = DIContainer()

    # Register core services
    register_core_services(container)

    # Register service providers
    register_service_providers(container)

    # Register graph services
    register_graph_services(container)

    logger.info("DI Container created and configured")
    return container


def register_core_services(container: DIContainer) -> None:
    """Register core infrastructure services."""
    logger.info("Registering core services...")

    # Configuration service (singleton)
    container.register_singleton(ConfigInterface, ConfigProvider)

    logger.info("Core services registered successfully")


def register_service_providers(container: DIContainer) -> None:
    """Register service providers."""
    logger.info("Registering service providers...")

    # Embedding service (singleton for performance)
    container.register_singleton(EmbeddingInterface, OllamaEmbeddingProvider)

    # Storage service (singleton for connection pooling)
    container.register_singleton(StorageInterface, Neo4jStorageProvider)

    logger.info("Service providers registered successfully")


def register_graph_services(container: DIContainer) -> None:
    """Register graph-related services in the DI container."""
    from .graph_builder import GraphBuilder
    from .strategies.storage.neo4j_storage import Neo4jStorageStrategy
    from .verification.engine import EnhancedGraphVerificationEngine

    logger.info("Registering graph services...")

    # Register GraphBuilder
    container.register_type(GraphBuilder, ServiceLifetime.SINGLETON)

    # Register Neo4j Storage Strategy
    container.register_type(Neo4jStorageStrategy, ServiceLifetime.SINGLETON)

    # Register Verification Engine
    container.register_type(
        EnhancedGraphVerificationEngine, ServiceLifetime.SINGLETON)

    logger.info("Graph services registered successfully")


def initialize_di_system() -> DIContainer:
    """Initialize the complete DI system."""
    logger.info("Initializing DI system...")

    # Create container
    container = create_container()

    logger.info("DI system initialized successfully")
    return container


def initialize_di() -> DIContainer:
    """
    Initialize the dependency injection system.

    Returns:
        Initialized DI container

    Raises:
        RuntimeError: If initialization fails
    """
    global _container

    try:
        logger.info("Initializing dependency injection system")

        # Create container
        _container = initialize_di_system()

        # Set global container reference
        set_container(_container)

        # Validate configuration
        config = _container.resolve(ConfigInterface)
        if not config.validate():
            raise RuntimeError("Configuration validation failed")

        # Test core services
        _test_core_services(_container)

        logger.info("Dependency injection system initialized successfully")
        return _container

    except Exception as e:
        logger.error("Failed to initialize dependency injection system: %s", e)
        raise RuntimeError(f"DI initialization failed: {e}")


def _test_core_services(container: DIContainer) -> None:
    """Test that core services can be resolved and are working."""
    try:
        logger.debug("Testing core services")

        # Test configuration service
        config = container.resolve(ConfigInterface)
        if not config.has_key('embedding.base_url'):
            raise RuntimeError("Configuration service not working properly")

        # Test embedding service
        embedding = container.resolve(EmbeddingInterface)
        if not embedding.is_available():
            logger.warning(
                "Embedding service not available - this may be expected in test environments")

        # Test storage service
        storage = container.resolve(StorageInterface)
        if not storage.is_connected():
            logger.warning(
                "Storage service not connected - this may be expected in test environments")

        logger.debug("Core services test completed")

    except Exception as e:
        logger.error("Core services test failed: %s", e)
        raise


def get_di_container() -> DIContainer:
    """
    Get the initialized DI container.

    Returns:
        DI container instance

    Raises:
        RuntimeError: If DI system not initialized
    """
    if _container is None:
        raise RuntimeError(
            "DI system not initialized. Call initialize_di() first.")

    return _container


def shutdown_di() -> None:
    """Shutdown the dependency injection system."""
    global _container

    try:
        logger.info("Shutting down dependency injection system")

        if _container:
            # Dispose of container and all managed instances
            _container.dispose()
            _container = None

        # Clear global container reference
        set_container(None)

        logger.info("Dependency injection system shutdown completed")

    except Exception as e:
        logger.error("Error during DI shutdown: %s", e)


def is_initialized() -> bool:
    """
    Check if DI system is initialized.

    Returns:
        True if initialized, False otherwise
    """
    return _container is not None


def reset_di() -> None:
    """Reset the DI system (useful for testing)."""
    logger.debug("Resetting dependency injection system")

    shutdown_di()
    initialize_di()

    logger.debug("Dependency injection system reset completed")


class DIContextManager:
    """Context manager for DI system lifecycle."""

    def __init__(self):
        self.container = None

    def __enter__(self) -> DIContainer:
        """Initialize DI system."""
        self.container = initialize_di()
        return self.container

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Shutdown DI system."""
        shutdown_di()


def with_di():
    """
    Decorator for functions that need DI system.

    Ensures DI system is initialized before function execution.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not is_initialized():
                initialize_di()
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Convenience functions for common operations
def resolve_config() -> ConfigInterface:
    """Resolve configuration service."""
    return get_di_container().resolve(ConfigInterface)


def resolve_embedding() -> EmbeddingInterface:
    """Resolve embedding service."""
    return get_di_container().resolve(EmbeddingInterface)


def resolve_storage() -> StorageInterface:
    """Resolve storage service."""
    return get_di_container().resolve(StorageInterface)


# Auto-initialization flag (can be disabled for testing)
AUTO_INITIALIZE = True

if AUTO_INITIALIZE and not is_initialized():
    try:
        initialize_di()
        logger.info("DI system auto-initialized")
    except Exception as e:
        logger.warning("DI system auto-initialization failed: %s", e)

"""
Dependency Injection Container.

Provides a comprehensive DI container with support for different service lifetimes,
factory registration, and automatic dependency resolution.
"""

from typing import TypeVar, Dict, Any, Callable, Type, Optional, get_type_hints
import logging
import inspect
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceLifetime(Enum):
    """Service lifetime management options."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


@dataclass
class ServiceDescriptor:
    """Describes a service registration."""
    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT
    dependencies: Optional[list[Type]] = None


class DIContainer:
    """
    Dependency Injection Container.

    Manages service registration, resolution, and lifetime management
    with support for automatic dependency injection.
    """

    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._scoped_instances: Dict[Type, Any] = {}
        self._logger = logging.getLogger(__name__)
        self._resolving: set[Type] = set()  # Circular dependency detection

    def register_singleton(self, service_type: Type[T], implementation_type: Type[T] = None) -> 'DIContainer':
        """
        Register a service as singleton.

        Args:
            service_type: Interface or base type
            implementation_type: Concrete implementation type

        Returns:
            Self for method chaining
        """
        impl_type = implementation_type or service_type
        dependencies = self._extract_dependencies(impl_type)

        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=impl_type,
            lifetime=ServiceLifetime.SINGLETON,
            dependencies=dependencies
        )

        self._services[service_type] = descriptor
        self._logger.debug("Registered singleton: %s -> %s",
                           service_type.__name__, impl_type.__name__)
        return self

    def register_transient(self, service_type: Type[T], implementation_type: Type[T] = None) -> 'DIContainer':
        """
        Register a service as transient.

        Args:
            service_type: Interface or base type
            implementation_type: Concrete implementation type

        Returns:
            Self for method chaining
        """
        impl_type = implementation_type or service_type
        dependencies = self._extract_dependencies(impl_type)

        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=impl_type,
            lifetime=ServiceLifetime.TRANSIENT,
            dependencies=dependencies
        )

        self._services[service_type] = descriptor
        self._logger.debug("Registered transient: %s -> %s",
                           service_type.__name__, impl_type.__name__)
        return self

    def register_scoped(self, service_type: Type[T], implementation_type: Type[T] = None) -> 'DIContainer':
        """
        Register a service as scoped.

        Args:
            service_type: Interface or base type
            implementation_type: Concrete implementation type

        Returns:
            Self for method chaining
        """
        impl_type = implementation_type or service_type
        dependencies = self._extract_dependencies(impl_type)

        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=impl_type,
            lifetime=ServiceLifetime.SCOPED,
            dependencies=dependencies
        )

        self._services[service_type] = descriptor
        self._logger.debug("Registered scoped: %s -> %s",
                           service_type.__name__, impl_type.__name__)
        return self

    def register_factory(self, service_type: Type[T], factory: Callable[[], T]) -> 'DIContainer':
        """
        Register a service with a factory function.

        Args:
            service_type: Service type
            factory: Factory function that creates the service

        Returns:
            Self for method chaining
        """
        descriptor = ServiceDescriptor(
            service_type=service_type,
            factory=factory,
            lifetime=ServiceLifetime.TRANSIENT
        )

        self._services[service_type] = descriptor
        self._logger.debug("Registered factory for: %s", service_type.__name__)
        return self

    def register_instance(self, service_type: Type[T], instance: T) -> 'DIContainer':
        """
        Register a service instance.

        Args:
            service_type: Service type
            instance: Service instance

        Returns:
            Self for method chaining
        """
        descriptor = ServiceDescriptor(
            service_type=service_type,
            instance=instance,
            lifetime=ServiceLifetime.SINGLETON
        )

        self._services[service_type] = descriptor
        self._singletons[service_type] = instance
        self._logger.debug("Registered instance for: %s",
                           service_type.__name__)
        return self

    def resolve(self, service_type: Type[T]) -> T:
        """
        Resolve a service instance.

        Args:
            service_type: Type of service to resolve

        Returns:
            Service instance

        Raises:
            ValueError: If service is not registered or circular dependency detected
        """
        if service_type in self._resolving:
            raise ValueError(
                f"Circular dependency detected for {service_type.__name__}")

        if service_type not in self._services:
            raise ValueError(
                f"Service {service_type.__name__} is not registered")

        descriptor = self._services[service_type]

        # Check for existing singleton
        if descriptor.lifetime == ServiceLifetime.SINGLETON and service_type in self._singletons:
            return self._singletons[service_type]

        # Check for existing scoped instance
        if descriptor.lifetime == ServiceLifetime.SCOPED and service_type in self._scoped_instances:
            return self._scoped_instances[service_type]

        # Mark as resolving for circular dependency detection
        self._resolving.add(service_type)

        try:
            instance = self._create_instance(descriptor)

            # Store singleton
            if descriptor.lifetime == ServiceLifetime.SINGLETON:
                self._singletons[service_type] = instance

            # Store scoped instance
            elif descriptor.lifetime == ServiceLifetime.SCOPED:
                self._scoped_instances[service_type] = instance

            self._logger.debug("Resolved: %s", service_type.__name__)
            return instance

        finally:
            self._resolving.discard(service_type)

    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create an instance from a service descriptor."""
        # Use existing instance
        if descriptor.instance is not None:
            return descriptor.instance

        # Use factory
        if descriptor.factory is not None:
            return descriptor.factory()

        # Create from implementation type
        if descriptor.implementation_type is not None:
            return self._create_from_type(descriptor.implementation_type, descriptor.dependencies)

        raise ValueError(
            f"Cannot create instance for {descriptor.service_type.__name__}")

    def _create_from_type(self, impl_type: Type, dependencies: Optional[list[Type]]) -> Any:
        """Create an instance from a type with dependency injection."""
        if not dependencies:
            return impl_type()

        # Resolve dependencies
        resolved_deps = []
        for dep_type in dependencies:
            resolved_dep = self.resolve(dep_type)
            resolved_deps.append(resolved_dep)

        return impl_type(*resolved_deps)

    def _extract_dependencies(self, impl_type: Type) -> Optional[list[Type]]:
        """Extract constructor dependencies from a type."""
        try:
            # Get constructor signature
            sig = inspect.signature(impl_type.__init__)

            # Get type hints
            type_hints = get_type_hints(impl_type.__init__)

            dependencies = []
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue

                # Get type annotation
                if param_name in type_hints:
                    param_type = type_hints[param_name]
                    dependencies.append(param_type)
                elif param.annotation != inspect.Parameter.empty:
                    dependencies.append(param.annotation)

            return dependencies if dependencies else None

        except Exception as e:
            self._logger.warning(
                "Failed to extract dependencies for %s: %s", impl_type.__name__, e)
            return None

    def clear_scoped(self) -> None:
        """Clear all scoped instances."""
        self._scoped_instances.clear()
        self._logger.debug("Cleared scoped instances")

    def is_registered(self, service_type: Type) -> bool:
        """Check if a service type is registered."""
        return service_type in self._services

    def get_registered_services(self) -> list[Type]:
        """Get list of all registered service types."""
        return list(self._services.keys())

    def dispose(self) -> None:
        """Dispose of the container and all managed instances."""
        # Dispose singletons that implement dispose
        for instance in self._singletons.values():
            if hasattr(instance, 'dispose') and callable(getattr(instance, 'dispose')):
                try:
                    instance.dispose()
                except Exception as e:
                    self._logger.error("Error disposing instance: %s", e)

        # Clear all instances
        self._singletons.clear()
        self._scoped_instances.clear()
        self._services.clear()

        self._logger.info("DI Container disposed")

"""
Dependency Injection Decorators.

Provides decorators for automatic dependency injection and service registration.
"""

import functools
import inspect
from typing import Type, TypeVar, Callable, Any, get_type_hints
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

# Global container reference - will be set by bootstrap
_container = None


def set_container(container):
    """Set the global DI container reference."""
    global _container
    _container = container


def get_container():
    """Get the global DI container reference."""
    if _container is None:
        raise RuntimeError(
            "DI Container not initialized. Call set_container() first.")
    return _container


def injectable(cls: Type[T]) -> Type[T]:
    """
    Mark a class as injectable.

    This decorator modifies the class constructor to automatically
    resolve dependencies from the DI container.

    Args:
        cls: Class to make injectable

    Returns:
        Modified class with dependency injection
    """
    original_init = cls.__init__

    # Get type hints for the constructor
    type_hints = get_type_hints(original_init)

    # Get constructor signature
    sig = inspect.signature(original_init)

    @functools.wraps(original_init)
    def new_init(self, *args, **kwargs):
        container = get_container()

        # If all dependencies are provided, use original constructor
        if len(args) + len(kwargs) >= len(sig.parameters) - 1:  # -1 for 'self'
            return original_init(self, *args, **kwargs)

        # Resolve missing dependencies
        resolved_args = list(args)
        param_names = list(sig.parameters.keys())[1:]  # Skip 'self'

        for i, param_name in enumerate(param_names):
            if i < len(args):
                continue  # Already provided

            if param_name in kwargs:
                continue  # Already provided

            # Try to resolve from container
            if param_name in type_hints:
                param_type = type_hints[param_name]
                if container.is_registered(param_type):
                    resolved_dep = container.resolve(param_type)
                    kwargs[param_name] = resolved_dep
                    logger.debug("Injected %s into %s.%s",
                                 param_type.__name__, cls.__name__, param_name)

        return original_init(self, *resolved_args, **kwargs)

    cls.__init__ = new_init
    cls._is_injectable = True

    logger.debug("Made class injectable: %s", cls.__name__)
    return cls


def inject(dependency_type: Type[T]) -> Callable[[Callable], Callable]:
    """
    Inject a specific dependency into a method parameter.

    Args:
        dependency_type: Type of dependency to inject

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            container = get_container()

            # Get function signature
            sig = inspect.signature(func)
            type_hints = get_type_hints(func)

            # Find parameter that matches dependency type
            for param_name, param in sig.parameters.items():
                if param_name in type_hints and type_hints[param_name] == dependency_type:
                    if param_name not in kwargs:
                        resolved_dep = container.resolve(dependency_type)
                        kwargs[param_name] = resolved_dep
                        logger.debug(
                            "Injected %s into %s.%s", dependency_type.__name__, func.__name__, param_name)
                        break

            return func(*args, **kwargs)

        return wrapper
    return decorator


def auto_inject(func: Callable) -> Callable:
    """
    Automatically inject all registered dependencies into a function.

    Args:
        func: Function to inject dependencies into

    Returns:
        Wrapped function with automatic dependency injection
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        container = get_container()

        # Get function signature and type hints
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Resolve missing dependencies
        for param_name, param in sig.parameters.items():
            if param_name in kwargs:
                continue  # Already provided

            if param_name in type_hints:
                param_type = type_hints[param_name]
                if container.is_registered(param_type):
                    resolved_dep = container.resolve(param_type)
                    kwargs[param_name] = resolved_dep
                    logger.debug("Auto-injected %s into %s.%s",
                                 param_type.__name__, func.__name__, param_name)

        return func(*args, **kwargs)

    return wrapper


def service(lifetime: str = "transient", interface: Type = None):
    """
    Register a class as a service in the DI container.

    Args:
        lifetime: Service lifetime ("singleton", "transient", "scoped")
        interface: Interface type to register against

    Returns:
        Class decorator
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # This decorator just marks the class for registration
        # Actual registration happens in the bootstrap process
        cls._service_lifetime = lifetime
        cls._service_interface = interface
        cls._is_service = True

        logger.debug("Marked class as service: %s (lifetime: %s)",
                     cls.__name__, lifetime)
        return cls

    return decorator


def singleton(interface: Type = None):
    """
    Register a class as a singleton service.

    Args:
        interface: Interface type to register against

    Returns:
        Class decorator
    """
    return service("singleton", interface)


def transient(interface: Type = None):
    """
    Register a class as a transient service.

    Args:
        interface: Interface type to register against

    Returns:
        Class decorator
    """
    return service("transient", interface)


def scoped(interface: Type = None):
    """
    Register a class as a scoped service.

    Args:
        interface: Interface type to register against

    Returns:
        Class decorator
    """
    return service("scoped", interface)


def lazy_inject(dependency_type: Type[T]) -> Callable[[], T]:
    """
    Create a lazy dependency resolver.

    Args:
        dependency_type: Type of dependency to resolve lazily

    Returns:
        Function that resolves the dependency when called
    """
    def resolver() -> T:
        container = get_container()
        return container.resolve(dependency_type)

    return resolver


class Injected:
    """
    Marker class for dependency injection in type annotations.

    Usage:
        def __init__(self, service: Injected[SomeService]):
            self.service = service
    """

    def __init__(self, dependency_type: Type[T]):
        self.dependency_type = dependency_type

    def __class_getitem__(cls, item):
        return cls(item)

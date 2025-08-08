"""
Dependency Injection Module.

Provides comprehensive dependency injection infrastructure including:
- DI Container with lifetime management
- Decorators for automatic injection
- Service registration utilities
- Bootstrap functionality
"""

from .container import DIContainer, ServiceLifetime, ServiceDescriptor
from .decorators import (
    injectable, inject, auto_inject, service, singleton, transient, scoped,
    lazy_inject, Injected, set_container, get_container
)

__all__ = [
    # Container
    'DIContainer',
    'ServiceLifetime',
    'ServiceDescriptor',

    # Decorators
    'injectable',
    'inject',
    'auto_inject',
    'service',
    'singleton',
    'transient',
    'scoped',
    'lazy_inject',
    'Injected',

    # Container access
    'set_container',
    'get_container'
]

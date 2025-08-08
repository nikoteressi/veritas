"""
Design Patterns Implementation для Graph Service.

Этот модуль содержит реализации паттернов проектирования для работы с графом фактов:
- Command Pattern: для операций с поддержкой undo/redo
- Observer Pattern: для уведомлений об изменениях
- Builder Pattern: для пошагового построения графа
"""

# Command Pattern
from .command import (
    Command,
    UndoableCommand,
    CompositeCommand,
    AddNodeCommand,
    RemoveNodeCommand,
    AddEdgeCommand,
    RemoveEdgeCommand,
    CreateClusterCommand,
    RemoveClusterCommand,
    OptimizeGraphCommand,
    UpdateNodeCommand,
    CommandInvoker,
    BatchCommandInvoker
)

# Observer Pattern
from .observer import (
    GraphEventType,
    GraphEvent,
    GraphObserver,
    FilteredGraphObserver,
    FunctionGraphObserver,
    GraphSubject,
    GraphEventBuilder,
    LoggingGraphObserver,
    MetricsGraphObserver,
    CacheInvalidationObserver
)

# Builder Pattern
from .builder import (
    BuilderValidationLevel,
    BuilderConfiguration,
    BuildStep,
    GraphBuilder,
    FactGraphBuilder,
    GraphBuilderDirector,
    GraphBuilderFactory
)

__all__ = [
    # Command Pattern
    'Command',
    'UndoableCommand',
    'CompositeCommand',
    'AddNodeCommand',
    'RemoveNodeCommand',
    'AddEdgeCommand',
    'RemoveEdgeCommand',
    'CreateClusterCommand',
    'RemoveClusterCommand',
    'OptimizeGraphCommand',
    'UpdateNodeCommand',
    'CommandInvoker',
    'BatchCommandInvoker',

    # Observer Pattern
    'GraphEventType',
    'GraphEvent',
    'GraphObserver',
    'FilteredGraphObserver',
    'FunctionGraphObserver',
    'GraphSubject',
    'GraphEventBuilder',
    'LoggingGraphObserver',
    'MetricsGraphObserver',
    'CacheInvalidationObserver',

    # Builder Pattern
    'BuilderValidationLevel',
    'BuilderConfiguration',
    'BuildStep',
    'GraphBuilder',
    'FactGraphBuilder',
    'GraphBuilderDirector',
    'GraphBuilderFactory'
]

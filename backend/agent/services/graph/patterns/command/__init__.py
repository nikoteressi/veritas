"""
Command Pattern Implementation для Graph Service.

Этот модуль реализует паттерн Command для операций с графом фактов,
обеспечивая поддержку undo/redo операций и пакетного выполнения команд.
"""

from .base_command import Command, UndoableCommand, CompositeCommand
from .graph_commands import (
    AddNodeCommand,
    RemoveNodeCommand,
    AddEdgeCommand,
    RemoveEdgeCommand,
    CreateClusterCommand,
    RemoveClusterCommand,
    OptimizeGraphCommand,
    UpdateNodeCommand
)
from .command_invoker import CommandInvoker, BatchCommandInvoker

__all__ = [
    # Base classes
    'Command',
    'UndoableCommand',
    'CompositeCommand',

    # Graph commands
    'AddNodeCommand',
    'RemoveNodeCommand',
    'AddEdgeCommand',
    'RemoveEdgeCommand',
    'CreateClusterCommand',
    'RemoveClusterCommand',
    'OptimizeGraphCommand',
    'UpdateNodeCommand',

    # Invokers
    'CommandInvoker',
    'BatchCommandInvoker'
]

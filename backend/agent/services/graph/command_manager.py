"""
Менеджер команд для управления операциями с графом.

Инкапсулирует логику Command Pattern для операций с undo/redo.
"""

import logging
from typing import Dict, List

from .patterns.command import (
    Command, CommandInvoker, BatchCommandInvoker
)

logger = logging.getLogger(__name__)


class CommandManager:
    """Менеджер для управления командами графа."""

    def __init__(self, history_size: int = 100):
        """
        Инициализация менеджера команд.

        Args:
            history_size: Размер истории команд
        """
        self._history_size = history_size
        self._invokers: Dict[str, CommandInvoker] = {}
        self._batch_invokers: Dict[str, BatchCommandInvoker] = {}

    def get_invoker(self, graph_id: str) -> CommandInvoker:
        """Получить инвокер команд для графа."""
        if graph_id not in self._invokers:
            self._invokers[graph_id] = CommandInvoker(self._history_size)
        return self._invokers[graph_id]

    def get_batch_invoker(self, graph_id: str) -> BatchCommandInvoker:
        """Получить пакетный инвокер команд для графа."""
        if graph_id not in self._batch_invokers:
            self._batch_invokers[graph_id] = BatchCommandInvoker(
                self._history_size)
        return self._batch_invokers[graph_id]

    async def execute_command(self, graph_id: str, command: Command) -> bool:
        """
        Выполнить команду.

        Args:
            graph_id: Идентификатор графа
            command: Команда для выполнения

        Returns:
            True если команда выполнена успешно
        """
        try:
            invoker = self.get_invoker(graph_id)
            await invoker.execute_command(command)
            logger.debug(
                f"Command executed for graph {graph_id}: {type(command).__name__}")
            return True
        except Exception as e:
            logger.error(
                f"Failed to execute command for graph {graph_id}: {e}")
            return False

    async def execute_batch_commands(self, graph_id: str, commands: List[Command]) -> bool:
        """
        Выполнить пакет команд.

        Args:
            graph_id: Идентификатор графа
            commands: Список команд для выполнения

        Returns:
            True если все команды выполнены успешно
        """
        try:
            batch_invoker = self.get_batch_invoker(graph_id)
            await batch_invoker.execute_batch(commands)
            logger.debug(
                f"Batch commands executed for graph {graph_id}: {len(commands)} commands")
            return True
        except Exception as e:
            logger.error(
                f"Failed to execute batch commands for graph {graph_id}: {e}")
            return False

    async def undo_command(self, graph_id: str) -> bool:
        """
        Отменить последнюю команду.

        Args:
            graph_id: Идентификатор графа

        Returns:
            True если команда отменена успешно
        """
        try:
            invoker = self.get_invoker(graph_id)
            await invoker.undo()
            logger.debug(f"Command undone for graph {graph_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to undo command for graph {graph_id}: {e}")
            return False

    async def redo_command(self, graph_id: str) -> bool:
        """
        Повторить отмененную команду.

        Args:
            graph_id: Идентификатор графа

        Returns:
            True если команда повторена успешно
        """
        try:
            invoker = self.get_invoker(graph_id)
            await invoker.redo()
            logger.debug(f"Command redone for graph {graph_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to redo command for graph {graph_id}: {e}")
            return False

    def get_command_history(self, graph_id: str) -> List[str]:
        """
        Получить историю команд.

        Args:
            graph_id: Идентификатор графа

        Returns:
            Список названий выполненных команд
        """
        if graph_id not in self._invokers:
            return []

        invoker = self._invokers[graph_id]
        return [type(cmd).__name__ for cmd in invoker.get_history()]

    def can_undo(self, graph_id: str) -> bool:
        """Проверить, можно ли отменить команду."""
        if graph_id not in self._invokers:
            return False
        return self._invokers[graph_id].can_undo()

    def can_redo(self, graph_id: str) -> bool:
        """Проверить, можно ли повторить команду."""
        if graph_id not in self._invokers:
            return False
        return self._invokers[graph_id].can_redo()

    def clear_history(self, graph_id: str) -> None:
        """Очистить историю команд для графа."""
        if graph_id in self._invokers:
            del self._invokers[graph_id]
        if graph_id in self._batch_invokers:
            del self._batch_invokers[graph_id]
        logger.debug(f"Command history cleared for graph {graph_id}")

    def get_stats(self) -> Dict[str, int]:
        """Получить статистику по командам."""
        return {
            "active_graphs": len(self._invokers),
            "total_commands": sum(len(invoker.get_history()) for invoker in self._invokers.values())
        }

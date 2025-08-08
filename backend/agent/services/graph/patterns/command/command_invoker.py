"""
Command Invoker для выполнения команд с поддержкой истории и undo/redo.

Этот модуль предоставляет инвокер для выполнения команд с автоматическим
управлением историей команд и возможностью отмены/повтора операций.
"""

from typing import List, Optional, Any
import logging
from .base_command import Command

logger = logging.getLogger(__name__)


class CommandInvoker:
    """Инвокер для выполнения команд с поддержкой истории"""

    def __init__(self, max_history_size: int = 100):
        self.max_history_size = max_history_size
        self._history: List[Command] = []
        self._current_position = -1

    async def execute_command(self, command: Command) -> Any:
        """Выполнить команду и добавить в историю"""
        try:
            logger.info(f"Executing command: {command.description}")
            result = await command.execute()

            # Добавляем в историю только успешно выполненные команды
            self._add_to_history(command)
            logger.info(f"Command executed successfully: {command.id}")

            return result

        except Exception as e:
            logger.error(f"Command execution failed: {command.id}, error: {e}")
            raise

    async def undo_last_command(self) -> Optional[Any]:
        """Отменить последнюю команду"""
        if not self.can_undo():
            logger.warning("No commands to undo")
            return None

        command = self._history[self._current_position]

        try:
            logger.info(f"Undoing command: {command.description}")
            result = await command.undo()
            self._current_position -= 1
            logger.info(f"Command undone successfully: {command.id}")
            return result

        except Exception as e:
            logger.error(f"Command undo failed: {command.id}, error: {e}")
            raise

    async def redo_next_command(self) -> Optional[Any]:
        """Повторить следующую команду"""
        if not self.can_redo():
            logger.warning("No commands to redo")
            return None

        self._current_position += 1
        command = self._history[self._current_position]

        try:
            logger.info(f"Redoing command: {command.description}")
            result = await command.execute()
            logger.info(f"Command redone successfully: {command.id}")
            return result

        except Exception as e:
            logger.error(f"Command redo failed: {command.id}, error: {e}")
            self._current_position -= 1
            raise

    async def undo_multiple_commands(self, count: int) -> List[Any]:
        """Отменить несколько команд"""
        results = []
        for _ in range(count):
            if not self.can_undo():
                break
            result = await self.undo_last_command()
            if result is not None:
                results.append(result)
        return results

    async def redo_multiple_commands(self, count: int) -> List[Any]:
        """Повторить несколько команд"""
        results = []
        for _ in range(count):
            if not self.can_redo():
                break
            result = await self.redo_next_command()
            if result is not None:
                results.append(result)
        return results

    def can_undo(self) -> bool:
        """Проверить возможность отмены"""
        return self._current_position >= 0

    def can_redo(self) -> bool:
        """Проверить возможность повтора"""
        return self._current_position < len(self._history) - 1

    def get_history(self) -> List[str]:
        """Получить историю команд"""
        return [cmd.description for cmd in self._history]

    def get_undo_stack(self) -> List[str]:
        """Получить стек команд для отмены"""
        if self._current_position < 0:
            return []
        return [cmd.description for cmd in self._history[:self._current_position + 1]]

    def get_redo_stack(self) -> List[str]:
        """Получить стек команд для повтора"""
        if self._current_position >= len(self._history) - 1:
            return []
        return [cmd.description for cmd in self._history[self._current_position + 1:]]

    def get_current_position(self) -> int:
        """Получить текущую позицию в истории"""
        return self._current_position

    def get_history_size(self) -> int:
        """Получить размер истории"""
        return len(self._history)

    def clear_history(self):
        """Очистить историю команд"""
        self._history.clear()
        self._current_position = -1
        logger.info("Command history cleared")

    def clear_redo_stack(self):
        """Очистить стек повтора (используется при выполнении новой команды после undo)"""
        if self._current_position < len(self._history) - 1:
            removed_count = len(self._history) - self._current_position - 1
            self._history = self._history[:self._current_position + 1]
            logger.debug(f"Cleared {removed_count} commands from redo stack")

    def get_command_by_id(self, command_id: str) -> Optional[Command]:
        """Найти команду по ID"""
        for command in self._history:
            if command.id == command_id:
                return command
        return None

    def get_commands_by_type(self, command_type: type) -> List[Command]:
        """Получить команды определенного типа"""
        return [cmd for cmd in self._history if isinstance(cmd, command_type)]

    def _add_to_history(self, command: Command):
        """Добавить команду в историю"""
        # Удаляем команды после текущей позиции (для случая undo -> new command)
        self.clear_redo_stack()

        # Добавляем новую команду
        self._history.append(command)
        self._current_position += 1

        # Ограничиваем размер истории
        if len(self._history) > self.max_history_size:
            removed_command = self._history.pop(0)
            self._current_position -= 1
            logger.debug(
                f"Removed old command from history: {removed_command.description}")


class BatchCommandInvoker(CommandInvoker):
    """Расширенный инвокер с поддержкой пакетного выполнения команд"""

    def __init__(self, max_history_size: int = 100):
        super().__init__(max_history_size)
        self._batch_mode = False
        self._batch_commands: List[Command] = []

    def start_batch(self):
        """Начать пакетное выполнение команд"""
        if self._batch_mode:
            logger.warning("Batch mode already active")
            return

        self._batch_mode = True
        self._batch_commands.clear()
        logger.info("Started batch command mode")

    async def execute_command(self, command: Command) -> Any:
        """Выполнить команду (в пакетном режиме добавляется в очередь)"""
        if self._batch_mode:
            self._batch_commands.append(command)
            logger.debug(f"Added command to batch: {command.description}")
            return None
        else:
            return await super().execute_command(command)

    async def commit_batch(self) -> List[Any]:
        """Выполнить все команды в пакете"""
        if not self._batch_mode:
            logger.warning("Not in batch mode")
            return []

        if not self._batch_commands:
            logger.info("No commands in batch to commit")
            self._batch_mode = False
            return []

        logger.info(
            f"Committing batch of {len(self._batch_commands)} commands")

        # Создаем составную команду из всех команд в пакете
        from .base_command import CompositeCommand
        composite_command = CompositeCommand(
            self._batch_commands.copy(),
            f"Batch of {len(self._batch_commands)} commands"
        )

        try:
            result = await super().execute_command(composite_command)
            self._batch_mode = False
            self._batch_commands.clear()
            logger.info("Batch committed successfully")
            return result

        except Exception as e:
            logger.error(f"Batch commit failed: {e}")
            self._batch_mode = False
            self._batch_commands.clear()
            raise

    async def rollback_batch(self):
        """Отменить пакет команд без выполнения"""
        if not self._batch_mode:
            logger.warning("Not in batch mode")
            return

        logger.info(
            f"Rolling back batch of {len(self._batch_commands)} commands")
        self._batch_mode = False
        self._batch_commands.clear()

    def is_batch_mode(self) -> bool:
        """Проверить, активен ли пакетный режим"""
        return self._batch_mode

    def get_batch_commands(self) -> List[str]:
        """Получить список команд в текущем пакете"""
        return [cmd.description for cmd in self._batch_commands]

    def get_batch_size(self) -> int:
        """Получить размер текущего пакета"""
        return len(self._batch_commands)

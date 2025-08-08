"""
Базовая инфраструктура Command Pattern для операций с графом.

Этот модуль предоставляет базовые классы для реализации паттерна Command
с поддержкой undo/redo операций для работы с графами фактов.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


class Command(ABC):
    """Базовый интерфейс для всех команд"""

    def __init__(self):
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.now()
        self.executed = False

    @abstractmethod
    async def execute(self) -> Any:
        """Выполнить команду"""
        pass

    @abstractmethod
    async def undo(self) -> Any:
        """Отменить команду"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Описание команды для логирования"""
        pass


class UndoableCommand(Command):
    """Базовый класс для команд с поддержкой отмены"""

    def __init__(self):
        super().__init__()
        self.undo_data: Optional[dict] = None

    async def execute(self) -> Any:
        """Выполнить команду с подготовкой данных для отмены"""
        if self.executed:
            raise RuntimeError(f"Command {self.id} already executed")

        logger.debug(f"Preparing undo data for command: {self.description}")
        # Сохраняем данные для отмены
        self.undo_data = await self._prepare_undo_data()

        logger.info(f"Executing command: {self.description}")
        # Выполняем команду
        result = await self._do_execute()
        self.executed = True

        logger.info(f"Command executed successfully: {self.id}")
        return result

    async def undo(self) -> Any:
        """Отменить команду"""
        if not self.executed:
            raise RuntimeError(f"Command {self.id} not executed yet")

        if self.undo_data is None:
            raise RuntimeError(f"No undo data for command {self.id}")

        logger.info(f"Undoing command: {self.description}")
        # Отменяем команду
        result = await self._do_undo()
        self.executed = False

        logger.info(f"Command undone successfully: {self.id}")
        return result

    @abstractmethod
    async def _prepare_undo_data(self) -> dict:
        """Подготовить данные для отмены"""
        pass

    @abstractmethod
    async def _do_execute(self) -> Any:
        """Выполнить основную логику команды"""
        pass

    @abstractmethod
    async def _do_undo(self) -> Any:
        """Выполнить отмену команды"""
        pass


class CompositeCommand(UndoableCommand):
    """Составная команда для выполнения нескольких команд как одной операции"""

    def __init__(self, commands: list[UndoableCommand], description: str):
        super().__init__()
        self.commands = commands
        self._description = description
        self.executed_commands: list[UndoableCommand] = []

    @property
    def description(self) -> str:
        return self._description

    async def _prepare_undo_data(self) -> dict:
        """Для составной команды данные для отмены хранятся в каждой подкоманде"""
        return {"composite": True}

    async def _do_execute(self) -> Any:
        """Выполнить все команды последовательно"""
        results = []

        try:
            for command in self.commands:
                result = await command.execute()
                self.executed_commands.append(command)
                results.append(result)

            return results

        except Exception as e:
            # Если одна из команд не выполнилась, отменяем все выполненные
            logger.error(
                f"Composite command failed, rolling back executed commands: {e}")
            await self._rollback_executed_commands()
            raise

    async def _do_undo(self) -> Any:
        """Отменить все команды в обратном порядке"""
        return await self._rollback_executed_commands()

    async def _rollback_executed_commands(self) -> list[Any]:
        """Откатить все выполненные команды"""
        results = []

        # Отменяем в обратном порядке
        for command in reversed(self.executed_commands):
            try:
                result = await command.undo()
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to undo command {command.id}: {e}")
                # Продолжаем откат остальных команд

        self.executed_commands.clear()
        return results

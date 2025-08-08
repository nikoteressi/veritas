"""
Observer Pattern Implementation для Graph Service.

Этот модуль реализует паттерн Observer для уведомлений об изменениях
в графе фактов, позволяя различным компонентам системы реагировать
на события изменения графа.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Callable
from enum import Enum
import logging
from datetime import datetime
import asyncio
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class GraphEventType(Enum):
    """Типы событий в графе"""
    NODE_ADDED = "node_added"
    NODE_REMOVED = "node_removed"
    NODE_UPDATED = "node_updated"
    EDGE_ADDED = "edge_added"
    EDGE_REMOVED = "edge_removed"
    EDGE_UPDATED = "edge_updated"
    CLUSTER_CREATED = "cluster_created"
    CLUSTER_REMOVED = "cluster_removed"
    CLUSTER_UPDATED = "cluster_updated"
    GRAPH_OPTIMIZED = "graph_optimized"
    GRAPH_CLEARED = "graph_cleared"
    VERIFICATION_COMPLETED = "verification_completed"
    BATCH_OPERATION_STARTED = "batch_operation_started"
    BATCH_OPERATION_COMPLETED = "batch_operation_completed"


@dataclass
class GraphEvent:
    """Событие изменения графа"""
    event_type: GraphEventType
    timestamp: datetime
    source: str  # Источник события (например, имя команды)
    data: Dict[str, Any]  # Данные события
    graph_id: Optional[str] = None
    node_id: Optional[str] = None
    edge_id: Optional[str] = None
    cluster_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class GraphObserver(ABC):
    """Базовый интерфейс для наблюдателей графа"""

    @abstractmethod
    async def on_graph_event(self, event: GraphEvent) -> None:
        """Обработать событие изменения графа"""
        pass

    @property
    @abstractmethod
    def observer_id(self) -> str:
        """Уникальный идентификатор наблюдателя"""
        pass

    @property
    def interested_events(self) -> Set[GraphEventType]:
        """События, которые интересуют данного наблюдателя"""
        return set(GraphEventType)  # По умолчанию все события


class FilteredGraphObserver(GraphObserver):
    """Наблюдатель с фильтрацией событий"""

    def __init__(self, observer_id: str, interested_events: Set[GraphEventType] = None):
        self._observer_id = observer_id
        self._interested_events = interested_events or set(GraphEventType)

    @property
    def observer_id(self) -> str:
        return self._observer_id

    @property
    def interested_events(self) -> Set[GraphEventType]:
        return self._interested_events

    async def on_graph_event(self, event: GraphEvent) -> None:
        """Базовая реализация - логирование события"""
        logger.info(
            f"Observer {self.observer_id} received event: {event.event_type}")


class FunctionGraphObserver(FilteredGraphObserver):
    """Наблюдатель на основе функции"""

    def __init__(
        self,
        observer_id: str,
        callback: Callable[[GraphEvent], None],
        interested_events: Set[GraphEventType] = None
    ):
        super().__init__(observer_id, interested_events)
        self._callback = callback

    async def on_graph_event(self, event: GraphEvent) -> None:
        """Вызвать callback функцию"""
        try:
            if asyncio.iscoroutinefunction(self._callback):
                await self._callback(event)
            else:
                self._callback(event)
        except Exception as e:
            logger.error(f"Error in observer {self.observer_id} callback: {e}")


class GraphSubject:
    """Субъект для уведомления наблюдателей об изменениях графа"""

    def __init__(self):
        self._observers: Dict[str, GraphObserver] = {}
        self._event_history: List[GraphEvent] = []
        self._max_history_size = 1000
        self._notification_enabled = True

    def attach_observer(self, observer: GraphObserver) -> None:
        """Подписать наблюдателя"""
        if observer.observer_id in self._observers:
            logger.warning(f"Observer {observer.observer_id} already attached")
            return

        self._observers[observer.observer_id] = observer
        logger.info(f"Observer {observer.observer_id} attached")

    def detach_observer(self, observer_id: str) -> bool:
        """Отписать наблюдателя"""
        if observer_id in self._observers:
            del self._observers[observer_id]
            logger.info(f"Observer {observer_id} detached")
            return True

        logger.warning(f"Observer {observer_id} not found")
        return False

    def get_observer(self, observer_id: str) -> Optional[GraphObserver]:
        """Получить наблюдателя по ID"""
        return self._observers.get(observer_id)

    def get_observers(self) -> List[GraphObserver]:
        """Получить всех наблюдателей"""
        return list(self._observers.values())

    def get_observer_count(self) -> int:
        """Получить количество наблюдателей"""
        return len(self._observers)

    async def notify_observers(self, event: GraphEvent) -> None:
        """Уведомить всех заинтересованных наблюдателей"""
        if not self._notification_enabled:
            return

        # Добавляем событие в историю
        self._add_to_history(event)

        # Уведомляем наблюдателей
        notification_tasks = []

        for observer in self._observers.values():
            if event.event_type in observer.interested_events:
                task = self._notify_observer_safe(observer, event)
                notification_tasks.append(task)

        if notification_tasks:
            await asyncio.gather(*notification_tasks, return_exceptions=True)

    async def _notify_observer_safe(self, observer: GraphObserver, event: GraphEvent) -> None:
        """Безопасно уведомить наблюдателя"""
        try:
            await observer.on_graph_event(event)
        except Exception as e:
            logger.error(
                f"Error notifying observer {observer.observer_id}: {e}")

    def enable_notifications(self) -> None:
        """Включить уведомления"""
        self._notification_enabled = True
        logger.info("Graph notifications enabled")

    def disable_notifications(self) -> None:
        """Отключить уведомления"""
        self._notification_enabled = False
        logger.info("Graph notifications disabled")

    def is_notifications_enabled(self) -> bool:
        """Проверить, включены ли уведомления"""
        return self._notification_enabled

    def get_event_history(self, limit: Optional[int] = None) -> List[GraphEvent]:
        """Получить историю событий"""
        if limit is None:
            return self._event_history.copy()
        return self._event_history[-limit:].copy()

    def get_events_by_type(self, event_type: GraphEventType, limit: Optional[int] = None) -> List[GraphEvent]:
        """Получить события определенного типа"""
        filtered_events = [
            e for e in self._event_history if e.event_type == event_type]
        if limit is None:
            return filtered_events
        return filtered_events[-limit:]

    def clear_event_history(self) -> None:
        """Очистить историю событий"""
        self._event_history.clear()
        logger.info("Event history cleared")

    def _add_to_history(self, event: GraphEvent) -> None:
        """Добавить событие в историю"""
        self._event_history.append(event)

        # Ограничиваем размер истории
        if len(self._event_history) > self._max_history_size:
            self._event_history.pop(0)


class GraphEventBuilder:
    """Builder для создания событий графа"""

    def __init__(self):
        self.reset()

    def reset(self) -> 'GraphEventBuilder':
        """Сбросить состояние builder'а"""
        self._event_type: Optional[GraphEventType] = None
        self._source: Optional[str] = None
        self._data: Dict[str, Any] = {}
        self._graph_id: Optional[str] = None
        self._node_id: Optional[str] = None
        self._edge_id: Optional[str] = None
        self._cluster_id: Optional[str] = None
        self._timestamp: Optional[datetime] = None
        return self

    def event_type(self, event_type: GraphEventType) -> 'GraphEventBuilder':
        """Установить тип события"""
        self._event_type = event_type
        return self

    def source(self, source: str) -> 'GraphEventBuilder':
        """Установить источник события"""
        self._source = source
        return self

    def data(self, data: Dict[str, Any]) -> 'GraphEventBuilder':
        """Установить данные события"""
        self._data = data.copy()
        return self

    def add_data(self, key: str, value: Any) -> 'GraphEventBuilder':
        """Добавить данные к событию"""
        self._data[key] = value
        return self

    def graph_id(self, graph_id: str) -> 'GraphEventBuilder':
        """Установить ID графа"""
        self._graph_id = graph_id
        return self

    def node_id(self, node_id: str) -> 'GraphEventBuilder':
        """Установить ID узла"""
        self._node_id = node_id
        return self

    def edge_id(self, edge_id: str) -> 'GraphEventBuilder':
        """Установить ID ребра"""
        self._edge_id = edge_id
        return self

    def cluster_id(self, cluster_id: str) -> 'GraphEventBuilder':
        """Установить ID кластера"""
        self._cluster_id = cluster_id
        return self

    def timestamp(self, timestamp: datetime) -> 'GraphEventBuilder':
        """Установить время события"""
        self._timestamp = timestamp
        return self

    def build(self) -> GraphEvent:
        """Создать событие"""
        if self._event_type is None:
            raise ValueError("Event type is required")
        if self._source is None:
            raise ValueError("Event source is required")

        event = GraphEvent(
            event_type=self._event_type,
            timestamp=self._timestamp or datetime.now(),
            source=self._source,
            data=self._data,
            graph_id=self._graph_id,
            node_id=self._node_id,
            edge_id=self._edge_id,
            cluster_id=self._cluster_id
        )

        return event


# Специализированные наблюдатели

class LoggingGraphObserver(FilteredGraphObserver):
    """Наблюдатель для логирования событий графа"""

    def __init__(self, log_level: int = logging.INFO):
        super().__init__("logging_observer")
        self._log_level = log_level

    async def on_graph_event(self, event: GraphEvent) -> None:
        """Логировать событие"""
        message = f"Graph event: {event.event_type.value}"
        if event.node_id:
            message += f", node: {event.node_id}"
        if event.edge_id:
            message += f", edge: {event.edge_id}"
        if event.cluster_id:
            message += f", cluster: {event.cluster_id}"

        logger.log(self._log_level, message)


class MetricsGraphObserver(FilteredGraphObserver):
    """Наблюдатель для сбора метрик графа"""

    def __init__(self):
        super().__init__("metrics_observer")
        self._metrics = {
            "nodes_added": 0,
            "nodes_removed": 0,
            "edges_added": 0,
            "edges_removed": 0,
            "clusters_created": 0,
            "clusters_removed": 0,
            "optimizations": 0,
            "verifications": 0
        }

    async def on_graph_event(self, event: GraphEvent) -> None:
        """Обновить метрики"""
        if event.event_type == GraphEventType.NODE_ADDED:
            self._metrics["nodes_added"] += 1
        elif event.event_type == GraphEventType.NODE_REMOVED:
            self._metrics["nodes_removed"] += 1
        elif event.event_type == GraphEventType.EDGE_ADDED:
            self._metrics["edges_added"] += 1
        elif event.event_type == GraphEventType.EDGE_REMOVED:
            self._metrics["edges_removed"] += 1
        elif event.event_type == GraphEventType.CLUSTER_CREATED:
            self._metrics["clusters_created"] += 1
        elif event.event_type == GraphEventType.CLUSTER_REMOVED:
            self._metrics["clusters_removed"] += 1
        elif event.event_type == GraphEventType.GRAPH_OPTIMIZED:
            self._metrics["optimizations"] += 1
        elif event.event_type == GraphEventType.VERIFICATION_COMPLETED:
            self._metrics["verifications"] += 1

    def get_metrics(self) -> Dict[str, int]:
        """Получить метрики"""
        return self._metrics.copy()

    def reset_metrics(self) -> None:
        """Сбросить метрики"""
        for key in self._metrics:
            self._metrics[key] = 0


class CacheInvalidationObserver(FilteredGraphObserver):
    """Наблюдатель для инвалидации кэша при изменениях графа"""

    def __init__(self, cache_manager):
        super().__init__("cache_invalidation_observer")
        self._cache_manager = cache_manager
        self._interested_events = {
            GraphEventType.NODE_ADDED,
            GraphEventType.NODE_REMOVED,
            GraphEventType.NODE_UPDATED,
            GraphEventType.EDGE_ADDED,
            GraphEventType.EDGE_REMOVED,
            GraphEventType.EDGE_UPDATED,
            GraphEventType.CLUSTER_CREATED,
            GraphEventType.CLUSTER_REMOVED,
            GraphEventType.CLUSTER_UPDATED,
            GraphEventType.GRAPH_OPTIMIZED
        }

    @property
    def interested_events(self) -> Set[GraphEventType]:
        return self._interested_events

    async def on_graph_event(self, event: GraphEvent) -> None:
        """Инвалидировать соответствующие кэши"""
        try:
            if event.node_id:
                await self._cache_manager.invalidate_node_cache(event.node_id)

            if event.edge_id:
                await self._cache_manager.invalidate_edge_cache(event.edge_id)

            if event.cluster_id:
                await self._cache_manager.invalidate_cluster_cache(event.cluster_id)

            # Инвалидируем общие кэши при структурных изменениях
            if event.event_type in {
                GraphEventType.NODE_ADDED, GraphEventType.NODE_REMOVED,
                GraphEventType.EDGE_ADDED, GraphEventType.EDGE_REMOVED,
                GraphEventType.CLUSTER_CREATED, GraphEventType.CLUSTER_REMOVED,
                GraphEventType.GRAPH_OPTIMIZED
            }:
                await self._cache_manager.invalidate_graph_structure_cache()

        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")

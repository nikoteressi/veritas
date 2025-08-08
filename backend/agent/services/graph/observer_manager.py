"""
Менеджер наблюдателей для уведомлений об изменениях в графе.

Инкапсулирует логику Observer Pattern для системы событий.
"""

import logging
from typing import Any, Dict, List, Optional

from .patterns.observer import (
    GraphObserver, GraphSubject, GraphEventType, GraphEventBuilder,
    LoggingGraphObserver, MetricsGraphObserver, CacheInvalidationObserver
)
from .cache_manager import GraphCacheManager

logger = logging.getLogger(__name__)


class ObserverManager:
    """Менеджер для управления наблюдателями графа."""

    def __init__(self):
        """Инициализация менеджера наблюдателей."""
        self._subject = GraphSubject()
        self._event_builder = GraphEventBuilder()
        self._observers: Dict[str, GraphObserver] = {}

    def add_observer(self, name: str, observer: GraphObserver) -> None:
        """
        Добавить наблюдателя.

        Args:
            name: Имя наблюдателя
            observer: Экземпляр наблюдателя
        """
        self._subject.attach_observer(observer)
        self._observers[name] = observer
        logger.debug(f"Observer added: {name}")

    def remove_observer(self, name: str) -> bool:
        """
        Удалить наблюдателя.

        Args:
            name: Имя наблюдателя

        Returns:
            True если наблюдатель удален успешно
        """
        if name in self._observers:
            observer = self._observers[name]
            self._subject.detach_observer(observer)
            del self._observers[name]
            logger.debug(f"Observer removed: {name}")
            return True
        return False

    def get_observer(self, name: str) -> Optional[GraphObserver]:
        """Получить наблюдателя по имени."""
        return self._observers.get(name)

    def list_observers(self) -> List[str]:
        """Получить список имен всех наблюдателей."""
        return list(self._observers.keys())

    async def notify_node_added(
        self,
        graph_id: str,
        node_id: str,
        source: str = "unknown",
        **kwargs
    ) -> None:
        """Уведомить о добавлении узла."""
        event = (self._event_builder
                 .reset()
                 .event_type(GraphEventType.NODE_ADDED)
                 .source(source)
                 .graph_id(graph_id)
                 .add_data("node_id", node_id))

        for key, value in kwargs.items():
            event.add_data(key, value)

        await self._subject.notify_observers(event.build())

    async def notify_node_removed(
        self,
        graph_id: str,
        node_id: str,
        source: str = "unknown",
        **kwargs
    ) -> None:
        """Уведомить об удалении узла."""
        event = (self._event_builder
                 .reset()
                 .event_type(GraphEventType.NODE_REMOVED)
                 .source(source)
                 .graph_id(graph_id)
                 .add_data("node_id", node_id))

        for key, value in kwargs.items():
            event.add_data(key, value)

        await self._subject.notify_observers(event.build())

    async def notify_edge_added(
        self,
        graph_id: str,
        edge_id: str,
        source_id: str,
        target_id: str,
        source: str = "unknown",
        **kwargs
    ) -> None:
        """Уведомить о добавлении ребра."""
        event = (self._event_builder
                 .reset()
                 .event_type(GraphEventType.EDGE_ADDED)
                 .source(source)
                 .graph_id(graph_id)
                 .add_data("edge_id", edge_id)
                 .add_data("source_id", source_id)
                 .add_data("target_id", target_id))

        for key, value in kwargs.items():
            event.add_data(key, value)

        await self._subject.notify_observers(event.build())

    async def notify_edge_removed(
        self,
        graph_id: str,
        edge_id: str,
        source: str = "unknown",
        **kwargs
    ) -> None:
        """Уведомить об удалении ребра."""
        event = (self._event_builder
                 .reset()
                 .event_type(GraphEventType.EDGE_REMOVED)
                 .source(source)
                 .graph_id(graph_id)
                 .add_data("edge_id", edge_id))

        for key, value in kwargs.items():
            event.add_data(key, value)

        await self._subject.notify_observers(event.build())

    async def notify_cluster_created(
        self,
        graph_id: str,
        cluster_id: str,
        source: str = "unknown",
        **kwargs
    ) -> None:
        """Уведомить о создании кластера."""
        event = (self._event_builder
                 .reset()
                 .event_type(GraphEventType.CLUSTER_CREATED)
                 .source(source)
                 .graph_id(graph_id)
                 .add_data("cluster_id", cluster_id))

        for key, value in kwargs.items():
            event.add_data(key, value)

        await self._subject.notify_observers(event.build())

    async def notify_verification_completed(
        self,
        graph_id: str,
        verification_id: str,
        result: str,
        source: str = "unknown",
        **kwargs
    ) -> None:
        """Уведомить о завершении верификации."""
        event = (self._event_builder
                 .reset()
                 .event_type(GraphEventType.VERIFICATION_COMPLETED)
                 .source(source)
                 .graph_id(graph_id)
                 .add_data("verification_id", verification_id)
                 .add_data("result", result))

        for key, value in kwargs.items():
            event.add_data(key, value)

        await self._subject.notify_observers(event.build())

    async def notify_graph_optimized(
        self,
        graph_id: str,
        optimization_type: str,
        source: str = "unknown",
        **kwargs
    ) -> None:
        """Уведомить об оптимизации графа."""
        event = (self._event_builder
                 .reset()
                 .event_type(GraphEventType.GRAPH_OPTIMIZED)
                 .source(source)
                 .graph_id(graph_id)
                 .add_data("optimization_type", optimization_type))

        for key, value in kwargs.items():
            event.add_data(key, value)

        await self._subject.notify_observers(event.build())

    async def notify_custom_event(
        self,
        event_type: GraphEventType,
        graph_id: str,
        source: str = "unknown",
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Уведомить о пользовательском событии.

        Args:
            event_type: Тип события
            graph_id: Идентификатор графа
            source: Источник события
            data: Дополнительные данные
        """
        event_builder = (self._event_builder
                         .reset()
                         .event_type(event_type)
                         .source(source)
                         .graph_id(graph_id))

        if data:
            for key, value in data.items():
                event_builder.add_data(key, value)

        await self._subject.notify_observers(event_builder.build())

    def setup_default_observers(self) -> None:
        """Настроить наблюдателей по умолчанию."""
        # Логирование
        logging_observer = LoggingGraphObserver()
        self.add_observer("logging", logging_observer)

        # Метрики
        metrics_observer = MetricsGraphObserver()
        self.add_observer("metrics", metrics_observer)

        # Инвалидация кеша
        cache_manager = GraphCacheManager()
        cache_observer = CacheInvalidationObserver(cache_manager)
        self.add_observer("cache", cache_observer)

        logger.info("Default observers set up")

    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику по наблюдателям."""
        stats = {
            "total_observers": len(self._observers),
            "observer_names": list(self._observers.keys())
        }

        # Добавляем статистику от MetricsGraphObserver если есть
        metrics_observer = self.get_observer("metrics")
        if isinstance(metrics_observer, MetricsGraphObserver):
            stats["metrics"] = metrics_observer.get_metrics()

        return stats

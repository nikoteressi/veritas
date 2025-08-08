"""
Менеджер билдеров для пошагового построения графов.

Инкапсулирует логику Builder Pattern для создания графов.
"""

import logging
from typing import Dict, List, Optional

from agent.models import FactGraph

from .patterns.builder import (
    FactGraphBuilder, GraphBuilderDirector, GraphBuilderFactory,
    BuilderConfiguration, BuilderValidationLevel
)

logger = logging.getLogger(__name__)


class BuilderManager:
    """Менеджер для управления билдерами графов."""

    def __init__(self):
        """Инициализация менеджера билдеров."""
        self._factory = GraphBuilderFactory()
        self._directors: Dict[str, GraphBuilderDirector] = {}
        self._active_builders: Dict[str, FactGraphBuilder] = {}

    def create_basic_builder(self, builder_id: Optional[str] = None) -> FactGraphBuilder:
        """
        Создать базовый билдер.

        Args:
            builder_id: Идентификатор билдера

        Returns:
            Экземпляр базового билдера
        """
        builder = self._factory.create_basic_builder()

        if builder_id:
            self._active_builders[builder_id] = builder
            logger.debug(f"Basic builder created with ID: {builder_id}")

        return builder

    def create_validation_builder(
        self,
        validation_level: BuilderValidationLevel = BuilderValidationLevel.STRICT,
        builder_id: Optional[str] = None
    ) -> FactGraphBuilder:
        """
        Создать билдер с валидацией.

        Args:
            validation_level: Уровень валидации
            builder_id: Идентификатор билдера

        Returns:
            Экземпляр билдера с валидацией
        """
        # Используем доступные методы фабрики в зависимости от уровня валидации
        if validation_level == BuilderValidationLevel.BASIC:
            builder = self._factory.create_basic_builder()
        elif validation_level == BuilderValidationLevel.STRICT:
            builder = self._factory.create_strict_builder()
        elif validation_level == BuilderValidationLevel.COMPREHENSIVE:
            builder = self._factory.create_comprehensive_builder()
        else:
            # По умолчанию используем строгий билдер
            builder = self._factory.create_strict_builder()

        if builder_id:
            self._active_builders[builder_id] = builder
            logger.debug(
                f"Validation builder created with ID: {builder_id}, level: {validation_level}")

        return builder

    def create_performance_builder(self, builder_id: Optional[str] = None) -> FactGraphBuilder:
        """
        Создать производительный билдер.

        Args:
            builder_id: Идентификатор билдера

        Returns:
            Экземпляр производительного билдера
        """
        # Используем комплексный билдер для производительности (с оптимизацией и кластеризацией)
        builder = self._factory.create_comprehensive_builder()

        if builder_id:
            self._active_builders[builder_id] = builder
            logger.debug(f"Performance builder created with ID: {builder_id}")

        return builder

    def create_custom_builder(
        self,
        config: BuilderConfiguration,
        builder_id: Optional[str] = None
    ) -> FactGraphBuilder:
        """
        Создать пользовательский билдер.

        Args:
            config: Конфигурация билдера
            builder_id: Идентификатор билдера

        Returns:
            Экземпляр пользовательского билдера
        """
        builder = self._factory.create_custom_builder(config)

        if builder_id:
            self._active_builders[builder_id] = builder
            logger.debug(f"Custom builder created with ID: {builder_id}")

        return builder

    def get_builder(self, builder_id: str) -> Optional[FactGraphBuilder]:
        """
        Получить активный билдер по ID.

        Args:
            builder_id: Идентификатор билдера

        Returns:
            Экземпляр билдера или None
        """
        return self._active_builders.get(builder_id)

    def remove_builder(self, builder_id: str) -> bool:
        """
        Удалить активный билдер.

        Args:
            builder_id: Идентификатор билдера

        Returns:
            True если билдер удален успешно
        """
        if builder_id in self._active_builders:
            del self._active_builders[builder_id]
            logger.debug(f"Builder removed: {builder_id}")
            return True
        return False

    def create_director(
        self,
        director_id: str,
        builder: Optional[FactGraphBuilder] = None
    ) -> GraphBuilderDirector:
        """
        Создать директора для управления билдером.

        Args:
            director_id: Идентификатор директора
            builder: Билдер для директора

        Returns:
            Экземпляр директора
        """
        if builder is None:
            builder = self.create_basic_builder()

        director = GraphBuilderDirector(builder)
        self._directors[director_id] = director

        logger.debug(f"Director created with ID: {director_id}")
        return director

    def get_director(self, director_id: str) -> Optional[GraphBuilderDirector]:
        """
        Получить директора по ID.

        Args:
            director_id: Идентификатор директора

        Returns:
            Экземпляр директора или None
        """
        return self._directors.get(director_id)

    def remove_director(self, director_id: str) -> bool:
        """
        Удалить директора.

        Args:
            director_id: Идентификатор директора

        Returns:
            True если директор удален успешно
        """
        if director_id in self._directors:
            del self._directors[director_id]
            logger.debug(f"Director removed: {director_id}")
            return True
        return False

    async def build_simple_graph(
        self,
        facts: List[str],
        builder_id: Optional[str] = None
    ) -> FactGraph:
        """
        Построить простой граф из списка фактов.

        Args:
            facts: Список фактов
            builder_id: Идентификатор билдера

        Returns:
            Построенный граф
        """
        builder = self.create_basic_builder(builder_id)
        director = GraphBuilderDirector(builder)

        return director.build_simple_graph(facts)

    async def build_hierarchical_graph(
        self,
        root_facts: List[str],
        child_facts: Dict[str, List[str]],
        builder_id: Optional[str] = None
    ) -> FactGraph:
        """
        Построить иерархический граф.

        Args:
            root_facts: Корневые факты
            child_facts: Дочерние факты для каждого корневого
            builder_id: Идентификатор билдера

        Returns:
            Построенный граф
        """
        builder = self.create_validation_builder(builder_id=builder_id)
        director = GraphBuilderDirector(builder)

        # Создаем иерархический граф используя build_connected_graph
        all_facts = root_facts.copy()
        relationships = []

        # Добавляем дочерние факты и создаем связи
        for root_fact in root_facts:
            if root_fact in child_facts:
                root_idx = all_facts.index(root_fact)
                for child_fact in child_facts[root_fact]:
                    if child_fact not in all_facts:
                        all_facts.append(child_fact)
                    child_idx = all_facts.index(child_fact)
                    relationships.append((root_idx, child_idx, "child_of"))

        return director.build_connected_graph(all_facts, relationships)

    async def build_clustered_graph(
        self,
        facts: List[str],
        cluster_size: int = 5,
        builder_id: Optional[str] = None
    ) -> FactGraph:
        """
        Построить кластеризованный граф.

        Args:
            facts: Список фактов
            cluster_size: Размер кластера
            builder_id: Идентификатор билдера

        Returns:
            Построенный граф
        """
        builder = self.create_performance_builder(builder_id)
        director = GraphBuilderDirector(builder)

        # Создаем простые связи между соседними фактами
        relationships = []
        for i in range(len(facts) - 1):
            relationships.append((i, i + 1, "related_to"))

        # Создаем кластеры на основе размера кластера
        clusters = []
        for i in range(0, len(facts), cluster_size):
            cluster_facts = list(range(i, min(i + cluster_size, len(facts))))
            if len(cluster_facts) >= 2:  # Кластер должен содержать минимум 2 узла
                clusters.append({
                    "node_indices": cluster_facts,
                    "cluster_type": "semantic",
                    "shared_context": f"Cluster {len(clusters) + 1}",
                    "metadata": {"size": len(cluster_facts)}
                })

        return director.build_clustered_graph(facts, relationships, clusters)

    def list_active_builders(self) -> List[str]:
        """Получить список активных билдеров."""
        return list(self._active_builders.keys())

    def list_active_directors(self) -> List[str]:
        """Получить список активных директоров."""
        return list(self._directors.keys())

    def clear_all(self) -> None:
        """Очистить все активные билдеры и директоров."""
        self._active_builders.clear()
        self._directors.clear()
        logger.debug("All builders and directors cleared")

    def get_stats(self) -> Dict[str, int]:
        """Получить статистику по билдерам."""
        return {
            "active_builders": len(self._active_builders),
            "active_directors": len(self._directors)
        }

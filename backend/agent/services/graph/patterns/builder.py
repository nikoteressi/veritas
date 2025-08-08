"""
Builder Pattern Implementation для Graph Service.

Этот модуль реализует паттерн Builder для пошагового построения
графа фактов с валидацией и оптимизацией.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import logging
from datetime import datetime
from dataclasses import dataclass, field

from agent.models import FactGraph, FactNode, FactEdge, FactCluster

logger = logging.getLogger(__name__)


class BuilderValidationLevel(Enum):
    """Уровни валидации при построении графа"""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"


@dataclass
class BuilderConfiguration:
    """Конфигурация для builder'а графа"""
    validation_level: BuilderValidationLevel = BuilderValidationLevel.BASIC
    auto_optimize: bool = True
    auto_cluster: bool = False
    max_nodes: Optional[int] = None
    max_edges: Optional[int] = None
    allow_duplicate_edges: bool = False
    allow_self_loops: bool = True
    enable_logging: bool = True
    custom_validators: List[Callable] = field(default_factory=list)


@dataclass
class BuildStep:
    """Шаг построения графа"""
    step_type: str
    description: str
    timestamp: datetime
    data: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None


class GraphBuilder(ABC):
    """Абстрактный базовый класс для builder'ов графа"""

    @abstractmethod
    def reset(self) -> 'GraphBuilder':
        """Сбросить состояние builder'а"""
        pass

    @abstractmethod
    def build(self) -> FactGraph:
        """Построить граф"""
        pass


class FactGraphBuilder(GraphBuilder):
    """Конкретный builder для построения графа фактов"""

    def __init__(self, config: Optional[BuilderConfiguration] = None):
        self._config = config or BuilderConfiguration()
        self._graph: Optional[FactGraph] = None
        self._build_steps: List[BuildStep] = []
        self._validation_errors: List[str] = []
        self._pending_nodes: Dict[str, FactNode] = {}
        self._pending_edges: List[FactEdge] = []
        self._pending_clusters: Dict[str, FactCluster] = {}
        self._node_id_counter = 0
        self._edge_id_counter = 0
        self._cluster_id_counter = 0
        self.reset()

    def reset(self) -> 'FactGraphBuilder':
        """Сбросить состояние builder'а"""
        self._graph = FactGraph()
        self._build_steps.clear()
        self._validation_errors.clear()
        self._pending_nodes.clear()
        self._pending_edges.clear()
        self._pending_clusters.clear()
        self._node_id_counter = 0
        self._edge_id_counter = 0
        self._cluster_id_counter = 0

        self._log_step("reset", "Builder reset", {})
        return self

    def with_configuration(self, config: BuilderConfiguration) -> 'FactGraphBuilder':
        """Установить конфигурацию"""
        self._config = config
        self._log_step("configure", "Configuration updated",
                       {"config": str(config)})
        return self

    def add_node(
        self,
        content: str,
        node_type: str = "fact",
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None
    ) -> 'FactGraphBuilder':
        """Добавить узел в граф"""
        if node_id is None:
            node_id = f"node_{self._node_id_counter}"
            self._node_id_counter += 1

        # Проверяем лимиты
        if self._config.max_nodes and len(self._pending_nodes) >= self._config.max_nodes:
            error = f"Maximum number of nodes ({self._config.max_nodes}) exceeded"
            self._validation_errors.append(error)
            self._log_step("add_node", "Failed to add node",
                           {"error": error}, success=False)
            return self

        # Проверяем дубликаты
        if node_id in self._pending_nodes:
            error = f"Node with id {node_id} already exists"
            self._validation_errors.append(error)
            self._log_step("add_node", "Failed to add node",
                           {"error": error}, success=False)
            return self

        node = FactNode(
            id=node_id,
            claim=content,
            domain=node_type,
            metadata=metadata or {},
            created_at=datetime.now()
        )

        # Валидация узла
        if not self._validate_node(node):
            return self

        self._pending_nodes[node_id] = node
        self._log_step("add_node", f"Added node {node_id}", {
            "node_id": node_id,
            "content": content[:50] + "..." if len(content) > 50 else content,
            "node_type": node_type
        })

        return self

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        edge_id: Optional[str] = None
    ) -> 'FactGraphBuilder':
        """Добавить ребро в граф"""
        if edge_id is None:
            edge_id = f"edge_{self._edge_id_counter}"
            self._edge_id_counter += 1

        # Проверяем лимиты
        if self._config.max_edges and len(self._pending_edges) >= self._config.max_edges:
            error = f"Maximum number of edges ({self._config.max_edges}) exceeded"
            self._validation_errors.append(error)
            self._log_step("add_edge", "Failed to add edge",
                           {"error": error}, success=False)
            return self

        # Проверяем существование узлов
        if source_id not in self._pending_nodes:
            error = f"Source node {source_id} not found"
            self._validation_errors.append(error)
            self._log_step("add_edge", "Failed to add edge",
                           {"error": error}, success=False)
            return self

        if target_id not in self._pending_nodes:
            error = f"Target node {target_id} not found"
            self._validation_errors.append(error)
            self._log_step("add_edge", "Failed to add edge",
                           {"error": error}, success=False)
            return self

        # Проверяем самозацикливание
        if not self._config.allow_self_loops and source_id == target_id:
            error = f"Self-loops are not allowed"
            self._validation_errors.append(error)
            self._log_step("add_edge", "Failed to add edge",
                           {"error": error}, success=False)
            return self

        edge = FactEdge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship,
            strength=weight,
            metadata=metadata or {},
            created_at=datetime.now()
        )

        # Проверяем дубликаты ребер
        if not self._config.allow_duplicate_edges:
            for existing_edge in self._pending_edges:
                if (existing_edge.source_id == source_id and
                    existing_edge.target_id == target_id and
                        existing_edge.relationship_type == relationship):
                    error = f"Duplicate edge not allowed: {source_id} -> {target_id} ({relationship})"
                    self._validation_errors.append(error)
                    self._log_step("add_edge", "Failed to add edge", {
                                   "error": error}, success=False)
                    return self

        # Валидация ребра
        if not self._validate_edge(edge):
            return self

        self._pending_edges.append(edge)
        self._log_step("add_edge", f"Added edge {edge_id}", {
            "edge_id": edge_id,
            "source_id": source_id,
            "target_id": target_id,
            "relationship": relationship,
            "weight": weight
        })

        return self

    def add_cluster(
        self,
        node_ids: List[str],
        cluster_type: str = "semantic",
        shared_context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        cluster_id: Optional[str] = None
    ) -> 'FactGraphBuilder':
        """Добавить кластер в граф"""
        if cluster_id is None:
            cluster_id = f"cluster_{self._cluster_id_counter}"
            self._cluster_id_counter += 1

        # Проверяем существование узлов
        missing_nodes = [
            nid for nid in node_ids if nid not in self._pending_nodes]
        if missing_nodes:
            error = f"Nodes not found for cluster: {missing_nodes}"
            self._validation_errors.append(error)
            self._log_step("add_cluster", "Failed to add cluster", {
                           "error": error}, success=False)
            return self

        # Получаем узлы для кластера
        cluster_nodes = [self._pending_nodes[nid] for nid in node_ids]

        cluster = FactCluster(
            id=cluster_id,
            nodes=cluster_nodes,
            cluster_type=cluster_type,
            shared_context=shared_context,
            metadata=metadata or {},
            created_at=datetime.now()
        )

        # Валидация кластера
        if not self._validate_cluster(cluster):
            return self

        self._pending_clusters[cluster_id] = cluster
        self._log_step("add_cluster", f"Added cluster {cluster_id}", {
            "cluster_id": cluster_id,
            "node_count": len(node_ids),
            "cluster_type": cluster_type
        })

        return self

    def add_nodes_batch(self, nodes_data: List[Dict[str, Any]]) -> 'FactGraphBuilder':
        """Добавить несколько узлов за раз"""
        for node_data in nodes_data:
            self.add_node(**node_data)
        return self

    def add_edges_batch(self, edges_data: List[Dict[str, Any]]) -> 'FactGraphBuilder':
        """Добавить несколько ребер за раз"""
        for edge_data in edges_data:
            self.add_edge(**edge_data)
        return self

    def auto_cluster_by_type(self) -> 'FactGraphBuilder':
        """Автоматически создать кластеры по типу узлов"""
        type_groups = {}
        for node in self._pending_nodes.values():
            node_type = node.domain
            if node_type not in type_groups:
                type_groups[node_type] = []
            type_groups[node_type].append(node.id)

        for node_type, node_ids in type_groups.items():
            if len(node_ids) > 1:  # Создаем кластер только если больше одного узла
                self.add_cluster(
                    node_ids=node_ids,
                    cluster_type="type_based",
                    shared_context=f"Nodes of type: {node_type}"
                )

        self._log_step("auto_cluster", f"Created {len(type_groups)} type-based clusters", {
            "cluster_types": list(type_groups.keys())
        })

        return self

    def validate(self) -> bool:
        """Выполнить валидацию построенного графа"""
        self._validation_errors.clear()

        if self._config.validation_level == BuilderValidationLevel.NONE:
            return True

        # Базовая валидация
        if self._config.validation_level.value in ["basic", "strict", "comprehensive"]:
            self._validate_basic()

        # Строгая валидация
        if self._config.validation_level.value in ["strict", "comprehensive"]:
            self._validate_strict()

        # Комплексная валидация
        if self._config.validation_level == BuilderValidationLevel.COMPREHENSIVE:
            self._validate_comprehensive()

        # Пользовательские валидаторы
        for validator in self._config.custom_validators:
            try:
                validator(self)
            except Exception as e:
                self._validation_errors.append(f"Custom validator error: {e}")

        is_valid = len(self._validation_errors) == 0
        self._log_step("validate", f"Validation {'passed' if is_valid else 'failed'}", {
            "errors_count": len(self._validation_errors),
            "validation_level": self._config.validation_level.value
        }, success=is_valid)

        return is_valid

    def build(self) -> FactGraph:
        """Построить граф"""
        # Валидация перед построением
        if not self.validate():
            error_msg = f"Validation failed with {len(self._validation_errors)} errors"
            self._log_step("build", error_msg, {
                           "errors": self._validation_errors}, success=False)
            raise ValueError(
                f"{error_msg}: {'; '.join(self._validation_errors)}")

        # Добавляем узлы в граф
        for node in self._pending_nodes.values():
            self._graph.add_node(node)

        # Добавляем ребра в граф
        for edge in self._pending_edges:
            self._graph.add_edge(edge)

        # Добавляем кластеры в граф
        for cluster in self._pending_clusters.values():
            self._graph.add_cluster(cluster)

        # Автоматическая оптимизация
        if self._config.auto_optimize:
            # Здесь можно добавить логику оптимизации
            pass

        # Автоматическая кластеризация
        if self._config.auto_cluster:
            self.auto_cluster_by_type()

        self._log_step("build", "Graph built successfully", {
            "nodes_count": len(self._pending_nodes),
            "edges_count": len(self._pending_edges),
            "clusters_count": len(self._pending_clusters)
        })

        return self._graph

    def get_build_steps(self) -> List[BuildStep]:
        """Получить шаги построения"""
        return self._build_steps.copy()

    def get_validation_errors(self) -> List[str]:
        """Получить ошибки валидации"""
        return self._validation_errors.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Получить статистику построения"""
        return {
            "pending_nodes": len(self._pending_nodes),
            "pending_edges": len(self._pending_edges),
            "pending_clusters": len(self._pending_clusters),
            "build_steps": len(self._build_steps),
            "validation_errors": len(self._validation_errors),
            "successful_steps": sum(1 for step in self._build_steps if step.success),
            "failed_steps": sum(1 for step in self._build_steps if not step.success)
        }

    def _validate_node(self, node: FactNode) -> bool:
        """Валидация узла"""
        if not node.claim.strip():
            error = f"Node {node.id} has empty claim"
            self._validation_errors.append(error)
            self._log_step("validate_node", "Node validation failed", {
                           "error": error}, success=False)
            return False

        return True

    def _validate_edge(self, edge: FactEdge) -> bool:
        """Валидация ребра"""
        if not str(edge.relationship_type).strip():
            error = f"Edge {edge.id} has empty relationship_type"
            self._validation_errors.append(error)
            self._log_step("validate_edge", "Edge validation failed", {
                           "error": error}, success=False)
            return False

        if edge.strength < 0:
            error = f"Edge {edge.id} has negative strength"
            self._validation_errors.append(error)
            self._log_step("validate_edge", "Edge validation failed", {
                           "error": error}, success=False)
            return False

        return True

    def _validate_cluster(self, cluster: FactCluster) -> bool:
        """Валидация кластера"""
        if len(cluster.nodes) < 2:
            error = f"Cluster {cluster.id} must have at least 2 nodes"
            self._validation_errors.append(error)
            self._log_step("validate_cluster", "Cluster validation failed", {
                           "error": error}, success=False)
            return False

        return True

    def _validate_basic(self):
        """Базовая валидация"""
        # Проверяем, что есть хотя бы один узел
        if not self._pending_nodes:
            self._validation_errors.append("Graph must have at least one node")

    def _validate_strict(self):
        """Строгая валидация"""
        # Проверяем связность графа
        if len(self._pending_nodes) > 1 and not self._pending_edges:
            self._validation_errors.append(
                "Graph with multiple nodes must have edges")

        # Проверяем, что все ребра ссылаются на существующие узлы
        for edge in self._pending_edges:
            if edge.source_id not in self._pending_nodes:
                self._validation_errors.append(
                    f"Edge {edge.id} references non-existent source node {edge.source_id}")
            if edge.target_id not in self._pending_nodes:
                self._validation_errors.append(
                    f"Edge {edge.id} references non-existent target node {edge.target_id}")

    def _validate_comprehensive(self):
        """Комплексная валидация"""
        # Проверяем на изолированные узлы
        connected_nodes = set()
        for edge in self._pending_edges:
            connected_nodes.add(edge.source_id)
            connected_nodes.add(edge.target_id)

        isolated_nodes = set(self._pending_nodes.keys()) - connected_nodes
        if isolated_nodes:
            self._validation_errors.append(
                f"Found isolated nodes: {list(isolated_nodes)}")

        # Проверяем на циклы (простая проверка)
        # Здесь можно добавить более сложную логику обнаружения циклов

    def _log_step(self, step_type: str, description: str, data: Dict[str, Any], success: bool = True):
        """Логировать шаг построения"""
        step = BuildStep(
            step_type=step_type,
            description=description,
            timestamp=datetime.now(),
            data=data,
            success=success,
            error_message=None if success else data.get("error")
        )

        self._build_steps.append(step)

        if self._config.enable_logging:
            level = logging.INFO if success else logging.WARNING
            logger.log(level, f"Builder step: {description}")


class GraphBuilderDirector:
    """Директор для управления процессом построения графа"""

    def __init__(self, builder: FactGraphBuilder):
        self._builder = builder

    def build_simple_graph(self, facts: List[str]) -> FactGraph:
        """Построить простой граф из списка фактов"""
        self._builder.reset()

        # Добавляем узлы для каждого факта
        for i, fact in enumerate(facts):
            self._builder.add_node(
                claim=fact,
                domain="fact",
                node_id=f"fact_{i}"
            )

        return self._builder.build()

    def build_connected_graph(self, facts: List[str], relationships: List[tuple]) -> FactGraph:
        """Построить связанный граф с заданными отношениями"""
        self._builder.reset()

        # Добавляем узлы
        for i, fact in enumerate(facts):
            self._builder.add_node(
                claim=fact,
                domain="fact",
                node_id=f"fact_{i}"
            )

        # Добавляем ребра
        for source_idx, target_idx, relationship in relationships:
            if 0 <= source_idx < len(facts) and 0 <= target_idx < len(facts):
                self._builder.add_edge(
                    source_id=f"fact_{source_idx}",
                    target_id=f"fact_{target_idx}",
                    relationship_type=relationship
                )

        return self._builder.build()

    def build_clustered_graph(
        self,
        facts: List[str],
        relationships: List[tuple],
        clusters: List[Dict[str, Any]]
    ) -> FactGraph:
        """Построить граф с кластерами"""
        self._builder.reset()

        # Добавляем узлы
        for i, fact in enumerate(facts):
            self._builder.add_node(
                claim=fact,
                domain="fact",
                node_id=f"fact_{i}"
            )

        # Добавляем ребра
        for source_idx, target_idx, relationship in relationships:
            if 0 <= source_idx < len(facts) and 0 <= target_idx < len(facts):
                self._builder.add_edge(
                    source_id=f"fact_{source_idx}",
                    target_id=f"fact_{target_idx}",
                    relationship_type=relationship
                )

        # Добавляем кластеры
        for cluster_data in clusters:
            node_indices = cluster_data.get("node_indices", [])
            node_ids = [
                f"fact_{i}" for i in node_indices if 0 <= i < len(facts)]

            if len(node_ids) >= 2:
                self._builder.add_cluster(
                    node_ids=node_ids,
                    cluster_type=cluster_data.get("cluster_type", "semantic"),
                    shared_context=cluster_data.get("shared_context"),
                    metadata=cluster_data.get("metadata", {})
                )

        return self._builder.build()


# Фабрика для создания builder'ов

class GraphBuilderFactory:
    """Фабрика для создания различных типов builder'ов"""

    @staticmethod
    def create_basic_builder() -> FactGraphBuilder:
        """Создать базовый builder"""
        config = BuilderConfiguration(
            validation_level=BuilderValidationLevel.BASIC,
            auto_optimize=False,
            auto_cluster=False
        )
        return FactGraphBuilder(config)

    @staticmethod
    def create_strict_builder() -> FactGraphBuilder:
        """Создать строгий builder с валидацией"""
        config = BuilderConfiguration(
            validation_level=BuilderValidationLevel.STRICT,
            auto_optimize=True,
            auto_cluster=False,
            allow_duplicate_edges=False,
            allow_self_loops=False
        )
        return FactGraphBuilder(config)

    @staticmethod
    def create_comprehensive_builder() -> FactGraphBuilder:
        """Создать комплексный builder"""
        config = BuilderConfiguration(
            validation_level=BuilderValidationLevel.COMPREHENSIVE,
            auto_optimize=True,
            auto_cluster=True,
            allow_duplicate_edges=False,
            allow_self_loops=True
        )
        return FactGraphBuilder(config)

    @staticmethod
    def create_custom_builder(config: BuilderConfiguration) -> FactGraphBuilder:
        """Создать builder с пользовательской конфигурацией"""
        return FactGraphBuilder(config)

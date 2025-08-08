"""
Базовые операции с графом фактов.

Инкапсулирует CRUD операции для работы с узлами, ребрами и кластерами.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent.models import FactCluster, FactEdge, FactGraph, FactNode
from agent.models.graph import ClusterType

logger = logging.getLogger(__name__)


class GraphOperations:
    """Класс для выполнения базовых операций с графом."""

    def __init__(self):
        """Инициализация операций с графом."""
        pass

    async def add_node(
        self,
        graph: FactGraph,
        content: str,
        node_type: str = "fact",
        metadata: Optional[Dict[str, Any]] = None
    ) -> FactNode:
        """
        Добавить узел в граф.

        Args:
            graph: Граф для добавления узла
            content: Содержимое узла
            node_type: Тип узла
            metadata: Метаданные узла

        Returns:
            Созданный узел
        """
        node_id = str(uuid.uuid4())
        node = FactNode(
            id=node_id,
            claim=content,
            domain=node_type,
            metadata=metadata or {},
            created_at=datetime.now()
        )

        graph.nodes[node_id] = node
        logger.debug(f"Node added to graph: {node_id}")
        return node

    async def remove_node(self, graph: FactGraph, node_id: str) -> bool:
        """
        Удалить узел из графа.

        Args:
            graph: Граф для удаления узла
            node_id: Идентификатор узла

        Returns:
            True если узел удален успешно
        """
        if node_id not in graph.nodes:
            logger.warning(f"Node not found for removal: {node_id}")
            return False

        # Удаляем все связанные ребра
        edges_to_remove = []
        for edge_id, edge in graph.edges.items():
            if edge.source_id == node_id or edge.target_id == node_id:
                edges_to_remove.append(edge_id)

        for edge_id in edges_to_remove:
            await self.remove_edge(graph, edge_id)

        # Удаляем узел
        del graph.nodes[node_id]
        logger.debug(f"Node removed from graph: {node_id}")
        return True

    async def update_node(
        self,
        graph: FactGraph,
        node_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Обновить узел в графе.

        Args:
            graph: Граф с узлом
            node_id: Идентификатор узла
            content: Новое содержимое узла
            metadata: Новые метаданные узла

        Returns:
            True если узел обновлен успешно
        """
        if node_id not in graph.nodes:
            logger.warning(f"Node not found for update: {node_id}")
            return False

        node = graph.nodes[node_id]

        if content is not None:
            node.content = content

        if metadata is not None:
            node.metadata.update(metadata)

        logger.debug(f"Node updated in graph: {node_id}")
        return True

    async def add_edge(
        self,
        graph: FactGraph,
        source_id: str,
        target_id: str,
        edge_type: str = "relates_to",
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FactEdge:
        """
        Добавить ребро в граф.

        Args:
            graph: Граф для добавления ребра
            source_id: Идентификатор исходного узла
            target_id: Идентификатор целевого узла
            edge_type: Тип ребра
            weight: Вес ребра
            metadata: Метаданные ребра

        Returns:
            Созданное ребро
        """
        # Проверяем существование узлов
        if source_id not in graph.nodes:
            raise ValueError(f"Source node not found: {source_id}")
        if target_id not in graph.nodes:
            raise ValueError(f"Target node not found: {target_id}")

        edge_id = str(uuid.uuid4())
        edge = FactEdge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relationship_type=edge_type,
            strength=weight,
            metadata=metadata or {},
            created_at=datetime.now()
        )

        graph.edges[edge_id] = edge
        logger.debug(f"Edge added to graph: {edge_id}")
        return edge

    async def remove_edge(self, graph: FactGraph, edge_id: str) -> bool:
        """
        Удалить ребро из графа.

        Args:
            graph: Граф для удаления ребра
            edge_id: Идентификатор ребра

        Returns:
            True если ребро удалено успешно
        """
        if edge_id not in graph.edges:
            logger.warning(f"Edge not found for removal: {edge_id}")
            return False

        del graph.edges[edge_id]
        logger.debug(f"Edge removed from graph: {edge_id}")
        return True

    async def update_edge(
        self,
        graph: FactGraph,
        edge_id: str,
        weight: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Обновить ребро в графе.

        Args:
            graph: Граф с ребром
            edge_id: Идентификатор ребра
            weight: Новый вес ребра
            metadata: Новые метаданные ребра

        Returns:
            True если ребро обновлено успешно
        """
        if edge_id not in graph.edges:
            logger.warning(f"Edge not found for update: {edge_id}")
            return False

        edge = graph.edges[edge_id]

        if weight is not None:
            edge.weight = weight

        if metadata is not None:
            edge.metadata.update(metadata)

        logger.debug(f"Edge updated in graph: {edge_id}")
        return True

    async def create_cluster(
        self,
        graph: FactGraph,
        node_ids: List[str],
        cluster_type: str = "semantic",
        metadata: Optional[Dict[str, Any]] = None
    ) -> FactCluster:
        """
        Создать кластер узлов.

        Args:
            graph: Граф для создания кластера
            node_ids: Список идентификаторов узлов
            cluster_type: Тип кластера
            metadata: Метаданные кластера

        Returns:
            Созданный кластер
        """
        # Проверяем существование всех узлов и получаем объекты узлов
        nodes = []
        for node_id in node_ids:
            if node_id not in graph.nodes:
                raise ValueError(f"Node not found: {node_id}")
            nodes.append(graph.nodes[node_id])

        # Преобразуем строковый cluster_type в enum
        if isinstance(cluster_type, str):
            cluster_type_enum = ClusterType.SIMILARITY_CLUSTER  # default
            for ct in ClusterType:
                if ct.value == cluster_type:
                    cluster_type_enum = ct
                    break
        else:
            cluster_type_enum = cluster_type

        cluster_id = str(uuid.uuid4())
        cluster = FactCluster(
            id=cluster_id,
            nodes=nodes,
            cluster_type=cluster_type_enum,
            metadata=metadata or {},
            created_at=datetime.now()
        )

        graph.clusters[cluster_id] = cluster
        logger.debug(f"Cluster created in graph: {cluster_id}")
        return cluster

    async def remove_cluster(self, graph: FactGraph, cluster_id: str) -> bool:
        """
        Удалить кластер из графа.

        Args:
            graph: Граф для удаления кластера
            cluster_id: Идентификатор кластера

        Returns:
            True если кластер удален успешно
        """
        if cluster_id not in graph.clusters:
            logger.warning(f"Cluster not found for removal: {cluster_id}")
            return False

        del graph.clusters[cluster_id]
        logger.debug(f"Cluster removed from graph: {cluster_id}")
        return True

    async def update_cluster(
        self,
        graph: FactGraph,
        cluster_id: str,
        node_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Обновить кластер в графе.

        Args:
            graph: Граф с кластером
            cluster_id: Идентификатор кластера
            node_ids: Новый список узлов кластера
            metadata: Новые метаданные кластера

        Returns:
            True если кластер обновлен успешно
        """
        if cluster_id not in graph.clusters:
            logger.warning(f"Cluster not found for update: {cluster_id}")
            return False

        cluster = graph.clusters[cluster_id]

        if node_ids is not None:
            # Проверяем существование всех узлов и получаем объекты узлов
            nodes = []
            for node_id in node_ids:
                if node_id not in graph.nodes:
                    raise ValueError(f"Node not found: {node_id}")
                nodes.append(graph.nodes[node_id])
            cluster.nodes = nodes

        if metadata is not None:
            cluster.metadata.update(metadata)

        logger.debug(f"Cluster updated in graph: {cluster_id}")
        return True

    def get_node_neighbors(self, graph: FactGraph, node_id: str) -> List[str]:
        """
        Получить соседей узла.

        Args:
            graph: Граф с узлом
            node_id: Идентификатор узла

        Returns:
            Список идентификаторов соседних узлов
        """
        neighbors = set()

        for edge in graph.edges.values():
            if edge.source_id == node_id:
                neighbors.add(edge.target_id)
            elif edge.target_id == node_id:
                neighbors.add(edge.source_id)

        return list(neighbors)

    def get_node_degree(self, graph: FactGraph, node_id: str) -> int:
        """
        Получить степень узла (количество связей).

        Args:
            graph: Граф с узлом
            node_id: Идентификатор узла

        Returns:
            Степень узла
        """
        return len(self.get_node_neighbors(graph, node_id))

    def get_graph_stats(self, graph: FactGraph) -> Dict[str, int]:
        """
        Получить статистику графа.

        Args:
            graph: Граф для анализа

        Returns:
            Словарь со статистикой
        """
        return {
            "nodes_count": len(graph.nodes),
            "edges_count": len(graph.edges),
            "clusters_count": len(graph.clusters)
        }

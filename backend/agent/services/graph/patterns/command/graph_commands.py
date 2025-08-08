"""
Конкретные команды для операций с графом фактов.

Этот модуль содержит реализации команд для добавления, удаления и модификации
узлов, ребер и кластеров в графе фактов с поддержкой undo/redo.
"""

from typing import Any
import logging
from agent.models.graph import FactGraph, FactNode, FactEdge, FactCluster
from .base_command import UndoableCommand

logger = logging.getLogger(__name__)


class AddNodeCommand(UndoableCommand):
    """Команда добавления узла в граф"""

    def __init__(self, graph: FactGraph, node: FactNode):
        super().__init__()
        self.graph = graph
        self.node = node

    @property
    def description(self) -> str:
        return f"Add node {self.node.id} to graph"

    async def _prepare_undo_data(self) -> dict:
        return {
            "node_existed": self.node.id in self.graph.nodes,
            "node_id": self.node.id
        }

    async def _do_execute(self) -> Any:
        self.graph.add_node(self.node)
        logger.debug(f"Added node {self.node.id} to graph")
        return self.node.id

    async def _do_undo(self) -> Any:
        if not self.undo_data["node_existed"]:
            self.graph.remove_node(self.node.id)
            logger.debug(f"Removed node {self.node.id} from graph (undo)")
        return self.node.id


class RemoveNodeCommand(UndoableCommand):
    """Команда удаления узла из графа"""

    def __init__(self, graph: FactGraph, node_id: str):
        super().__init__()
        self.graph = graph
        self.node_id = node_id

    @property
    def description(self) -> str:
        return f"Remove node {self.node_id} from graph"

    async def _prepare_undo_data(self) -> dict:
        node = self.graph.nodes.get(self.node_id)
        # Сохраняем связанные ребра для восстановления
        connected_edges = [
            edge for edge in self.graph.edges.values()
            if edge.source_id == self.node_id or edge.target_id == self.node_id
        ]

        # Сохраняем кластеры, содержащие этот узел
        node_clusters = self.graph.get_node_clusters(
            self.node_id) if node else []

        return {
            "node": node,
            "connected_edges": connected_edges,
            "node_clusters": node_clusters
        }

    async def _do_execute(self) -> Any:
        if self.node_id not in self.graph.nodes:
            raise ValueError(f"Node {self.node_id} not found in graph")

        # Удаляем узел (метод remove_node автоматически удаляет связанные ребра и обновляет кластеры)
        self.graph.remove_node(self.node_id)
        logger.debug(f"Removed node {self.node_id} from graph")
        return self.node_id

    async def _do_undo(self) -> Any:
        # Восстанавливаем узел
        if self.undo_data["node"]:
            self.graph.add_node(self.undo_data["node"])
            logger.debug(f"Restored node {self.node_id} to graph")

        # Восстанавливаем связанные ребра
        for edge in self.undo_data["connected_edges"]:
            self.graph.add_edge(edge)
            logger.debug(f"Restored edge {edge.id}")

        # Восстанавливаем кластеры
        for cluster in self.undo_data["node_clusters"]:
            if cluster.id not in self.graph.clusters:
                self.graph.add_cluster(cluster)
            else:
                # Добавляем узел обратно в существующий кластер
                existing_cluster = self.graph.clusters[cluster.id]
                if self.undo_data["node"] not in existing_cluster.nodes:
                    existing_cluster.add_node(self.undo_data["node"])

        return self.node_id


class AddEdgeCommand(UndoableCommand):
    """Команда добавления ребра в граф"""

    def __init__(self, graph: FactGraph, edge: FactEdge):
        super().__init__()
        self.graph = graph
        self.edge = edge

    @property
    def description(self) -> str:
        return f"Add edge {self.edge.id} ({self.edge.source_id} -> {self.edge.target_id})"

    async def _prepare_undo_data(self) -> dict:
        return {
            "edge_existed": self.edge.id in self.graph.edges,
            "edge_id": self.edge.id
        }

    async def _do_execute(self) -> Any:
        # Проверяем существование узлов
        if self.edge.source_id not in self.graph.nodes:
            raise ValueError(f"Source node {self.edge.source_id} not found")
        if self.edge.target_id not in self.graph.nodes:
            raise ValueError(f"Target node {self.edge.target_id} not found")

        self.graph.add_edge(self.edge)
        logger.debug(f"Added edge {self.edge.id} to graph")
        return self.edge.id

    async def _do_undo(self) -> Any:
        if not self.undo_data["edge_existed"]:
            self.graph.remove_edge(self.edge.id)
            logger.debug(f"Removed edge {self.edge.id} from graph (undo)")
        return self.edge.id


class RemoveEdgeCommand(UndoableCommand):
    """Команда удаления ребра из графа"""

    def __init__(self, graph: FactGraph, edge_id: str):
        super().__init__()
        self.graph = graph
        self.edge_id = edge_id

    @property
    def description(self) -> str:
        return f"Remove edge {self.edge_id} from graph"

    async def _prepare_undo_data(self) -> dict:
        edge = self.graph.edges.get(self.edge_id)
        return {
            "edge": edge,
            "edge_existed": edge is not None
        }

    async def _do_execute(self) -> Any:
        if self.edge_id not in self.graph.edges:
            raise ValueError(f"Edge {self.edge_id} not found in graph")

        self.graph.remove_edge(self.edge_id)
        logger.debug(f"Removed edge {self.edge_id} from graph")
        return self.edge_id

    async def _do_undo(self) -> Any:
        if self.undo_data["edge_existed"] and self.undo_data["edge"]:
            self.graph.add_edge(self.undo_data["edge"])
            logger.debug(f"Restored edge {self.edge_id} to graph")
        return self.edge_id


class CreateClusterCommand(UndoableCommand):
    """Команда создания кластера"""

    def __init__(self, graph: FactGraph, cluster: FactCluster):
        super().__init__()
        self.graph = graph
        self.cluster = cluster

    @property
    def description(self) -> str:
        return f"Create cluster {self.cluster.id} with {len(self.cluster.nodes)} nodes"

    async def _prepare_undo_data(self) -> dict:
        return {
            "cluster_existed": self.cluster.id in self.graph.clusters,
            "cluster_id": self.cluster.id
        }

    async def _do_execute(self) -> Any:
        # Проверяем существование узлов кластера
        for node in self.cluster.nodes:
            if node.id not in self.graph.nodes:
                raise ValueError(f"Node {node.id} not found in graph")

        self.graph.add_cluster(self.cluster)
        logger.debug(
            f"Created cluster {self.cluster.id} with {len(self.cluster.nodes)} nodes")
        return self.cluster.id

    async def _do_undo(self) -> Any:
        if not self.undo_data["cluster_existed"]:
            self.graph.remove_cluster(self.cluster.id)
            logger.debug(f"Removed cluster {self.cluster.id} (undo)")
        return self.cluster.id


class RemoveClusterCommand(UndoableCommand):
    """Команда удаления кластера"""

    def __init__(self, graph: FactGraph, cluster_id: str):
        super().__init__()
        self.graph = graph
        self.cluster_id = cluster_id

    @property
    def description(self) -> str:
        return f"Remove cluster {self.cluster_id} from graph"

    async def _prepare_undo_data(self) -> dict:
        cluster = self.graph.clusters.get(self.cluster_id)
        return {
            "cluster": cluster,
            "cluster_existed": cluster is not None
        }

    async def _do_execute(self) -> Any:
        if self.cluster_id not in self.graph.clusters:
            raise ValueError(f"Cluster {self.cluster_id} not found in graph")

        self.graph.remove_cluster(self.cluster_id)
        logger.debug(f"Removed cluster {self.cluster_id} from graph")
        return self.cluster_id

    async def _do_undo(self) -> Any:
        if self.undo_data["cluster_existed"] and self.undo_data["cluster"]:
            self.graph.add_cluster(self.undo_data["cluster"])
            logger.debug(f"Restored cluster {self.cluster_id} to graph")
        return self.cluster_id


class OptimizeGraphCommand(UndoableCommand):
    """Команда оптимизации графа"""

    def __init__(self, graph: FactGraph, optimization_config: dict):
        super().__init__()
        self.graph = graph
        self.config = optimization_config

    @property
    def description(self) -> str:
        return f"Optimize graph with config: {self.config}"

    async def _prepare_undo_data(self) -> dict:
        # Сохраняем полное состояние графа
        return {
            "nodes": dict(self.graph.nodes),
            "edges": dict(self.graph.edges),
            "clusters": dict(self.graph.clusters),
            "node_to_clusters": {k: v.copy() for k, v in self.graph.node_to_clusters.items()}
        }

    async def _do_execute(self) -> Any:
        optimization_stats = {
            "removed_edges": 0,
            "merged_nodes": 0,
            "removed_isolated_nodes": 0
        }

        # Удаление слабых ребер
        if self.config.get("remove_weak_edges", True):
            weak_threshold = self.config.get("weak_edge_threshold", 0.3)
            weak_edges = [
                edge_id for edge_id, edge in self.graph.edges.items()
                if edge.strength < weak_threshold
            ]
            for edge_id in weak_edges:
                del self.graph.edges[edge_id]
            optimization_stats["removed_edges"] = len(weak_edges)
            logger.debug(f"Removed {len(weak_edges)} weak edges")

        # Удаление изолированных узлов
        if self.config.get("remove_isolated_nodes", True):
            connected_nodes = set()
            for edge in self.graph.edges.values():
                connected_nodes.add(edge.source_id)
                connected_nodes.add(edge.target_id)

            isolated_nodes = [
                node_id for node_id in self.graph.nodes.keys()
                if node_id not in connected_nodes
            ]

            for node_id in isolated_nodes:
                # Удаляем из кластеров
                cluster_ids = self.graph.node_to_clusters.get(
                    node_id, set()).copy()
                for cluster_id in cluster_ids:
                    if cluster_id in self.graph.clusters:
                        cluster = self.graph.clusters[cluster_id]
                        node = self.graph.nodes[node_id]
                        cluster.remove_node(node)

                # Удаляем узел
                del self.graph.nodes[node_id]
                if node_id in self.graph.node_to_clusters:
                    del self.graph.node_to_clusters[node_id]

            optimization_stats["removed_isolated_nodes"] = len(isolated_nodes)
            logger.debug(f"Removed {len(isolated_nodes)} isolated nodes")

        logger.info(f"Graph optimization completed: {optimization_stats}")
        return optimization_stats

    async def _do_undo(self) -> Any:
        # Восстанавливаем полное состояние графа
        self.graph.nodes = self.undo_data["nodes"]
        self.graph.edges = self.undo_data["edges"]
        self.graph.clusters = self.undo_data["clusters"]
        self.graph.node_to_clusters = self.undo_data["node_to_clusters"]
        logger.info("Graph state restored from optimization")
        return "Graph state restored"


class UpdateNodeCommand(UndoableCommand):
    """Команда обновления узла в графе"""

    def __init__(self, graph: FactGraph, node_id: str, updates: dict):
        super().__init__()
        self.graph = graph
        self.node_id = node_id
        self.updates = updates

    @property
    def description(self) -> str:
        return f"Update node {self.node_id} with {list(self.updates.keys())}"

    async def _prepare_undo_data(self) -> dict:
        node = self.graph.nodes.get(self.node_id)
        if not node:
            raise ValueError(f"Node {self.node_id} not found")

        # Сохраняем текущие значения обновляемых полей
        original_values = {}
        for field in self.updates.keys():
            if hasattr(node, field):
                original_values[field] = getattr(node, field)

        return {
            "original_values": original_values,
            "node_id": self.node_id
        }

    async def _do_execute(self) -> Any:
        node = self.graph.nodes.get(self.node_id)
        if not node:
            raise ValueError(f"Node {self.node_id} not found")

        # Применяем обновления
        updated_fields = []
        for field, value in self.updates.items():
            if hasattr(node, field):
                setattr(node, field, value)
                updated_fields.append(field)
            else:
                logger.warning(
                    f"Field {field} not found in node {self.node_id}")

        logger.debug(f"Updated node {self.node_id} fields: {updated_fields}")
        return updated_fields

    async def _do_undo(self) -> Any:
        node = self.graph.nodes.get(self.node_id)
        if not node:
            logger.warning(f"Node {self.node_id} not found during undo")
            return []

        # Восстанавливаем оригинальные значения
        restored_fields = []
        for field, value in self.undo_data["original_values"].items():
            if hasattr(node, field):
                setattr(node, field, value)
                restored_fields.append(field)

        logger.debug(f"Restored node {self.node_id} fields: {restored_fields}")
        return restored_fields

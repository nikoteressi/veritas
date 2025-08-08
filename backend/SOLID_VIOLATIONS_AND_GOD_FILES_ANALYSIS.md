# Анализ "God Files" и нарушений принципов SOLID

## Обзор

Данный анализ выявляет крупные файлы ("god files") и нарушения принципов SOLID в модуле `graph` services, а также предлагает применимые паттерны проектирования для улучшения архитектуры.

## God Files (Файлы-монстры)

### Выявленные крупные файлы

1. **graph_fact_checking.py** - 823 строки
   - Основной сервис для проверки фактов
   - Множественные обязанности: кэширование, построение графов, кластеризация, анализ

2. **graph_builder.py** - 751 строка  
   - Сервис для построения графов
   - Отвечает за создание узлов, отношений, кластеров, оптимизацию

3. **verification/source_manager.py** - 607 строк
   - Менеджер источников
   - Скрапинг, кэширование, анализ релевантности

4. **verification/engine.py** - 583 строки
   - Движок верификации
   - Координация процесса верификации

5. **graph_storage.py** - 540 строк
   - Хранилище графов
   - Операции с Neo4j базой данных

## Нарушения принципов SOLID

### 1. Single Responsibility Principle (SRP) - КРИТИЧЕСКИ НАРУШЕН

#### GraphFactCheckingService

- ❌ Кэширование результатов
- ❌ Построение графов  
- ❌ Кластеризация фактов
- ❌ Анализ отношений
- ❌ Управление репутацией источников
- ❌ Верификация фактов
- ❌ Байесовский анализ неопределенности
- ❌ Обновление хранилища

#### GraphBuilder

- ❌ Создание узлов графа
- ❌ Обнаружение отношений
- ❌ Формирование кластеров (4 разных типа)
- ❌ Оптимизация кластеров
- ❌ Генерация эмбеддингов
- ❌ Управление кэшем

#### EnhancedSourceManager

- ❌ Веб-скрапинг
- ❌ Кэширование контента
- ❌ Анализ релевантности
- ❌ Управление адаптивными порогами
- ❌ Обработка метаданных

### 2. Open/Closed Principle (OCP) - НАРУШЕН

```python
# Проблема: жестко закодированные алгоритмы
async def _form_clusters(self, nodes, edges, graph):
    # Similarity-based clustering - жестко закодировано
    similarity_clusters = await self._create_similarity_clusters(nodes, graph)
    
    # Domain-based clustering - жестко закодировано  
    if self.config.enable_domain_clustering:
        domain_clusters = await self._create_domain_clusters(nodes, graph)
```

**Проблемы:**

- Невозможно добавить новые алгоритмы кластеризации без изменения кода
- Фиксированные типы отношений
- Жестко привязанные стратегии верификации

### 3. Liskov Substitution Principle (LSP) - НАРУШЕН

**Проблемы:**

- Отсутствие абстракций и интерфейсов
- Прямые зависимости от конкретных классов
- Невозможность подстановки реализаций

### 4. Interface Segregation Principle (ISP) - НАРУШЕН

**Проблемы:**

- Отсутствие интерфейсов
- Монолитные классы с множественными обязанностями
- Клиенты вынуждены зависеть от методов, которые не используют

### 5. Dependency Inversion Principle (DIP) - КРИТИЧЕСКИ НАРУШЕН

```python
# Проблема: прямые импорты конкретных классов
from agent.llm.embeddings import OllamaEmbeddingFunction
from .graph_storage import Neo4jGraphStorage
from .clustering import AdvancedClusteringSystem

# Жесткая связанность в конструкторе
def __init__(self):
    self.embeddings = OllamaEmbeddingFunction(...)  # Прямая зависимость
    self.storage = Neo4jGraphStorage(...)           # Прямая зависимость
```

## Применимые паттерны проектирования

### 1. Strategy Pattern - для алгоритмов

```python
from abc import ABC, abstractmethod

class ClusteringStrategy(ABC):
    @abstractmethod
    async def create_clusters(self, nodes: list[FactNode]) -> list[FactCluster]:
        pass

class SimilarityClusteringStrategy(ClusteringStrategy):
    async def create_clusters(self, nodes: list[FactNode]) -> list[FactCluster]:
        # Реализация DBSCAN кластеризации
        embeddings = await self._get_embeddings([node.claim for node in nodes])
        clustering = DBSCAN(eps=self.config.dbscan_eps).fit(embeddings)
        return self._build_clusters_from_labels(nodes, clustering.labels_)

class DomainClusteringStrategy(ClusteringStrategy):
    async def create_clusters(self, nodes: list[FactNode]) -> list[FactCluster]:
        # Реализация доменной кластеризации
        domain_groups = self._group_by_domain(nodes)
        return self._build_domain_clusters(domain_groups)

class TemporalClusteringStrategy(ClusteringStrategy):
    async def create_clusters(self, nodes: list[FactNode]) -> list[FactCluster]:
        # Реализация временной кластеризации
        temporal_groups = self._find_temporal_relationships(nodes)
        return self._build_temporal_clusters(temporal_groups)
```

### 2. Factory Pattern - для создания компонентов

```python
class GraphComponentFactory:
    @staticmethod
    def create_clustering_strategy(strategy_type: str, config: ClusteringConfig) -> ClusteringStrategy:
        strategies = {
            "similarity": SimilarityClusteringStrategy,
            "domain": DomainClusteringStrategy,
            "temporal": TemporalClusteringStrategy,
            "causal": CausalClusteringStrategy,
        }
        
        if strategy_type not in strategies:
            raise ValueError(f"Unknown clustering strategy: {strategy_type}")
            
        return strategies[strategy_type](config)
    
    @staticmethod
    def create_verification_engine(engine_type: str) -> VerificationEngine:
        engines = {
            "standard": StandardVerificationEngine,
            "enhanced": EnhancedVerificationEngine,
            "bayesian": BayesianVerificationEngine,
        }
        return engines[engine_type]()
```

### 3. Repository Pattern - для абстракции хранения

```python
from abc import ABC, abstractmethod

class GraphRepository(ABC):
    @abstractmethod
    async def save_graph(self, graph: FactGraph) -> None:
        pass
    
    @abstractmethod
    async def load_graph(self, graph_id: str) -> FactGraph | None:
        pass
    
    @abstractmethod
    async def delete_graph(self, graph_id: str) -> bool:
        pass
    
    @abstractmethod
    async def search_graphs(self, criteria: dict) -> list[FactGraph]:
        pass

class Neo4jGraphRepository(GraphRepository):
    def __init__(self, connection_config: Neo4jConfig):
        self.config = connection_config
        self.driver = None
    
    async def save_graph(self, graph: FactGraph) -> None:
        async with self.driver.session() as session:
            # Реализация сохранения в Neo4j
            pass
    
    async def load_graph(self, graph_id: str) -> FactGraph | None:
        async with self.driver.session() as session:
            # Реализация загрузки из Neo4j
            pass

class InMemoryGraphRepository(GraphRepository):
    def __init__(self):
        self._graphs: dict[str, FactGraph] = {}
    
    async def save_graph(self, graph: FactGraph) -> None:
        self._graphs[graph.id] = graph
    
    async def load_graph(self, graph_id: str) -> FactGraph | None:
        return self._graphs.get(graph_id)
```

### 4. Builder Pattern - для сложных объектов

```python
class FactGraphBuilder:
    def __init__(self):
        self._graph = FactGraph()
        self._clustering_strategies: list[ClusteringStrategy] = []
        self._relationship_analyzers: list[RelationshipAnalyzer] = []
    
    def add_clustering_strategy(self, strategy: ClusteringStrategy) -> 'FactGraphBuilder':
        self._clustering_strategies.append(strategy)
        return self
    
    def add_relationship_analyzer(self, analyzer: RelationshipAnalyzer) -> 'FactGraphBuilder':
        self._relationship_analyzers.append(analyzer)
        return self
    
    async def build_from_facts(self, facts: list[Fact]) -> FactGraph:
        # Создание узлов
        nodes = await self._create_nodes(facts)
        
        # Анализ отношений
        edges = await self._analyze_relationships(nodes)
        
        # Кластеризация
        clusters = await self._create_clusters(nodes, edges)
        
        # Сборка графа
        self._graph.nodes = {node.id: node for node in nodes}
        self._graph.edges = {edge.id: edge for edge in edges}
        self._graph.clusters = {cluster.id: cluster for cluster in clusters}
        
        return self._graph
```

### 5. Command Pattern - для операций с графом

```python
class GraphCommand(ABC):
    @abstractmethod
    async def execute(self) -> Any:
        pass
    
    @abstractmethod
    async def undo(self) -> Any:
        pass

class AddNodeCommand(GraphCommand):
    def __init__(self, graph: FactGraph, node: FactNode):
        self.graph = graph
        self.node = node
        self._executed = False
    
    async def execute(self):
        if not self._executed:
            self.graph.add_node(self.node)
            self._executed = True
    
    async def undo(self):
        if self._executed:
            self.graph.remove_node(self.node.id)
            self._executed = False

class CreateClusterCommand(GraphCommand):
    def __init__(self, graph: FactGraph, cluster: FactCluster):
        self.graph = graph
        self.cluster = cluster
        self._executed = False
    
    async def execute(self):
        if not self._executed:
            self.graph.add_cluster(self.cluster)
            self._executed = True
    
    async def undo(self):
        if self._executed:
            self.graph.remove_cluster(self.cluster.id)
            self._executed = False

class GraphCommandInvoker:
    def __init__(self):
        self._history: list[GraphCommand] = []
    
    async def execute_command(self, command: GraphCommand):
        await command.execute()
        self._history.append(command)
    
    async def undo_last_command(self):
        if self._history:
            command = self._history.pop()
            await command.undo()
```

### 6. Observer Pattern - для уведомлений

```python
class GraphObserver(ABC):
    @abstractmethod
    async def on_node_added(self, node: FactNode):
        pass
    
    @abstractmethod
    async def on_cluster_created(self, cluster: FactCluster):
        pass
    
    @abstractmethod
    async def on_verification_completed(self, results: VerificationResults):
        pass

class GraphEventPublisher:
    def __init__(self):
        self._observers: list[GraphObserver] = []
    
    def subscribe(self, observer: GraphObserver):
        self._observers.append(observer)
    
    def unsubscribe(self, observer: GraphObserver):
        self._observers.remove(observer)
    
    async def notify_node_added(self, node: FactNode):
        for observer in self._observers:
            await observer.on_node_added(node)
    
    async def notify_cluster_created(self, cluster: FactCluster):
        for observer in self._observers:
            await observer.on_cluster_created(cluster)

class CacheInvalidationObserver(GraphObserver):
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
    
    async def on_node_added(self, node: FactNode):
        # Инвалидация кэша при добавлении узла
        await self.cache_manager.invalidate_graph_cache()
    
    async def on_cluster_created(self, cluster: FactCluster):
        # Инвалидация кэша при создании кластера
        await self.cache_manager.invalidate_cluster_cache()
```

### 7. Dependency Injection Container

```python
from typing import TypeVar, Type, Callable, Dict, Any

T = TypeVar('T')

class DIContainer:
    def __init__(self):
        self._services: Dict[Type, Callable] = {}
        self._singletons: Dict[Type, Any] = {}
    
    def register_transient(self, interface: Type[T], implementation: Type[T]):
        """Регистрация transient сервиса (новый экземпляр каждый раз)"""
        self._services[interface] = implementation
    
    def register_singleton(self, interface: Type[T], implementation: Type[T]):
        """Регистрация singleton сервиса (один экземпляр)"""
        def singleton_factory():
            if interface not in self._singletons:
                self._singletons[interface] = implementation()
            return self._singletons[interface]
        
        self._services[interface] = singleton_factory
    
    def register_instance(self, interface: Type[T], instance: T):
        """Регистрация готового экземпляра"""
        self._singletons[interface] = instance
        self._services[interface] = lambda: instance
    
    def resolve(self, interface: Type[T]) -> T:
        """Разрешение зависимости"""
        if interface not in self._services:
            raise ValueError(f"Service {interface} not registered")
        
        return self._services[interface]()

# Использование:
container = DIContainer()

# Регистрация сервисов
container.register_singleton(GraphRepository, Neo4jGraphRepository)
container.register_transient(ClusteringStrategy, SimilarityClusteringStrategy)
container.register_instance(ClusteringConfig, ClusteringConfig())

# Разрешение зависимостей
repository = container.resolve(GraphRepository)
strategy = container.resolve(ClusteringStrategy)
```

## Рекомендуемый план рефакторинга

### Фаза 1: Разделение монолитных классов (2-3 недели)

#### 1.1 Разделение GraphFactCheckingService

```python
# Вместо одного монолитного класса:
class FactVerificationOrchestrator:
    """Координация процесса верификации"""
    def __init__(self, 
                 graph_builder: GraphBuilder,
                 verification_engine: VerificationEngine,
                 uncertainty_analyzer: UncertaintyAnalyzer,
                 result_compiler: ResultCompiler):
        self.graph_builder = graph_builder
        self.verification_engine = verification_engine
        self.uncertainty_analyzer = uncertainty_analyzer
        self.result_compiler = result_compiler

class GraphCacheManager:
    """Управление кэшем графов"""
    async def get_cached_graph(self, cache_key: str) -> FactGraph | None:
        pass
    
    async def cache_graph(self, cache_key: str, graph: FactGraph):
        pass

class VerificationResultCompiler:
    """Компиляция результатов верификации"""
    async def compile_results(self, 
                            verification_results: list[VerificationResult],
                            uncertainty_analysis: UncertaintyAnalysis) -> FactCheckingResult:
        pass

class UncertaintyAnalyzer:
    """Байесовский анализ неопределенности"""
    async def analyze_uncertainty(self, graph: FactGraph) -> UncertaintyAnalysis:
        pass
```

#### 1.2 Разделение GraphBuilder

```python
class NodeFactory:
    """Создание узлов графа"""
    async def create_nodes(self, facts: list[Fact]) -> list[FactNode]:
        pass

class RelationshipDetector:
    """Обнаружение отношений между узлами"""
    async def detect_relationships(self, nodes: list[FactNode]) -> list[FactEdge]:
        pass

class ClusterManager:
    """Управление кластеризацией"""
    def __init__(self, strategies: list[ClusteringStrategy]):
        self.strategies = strategies
    
    async def create_clusters(self, nodes: list[FactNode], edges: list[FactEdge]) -> list[FactCluster]:
        pass

class GraphAssembler:
    """Сборка финального графа"""
    async def assemble_graph(self, 
                           nodes: list[FactNode], 
                           edges: list[FactEdge], 
                           clusters: list[FactCluster]) -> FactGraph:
        pass
```

### Фаза 2: Внедрение паттернов (1-2 недели)

#### 2.1 Strategy Pattern для кластеризации

#### 2.2 Repository Pattern для хранения

#### 2.3 Factory Pattern для создания компонентов

### Фаза 3: Dependency Injection (1 неделя)

#### 3.1 Создание DI контейнера

#### 3.2 Регистрация всех сервисов

#### 3.3 Рефакторинг конструкторов

### Фаза 4: Дополнительные паттерны (1 неделя)

#### 4.1 Command Pattern для операций

#### 4.2 Observer Pattern для уведомлений

#### 4.3 Builder Pattern для сложных объектов

## Ожидаемые результаты

### Улучшения архитектуры

- ✅ Соблюдение принципов SOLID
- ✅ Высокая тестируемость
- ✅ Легкость расширения
- ✅ Слабая связанность компонентов
- ✅ Возможность замены реализаций

### Улучшения качества кода

- ✅ Уменьшение размера классов
- ✅ Четкое разделение обязанностей
- ✅ Улучшенная читаемость
- ✅ Упрощенное тестирование
- ✅ Лучшая поддерживаемость

### Метрики качества

- **Цикломатическая сложность**: снижение с 15-20 до 5-8
- **Размер классов**: снижение с 500+ до 100-200 строк
- **Связанность**: снижение с высокой до низкой
- **Сплоченность**: повышение с низкой до высокой
- **Покрытие тестами**: увеличение с 0% до 80%+

## Заключение

Текущая архитектура модуля `graph` services страдает от серьезных нарушений принципов SOLID и содержит несколько "god files". Предложенный план рефакторинга с применением паттернов проектирования позволит значительно улучшить качество кода, тестируемость и расширяемость системы.

Рекомендуется начать с разделения монолитных классов (Фаза 1), так как это даст наибольший эффект и создаст основу для дальнейших улучшений.

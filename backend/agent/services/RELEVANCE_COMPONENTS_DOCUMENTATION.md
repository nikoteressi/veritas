# Документация по компонентам релевантности

## Обзор

Система компонентов релевантности предоставляет комплексное решение для анализа и оценки релевантности документов с использованием современных методов машинного обучения, кэширования и временного анализа.

## Архитектура компонентов

### 1. EnhancedOllamaEmbeddings
**Файл:** `enhanced_ollama_embeddings.py`

Улучшенный компонент для работы с эмбеддингами Ollama с интеллектуальным кэшированием.

**Основные возможности:**
- Генерация эмбеддингов с кэшированием
- Поиск по семантическому сходству
- Пакетная обработка документов
- Метрики производительности
- Автоматическая оптимизация

**Пример использования:**
```python
from backend.agent.services.enhanced_ollama_embeddings import EnhancedOllamaEmbeddings
from backend.agent.services.intelligent_cache import IntelligentCache

cache = IntelligentCache()
embeddings = EnhancedOllamaEmbeddings(
    ollama_host="http://localhost:11434",
    model_name="nomic-embed-text",
    cache=cache
)

# Генерация эмбеддингов
documents = ["Document 1", "Document 2"]
embeddings_result = await embeddings.embed_documents(documents)

# Поиск по сходству
results = await embeddings.similarity_search(
    query="search query",
    documents=documents,
    k=5
)
```

### 2. CachedHybridRelevanceScorer
**Файл:** `cached_hybrid_relevance_scorer.py`

Гибридный скорер релевантности, объединяющий BM25 и семантическое сходство.

**Основные возможности:**
- Комбинированный скоринг (BM25 + семантический)
- Интеллектуальное кэширование результатов
- Ранжирование документов
- Настраиваемые веса компонентов
- Метрики производительности

**Пример использования:**
```python
from backend.agent.services.cached_hybrid_relevance_scorer import CachedHybridRelevanceScorer

scorer = CachedHybridRelevanceScorer(
    embeddings=embeddings,
    cache=cache,
    bm25_weight=0.3,
    semantic_weight=0.7
)

# Скоринг документов
results = await scorer.score_documents(
    query="climate change",
    documents=documents
)

# Ранжирование
ranked = await scorer.rank_documents(
    query="climate change",
    documents=documents
)
```

### 3. TemporalAnalysisCache
**Файл:** `temporal_analysis_cache.py`

Компонент для временного анализа релевантности с учетом актуальности и трендов.

**Основные возможности:**
- Анализ актуальности документов
- Выявление временных трендов
- Временное затухание релевантности
- Пакетная обработка
- Настраиваемые временные веса

**Пример использования:**
```python
from backend.agent.services.temporal_analysis_cache import TemporalAnalysisCache

temporal_cache = TemporalAnalysisCache(cache=cache)

# Анализ временной релевантности
results = await temporal_cache.analyze_temporal_relevance(
    documents=documents,
    metadata=metadata,
    query="climate change"
)

# Пакетный анализ
batch_results = await temporal_cache.batch_analyze(
    documents_list=[docs1, docs2],
    metadata_list=[meta1, meta2],
    queries=["query1", "query2"]
)
```

### 4. ExplainableRelevanceScorer
**Файл:** `explainable_relevance_scorer.py`

Компонент для создания объяснимых оценок релевантности с детальным анализом.

**Основные возможности:**
- Базовые объяснения релевантности
- Детальный лингвистический анализ
- Анализ важности признаков
- Механизмы внимания
- Рекомендации по улучшению

**Пример использования:**
```python
from backend.agent.services.explainable_relevance_scorer import ExplainableRelevanceScorer

explainer = ExplainableRelevanceScorer(
    hybrid_scorer=scorer,
    temporal_cache=temporal_cache,
    cache=cache
)

# Базовое объяснение
basic = await explainer.generate_basic_explanation(
    query="climate change",
    document="document text",
    metadata={"timestamp": "2024-01-15T10:00:00Z"}
)

# Детальное объяснение
detailed = await explainer.generate_detailed_explanation(
    query="climate change",
    document="document text",
    metadata=metadata
)

# Комплексное объяснение
comprehensive = await explainer.generate_comprehensive_explanation(
    query="climate change",
    document="document text",
    metadata=metadata
)
```

### 5. RelevanceIntegrationManager
**Файл:** `relevance_integration.py`

Центральный менеджер для интеграции всех компонентов релевантности.

**Основные возможности:**
- Комплексный анализ релевантности
- Пакетная обработка запросов
- Управление производительностью
- Автоматическая оптимизация
- Отчеты о производительности

**Пример использования:**
```python
from backend.agent.services.relevance_integration import get_relevance_manager, close_relevance_manager

# Получение менеджера
manager = await get_relevance_manager()

# Комплексный анализ
results = await manager.calculate_comprehensive_relevance(
    query="climate change",
    documents=documents,
    document_metadata=metadata,
    explain=True
)

# Пакетный анализ
batch_results = await manager.batch_analyze_relevance(
    queries=["query1", "query2"],
    document_sets=[docs1, docs2],
    metadata_sets=[meta1, meta2],
    explain=False
)

# Отчет о производительности
report = await manager.get_performance_report()

# Закрытие менеджера
await close_relevance_manager()
```

## Интеграция с существующими компонентами

### EnhancedSourceManager
Компонент `EnhancedSourceManager` был обновлен для использования новых компонентов релевантности:

```python
# В методе filter_evidence_for_cluster
await self._ensure_relevance_manager()
if self.relevance_manager:
    relevance_results = await self.relevance_manager.calculate_comprehensive_relevance(
        query=cluster_query,
        documents=[item.get('content', '') for item in scraped_content],
        document_metadata=[{
            'url': item.get('url', ''),
            'timestamp': item.get('timestamp', ''),
            'title': item.get('title', '')
        } for item in scraped_content],
        explain=False
    )
```

## Конфигурация и настройка

### Переменные окружения
```bash
# Ollama настройки
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=nomic-embed-text

# Кэш настройки
CACHE_TTL=3600
CACHE_MAX_SIZE=10000

# Релевантность настройки
BM25_WEIGHT=0.3
SEMANTIC_WEIGHT=0.7
TEMPORAL_WEIGHT=0.2
```

### Настройки компонентов
```python
# Настройки гибридного скорера
scorer_config = {
    "bm25_weight": 0.3,
    "semantic_weight": 0.7,
    "min_score_threshold": 0.1
}

# Настройки временного анализа
temporal_config = {
    "recency_weight": 0.4,
    "trend_weight": 0.3,
    "decay_factor": 0.1,
    "time_window_days": 30
}

# Настройки объяснений
explanation_config = {
    "max_features": 100,
    "attention_threshold": 0.1,
    "linguistic_features": True
}
```

## Метрики и мониторинг

### Доступные метрики
- **Производительность кэша:** hit rate, miss rate, размер кэша
- **Производительность эмбеддингов:** время генерации, количество запросов
- **Качество релевантности:** средние скоры, распределение скоров
- **Временные метрики:** время обработки, пропускная способность

### Получение метрик
```python
# Метрики компонентов
embeddings_metrics = await embeddings.get_performance_metrics()
scorer_metrics = await scorer.get_performance_metrics()
temporal_metrics = await temporal_cache.get_performance_metrics()

# Общий отчет
performance_report = await manager.get_performance_report()
```

## Оптимизация производительности

### Автоматическая оптимизация
```python
# Оптимизация через менеджер
optimization_results = await manager.optimize_performance()

# Ручная оптимизация компонентов
await embeddings.optimize_cache()
await scorer.optimize_performance()
await temporal_cache.optimize_cache()
```

### Рекомендации по производительности
1. **Кэширование:** Используйте подходящие TTL для разных типов данных
2. **Пакетная обработка:** Обрабатывайте документы пакетами для лучшей производительности
3. **Мониторинг:** Регулярно проверяйте метрики производительности
4. **Оптимизация:** Запускайте оптимизацию в периоды низкой нагрузки

## Обработка ошибок

### Типичные ошибки и решения
```python
try:
    results = await manager.calculate_comprehensive_relevance(...)
except ConnectionError:
    # Проблемы с подключением к Ollama
    logger.error("Ollama connection failed, using fallback")
    # Использование fallback метода
except CacheError:
    # Проблемы с кэшем
    logger.error("Cache error, clearing cache")
    await cache.clear()
except Exception as e:
    # Общие ошибки
    logger.error(f"Unexpected error: {e}")
    # Fallback к базовому методу
```

## Тестирование

Для тестирования компонентов используйте файл `test_relevance_components.py`:

```bash
cd backend/agent/services
python test_relevance_components.py
```

Тесты покрывают:
- Функциональность каждого компонента
- Интеграцию между компонентами
- Производительность и метрики
- Обработку ошибок

## Миграция и обновление

### Обновление зависимостей
```bash
pip install -r requirements.txt
```

### Миграция данных кэша
При обновлении компонентов может потребоваться очистка кэша:
```python
await cache.clear()
await cache.optimize()
```

## Поддержка и отладка

### Логирование
Все компоненты используют структурированное логирование:
```python
import logging
logger = logging.getLogger(__name__)

# Настройка уровня логирования
logging.basicConfig(level=logging.INFO)
```

### Отладочная информация
```python
# Включение детального логирования
logger.setLevel(logging.DEBUG)

# Получение отладочной информации
debug_info = await manager.get_debug_info()
```

## Заключение

Система компонентов релевантности предоставляет мощные инструменты для анализа и оценки релевантности документов. Компоненты спроектированы для высокой производительности, масштабируемости и простоты использования.

Для получения дополнительной помощи обращайтесь к исходному коду компонентов или запускайте тесты для проверки функциональности.
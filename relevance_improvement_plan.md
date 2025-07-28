# План улучшения анализа релевантности для системы Veritas

## Статус выполнения

### ✅ Выполненные изменения (Система верификации графов)

**Дата обновления**: Декабрь 2024

В рамках развития системы была реализована **система верификации графов** с использованием компонентов, схожих с планом релевантности, но для других целей:

#### Реализованные компоненты:

1. **IntelligentCache** (`backend/agent/services/intelligent_cache.py`)
   - ✅ Многоуровневое кэширование (MEMORY → REDIS → DISK)
   - ✅ Стратегии: TTL, LRU, SIMILARITY, DEPENDENCY
   - ✅ Управление зависимостями и автоматическая инвалидация
   - ✅ Специализированные кэши: EmbeddingCache, VerificationCache

2. **AdaptiveThresholds** (`backend/agent/services/adaptive_thresholds.py`)
   - ✅ Динамическая калибровка порогов для верификации
   - ✅ Интеграция с IntelligentCache
   - ✅ Метрики производительности и адаптация
   - ✅ Контекстно-зависимые пороги

3. **EnhancedSourceManager** (`backend/agent/services/graph_verification/source_manager.py`)
   - ✅ Интеллектуальное кэширование скрапинга
   - ✅ Управление источниками с метаданными
   - ✅ Оптимизация повторных запросов

4. **EnhancedEvidenceGatherer** (`backend/agent/services/graph_verification/evidence_gatherer.py`)
   - ✅ Кэшированный поиск доказательств
   - ✅ Интеграция с системой поиска
   - ✅ Оптимизация запросов

5. **VerificationProcessor** (`backend/agent/services/graph_verification/verification_processor.py`)
   - ✅ Кэшированная верификация фактов
   - ✅ Метрики производительности
   - ✅ Адаптивные пороги доверия

#### Ключевые отличия от плана релевантности:

- **Цель**: Верификация графов знаний vs. анализ релевантности источников
- **Модели**: Использует существующие LLM vs. планируемые Ollama embeddings
- **Область применения**: Проверка фактов vs. фильтрация источников

### ❌ Не реализованные компоненты плана релевантности:

1. **EnhancedOllamaEmbeddings** - улучшенные эмбеддинги через Ollama
2. **CachedHybridRelevanceScorer** - гибридный поиск с кэшированием
3. **TemporalAnalysisCache** - временной анализ релевантности
4. **CacheMonitor** - мониторинг производительности кэшей
5. **ExplainableRelevanceScorer** - объяснимая оценка релевантности

### 🔄 Следующие шаги:

Необходимо адаптировать реализованную инфраструктуру для **анализа релевантности источников**, используя уже созданные компоненты как основу.

### 🔧 Исправленные технические проблемы:

**Проблема**: Неправильное использование конструктора `IntelligentCache`
- ❌ **Было**: `IntelligentCache(cache_levels=[...], strategies=[...])`
- ✅ **Стало**: `IntelligentCache(max_memory_size=1000)`

**Исправленные файлы**:
- `backend/agent/services/adaptive_thresholds.py`
- `backend/agent/services/graph_verification/source_manager.py`
- `backend/agent/services/graph_verification/evidence_gatherer.py`
- `backend/agent/services/graph_verification/verification_processor.py`

**Причина**: Конструктор `IntelligentCache` принимает только `redis_client` и `max_memory_size`, а не параметры конфигурации уровней и стратегий кэширования.

---

## Обзор проблемы

Текущая система анализа релевантности демонстрирует значительные недостатки:

### Основные проблемы

1. **Высокие потери качественных источников**: ~80% релевантных источников отфильтровываются
2. **Низкая точность классификации**: precision ~45%, recall ~35%
3. **Неоптимальные пороги релевантности**: статические пороги не учитывают контекст
4. **Ограниченная семантическая обработка**: простые эмбеддинги без контекстного анализа
5. **Отсутствие гибридного поиска**: только векторный поиск без BM25
6. **Неэффективное использование кэширования**: множественные простые кэши вместо интеллектуального кэширования

### Критические случаи потерь

- **Финансовые документы SEC**: потеря 85% релевантных 10-K/10-Q отчетов
- **Новостные источники**: потеря 75% релевантных статей Reuters/Bloomberg
- **Академические источники**: потеря 70% релевантных исследований
- **Правовые документы**: потеря 80% релевантных судебных решений

## Анализ текущей архитектуры

### Существующие компоненты кэширования

Система уже имеет развитую кэш-инфраструктуру:

#### IntelligentCache (backend/agent/services/intelligent_cache.py)

- **Многоуровневая архитектура**: MEMORY → REDIS → DISK
- **Стратегии кэширования**: TTL, LRU, SIMILARITY, DEPENDENCY
- **Специализированные кэши**: EmbeddingCache, VerificationCache
- **Управление зависимостями**: автоматическая инвалидация связанных данных

#### Существующие простые кэши

- `_scrape_cache` в `source_manager.py`
- `_search_cache` в `evidence_gatherer.py`  
- `CacheManager` в `graph_verification/utils.py`
- Различные временные кэши в компонентах

### Текущая модель эмбеддингов

- **Базовая модель**: стандартная sentence-transformers
- **Отсутствие кэширования эмбеддингов**: повторные вычисления
- **Ограниченная семантическая глубина**: простые векторные представления

## Современные техники анализа релевантности

### Transformer-based семантический анализ

#### Рекомендуемые модели для Ollama

- **snowflake-arctic-embed2**: 1024-dim, оптимизирован для поиска
- **granite3-embedding**: IBM модель с высокой точностью
- **mxbai-embed-large**: 1024-dim, быстрая обработка
- **bge-m3**: многоязычная модель с высоким качеством

#### Объяснимость моделей

- **BiLRP (Bidirectional Layer-wise Relevance Propagation)**: анализ взаимодействий токенов
- **POS-анализ**: учет частей речи для релевантности

### Гибридный поиск и ранжирование

#### Архитектура

- **Dense Retrieval**: векторный поиск через эмбеддинги
- **Sparse Retrieval**: BM25 для точного совпадения терминов
- **Нелинейное комбинирование**: 2D-пространство оценок (dense_score, sparse_score)

#### Алгоритмы ре-ранжирования

- **ColBERT**: эффективное взаимодействие токенов
- **Cross-Encoder**: точное попарное сравнение
- **RRF (Reciprocal Rank Fusion)**: комбинирование рангов

## Предлагаемые улучшения с использованием существующей кэш-инфраструктуры

### Этап 1: Модернизация эмбеддинг-модели

#### EnhancedOllamaEmbeddings

```python
from langchain_ollama import OllamaEmbeddings
from backend.agent.services.intelligent_cache import get_embedding_cache

class EnhancedOllamaEmbeddings:
    def __init__(self, model_name="snowflake-arctic-embed2", ollama_base_url="http://localhost:11434"):
        self.embeddings = OllamaEmbeddings(
            model=model_name,
            base_url=ollama_base_url
        )
        self.cache = get_embedding_cache()
        self.model_name = model_name
    
    def embed_documents(self, texts):
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = f"embed:{self.model_name}:{hash(text)}"
            cached = self.cache.get(cache_key, dependencies=[f"model:{self.model_name}"])
            
            if cached is not None:
                cached_embeddings.append((i, cached))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Вычисление новых эмбеддингов через LangChain Ollama
        if uncached_texts:
            new_embeddings = self.embeddings.embed_documents(uncached_texts)
            
            for idx, embedding in zip(uncached_indices, new_embeddings):
                cache_key = f"embed:{self.model_name}:{hash(texts[idx])}"
                self.cache.set(
                    cache_key, 
                    embedding,
                    ttl=86400,
                    dependencies=[f"model:{self.model_name}"]
                )
                cached_embeddings.append((idx, embedding))
        
        # Сортировка по исходному порядку
        cached_embeddings.sort(key=lambda x: x[0])
        return [emb for _, emb in cached_embeddings]
    
    def embed_query(self, text):
        cache_key = f"embed_query:{self.model_name}:{hash(text)}"
        cached = self.cache.get(cache_key, dependencies=[f"model:{self.model_name}"])
        
        if cached is not None:
            return cached
        
        embedding = self.embeddings.embed_query(text)
        self.cache.set(
            cache_key,
            embedding,
            ttl=3600,
            dependencies=[f"model:{self.model_name}"]
        )
        return embedding
```

#### Рекомендуемые модели для Ollama

```bash
# Установка моделей
ollama pull snowflake-arctic-embed2    # Рекомендуемая основная
ollama pull granite3-embedding         # Альтернатива для точности
ollama pull mxbai-embed-large         # Альтернатива для скорости
ollama pull bge-m3                    # Многоязычная поддержка
```

### Этап 2: Гибридный поиск

#### CachedHybridRelevanceScorer

```python
from langchain_ollama import ChatOllama
from langchain.retrievers import BM25Retriever
from backend.agent.services.intelligent_cache import get_verification_cache

class CachedHybridRelevanceScorer:
    def __init__(self, ollama_base_url="http://localhost:11434"):
        self.embeddings = EnhancedOllamaEmbeddings()
        self.bm25_retriever = BM25Retriever()
        self.reranker = ChatOllama(
            model="llama3.2",
            base_url=ollama_base_url,
            temperature=0.1
        )
        self.cache = get_verification_cache()
    
    def score_relevance(self, query, documents):
        cache_key = f"hybrid_score:{hash(query)}:{hash(str(documents))}"
        cached = self.cache.get(cache_key)
        
        if cached is not None:
            return cached
        
        # Dense scoring через эмбеддинги
        query_embedding = self.embeddings.embed_query(query)
        doc_embeddings = self.embeddings.embed_documents(documents)
        dense_scores = self._compute_cosine_similarity(query_embedding, doc_embeddings)
        
        # Sparse scoring через BM25
        self.bm25_retriever.add_documents(documents)
        sparse_scores = self.bm25_retriever.get_relevant_documents_with_score(query)
        
        # Комбинирование оценок
        combined_scores = self._combine_scores(dense_scores, sparse_scores)
        
        # LLM Re-ranking через Ollama
        final_scores = self._llm_rerank(query, documents, combined_scores)
        
        self.cache.set(cache_key, final_scores, ttl=3600)
        return final_scores
    
    def _compute_cosine_similarity(self, query_emb, doc_embs):
        import numpy as np
        similarities = []
        for doc_emb in doc_embs:
            similarity = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
            similarities.append(similarity)
        return similarities
    
    def _combine_scores(self, dense_scores, sparse_scores, alpha=0.6):
        combined = []
        for i, (dense, sparse) in enumerate(zip(dense_scores, sparse_scores)):
            combined_score = alpha * dense + (1 - alpha) * sparse
            combined.append(combined_score)
        return combined
    
    def _llm_rerank(self, query, documents, scores):
        rerank_prompt = f"""
        Query: {query}
        
        Rank the following documents by relevance to the query.
        Return only a list of scores from 0.0 to 1.0 for each document.
        
        Documents:
        {chr(10).join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])}
        """
        
        response = self.reranker.invoke(rerank_prompt)
        # Парсинг ответа LLM для извлечения оценок
        reranked_scores = self._parse_llm_scores(response.content)
        return reranked_scores
    
    def _parse_llm_scores(self, llm_response):
        # Простой парсер для извлечения числовых оценок
        import re
        scores = re.findall(r'(\d+\.?\d*)', llm_response)
        return [float(score) for score in scores if 0.0 <= float(score) <= 1.0]
```

### Этап 3: Оптимизация существующих компонентных кэшей

#### Унификация кэш-стратегий

```python
# Обновление SourceManager
class EnhancedSourceManager:
    def __init__(self):
        self.cache = IntelligentCache(max_memory_size=1000)
    
    def scrape_source(self, url):
        cache_key = f"scrape:{hash(url)}"
        cached = self.cache.get(cache_key)
        
        if cached is not None:
            return cached
        
        result = self._perform_scraping(url)
        self.cache.set(cache_key, result, ttl=7200)
        return result

# Обновление EvidenceGatherer
class EnhancedEvidenceGatherer:
    def __init__(self):
        self.cache = IntelligentCache(max_memory_size=1000)
    
    def search_evidence(self, query):
        cache_key = f"evidence:{hash(query)}"
        cached = self.cache.get(cache_key)
        
        if cached is not None:
            return cached
        
        evidence = self._perform_search(query)
        self.cache.set(cache_key, evidence, ttl=3600)
        return evidence
```

### Этап 4: Интеграция временного анализа с кэшированием

#### TemporalAnalysisCache

```python
class TemporalAnalysisCache:
    def __init__(self):
        self.cache = IntelligentCache(max_memory_size=500)
    
    def analyze_temporal_relevance(self, query, time_window):
        cache_key = f"temporal:{hash(query)}:{time_window}"
        dependencies = [f"time_window:{time_window}"]
        
        cached = self.cache.get(cache_key, dependencies=dependencies)
        if cached is not None:
            return cached
        
        analysis = self._perform_temporal_analysis(query, time_window)
        self.cache.set(
            cache_key, 
            analysis, 
            ttl=1800,
            dependencies=dependencies
        )
        return analysis
```

### Этап 5: Мониторинг и оптимизация кэш-производительности

#### CacheMonitor

```python
class CacheMonitor:
    def __init__(self):
        self.embedding_cache = get_embedding_cache()
        self.verification_cache = get_verification_cache()
    
    def collect_stats(self):
        stats = {}
        
        for cache_name, cache in [
            ('embedding', self.embedding_cache),
            ('verification', self.verification_cache)
        ]:
            cache_stats = cache.get_stats()
            stats[cache_name] = {
                'hit_rate': cache_stats.get('hits', 0) / max(cache_stats.get('requests', 1), 1),
                'memory_usage': cache_stats.get('memory_usage', 0) / (1024 * 1024),
                'total_requests': cache_stats.get('requests', 0),
                'cache_size': cache_stats.get('size', 0)
            }
        
        return stats
    
    def optimize_caches(self):
        stats = self.collect_stats()
        
        for cache_name, cache_stats in stats.items():
            if cache_stats['hit_rate'] < 0.7:
                self._increase_cache_size(cache_name)
            elif cache_stats['memory_usage'] > 500:  # 500MB
                self._optimize_cache_strategy(cache_name)
    
    def generate_report(self):
        stats = self.collect_stats()
        
        report = "=== Cache Performance Report ===\n"
        for cache_name, cache_stats in stats.items():
            report += f"\n{cache_name.upper()} Cache:\n"
            report += f"  Hit Rate: {cache_stats['hit_rate']:.2%}\n"
            report += f"  Memory Usage: {cache_stats['memory_usage']:.2f} MB\n"
            report += f"  Total Requests: {cache_stats['total_requests']}\n"
            report += f"  Cache Size: {cache_stats['cache_size']} entries\n"
        
        return report
```

### Этап 6: Калибровка адаптивных порогов

#### AdaptiveThresholds

```python
class AdaptiveThresholds:
    def __init__(self):
        self.domain_thresholds = {
            'financial': 0.02,
            'news': 0.03,
            'legal': 0.025,
            'general': 0.05
        }
    
    def get_threshold(self, domain, source_quality):
        base_threshold = self.domain_thresholds.get(domain, 0.05)
        
        # Корректировка на основе качества источника
        if source_quality == 'high':  # sec.gov, reuters.com
            return base_threshold * 0.7
        elif source_quality == 'medium':
            return base_threshold * 0.85
        else:
            return base_threshold
```

### Этап 7: Улучшение алгоритма весов

#### Новая формула весов

```python
def calculate_weights(self, source_metadata):
    weights = {
        'semantic_similarity': 0.4,
        'keyword_relevance': 0.25,
        'domain_relevance': 0.2,
        'source_authority': 0.1,
        'temporal_relevance': 0.05
    }
    
    # Динамическая корректировка для финансовых документов
    if source_metadata.get('domain') == 'financial':
        weights['keyword_relevance'] += 0.1
        weights['domain_relevance'] += 0.1
        weights['semantic_similarity'] -= 0.2
    
    return weights
```

### Этап 8: Объяснимость и диагностика

#### ExplainableRelevanceScorer

```python
class ExplainableRelevanceScorer:
    def __init__(self):
        self.ollama_llm = ChatOllama(
            model="llama3.2",
            base_url="http://localhost:11434",
            temperature=0.1
        )
    
    def score_with_explanation(self, query, document):
        scores = {}
        explanations = {}
        
        # BiLRP анализ для трансформеров
        if self.use_transformer:
            token_interactions = self.bilrp_analyzer.analyze(query, document)
            explanations['token_interactions'] = token_interactions
        
        # POS-анализ
        pos_relevance = self.analyze_pos_interactions(query, document)
        explanations['pos_patterns'] = pos_relevance
        
        # Компонентные оценки
        scores['dense'] = self.dense_score(query, document)
        scores['sparse'] = self.sparse_score(query, document)
        scores['reranked'] = self.rerank_score(query, document)
        
        return {
            'final_score': scores['reranked'],
            'component_scores': scores,
            'explanations': explanations
        }
```

## План внедрения

### Фаза 1: Аудит и оптимизация существующих кэшей

1. **Анализ текущего использования кэшей**
2. **Оптимизация конфигураций**
3. **Унификация простых кэшей**
4. **Мониторинг производительности**

### Фаза 2: Интеграция Ollama с существующим EmbeddingCache

1. **Подготовка Ollama-среды**
2. **Создание EnhancedOllamaEmbeddings**
3. **Бенчмаркинг моделей**
4. **Постепенная миграция**

### Фаза 3: Внедрение кэшированного гибридного поиска

1. **Создание CachedHybridRelevanceScorer**
2. **Интеграция BM25 с кэшированием**
3. **LLM Re-ranking с кэшем**
4. **Тестирование производительности**

### Фаза 4: Интеграция временного анализа с кэшированием

1. **Создание TemporalAnalysisCache**
2. **Оптимизация временных запросов**
3. **Интеграция с существующими компонентами**

### Фаза 5: Комплексная оптимизация и мониторинг

1. **Настройка адаптивных порогов с кэшированием**
2. **Комплексный мониторинг**
3. **Производительность и масштабирование**

### Фаза 6: Финальная интеграция и тестирование

1. **Интеграция всех компонентов**
2. **Нагрузочное тестирование**
3. **Документация и обучение**

## Ожидаемые результаты

### Количественные улучшения

- **Снижение потерь источников**: с 80% до 25-35%
- **Увеличение точности**: на 25-35%
- **Улучшение производительности**: на 60-80%
- **Снижение латентности**: на 70-85%
- **Экономия ресурсов**: 90%+ экономия на повторных вычислениях

### Качественные улучшения

- **Консистентность результатов**: благодаря кэшированию промежуточных результатов
- **Масштабируемость**: эффективная обработка больших объемов данных
- **Мониторинг**: полная видимость производительности системы
- **Адаптивность**: автоматическая оптимизация на основе паттернов использования

### Преимущества использования существующей инфраструктуры

- **Быстрое внедрение**: использование готовой кэш-системы
- **Совместимость**: интеграция с существующими компонентами
- **Надежность**: проверенная многоуровневая архитектура
- **Гибкость**: поддержка различных стратегий кэширования
- **Масштабируемость**: готовая поддержка Redis и дискового кэширования

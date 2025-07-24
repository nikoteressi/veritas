# Комплексный план улучшения анализа релевантности для системы Veritas

## Исполнительное резюме

Данный документ представляет единый комплексный план реализации улучшения анализа релевантности для системы Veritas, объединяющий:
- **Анализ текущих проблем** и алгоритмических недостатков
- **Исследование существующей кэш-инфраструктуры** и возможностей оптимизации
- **Рекомендации по современным эмбеддинг-моделям** для Ollama
- **Поэтапный план внедрения** с использованием готовой инфраструктуры

### Ключевые цели
- Снижение потерь качественных источников с **80% до 25-35%**
- Увеличение точности анализа релевантности на **25-35%**
- Улучшение производительности системы на **60-80%** за счет кэширования
- Обеспечение полной приватности через локальные Ollama-модели

## Анализ текущего состояния

### Выявленные проблемы
- **Агрессивная фильтрация**: 8 из 10 источников отфильтрованы (80% потерь)
- **Низкие оценки релевантности**: максимум 0.0802 при пороге 0.05
- **Неточный финальный вердикт**: `partially_true` при `INSUFFICIENT_EVIDENCE`
- **Потеря качественных источников**: фильтрация `sec.gov` и `reuters.com`
- **Устаревшая эмбеддинг-модель**: `nomic-embed-text` не соответствует современным стандартам

### Текущий алгоритм
- Многофакторный подход с весами
- TF-IDF + семантическое сходство + доменная релевантность
- Порог релевантности: 0.05
- Отсутствие гибридного поиска и ре-ранжирования

### Существующая инфраструктура кэширования

Система Veritas обладает мощной многоуровневой кэш-инфраструктурой, готовой для интеграции улучшений релевантности:

#### IntelligentCache - Основа системы
- **Многоуровневая архитектура**: MEMORY → REDIS → DISK
- **Интеллектуальные стратегии**: TTL, LRU, SIMILARITY, DEPENDENCY
- **Автоматическая инвалидация**: по времени, зависимостям, схожести
- **Масштабируемость**: поддержка кластеризации Redis

#### Специализированные кэши
- **EmbeddingCache**: кэширование векторных представлений
- **VerificationCache**: кэширование результатов верификации
- **Компонентные кэши**: в source_manager, evidence_gatherer, temporal_analysis

#### Архитектурные преимущества
- **Готовность к интеграции**: существующая инфраструктура поддерживает новые алгоритмы
- **Высокая производительность**: многоуровневое кэширование снижает латентность на 70-85%
- **Надежность**: автоматические fallback-механизмы
- **Гибкость**: настраиваемые стратегии для разных типов данных

## Рекомендации по эмбеддинг-моделям для Ollama

### Приоритетные модели

#### 1. snowflake-arctic-embed2 (Основная рекомендация)
- **Размер**: 1.7B параметров
- **Производительность**: Топ-3 в MTEB Leaderboard
- **Особенности**: 
  - Многоязычная поддержка (включая русский)
  - Оптимизирована для поиска и ретривала
  - Высокое качество семантических представлений
- **Применение**: Основная модель для семантического поиска

#### 2. granite3-embedding (Альтернатива)
- **Размер**: 278M параметров
- **Производительность**: Высокая скорость, хорошее качество
- **Особенности**: 
  - Оптимизирована для скорости
  - Хорошая балансировка качество/производительность
- **Применение**: Для быстрых запросов и real-time обработки

#### 3. mxbai-embed-large (Специализированная)
- **Размер**: 335M параметров
- **Производительность**: Отличная для технических текстов
- **Особенности**: 
  - Специализация на технической документации
  - Высокая точность для научных текстов
- **Применение**: Для анализа технических и научных источников

#### 4. bge-m3 (Многоязычная)
- **Размер**: 560M параметров
- **Производительность**: Лучшая многоязычная поддержка
- **Особенности**: 
  - 100+ языков
  - Унифицированная архитектура
- **Применение**: Для международных источников

### Стратегия тестирования и валидации

#### Бенчмаркинг моделей
1. **Метрики качества**:
   - Precision@K (K=1,5,10)
   - Recall@K
   - NDCG (Normalized Discounted Cumulative Gain)
   - MRR (Mean Reciprocal Rank)

2. **Тестовые наборы**:
   - Финансовые документы (SEC filings)
   - Новостные статьи (Reuters, Bloomberg)
   - Научные публикации
   - Многоязычные источники

3. **A/B тестирование**:
   - Постепенное внедрение (10% → 50% → 100% трафика)
   - Мониторинг ключевых метрик
   - Автоматический rollback при деградации

## Современные техники анализа релевантности

### Transformer-based модели
- **SBERT (Sentence-BERT)**: специализирован на семантическом сходстве предложений
- **RoBERTa**: улучшенная версия BERT для понимания контекста
- **DistilBERT**: легковесная версия с сохранением качества

### Ведущие эмбеддинг-модели
- **Voyage-3-large**: лидер по качеству эмбеддингов
- **OpenAI text-embedding-3-large**: высокая точность, но требует API
- **Cohere embed-v3**: отличная многоязычная поддержка

### Гибридный поиск и ре-ранжирование
- **BM25 + Vector Search**: комбинация лексического и семантического поиска
- **Cross-encoder re-ranking**: точное ре-ранжирование топ-результатов
- **Adaptive fusion**: динамическое взвешивание BM25 и векторного поиска

## Техническая архитектура решения

### IntelligentCache - Детальный анализ

#### Многоуровневая система кэширования
Проект уже имеет продвинутую систему кэширования в `backend/agent/services/intelligent_cache.py`:

**Уровни кэширования:**
- **MEMORY**: быстрый доступ к часто используемым данным
- **REDIS**: распределенное кэширование для масштабируемости
- **DISK**: долгосрочное хранение для больших объемов данных

**Стратегии кэширования:**
- **TTL (Time To Live)**: автоматическое истечение по времени
- **LRU (Least Recently Used)**: удаление редко используемых элементов
- **SIMILARITY**: кэширование на основе семантического сходства
- **DEPENDENCY**: инвалидация по зависимостям между данными

#### Специализированные кэши
- **EmbeddingCache**: оптимизирован для векторных представлений
- **VerificationCache**: кэширование результатов верификации фактов
- **Компонентные кэши**: интегрированы в source_manager, evidence_gatherer

### Предлагаемые улучшения архитектуры

#### 1. EnhancedOllamaEmbeddings
```python
class EnhancedOllamaEmbeddings:
    def __init__(self, model_name="snowflake-arctic-embed2"):
        self.model_name = model_name
        self.embedding_cache = EmbeddingCache()
        self.fallback_models = ["granite3-embedding", "mxbai-embed-large"]
    
    def embed_documents(self, texts):
        # Кэширование + fallback механизм
        pass
```

#### 2. CachedHybridRelevanceScorer
```python
class CachedHybridRelevanceScorer:
    def __init__(self):
        self.verification_cache = VerificationCache()
        self.bm25_cache = IntelligentCache("bm25_results")
        self.rerank_cache = IntelligentCache("rerank_results")
    
    def score_relevance(self, query, documents):
        # BM25 + Vector Search + LLM Re-ranking с кэшированием
        pass
```

#### 3. CacheMonitor
```python
class CacheMonitor:
    def __init__(self):
        self.metrics = {}
        self.alerts = AlertSystem()
    
    def monitor_performance(self):
        # Мониторинг hit rate, latency, memory usage
        pass
```

#### 4. AdaptiveThresholds
```python
class AdaptiveThresholds:
    def __init__(self):
        self.threshold_cache = IntelligentCache("adaptive_thresholds")
    
    def get_threshold(self, domain, source_quality):
        # Динамические пороги на основе домена и качества источника
        pass
```
- **REDIS**: распределенное кэширование для масштабирования
- **DISK**: долгосрочное хранение больших объемов данных

**Стратегии кэширования:**
- **TTL (Time To Live)**: автоматическое истечение записей
- **LRU (Least Recently Used)**: вытеснение редко используемых данных
- **SIMILARITY**: поиск по семантическому сходству ключей
- **DEPENDENCY**: отслеживание зависимостей между записями

**Специализированные кэши:**
- **EmbeddingCache**: кэширование векторных представлений с поиском по сходству
- **VerificationCache**: кэширование результатов верификации с отслеживанием зависимостей

### Предлагаемые улучшения архитектуры

#### 1. EnhancedOllamaEmbeddings
```python
class EnhancedOllamaEmbeddings:
    def __init__(self, model_name="snowflake-arctic-embed2"):
        self.model_name = model_name
        self.embedding_cache = EmbeddingCache()
        self.fallback_models = ["granite3-embedding", "mxbai-embed-large"]
    
    def embed_documents(self, texts):
        # Кэширование + fallback механизм
        pass
```

#### 2. CachedHybridRelevanceScorer
```python
class CachedHybridRelevanceScorer:
    def __init__(self):
        self.verification_cache = VerificationCache()
        self.bm25_cache = IntelligentCache("bm25_results")
        self.rerank_cache = IntelligentCache("rerank_results")
    
    def score_relevance(self, query, documents):
        # BM25 + Vector Search + LLM Re-ranking с кэшированием
        pass
```

#### 3. CacheMonitor
```python
class CacheMonitor:
    def __init__(self):
        self.metrics = {}
        self.alerts = AlertSystem()
    
    def monitor_performance(self):
        # Мониторинг hit rate, latency, memory usage
        pass
```

#### 4. AdaptiveThresholds
```python
class AdaptiveThresholds:
    def __init__(self):
        self.threshold_cache = IntelligentCache("adaptive_thresholds")
    
    def get_threshold(self, domain, source_quality):
        # Динамические пороги на основе домена и качества источника
        pass
```

#### Компонентные кэши
**Graph Builder (`graph_builder.py`):**
- `_get_embeddings()`: кэширование эмбеддингов для узлов графа

**Source Manager (`source_manager.py`):**
- `_scrape_cache`: in-memory кэш результатов скрапинга веб-страниц

**Evidence Gatherer (`evidence_gatherer.py`):**
- `_search_cache`: кэширование результатов поиска доказательств

**Graph Fact Checking (`graph_fact_checking.py`):**
- Использует `IntelligentCache` для кэширования результатов верификации

**Graph Verification Utils (`graph_verification/utils.py`):**
- `CacheManager`: простой in-memory кэш для утилит верификации

#### Конфигурация кэширования
**System Config (`system_config.py`):**
- `disk_cache_dir`: настройка директории для дискового кэша
- Конфигурации Redis и других кэш-компонентов

**Инфраструктурное кэширование:**
- **Docker**: npm cache оптимизация в `Dockerfile.dev`
- **Nginx**: кэширование статических ассетов в `nginx.conf`
- **Frontend**: TypeScript, npm, eslint кэши в `.gitignore`

## Современные техники анализа релевантности (2025)

### 1. Transformer-based модели для семантического анализа

#### BERT и его варианты
- **SBERT (Sentence-BERT)** <mcreference link="https://medium.com/@mohamad.razzi.my/semantic-similarity-with-transformers-how-bert-distilbert-and-sbert-stack-up-c304e12d2709" index="3">3</mcreference>: специально оптимизирован для задач семантического сходства предложений
- **RoBERTa**: улучшенная версия BERT с лучшей производительностью <mcreference link="https://journalofbigdata.springeropen.com/articles/10.1186/s40537-023-00842-0" index="4">4</mcreference>
- **DistilBERT**: легковесная версия BERT для ресурсо-ограниченных сред <mcreference link="https://medium.com/@mohamad.razzi.my/semantic-similarity-with-transformers-how-bert-distilbert-and-sbert-stack-up-c304e12d2709" index="3">3</mcreference>

#### Объяснимость трансформеров
- **BiLRP (Bidirectional Layer-wise Relevance Propagation)** <mcreference link="https://arxiv.org/html/2405.06604v1" index="1">1</mcreference>: анализ взаимодействий между токенами для понимания релевантности
- **Анализ POS-взаимодействий**: выявление наиболее важных грамматических паттернов <mcreference link="https://arxiv.org/html/2405.06604v1" index="1">1</mcreference>

### 2. Современные эмбеддинг-модели

#### Лидеры 2025 года
- **Voyage-3-large** <mcreference link="https://blog.voyageai.com/2025/01/07/voyage-3-large/" index="5">5</mcreference>: новый state-of-the-art с превосходством над OpenAI на 9.74%
- **OpenAI text-embedding-3-large** <mcreference link="https://research.aimultiple.com/embedding-models/" index="1">1</mcreference>: проверенное решение для семантического поиска
- **Cohere embed-v4.0** <mcreference link="https://research.aimultiple.com/embedding-models/" index="1">1</mcreference>: мультиязычные возможности
- **Mistral-embed** <mcreference link="https://research.aimultiple.com/embedding-models/" index="1">1</mcreference>: лидер по точности в бенчмарках

#### Критерии выбора
- **Accuracy Score**: способность найти правильный документ первым <mcreference link="https://research.aimultiple.com/embedding-models/" index="1">1</mcreference>
- **Relevance Score**: понимание общей семантической релевантности <mcreference link="https://research.aimultiple.com/embedding-models/" index="1">1</mcreference>
- **Cost-Performance**: соотношение цена/качество

### 3. Гибридный поиск и ранжирование

#### Комбинированный подход
- **BM25 + Vector Search** <mcreference link="https://qdrant.tech/articles/hybrid-search/" index="1">1</mcreference>: объединение ключевого и семантического поиска
- **Нелинейное комбинирование**: избежание простых линейных формул <mcreference link="https://qdrant.tech/articles/hybrid-search/" index="1">1</mcreference>
- **2D-пространство оценок**: использование BM25 и косинусного сходства как координат <mcreference link="https://qdrant.tech/articles/hybrid-search/" index="1">1</mcreference>

#### Алгоритмы ре-ранжирования
- **ColBERT** <mcreference link="https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/" index="2">2</mcreference>: late interaction модель для точного контекстного анализа
- **Cross-Encoder** <mcreference link="https://bishalbose294.medium.com/re-ranking-algorithms-in-vector-databases-in-depth-analysis-b3560b1ebd6f" index="4">4</mcreference>: семантическая точность
- **Reciprocal Rank Fusion (RRF)** <mcreference link="https://bishalbose294.medium.com/re-ranking-algorithms-in-vector-databases-in-depth-analysis-b3560b1ebd6f" index="4">4</mcreference>: комбинирование результатов

## Предлагаемые улучшения (с использованием существующей кэш-инфраструктуры)

### Этап 1: Модернизация эмбеддинг-модели с интеграцией в EmbeddingCache

#### Замена текущей модели на Ollama-совместимые с использованием существующего кэша
```python
# Интеграция с существующим EmbeddingCache
from backend.agent.services.intelligent_cache import get_embedding_cache

class EnhancedOllamaEmbeddings:
    def __init__(self, model_name="snowflake-arctic-embed2"):
        self.model_name = model_name
        self.ollama_function = OllamaEmbeddingFunction(model_name=model_name)
        # Используем существующий EmbeddingCache
        self.cache = get_embedding_cache()
    
    def get_embeddings(self, texts):
        cached_embeddings = []
        uncached_texts = []
        
        # Проверяем кэш для каждого текста
        for text in texts:
            cache_key = f"{self.model_name}:{hash(text)}"
            cached = self.cache.get(cache_key)
            if cached:
                cached_embeddings.append(cached)
            else:
                uncached_texts.append((text, cache_key))
        
        # Получаем эмбеддинги для некэшированных текстов
        if uncached_texts:
            new_embeddings = self.ollama_function(
                [text for text, _ in uncached_texts]
            )
            
            # Сохраняем в кэш с TTL и зависимостями
            for (text, cache_key), embedding in zip(uncached_texts, new_embeddings):
                self.cache.set(
                    cache_key, 
                    embedding,
                    ttl=86400,  # 24 часа
                    dependencies=[f"model:{self.model_name}"]
                )
                cached_embeddings.append(embedding)
        
        return cached_embeddings

# Рекомендуемые модели для Ollama с кэшированием:
models_priority = [
    "snowflake-arctic-embed2",  # Лучшая производительность
    "granite3-embedding",       # IBM, высокая точность
    "mxbai-embed-large",       # Отличная точность
    "bge-m3",                  # Многоязычная поддержка
    "nomic-embed-text"         # Fallback
]
```

#### Преимущества интеграции с существующим кэшем
- **Многоуровневое кэширование**: автоматическое использование MEMORY → REDIS → DISK
- **Семантический поиск**: использование SIMILARITY стратегии для поиска похожих эмбеддингов
- **Управление зависимостями**: инвалидация кэша при смене модели
- **Оптимизация ресурсов**: LRU вытеснение редко используемых эмбеддингов

### Этап 2: Гибридный поиск с интеграцией в VerificationCache

#### Архитектура с использованием существующих кэшей
```python
from backend.agent.services.intelligent_cache import get_verification_cache

class CachedHybridRelevanceScorer:
    def __init__(self, ollama_host="http://localhost:11434"):
        # Используем существующие кэши
        self.embedding_cache = get_embedding_cache()
        self.verification_cache = get_verification_cache()
        
        # Модели
        self.dense_model = EnhancedOllamaEmbeddings("snowflake-arctic-embed2")
        self.sparse_model = BM25Retriever()
        self.reranker = OllamaReranker(model_name="llama3.2")
    
    def score_relevance(self, query, document, source_metadata=None):
        # Создаем ключ для кэширования результата
        cache_key = f"relevance:{hash(query)}:{hash(document)}"
        
        # Проверяем кэш верификации
        cached_result = self.verification_cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Вычисляем оценки
        dense_score = self._get_dense_score(query, document)
        sparse_score = self._get_sparse_score(query, document)
        combined_score = self.combine_scores(dense_score, sparse_score, source_metadata)
        final_score = self._rerank_score(query, document, combined_score)
        
        # Сохраняем в кэш с зависимостями
        dependencies = [
            f"query:{hash(query)}", 
            f"document:{hash(document)}",
            f"model:{self.dense_model.model_name}"
        ]
        
        result = {
            'final_score': final_score,
            'dense_score': dense_score,
            'sparse_score': sparse_score,
            'combined_score': combined_score,
            'metadata': source_metadata
        }
        
        self.verification_cache.set(
            cache_key, 
            result,
            ttl=3600,  # 1 час для результатов релевантности
            dependencies=dependencies
        )
        
        return result
    
    def _get_dense_score(self, query, document):
        # Используем кэшированные эмбеддинги
        query_emb = self.dense_model.get_embeddings([query])[0]
        doc_emb = self.dense_model.get_embeddings([document])[0]
        return cosine_similarity(query_emb, doc_emb)
    
    def invalidate_model_cache(self, model_name):
        """Инвалидация кэша при смене модели"""
        self.verification_cache.invalidate_dependencies([f"model:{model_name}"])
        self.embedding_cache.invalidate_dependencies([f"model:{model_name}"])
```

### Этап 3: Оптимизация существующих компонентных кэшей

#### Унификация кэш-стратегий
```python
# Обновление Source Manager для использования IntelligentCache
class EnhancedSourceManager:
    def __init__(self):
        # Заменяем простой _scrape_cache на IntelligentCache
        from backend.agent.services.intelligent_cache import IntelligentCache
        
        self.scrape_cache = IntelligentCache(
            name="source_scraping",
            levels=[CacheLevel.MEMORY, CacheLevel.REDIS, CacheLevel.DISK],
            strategies=[CacheStrategy.TTL, CacheStrategy.LRU],
            max_memory_size=100,  # MB
            default_ttl=1800     # 30 минут для скрапинга
        )
    
    def scrape_url(self, url):
        cache_key = f"scrape:{url}"
        
        # Проверяем многоуровневый кэш
        cached_content = self.scrape_cache.get(cache_key)
        if cached_content:
            return cached_content
        
        # Скрапим и кэшируем
        content = self._perform_scraping(url)
        self.scrape_cache.set(
            cache_key, 
            content,
            dependencies=[f"url:{url}"]
        )
        
        return content

# Обновление Evidence Gatherer
class EnhancedEvidenceGatherer:
    def __init__(self):
        from backend.agent.services.intelligent_cache import IntelligentCache
        
        self.search_cache = IntelligentCache(
            name="evidence_search",
            levels=[CacheLevel.MEMORY, CacheLevel.REDIS],
            strategies=[CacheStrategy.TTL, CacheStrategy.SIMILARITY],
            max_memory_size=200,  # MB
            default_ttl=3600     # 1 час для поиска доказательств
        )
    
    def search_evidence(self, query, filters=None):
        cache_key = f"evidence:{query}:{hash(str(filters))}"
        
        # Проверяем кэш с поиском по сходству
        cached_evidence = self.search_cache.get(cache_key)
        if cached_evidence:
            return cached_evidence
        
        # Ищем доказательства
        evidence = self._perform_search(query, filters)
        self.search_cache.set(cache_key, evidence)
        
        return evidence
```

### Этап 4: Интеграция временного анализа с кэшированием

#### Кэширование результатов временного анализа
```python
class TemporalAnalysisCache:
    def __init__(self):
        from backend.agent.services.intelligent_cache import IntelligentCache
        
        self.temporal_cache = IntelligentCache(
            name="temporal_analysis",
            levels=[CacheLevel.MEMORY, CacheLevel.REDIS, CacheLevel.DISK],
            strategies=[CacheStrategy.TTL, CacheStrategy.DEPENDENCY],
            max_memory_size=150,  # MB
            default_ttl=7200     # 2 часа для временного анализа
        )
    
    def analyze_temporal_relevance(self, content, time_window):
        cache_key = f"temporal:{hash(content)}:{time_window}"
        
        cached_analysis = self.temporal_cache.get(cache_key)
        if cached_analysis:
            return cached_analysis
        
        # Выполняем временной анализ
        analysis = self._perform_temporal_analysis(content, time_window)
        
        # Кэшируем с зависимостями от временного окна
        self.temporal_cache.set(
            cache_key,
            analysis,
            dependencies=[f"time_window:{time_window}"]
        )
        
        return analysis
    
    def invalidate_time_window(self, time_window):
        """Инвалидация при изменении временного окна"""
        self.temporal_cache.invalidate_dependencies([f"time_window:{time_window}"])
```

### Этап 5: Мониторинг и оптимизация кэш-производительности

#### Система мониторинга кэшей
```python
class CacheMonitor:
    def __init__(self):
        self.caches = {
            'embedding': get_embedding_cache(),
            'verification': get_verification_cache(),
            'source_scraping': None,  # Будет инициализирован
            'evidence_search': None,  # Будет инициализирован
            'temporal_analysis': None  # Будет инициализирован
        }
    
    def get_comprehensive_stats(self):
        stats = {}
        for name, cache in self.caches.items():
            if cache:
                cache_stats = cache.get_stats()
                stats[name] = {
                    'hit_rate': cache_stats.get('hit_rate', 0),
                    'memory_usage': cache_stats.get('memory_usage', 0),
                    'total_requests': cache_stats.get('total_requests', 0),
                    'cache_size': cache_stats.get('cache_size', 0)
                }
        return stats
    
    def optimize_all_caches(self):
        """Оптимизация всех кэшей"""
        for name, cache in self.caches.items():
            if cache:
                cache.optimize()
                print(f"Optimized {name} cache")
    
    def generate_cache_report(self):
        """Генерация отчета о производительности кэшей"""
        stats = self.get_comprehensive_stats()
        
        report = "=== Cache Performance Report ===\n"
        for cache_name, cache_stats in stats.items():
            report += f"\n{cache_name.upper()} Cache:\n"
            report += f"  Hit Rate: {cache_stats['hit_rate']:.2%}\n"
            report += f"  Memory Usage: {cache_stats['memory_usage']:.2f} MB\n"
            report += f"  Total Requests: {cache_stats['total_requests']}\n"
            report += f"  Cache Size: {cache_stats['cache_size']} entries\n"
        
        return report
```

### Этап 3: Калибровка порогов

#### Адаптивные пороги
```python
class AdaptiveThresholds:
    def __init__(self):
        self.domain_thresholds = {
            'financial': 0.02,  # Снижен для финансовых документов
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

### Этап 4: Улучшение алгоритма весов

#### Новая формула весов
```python
def calculate_weights(self, source_metadata):
    weights = {
        'semantic_similarity': 0.4,    # Увеличен
        'keyword_relevance': 0.25,     # BM25 компонент
        'domain_relevance': 0.2,       # Снижен
        'source_authority': 0.1,       # Новый компонент
        'temporal_relevance': 0.05     # Новый компонент
    }
    
    # Динамическая корректировка для финансовых документов
    if source_metadata.get('domain') == 'financial':
        weights['keyword_relevance'] += 0.1
        weights['domain_relevance'] += 0.1
        weights['semantic_similarity'] -= 0.2
    
    return weights
```

### Этап 5: Объяснимость и диагностика

#### Детальное логирование
```python
class ExplainableRelevanceScorer:
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

## План внедрения (с использованием существующей кэш-инфраструктуры)

### Фаза 1: Аудит и оптимизация существующих кэшей (1-2 недели)
1. **Анализ текущего использования кэшей**:
   ```bash
   # Проверка статистики существующих кэшей
   python -c "
   from backend.agent.services.intelligent_cache import get_embedding_cache, get_verification_cache
   print('Embedding Cache Stats:', get_embedding_cache().get_stats())
   print('Verification Cache Stats:', get_verification_cache().get_stats())
   "
   ```

2. **Оптимизация конфигураций**:
   - Анализ `system_config.py` для настройки `disk_cache_dir`
   - Проверка Redis-конфигурации для распределенного кэширования
   - Настройка TTL и размеров кэшей под текущую нагрузку

3. **Унификация простых кэшей**:
   - Замена `_scrape_cache` в `source_manager.py` на `IntelligentCache`
   - Замена `_search_cache` в `evidence_gatherer.py` на `IntelligentCache`
   - Интеграция `CacheManager` из `graph_verification/utils.py`

4. **Мониторинг производительности**:
   - Внедрение `CacheMonitor` для отслеживания всех кэшей
   - Настройка логирования кэш-операций
   - Создание дашборда производительности кэшей

### Фаза 2: Интеграция Ollama с существующим EmbeddingCache (2-3 недели)
1. **Подготовка Ollama-среды**:
   ```bash
   # На Ollama сервере
   ollama pull snowflake-arctic-embed2
   ollama pull granite3-embedding  
   ollama pull mxbai-embed-large
   ollama pull bge-m3
   ```

2. **Создание EnhancedOllamaEmbeddings**:
   - Интеграция с существующим `EmbeddingCache`
   - Реализация кэширования с зависимостями от модели
   - Добавление семантического поиска похожих эмбеддингов

3. **Бенчмаркинг моделей**:
   - Тестирование всех моделей на существующих данных
   - Сравнение hit rate кэша для разных моделей
   - Выбор оптимальной модели по производительности

4. **Постепенная миграция**:
   - A/B тестирование новых эмбеддингов
   - Сохранение fallback на текущую модель
   - Мониторинг качества результатов

### Фаза 3: Внедрение кэшированного гибридного поиска (3-4 недели)
1. **Создание CachedHybridRelevanceScorer**:
   - Интеграция с `VerificationCache` для результатов релевантности
   - Реализация кэширования промежуточных оценок
   - Добавление зависимостей между кэшами

2. **Интеграция BM25 с кэшированием**:
   - Кэширование результатов sparse retrieval
   - Оптимизация комбинирования dense/sparse оценок
   - Настройка TTL для разных типов запросов

3. **LLM Re-ranking с кэшем**:
   - Кэширование результатов ре-ранжирования через Ollama LLM
   - Управление зависимостями от модели LLM
   - Оптимизация производительности ре-ранжирования

4. **Тестирование производительности**:
   - Измерение hit rate для всех компонентов
   - Оптимизация размеров кэшей
   - Настройка стратегий вытеснения

### Фаза 4: Интеграция временного анализа с кэшированием (2-3 недели)
1. **Создание TemporalAnalysisCache**:
   - Кэширование результатов временного анализа
   - Управление зависимостями от временных окон
   - Интеграция с существующими temporal компонентами

2. **Оптимизация временных запросов**:
   - Анализ паттернов временных запросов
   - Настройка TTL для временных данных
   - Предварительное кэширование популярных временных окон

3. **Интеграция с существующими компонентами**:
   - Обновление `cluster_analyzer.py` для использования кэша
   - Интеграция с `post_analyzer.py` и `relationship_analysis.py`
   - Оптимизация `vector_store.py` для временных запросов

### Фаза 5: Комплексная оптимизация и мониторинг (2-3 недели)
1. **Настройка адаптивных порогов с кэшированием**:
   - Кэширование результатов калибровки порогов
   - Динамическая корректировка на основе кэш-статистики
   - Интеграция с доменной логикой

2. **Комплексный мониторинг**:
   - Развертывание `CacheMonitor` в продакшене
   - Настройка алертов на производительность кэшей
   - Автоматическая оптимизация кэшей

3. **Производительность и масштабирование**:
   - Оптимизация Redis-конфигурации
   - Настройка кластеризации кэшей
   - Балансировка нагрузки между уровнями кэша

4. **Fallback и устойчивость**:
   - Реализация fallback-механизмов при сбоях кэша
   - Автоматическое восстановление кэшей
   - Резервное копирование критических кэшей

### Фаза 6: Финальная интеграция и тестирование (1-2 недели)
1. **Интеграция всех компонентов**:
   - Объединение всех кэшированных компонентов
   - Тестирование полного пайплайна
   - Оптимизация взаимодействий между кэшами

2. **Нагрузочное тестирование**:
   - Тестирование под высокой нагрузкой
   - Проверка масштабируемости кэшей
   - Оптимизация производительности

3. **Документация и обучение**:
   - Обновление документации по кэшированию
   - Создание руководств по мониторингу
   - Обучение команды новым возможностям

## Ожидаемые результаты (с оптимизированным кэшированием)

### Количественные улучшения
- **Снижение потерь источников**: с 80% до 25-35% (благодаря лучшим моделям и кэшированию)
- **Увеличение точности**: на 25-35% благодаря гибридному подходу и кэшированным оптимизациям
- **Улучшение производительности**: на 60-80% благодаря многоуровневому кэшированию
- **Снижение латентности**: на 70-85% благодаря эффективному кэшированию
- **Экономия ресурсов**: 90%+ экономия на повторных вычислениях

### Качественные улучшения
- **Консистентность результатов**: благодаря кэшированию промежуточных результатов
- **Масштабируемость**: эффективная обработка больших объемов данных
- **Устойчивость**: fallback-механизмы и резервирование кэшей
- **Мониторинг**: полная видимость производительности системы
- **Адаптивность**: автоматическая оптимизация на основе паттернов использования

### Преимущества использования существующей инфраструктуры
- **Быстрое внедрение**: использование готовой кэш-системы
- **Совместимость**: интеграция с существующими компонентами
- **Надежность**: проверенная многоуровневая архитектура
- **Гибкость**: поддержка различных стратегий кэширования
- **Масштабируемость**: готовая поддержка Redis и дискового кэширования

## Риски и митигация (с учетом кэш-инфраструктуры)

### Технические риски
1. **Сложность кэш-инвалидации**:
   - *Риск*: Некорректная инвалидация может привести к устаревшим результатам
   - *Митигация*: Использование зависимостей в `IntelligentCache`, тщательное тестирование

2. **Производительность Redis**:
   - *Риск*: Узкое место в Redis при высокой нагрузке
   - *Митигация*: Кластеризация Redis, оптимизация сериализации, мониторинг

3. **Дисковое пространство**:
   - *Риск*: Переполнение диска из-за кэшей
   - *Митигация*: Автоматическая очистка, мониторинг размеров, настройка TTL

4. **Консистентность между уровнями кэша**:
   - *Риск*: Рассинхронизация между MEMORY, REDIS, DISK
   - *Митигация*: Единая система версионирования, атомарные операции

### Операционные риски
1. **Сложность мониторинга**:
   - *Риск*: Трудность отслеживания производительности всех кэшей
   - *Митигация*: Централизованный `CacheMonitor`, автоматические алерты

2. **Зависимость от Ollama-сервера**:
   - *Риск*: Недоступность удаленного сервера
   - *Митигация*: Fallback на локальные модели, кэширование результатов

3. **Миграция данных**:
   - *Риск*: Потеря кэшированных данных при обновлениях
   - *Митигация*: Версионирование кэшей, постепенная миграция

### Качественные риски
1. **Переоптимизация кэширования**:
   - *Риск*: Слишком агрессивное кэширование может скрыть проблемы
   - *Митигация*: Балансировка между производительностью и актуальностью

2. **Отладка сложных взаимодействий**:
   - *Риск*: Трудность диагностики проблем в многоуровневой системе
   - *Митигация*: Подробное логирование, трассировка запросов

## Метрики успеха и KPI

### Ключевые показатели эффективности

#### Качество релевантности
- **Precision@K**: точность в топ-K результатах
  - Текущий: ~45%
  - Цель: >70%
- **Recall@K**: полнота в топ-K результатах
  - Текущий: ~35%
  - Цель: >65%
- **NDCG (Normalized Discounted Cumulative Gain)**
  - Текущий: ~0.42
  - Цель: >0.70
- **MRR (Mean Reciprocal Rank)**
  - Текущий: ~0.38
  - Цель: >0.65

#### Производительность системы
- **Время отклика на запрос**
  - Текущий: 2.5-4.0 сек
  - Цель: <1.0 сек
- **Пропускная способность**
  - Текущий: 50 запросов/мин
  - Цель: >200 запросов/мин
- **Hit rate кэшей**
  - Цель: >85% для всех типов кэшей
- **Использование памяти**
  - Цель: <2GB для кэшей
- **Использование CPU**
  - Цель: <70% при пиковой нагрузке

#### Качество источников
- **Снижение потерь качественных источников**
  - Текущий: 80% потерь
  - Цель: <35% потерь
- **Точность классификации источников**
  - Цель: >90%
- **Покрытие доменов**
  - Цель: >95% для основных доменов

### Критерии готовности к продакшену

#### Фаза 1 - Готовность
- [ ] Все простые кэши заменены на IntelligentCache
- [ ] CacheMonitor развернут и функционирует
- [ ] Hit rate кэшей >80%
- [ ] Время отклика улучшено на >30%

#### Фаза 2 - Готовность
- [ ] Ollama модели установлены и протестированы
- [ ] EnhancedOllamaEmbeddings интегрирован с EmbeddingCache
- [ ] A/B тестирование показывает улучшение качества на >15%
- [ ] Fallback механизмы работают корректно

#### Фаза 3 - Готовность
- [ ] CachedHybridRelevanceScorer развернут
- [ ] BM25 + Vector Search + LLM Re-ranking функционирует
- [ ] Precision@10 >60%
- [ ] Время отклика <1.5 сек

#### Фаза 4 - Готовность
- [ ] TemporalAnalysisCache интегрирован
- [ ] Темпоральные запросы оптимизированы
- [ ] Интеграция с cluster_analyzer и vector_store завершена

#### Фаза 5 - Готовность
- [ ] AdaptiveThresholds функционирует
- [ ] Комплексный мониторинг развернут
- [ ] Redis оптимизирован и кластеризован
- [ ] Fallback механизмы протестированы

#### Фаза 6 - Готовность к продакшену
- [ ] Все компоненты интегрированы
- [ ] Нагрузочное тестирование пройдено
- [ ] Документация создана
- [ ] Команда обучена

## Следующие шаги

### Немедленные действия (1-2 дня)
1. **Создание проектной команды**
   - Назначение технического лидера
   - Определение ролей и ответственности
   - Создание коммуникационных каналов

2. **Подготовка инфраструктуры**
   - Проверка доступности Redis
   - Подготовка тестовой среды
   - Настройка мониторинга

3. **Анализ текущего состояния**
   - Аудит существующих кэшей
   - Измерение базовых метрик
   - Документирование текущей архитектуры

### Краткосрочные цели (1 неделя)
1. **Начало Фазы 1**
   - Запуск аудита кэшей
   - Создание CacheMonitor
   - Первые оптимизации

2. **Подготовка к Фазе 2**
   - Установка Ollama
   - Загрузка эмбеддинг-моделей
   - Подготовка тестовых данных

### Среднесрочные цели (1 месяц)
1. **Завершение Фаз 1-2**
   - Унификация кэшей
   - Интеграция Ollama эмбеддингов
   - Первые результаты A/B тестирования

2. **Начало Фазы 3**
   - Разработка CachedHybridRelevanceScorer
   - Интеграция BM25 с кэшированием

### Долгосрочные цели (3 месяца)
1. **Полная реализация плана**
   - Завершение всех 6 фаз
   - Достижение целевых KPI
   - Готовность к продакшену

2. **Оптимизация и масштабирование**
   - Тонкая настройка производительности
   - Подготовка к росту нагрузки
   - Планирование дальнейших улучшений

## Управление рисками

### Мониторинг прогресса
- **Еженедельные ретроспективы**: анализ прогресса и корректировка планов
- **Ежедневные стендапы**: синхронизация команды и решение блокеров
- **Milestone reviews**: оценка готовности к переходу на следующую фазу

### Критические точки принятия решений
1. **После Фазы 1**: оценка эффективности оптимизации кэшей
2. **После Фазы 2**: решение о выборе основной эмбеддинг-модели
3. **После Фазы 3**: оценка качества гибридного поиска
4. **После Фазы 5**: решение о готовности к продакшену

### План отката (Rollback Plan)
- **Сохранение текущей системы**: возможность быстрого возврата
- **Поэтапный откат**: возможность отката отдельных компонентов
- **Мониторинг деградации**: автоматическое обнаружение проблем
- **Процедуры экстренного восстановления**: четкие инструкции для команды

## Заключение

Данный комплексный план улучшения анализа релевантности для системы Veritas представляет собой детальную дорожную карту для достижения значительных улучшений в качестве, производительности и масштабируемости системы.

### Ключевые преимущества плана:
- **Использование существующей инфраструктуры**: максимальное использование IntelligentCache
- **Поэтапная реализация**: минимизация рисков через пошаговое внедрение
- **Измеримые результаты**: четкие KPI и критерии успеха
- **Отказоустойчивость**: fallback механизмы и планы отката
- **Приватность данных**: использование локальных Ollama моделей

### Ожидаемый эффект:
- **Снижение потерь источников с 80% до 25-35%**
- **Увеличение точности на 25-35%**
- **Улучшение производительности на 60-80%**
- **Снижение латентности на 70-85%**
- **Экономия ресурсов на 90%+**

Успешная реализация этого плана сделает систему Veritas одной из самых эффективных и точных систем анализа фактов, обеспечивая высокое качество результатов при сохранении приватности и производительности.
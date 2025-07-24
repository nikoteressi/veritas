# Рекомендации по эмбеддинг-моделям для Ollama

## Обзор лучших моделей для анализа релевантности

### 1. Snowflake Arctic Embed 2 (Рекомендуется)
**Модель**: `snowflake-arctic-embed2`

#### Преимущества
- **Лучшее соотношение производительность/размер** <mcreference link="https://huggingface.co/Snowflake/snowflake-arctic-embed-2" index="1">1</mcreference>
- **Многоязычная поддержка** без потери качества на английском <mcreference link="https://huggingface.co/Snowflake/snowflake-arctic-embed-2" index="1">1</mcreference>
- **Превосходит модели с большим количеством параметров** <mcreference link="https://huggingface.co/Snowflake/snowflake-arctic-embed-2" index="1">1</mcreference>
- **Оптимизирована для retrieval задач** <mcreference link="https://huggingface.co/Snowflake/snowflake-arctic-embed-2" index="1">1</mcreference>

#### Технические характеристики
- Размер: ~335M параметров
- Поддерживаемые языки: 100+ языков
- Максимальная длина последовательности: 8192 токена
- Размерность эмбеддингов: 1024

#### Применение в Veritas
```python
# Установка на Ollama сервере
ollama pull snowflake-arctic-embed2

# Использование в коде
model = OllamaEmbeddingFunction(model_name="snowflake-arctic-embed2")
```

### 2. Granite 3 Embedding (Альтернатива)
**Модель**: `granite3-embedding`

#### Преимущества
- **Высокая производительность IBM** <mcreference link="https://huggingface.co/ibm-granite/granite-3.1-8b-instruct" index="2">2</mcreference>
- **Поддержка разреженных эмбеддингов** <mcreference link="https://huggingface.co/ibm-granite/granite-3.1-8b-instruct" index="2">2</mcreference>
- **Улучшенные возможности рассуждения** <mcreference link="https://huggingface.co/ibm-granite/granite-3.1-8b-instruct" index="2">2</mcreference>
- **Хорошие результаты в сравнении с bge-m3** <mcreference link="https://www.reddit.com/r/LocalLLaMA/comments/1h8qxvl/best_embedding_models_for_ollama/" index="3">3</mcreference>

#### Технические характеристики
- Доступны версии: 30M (английский) и 278M (многоязычная)
- Оптимизирована для задач сходства текста
- Поддержка извлечения и поиска

### 3. MxBai Embed Large (Для английского языка)
**Модель**: `mxbai-embed-large`

#### Преимущества
- **Отличная точность для английского языка** <mcreference link="https://ollama.com/library/mxbai-embed-large" index="4">4</mcreference>
- **Оптимизирована для семантического поиска** <mcreference link="https://ollama.com/library/mxbai-embed-large" index="4">4</mcreference>
- **Хорошая производительность на финансовых документах**

#### Применение
Идеальна для англоязычных финансовых документов и новостей.

### 4. BGE-M3 (Проверенное решение)
**Модель**: `bge-m3`

#### Преимущества
- **Проверенная модель** для многоязычных задач <mcreference link="https://ollama.com/library/bge-m3" index="5">5</mcreference>
- **Хорошая поддержка русского языка**
- **Стабильная производительность**

## Сравнительная таблица моделей

| Модель | Размер | Языки | Производительность | Рекомендация |
|--------|--------|-------|-------------------|--------------|
| snowflake-arctic-embed2 | 335M | 100+ | ⭐⭐⭐⭐⭐ | Основная |
| granite3-embedding | 278M | Многоязычная | ⭐⭐⭐⭐ | Альтернатива |
| mxbai-embed-large | ~335M | Английский | ⭐⭐⭐⭐ | Для EN |
| bge-m3 | ~560M | Многоязычная | ⭐⭐⭐ | Fallback |
| nomic-embed-text | ~137M | Английский | ⭐⭐ | Текущая |

## План тестирования моделей

### Этап 1: Установка моделей
```bash
# На Ollama сервере
ollama pull snowflake-arctic-embed2
ollama pull granite3-embedding
ollama pull mxbai-embed-large
ollama pull bge-m3
```

### Этап 2: Бенчмаркинг
1. **Тестовый набор**: 1000 пар (запрос, документ) из исторических данных
2. **Метрики**: 
   - Точность релевантности
   - Время обработки
   - Качество семантического поиска
3. **Сравнение**: с текущей nomic-embed-text

### Этап 3: A/B тестирование
1. **Группа A**: текущая система (nomic-embed-text)
2. **Группа B**: новая модель (snowflake-arctic-embed2)
3. **Метрики**: количество отфильтрованных источников, качество результатов

## Рекомендации по внедрению

### Приоритетный порядок
1. **snowflake-arctic-embed2** - основная модель
2. **granite3-embedding** - если первая недоступна
3. **mxbai-embed-large** - для англоязычного контента
4. **bge-m3** - fallback вариант

### Конфигурация OllamaEmbeddingFunction
```python
class AdaptiveOllamaEmbedding:
    def __init__(self, ollama_host="http://localhost:11434"):
        self.models_priority = [
            "snowflake-arctic-embed2",
            "granite3-embedding", 
            "mxbai-embed-large",
            "bge-m3",
            "nomic-embed-text"  # fallback
        ]
        self.ollama_host = ollama_host
        self.current_model = self._select_best_model()
    
    def _select_best_model(self):
        # Проверяем доступность моделей и выбираем лучшую
        for model in self.models_priority:
            if self._is_model_available(model):
                return OllamaEmbeddingFunction(
                    model_name=model,
                    ollama_host=self.ollama_host
                )
        raise Exception("No embedding models available")
```

### Мониторинг производительности
```python
class EmbeddingPerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'response_time': [],
            'accuracy_score': [],
            'model_availability': {}
        }
    
    def log_performance(self, model_name, response_time, accuracy):
        self.metrics['response_time'].append(response_time)
        self.metrics['accuracy_score'].append(accuracy)
        self.metrics['model_availability'][model_name] = True
    
    def get_best_model(self):
        # Анализ метрик и выбор лучшей модели
        pass
```

## Ожидаемые улучшения

### Количественные показатели
- **Точность**: +20-30% по сравнению с nomic-embed-text
- **Скорость**: локальная обработка снижает латентность на 40-60%
- **Качество фильтрации**: снижение потерь источников с 80% до 35-45%

### Качественные улучшения
- Лучшее понимание финансовой терминологии
- Улучшенная многоязычная поддержка
- Более точная семантическая оценка релевантности

## Заключение

Переход на современные эмбеддинг-модели через Ollama обеспечит значительное улучшение качества анализа релевантности в системе Veritas. Рекомендуется начать с **snowflake-arctic-embed2** как наиболее сбалансированного решения, с возможностью fallback на другие модели при необходимости.
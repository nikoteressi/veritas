"""
Relevance Integration Module

Интегрирует все компоненты релевантности в единую систему для улучшения
анализа релевантности в системе верификации фактов.
"""
import gc
import asyncio
import logging
from datetime import datetime
from typing import Any

import aiohttp

from app.config import settings

from ..analysis.adaptive_thresholds import AdaptiveThresholds
from ..cache.cache_monitor import CacheMonitor
from ..relevance.cached_hybrid_relevance_scorer import CachedHybridRelevanceScorer
from ..infrastructure.enhanced_ollama_embeddings import EnhancedOllamaEmbeddings
from ..relevance.explainable_relevance_scorer import ExplainableRelevanceScorer
from ..cache.intelligent_cache import IntelligentCache
from ..cache.temporal_analysis_cache import TemporalAnalysisCache

logger = logging.getLogger(__name__)


class RelevanceIntegrationManager:
    """
    Менеджер интеграции всех компонентов релевантности.

    Объединяет:
    - EnhancedOllamaEmbeddings для семантических эмбеддингов
    - CachedHybridRelevanceScorer для гибридного скоринга
    - TemporalAnalysisCache для временного анализа
    - ExplainableRelevanceScorer для объяснимых оценок
    - IntelligentCache для кэширования
    - AdaptiveThresholds для адаптивных порогов
    - CacheMonitor для мониторинга производительности
    """

    def __init__(
        self,
        ollama_host: str | None = None,
        embedding_model: str | None = None,
        cache_config: dict[str, Any] | None = None,
        enable_monitoring: bool = True,
    ):
        """
        Инициализация менеджера интеграции релевантности.

        Args:
            ollama_host: Хост Ollama сервера (если None, используется из настроек)
            embedding_model: Модель для эмбеддингов (если None, используется из настроек)
            cache_config: Конфигурация кэша
            enable_monitoring: Включить мониторинг
        """
        # Import settings here to avoid circular imports
        # Use settings if parameters not provided
        if ollama_host is None:
            ollama_host = settings.ollama_base_url
        if embedding_model is None:
            embedding_model = settings.embedding_model_name

        self.ollama_host = ollama_host
        self.embedding_model = embedding_model
        self.enable_monitoring = enable_monitoring

        # Инициализация базовых компонентов
        self.cache = IntelligentCache(**(cache_config or {}))
        self.adaptive_thresholds = AdaptiveThresholds()

        if enable_monitoring:
            self.cache_monitor = CacheMonitor()

        # Инициализация компонентов релевантности
        self.enhanced_embeddings = None
        self.hybrid_scorer = None
        self.temporal_cache = None
        self.explainable_scorer = None
        self._monitoring_task = None

        self._initialized = False

    async def initialize(self) -> None:
        """Асинхронная инициализация всех компонентов с проверкой доступности Ollama."""
        if self._initialized:
            logger.info("Components already initialized, skipping")
            return

        try:
            logger.info("Starting relevance components initialization...")

            # First check if Ollama is available
            logger.info("Checking Ollama health at %s", self.ollama_host)
            ollama_healthy = await self._check_ollama_health()
            logger.info("Ollama health check result: %s", ollama_healthy)

            if not ollama_healthy:
                logger.error(
                    "Ollama server at %s not available. Cannot initialize relevance components.",
                    self.ollama_host,
                )
                # Set initialized to True even if components are None to prevent repeated attempts
                self._initialized = True
                return

            logger.info(
                "Ollama is healthy, proceeding with component initialization")

            # Инициализация улучшенных эмбеддингов с уменшенным кэшем для Docker
            logger.info("Initializing enhanced embeddings...")
            self.enhanced_embeddings = EnhancedOllamaEmbeddings(
                model=self.embedding_model,
                base_url=self.ollama_host,
                cache_size=150,  # Further reduced for Docker environment
            )
            logger.info("Enhanced embeddings initialized successfully")

            # Инициализация гибридного скорера с общими эмбеддингами
            logger.info("Initializing hybrid scorer...")
            self.hybrid_scorer = CachedHybridRelevanceScorer(
                embeddings_model=self.embedding_model,
                embeddings_url=self.ollama_host,
                cache_size=100,  # Further reduced for Docker environment
                shared_embeddings=self.enhanced_embeddings,  # Share embeddings to reduce memory
            )
            logger.info("Hybrid scorer initialized successfully")

            # Инициализация временного анализа
            logger.info("Initializing temporal cache...")
            self.temporal_cache = TemporalAnalysisCache()
            logger.info("Temporal cache initialized successfully")

            # Инициализация объяснимого скорера
            logger.info("Initializing explainable scorer...")
            self.explainable_scorer = ExplainableRelevanceScorer(
                hybrid_scorer=self.hybrid_scorer, temporal_analyzer=self.temporal_cache
            )
            logger.info("Explainable scorer initialized successfully")

            # Запуск мониторинга как фоновой задачи
            if self.enable_monitoring and self.cache_monitor:
                logger.info("Starting cache monitoring...")
                self._monitoring_task = asyncio.create_task(
                    self.cache_monitor.start_monitoring())
                logger.info("Cache monitoring started as background task")

            self._initialized = True
            logger.info("All relevance components successfully initialized")

            # Trigger garbage collection after initialization to reduce memory pressure
            gc.collect()
            logger.debug(
                "Memory cleanup completed after relevance components initialization"
            )

        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError, OSError) as e:
            logger.error(
                "Error initializing relevance components: %s", e, exc_info=True
            )
            # Set initialized to True to prevent repeated attempts, but components remain None
            self._initialized = True
            logger.warning(
                "Initialization failed, components will remain None for graceful degradation"
            )

    async def _check_ollama_health(self) -> bool:
        """Проверка доступности Ollama сервера."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.ollama_host}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [
                            model.get("name", "") for model in data.get("models", [])
                        ]
                        if self.embedding_model in models:
                            logger.info(
                                "Ollama server is healthy and model '%s' is available",
                                self.embedding_model,
                            )
                            return True
                        else:
                            logger.warning(
                                "Model '%s' not found on Ollama server. Available models: %s",
                                self.embedding_model,
                                models,
                            )
                            return False
                    else:
                        logger.warning(
                            "Ollama server returned status %s", response.status
                        )
                        return False
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError) as e:
            logger.warning("Failed to check Ollama health: %s", e)
            return False

    async def calculate_comprehensive_relevance(
        self,
        query: str,
        documents: list[str],
        document_metadata: list[dict[str, Any]] | None = None,
        explain: bool = False,
    ) -> dict[str, Any]:
        """
        Вычисляет комплексную релевантность с использованием всех компонентов.

        Args:
            query: Поисковый запрос
            documents: Список документов для анализа
            document_metadata: Метаданные документов (включая временные метки)
            explain: Включить объяснения

        Returns:
            Словарь с результатами анализа релевантности
        """
        if not self._initialized:
            await self.initialize()

        # Validate that critical components are initialized
        if self.hybrid_scorer is None:
            logger.error(
                "Hybrid scorer is None - initialization may have failed")
            raise RuntimeError(
                "Hybrid scorer not initialized - cannot calculate relevance")

        if self.temporal_cache is None:
            logger.error(
                "Temporal cache is None - initialization may have failed")
            raise RuntimeError(
                "Temporal cache not initialized - cannot calculate relevance")

        if self.explainable_scorer is None:
            logger.error(
                "Explainable scorer is None - initialization may have failed")
            raise RuntimeError(
                "Explainable scorer not initialized - cannot calculate relevance")

        logger.debug(
            "All components validated, proceeding with relevance calculation")

        try:
            # Получение адаптивных порогов
            thresholds = await self.adaptive_thresholds.get_adaptive_threshold(
                query_type="relevance_analysis",
                source_type="document",
                context={"query_complexity": len(query.split())},
            )

            # Гибридный скоринг
            logger.info("Getting hybrid scores from scorer...")
            logger.debug(
                f"About to call score_documents with query: {query[:50]}...")
            logger.info(f"Documents count: {len(documents)}")
            logger.debug(f"Hybrid scorer type: {type(self.hybrid_scorer)}")

            hybrid_results = await self.hybrid_scorer.score_documents(
                query=query, documents=documents
            )

            logger.info(f"score_documents call completed")
            logger.debug(
                f"Received hybrid_results: type={type(hybrid_results)}")
            logger.debug(
                f"hybrid_results is coroutine: {asyncio.iscoroutine(hybrid_results)}")

            if asyncio.iscoroutine(hybrid_results):
                logger.error("ERROR: score_documents returned a coroutine!")
                raise RuntimeError(
                    "score_documents returned an unawaited coroutine")

            logger.debug(
                f"hybrid_results length: {len(hybrid_results) if hasattr(hybrid_results, '__len__') else 'N/A'}")

            # Debug each result
            for i, result in enumerate(hybrid_results):
                logger.debug(
                    f"hybrid_results[{i}]: type={type(result)}, value={result}")
                if asyncio.iscoroutine(result):
                    logger.error(
                        f"ERROR: hybrid_results[{i}] is a coroutine: {result}")
                    raise RuntimeError(
                        f"hybrid_results[{i}] is an unawaited coroutine")

            # Временной анализ (если есть метаданные)
            temporal_scores = {}
            if document_metadata:
                logger.info("Getting temporal scores...")
                for i, (doc, metadata) in enumerate(
                    zip(documents, document_metadata, strict=False)
                ):
                    try:
                        content_date = metadata.get(
                            "date", metadata.get("timestamp", datetime.now())
                        )
                        temporal_result = (
                            await self.temporal_cache.analyze_temporal_relevance(
                                query=query,
                                content=doc,
                                content_date=content_date,
                                base_relevance=0.5,  # Будет обновлено позже
                            )
                        )
                        logger.debug(
                            f"Temporal result for doc_{i}: type={type(temporal_result)}, value={temporal_result}")
                        # Always ensure temporal_scores contains dict structure
                        if isinstance(temporal_result, dict):
                            temporal_scores[f"doc_{i}"] = temporal_result
                        else:
                            # Convert any non-dict result to proper dict structure
                            temporal_scores[f"doc_{i}"] = {"adjusted_relevance": float(
                                temporal_result) if temporal_result else 1.0}
                    except Exception as e:
                        logger.error(
                            f"Error getting temporal score for doc_{i}: {e}")
                        temporal_scores[f"doc_{i}"] = {
                            "adjusted_relevance": 1.0}

            # Объединение результатов
            final_scores = []
            explanations = []

            for i, doc in enumerate(documents):
                # Базовый гибридный скор
                hybrid_result = hybrid_results[i]
                # Handle both score-only and (score, explanation) tuple formats
                hybrid_score = hybrid_result[0] if isinstance(
                    hybrid_result, tuple) else hybrid_result

                # Временной скор (если доступен)
                temporal_score = 1.0
                if temporal_scores and isinstance(temporal_scores, dict):
                    temporal_result = temporal_scores.get(f"doc_{i}", {})
                    if isinstance(temporal_result, dict):
                        temporal_score = temporal_result.get(
                            "adjusted_relevance", 1.0)
                    else:
                        temporal_score = float(
                            temporal_result) if temporal_result else 1.0
                elif temporal_scores and not isinstance(temporal_scores, dict):
                    logger.error(
                        f"temporal_scores is not a dict: type={type(temporal_scores)}, value={temporal_scores}")
                    temporal_score = 1.0

                # Комбинированный скор
                combined_score = (
                    hybrid_score * 0.7  # Гибридный скор (70%)
                    + temporal_score * 0.3  # Временной скор (30%)
                )

                final_scores.append(
                    {
                        "document_index": i,
                        "hybrid_score": hybrid_score,
                        "temporal_score": temporal_score,
                        "combined_score": combined_score,
                        "above_threshold": combined_score
                        >= (thresholds if isinstance(thresholds, (int, float)) else 0.5),
                    }
                )

                # Генерация объяснений
                if explain:
                    explanation = await self.explainable_scorer.explain_relevance(
                        query=query,
                        text=doc,
                        content_date=(
                            document_metadata[i].get("date")
                            if document_metadata and i < len(document_metadata) and isinstance(document_metadata[i], dict)
                            else None
                        ),
                        depth="comprehensive",
                    )
                    explanations.append(explanation)

            # Сортировка по комбинированному скору
            final_scores.sort(key=lambda x: x["combined_score"], reverse=True)

            result = {
                "query": query,
                "total_documents": len(documents),
                "scores": final_scores,
                "thresholds": thresholds,
                "performance_metrics": {
                    "hybrid_scorer": self.hybrid_scorer.get_performance_metrics(),
                    "temporal_analysis": (
                        {}
                        if not temporal_scores or not isinstance(temporal_scores, dict)
                        else {}
                    ),
                    "cache_stats": await self.cache.get_stats(),
                },
                "timestamp": datetime.now().isoformat(),
            }

            if explain:
                result["explanations"] = explanations

            return result

        except Exception as e:
            logger.error(
                "Error calculating comprehensive relevance: %s", e
            )
            raise

    async def batch_analyze_relevance(
        self,
        queries: list[str],
        document_sets: list[list[str]],
        metadata_sets: list[list[dict[str, Any]]] | None = None,
        explain: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Пакетный анализ релевантности для множественных запросов.

        Args:
            queries: Список запросов
            document_sets: Список наборов документов для каждого запроса
            metadata_sets: Список наборов метаданных
            explain: Включить объяснения

        Returns:
            Список результатов анализа релевантности
        """
        if not self._initialized:
            await self.initialize()

        tasks = []
        for i, query in enumerate(queries):
            documents = document_sets[i]
            metadata = metadata_sets[i] if metadata_sets else None

            task = self.calculate_comprehensive_relevance(
                query=query,
                documents=documents,
                document_metadata=metadata,
                explain=explain,
            )
            tasks.append(task)

        return await asyncio.gather(*tasks)

    async def get_performance_report(self) -> dict[str, Any]:
        """
        Получает отчет о производительности всех компонентов.

        Returns:
            Словарь с метриками производительности
        """
        if not self._initialized:
            await self.initialize()

        report = {
            "timestamp": datetime.now().isoformat(),
            "cache_stats": await self.cache.get_stats(),
            "adaptive_thresholds_stats": await self.adaptive_thresholds.get_performance_summary(),
        }

        if self.enhanced_embeddings:
            report["embeddings_stats"] = (
                self.enhanced_embeddings.get_performance_metrics()
            )

        if self.hybrid_scorer:
            report["hybrid_scorer_stats"] = (
                self.hybrid_scorer.get_performance_metrics()
            )

        if self.temporal_cache:
            report["temporal_cache_stats"] = (
                self.temporal_cache.get_performance_metrics()
            )

        if self.explainable_scorer:
            report["explainable_scorer_stats"] = (
                self.explainable_scorer.get_performance_metrics()
            )

        if self.enable_monitoring and self.cache_monitor:
            report["cache_monitor_stats"] = (
                await self.cache_monitor.generate_performance_report()
            )

        return report

    async def optimize_performance(self) -> dict[str, Any]:
        """
        Оптимизирует производительность всех компонентов.

        Returns:
            Отчет об оптимизации
        """
        if not self._initialized:
            await self.initialize()

        optimization_results = {}

        # Оптимизация кэша
        cache_optimization = await self.cache.optimize()
        optimization_results["cache"] = cache_optimization

        # Оптимизация адаптивных порогов
        thresholds_optimization = (
            await self.adaptive_thresholds.get_threshold_recommendations()
        )
        optimization_results["adaptive_thresholds"] = thresholds_optimization

        # Оптимизация компонентов релевантности
        if self.enhanced_embeddings:
            embeddings_optimization = await self.enhanced_embeddings.optimize_cache()
            optimization_results["embeddings"] = embeddings_optimization

        if self.hybrid_scorer:
            scorer_optimization = await self.hybrid_scorer.optimize_cache()
            optimization_results["hybrid_scorer"] = scorer_optimization

        if self.temporal_cache:
            temporal_optimization = await self.temporal_cache.optimize_cache()
            optimization_results["temporal_cache"] = temporal_optimization

        if self.explainable_scorer:
            explainable_optimization = await self.explainable_scorer.optimize_cache()
            optimization_results["explainable_scorer"] = explainable_optimization

        return {
            "timestamp": datetime.now().isoformat(),
            "optimization_results": optimization_results,
        }

    async def close(self) -> None:
        """Закрывает все компоненты и освобождает ресурсы."""
        if not self._initialized:
            return

        try:
            # Остановка мониторинга и отмена фоновой задачи
            if self.enable_monitoring and self.cache_monitor:
                await self.cache_monitor.stop_monitoring()

                # Отмена фоновой задачи мониторинга
                if hasattr(self, '_monitoring_task') and self._monitoring_task:
                    self._monitoring_task.cancel()
                    try:
                        await self._monitoring_task
                    except asyncio.CancelledError:
                        logger.info(
                            "Cache monitoring task cancelled successfully")

            # Закрытие компонентов
            if self.enhanced_embeddings:
                await self.enhanced_embeddings.close()

            if self.hybrid_scorer:
                await self.hybrid_scorer.close()

            if self.temporal_cache:
                await self.temporal_cache.close()

            if self.explainable_scorer:
                await self.explainable_scorer.close()

            # Закрытие базовых компонентов
            await self.cache.close()

            self._initialized = False
            logger.info("All relevance components closed successfully")

        except Exception as e:
            logger.error("Error closing relevance components: %s", e)
            raise


# Global instance of the manager
_relevance_manager: RelevanceIntegrationManager | None = None


async def get_relevance_manager(
    ollama_host: str | None = None,
    embedding_model: str | None = None,
    cache_config: dict[str, Any] | None = None,
    enable_monitoring: bool = True,
) -> RelevanceIntegrationManager:
    """
    Получает глобальный экземпляр менеджера релевантности.

    Args:
        ollama_host: Хост Ollama сервера (если None, используется из настроек)
        embedding_model: Модель для эмбеддингов (если None, используется из настроек)
        cache_config: Конфигурация кэша
        enable_monitoring: Включить мониторинг

    Returns:
        Экземпляр RelevanceIntegrationManager
    """
    global _relevance_manager

    if _relevance_manager is None:
        # Import settings here to avoid circular imports
        # Use settings if parameters not provided
        if ollama_host is None:
            ollama_host = settings.ollama_base_url
        if embedding_model is None:
            embedding_model = settings.embedding_model_name

        _relevance_manager = RelevanceIntegrationManager(
            ollama_host=ollama_host,
            embedding_model=embedding_model,
            cache_config=cache_config,
            enable_monitoring=enable_monitoring,
        )

        # Initialize the manager
        await _relevance_manager.initialize()

    return _relevance_manager


async def close_relevance_manager() -> None:
    """Закрывает глобальный менеджер релевантности."""
    global _relevance_manager

    if _relevance_manager is not None:
        await _relevance_manager.close()
        _relevance_manager = None

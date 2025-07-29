"""
Source manager for web scraping and source management.

Handles scraping of web sources and caching for efficient verification.
Enhanced with intelligent caching and adaptive relevance scoring.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections import Counter
from typing import Any

from ...analysis.adaptive_thresholds import get_adaptive_thresholds
from ...cache.intelligent_cache import IntelligentCache
from ...infrastructure.web_scraper import WebScraper
from ...relevance.relevance_orchestrator import get_relevance_manager

logger = logging.getLogger(__name__)


class EnhancedSourceManager:
    """Enhanced source manager with intelligent caching and adaptive relevance scoring."""

    def __init__(self, max_concurrent_scrapes: int = 3):
        """Initialize EnhancedSourceManager with intelligent caching and controlled concurrency.

        Args:
            max_concurrent_scrapes: Maximum number of concurrent scraping operations (default: 3)
        """
        self.web_scraper = WebScraper(max_concurrent_scrapes=max_concurrent_scrapes)

        # Use IntelligentCache instead of simple dict
        self.cache = IntelligentCache(max_memory_size=1000)

        # Get adaptive thresholds instance
        self.adaptive_thresholds = get_adaptive_thresholds()

        # Initialize relevance manager (will be set up async)
        self.relevance_manager = None
        self._relevance_initialized = False

        # Enhanced stop words for better relevance calculation
        self.stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "what",
            "where",
            "when",
            "why",
            "how",
            "which",
            "who",
            "whom",
        }

    async def _ensure_relevance_manager(self):
        """Ensure relevance manager is initialized."""
        if not self._relevance_initialized:
            try:
                logger.info("Getting relevance manager singleton for source manager...")
                self.relevance_manager = get_relevance_manager()

                # Check if the singleton is already initialized
                if self.relevance_manager.is_initialized:
                    logger.info("Relevance manager singleton already initialized, using existing instance")
                    self._relevance_initialized = True
                else:
                    # Initialize the relevance manager only if not already initialized
                    logger.info("Initializing relevance manager singleton...")
                    initialization_success = await self.relevance_manager.initialize()

                    if initialization_success:
                        self._relevance_initialized = True
                        logger.info("Relevance manager singleton initialized successfully")
                    else:
                        logger.warning("Relevance manager could not be initialized (Ollama not available)")
                        self.relevance_manager = None
                        self._relevance_initialized = False

            except (ConnectionError, TimeoutError) as e:
                logger.error("Failed to initialize relevance manager for source manager: %s", e)
                self.relevance_manager = None
                self._relevance_initialized = False

    async def scrape_sources_batch(self, urls: list[str], query_context: str | None = None) -> dict[str, str]:
        """
        Scrape multiple URLs concurrently with intelligent caching.

        Args:
            urls: List of URLs to scrape
            query_context: Optional query context for cache optimization

        Returns:
            Dictionary mapping URLs to their scraped content
        """
        logger.info("Starting batch scraping for %d URLs", len(urls))

        results = {}
        cache_hits = 0
        urls_to_scrape = []

        # First pass: Check cache for all URLs
        for url in urls:
            try:
                # Generate cache key
                cache_key = f"source_content:{hashlib.md5(url.encode()).hexdigest()}"

                # Check intelligent cache first
                cached_content = await self.cache.get(cache_key)
                if cached_content:
                    logger.debug("Cache hit for URL: %s", url)
                    results[url] = cached_content
                    cache_hits += 1
                else:
                    urls_to_scrape.append(url)

            except (ConnectionError, TimeoutError) as e:
                logger.error("Error checking cache for URL %s: %s", url, str(e))
                urls_to_scrape.append(url)

        # Second pass: Scrape all non-cached URLs concurrently
        if urls_to_scrape:
            logger.info("Scraping %d URLs concurrently", len(urls_to_scrape))
            try:
                scrape_results = await self.web_scraper.scrape_urls(urls_to_scrape)

                # Process all scrape results
                for scrape_result in scrape_results:
                    url = scrape_result.get("url")
                    if not url:
                        logger.warning("Scrape result missing URL")
                        continue

                    try:
                        if scrape_result.get("status") == "success" and scrape_result.get("content"):
                            content = scrape_result["content"]

                            # Store in intelligent cache with context
                            cache_key = f"source_content:{hashlib.md5(url.encode()).hexdigest()}"
                            cache_metadata = {
                                "url": url,
                                "query_context": query_context,
                                "content_length": len(content),
                                "scraped_at": "now",
                            }

                            await self.cache.set(
                                cache_key,
                                content,
                                ttl_seconds=3600,  # 1 hour TTL
                                dependencies=cache_metadata,
                            )

                            results[url] = content
                            logger.debug("Successfully scraped and cached: %s", url)
                        else:
                            error_msg = scrape_result.get("error_message", "Unknown error")
                            logger.warning("Failed to scrape URL %s: %s", url, error_msg)
                            results[url] = f"Failed to scrape: {error_msg}"

                    except (ConnectionError, TimeoutError) as e:
                        logger.error("Error processing scrape result for URL %s: %s", url, str(e))
                        results[url] = f"Failed to scrape: {str(e)}"

            except (ConnectionError, TimeoutError) as e:
                logger.error("Error during batch scraping: %s", str(e))
                # Fallback: mark all non-cached URLs as failed
                for url in urls_to_scrape:
                    results[url] = f"Failed to scrape: {str(e)}"

        cache_hit_rate = cache_hits / len(urls) if urls else 0
        successful_scrapes = len([r for r in results.values() if r and not r.startswith("Failed to scrape")])
        logger.info(
            "Batch scraping completed. Cache hit rate: %.2f%%. Successfully scraped %d out of %d URLs",
            cache_hit_rate * 100,
            successful_scrapes,
            len(urls),
        )

        return results

    async def calculate_relevance(
        self,
        content: str,
        query: str,
        source_type: str = "web",
        query_type: str = "general",
    ) -> float:
        """
        Calculate enhanced relevance score between content and query using adaptive thresholds.

        Args:
            content: The content to evaluate
            query: The query to match against
            source_type: Type of source (web, academic, news, etc.)
            query_type: Type of query (factual, opinion, analysis, etc.)

        Returns:
            Relevance score between 0 and 1
        """
        if not content or not query:
            return 0.0

        # Generate cache key for relevance calculation
        inner = content[:100] + query + source_type + query_type
        hash_ = hashlib.md5(inner.encode()).hexdigest()
        cache_key = f"relevance:{hash_}"

        # Check cache first
        cached_score = await self.cache.get(cache_key)
        if cached_score is not None:
            return float(cached_score)

        # Enhanced relevance calculation
        score = await self._calculate_enhanced_relevance(content, query, source_type, query_type)

        # Cache the result
        # 30 minutes TTL
        await self.cache.set(cache_key, score, ttl_seconds=1800)

        return score

    async def _calculate_enhanced_relevance(self, content: str, query: str, source_type: str, query_type: str) -> float:
        """Calculate enhanced relevance score with multiple factors."""
        content_lower = content.lower()
        query_lower = query.lower()

        # Tokenize and clean
        content_words = self._tokenize_and_clean(content_lower)
        query_words = self._tokenize_and_clean(query_lower)

        if not query_words:
            return 0.0

        # 1. Exact phrase matching (highest weight)
        phrase_score = self._calculate_phrase_score(content_lower, query_lower)

        # 2. Keyword frequency and density
        keyword_score = self._calculate_keyword_score(content_words, query_words)

        # 3. Semantic proximity (word co-occurrence)
        proximity_score = self._calculate_proximity_score(content_words, query_words)

        # 4. Content structure analysis
        structure_score = self._calculate_structure_score(content, query_words)

        # 5. Source type adjustment
        source_weight = self._get_source_weight(source_type)

        # Combine scores with weights
        combined_score = (
            phrase_score * 0.35 + keyword_score * 0.25 + proximity_score * 0.20 + structure_score * 0.20
        ) * source_weight

        # Apply adaptive thresholds
        threshold = await self.adaptive_thresholds.get_adaptive_threshold(
            query_type=query_type,
            source_type=source_type,
            context={"content_length": len(content)},
        )

        # Normalize and apply threshold
        final_score = min(1.0, max(0.0, combined_score))

        # Record performance metrics
        await self.adaptive_thresholds.record_performance_metrics(
            precision=0.8,  # Placeholder precision
            recall=0.7,  # Placeholder recall
            f1_score=0.75,  # Placeholder F1 score
            source_retention_rate=0.9,  # Placeholder retention rate
            query_type=query_type,
            source_type=source_type,
        )

        return final_score

    def _tokenize_and_clean(self, text: str) -> list[str]:
        """Tokenize text and remove stop words."""
        # Simple tokenization
        words = re.findall(r"\b\w+\b", text)
        # Remove stop words and short words
        return [word for word in words if word not in self.stop_words and len(word) > 2]

    def _calculate_phrase_score(self, content: str, query: str) -> float:
        """Calculate score based on exact phrase matches."""
        if query in content:
            return 1.0

        # Check for partial phrase matches
        query_words = query.split()
        if len(query_words) > 1:
            phrases = [" ".join(query_words[i : i + 2]) for i in range(len(query_words) - 1)]
            matches = sum(1 for phrase in phrases if phrase in content)
            return matches / len(phrases)

        return 0.0

    def _calculate_keyword_score(self, content_words: list[str], query_words: list[str]) -> float:
        """Calculate score based on keyword frequency and density."""
        if not query_words or not content_words:
            return 0.0

        content_counter = Counter(content_words)
        total_content_words = len(content_words)

        score = 0.0
        for word in query_words:
            if word in content_counter:
                # TF-IDF like scoring
                tf = content_counter[word] / total_content_words
                # Simple IDF approximation (rare words get higher scores)
                idf = 1.0 / (1.0 + content_counter[word] / 10.0)
                score += tf * idf

        return min(1.0, score / len(query_words))

    def _calculate_proximity_score(self, content_words: list[str], query_words: list[str]) -> float:
        """Calculate score based on word proximity in content."""
        if not query_words or not content_words:
            return 0.0

        # Find positions of query words in content
        word_positions = {}
        for i, word in enumerate(content_words):
            if word in query_words:
                if word not in word_positions:
                    word_positions[word] = []
                word_positions[word].append(i)

        if len(word_positions) < 2:
            return 0.0

        # Calculate average distance between query words
        distances = []
        words_found = list(word_positions.keys())

        for i, word1 in enumerate(words_found):
            for word2 in words_found[i + 1 :]:
                word1_positions = word_positions[word1]
                word2_positions = word_positions[word2]

                min_distance = min(abs(pos1 - pos2) for pos1 in word1_positions for pos2 in word2_positions)
                distances.append(min_distance)

        if distances:
            avg_distance = sum(distances) / len(distances)
            # Closer words get higher scores
            return max(0.0, 1.0 - (avg_distance / 50.0))

        return 0.0

    def _calculate_structure_score(self, content: str, query_words: list[str]) -> float:
        """Calculate score based on content structure (titles, headers, etc.)."""
        score = 0.0

        # Check for query words in titles/headers (simple heuristic)
        lines = content.split("\n")
        for line in lines[:5]:  # Check first 5 lines for titles
            line_lower = line.lower()
            if any(word in line_lower for word in query_words):
                if len(line.strip()) < 100:  # Likely a title/header
                    score += 0.3

        # Check for query words in emphasized text (caps, etc.)
        caps_words = re.findall(r"\b[A-Z]{2,}\b", content)
        caps_lower = [word.lower() for word in caps_words]
        for word in query_words:
            if word in caps_lower:
                score += 0.2

        return min(1.0, score)

    def _get_source_weight(self, source_type: str) -> float:
        """Get weight multiplier based on source type."""
        weights = {
            "academic": 1.2,
            "news": 1.1,
            "government": 1.15,
            "encyclopedia": 1.1,
            "web": 1.0,
            "social": 0.8,
            "forum": 0.7,
        }
        return weights.get(source_type, 1.0)

    def extract_sources_from_search_results(self, search_results: list[dict[str, Any]]) -> set[str]:
        """Extract unique source URLs from search results."""
        sources = set()

        for result in search_results:
            if isinstance(result, dict):
                # Handle different result formats
                if "url" in result:
                    sources.add(result["url"])
                elif "link" in result:
                    sources.add(result["link"])
                elif "href" in result:
                    sources.add(result["href"])

                # Handle nested results
                if "results" in result and isinstance(result["results"], list):
                    for nested_result in result["results"]:
                        if isinstance(nested_result, dict) and "url" in nested_result:
                            sources.add(nested_result["url"])

        return sources

    async def filter_evidence_for_cluster(
        self, cluster_queries: list[str], scraped_content: dict[str, str]
    ) -> list[dict[str, Any]]:
        """Filter scraped content to find evidence relevant to cluster queries."""
        # Ensure relevance manager is initialized
        await self._ensure_relevance_manager()

        evidence = []

        logger.info(
            "Filtering evidence for %d queries from %d sources",
            len(cluster_queries),
            len(scraped_content),
        )
        logger.debug("Cluster queries: %s", cluster_queries)

        for url, content in scraped_content.items():
            if not content or content.startswith("Failed to scrape"):
                logger.debug("Skipping %s: invalid content", url)
                continue

            # Use enhanced relevance calculation if available
            if self.relevance_manager:
                try:
                    # Use comprehensive relevance analysis
                    logger.debug(
                        f"About to call calculate_comprehensive_relevance from source_manager for URL: {url[:50]}"
                    )
                    logger.debug(f"Query: {' '.join(cluster_queries)[:100]}")
                    logger.debug(f"Content length: {len(content)}")

                    relevance_result = await self.relevance_manager.calculate_comprehensive_relevance(
                        query=" ".join(cluster_queries),
                        document=content,
                        metadata=[{"url": url, "timestamp": "now"}],
                    )

                    logger.debug(f"Got relevance_result type: {type(relevance_result)}")
                    logger.debug(f"Is relevance_result a coroutine? {asyncio.iscoroutine(relevance_result)}")
                    if asyncio.iscoroutine(relevance_result):
                        logger.error("ERROR: relevance_result is still a coroutine!")
                        raise RuntimeError("relevance_result is a coroutine instead of a result")
                    # Extract the combined score from the relevance result
                    if relevance_result and "scores" in relevance_result and relevance_result["scores"]:
                        relevance_score = relevance_result["scores"][0]["combined_score"]
                    else:
                        relevance_score = 0.0
                    logger.debug(
                        "Enhanced relevance score for %s: %.4f",
                        url[:50],
                        relevance_score,
                    )
                except (ConnectionError, TimeoutError) as e:
                    logger.warning(
                        "Enhanced relevance calculation failed for %s: %s. Using fallback.",
                        url,
                        e,
                    )
                    relevance_score = await self._calculate_relevance_fallback(cluster_queries, content)
            else:
                # Fallback to basic calculation
                relevance_score = await self._calculate_relevance_fallback(cluster_queries, content)

            logger.debug("URL: %s - Relevance score: %.4f", url[:50], relevance_score)

            # Use adaptive threshold
            threshold = await self.adaptive_thresholds.get_adaptive_threshold(
                query_type="fact_verification",
                source_type="web",
                context={"content_length": len(content)},
            )

            if relevance_score > threshold:
                evidence.append(
                    {
                        "url": url,
                        "content": content[:2000],  # Limit content length
                        "relevance_score": relevance_score,
                        "source_type": "web",
                        "threshold_used": threshold,
                    }
                )
                logger.info(
                    "Added evidence from %s with score %.4f (threshold: %.4f)",
                    url,
                    relevance_score,
                    threshold,
                )
            else:
                logger.debug(
                    "Filtered out %s - score %.4f below threshold %.4f",
                    url,
                    relevance_score,
                    threshold,
                )

        # Sort by relevance score
        evidence.sort(key=lambda x: x["relevance_score"], reverse=True)

        logger.info("Found %d relevant sources before limiting", len(evidence))

        # Limit to top 10 most relevant sources
        limited_evidence = evidence[:10]

        if len(evidence) > 10:
            logger.info("Limited evidence to top 10 sources (from %d)", len(evidence))

        logger.info("Final evidence count: %d", len(limited_evidence))
        for i, ev in enumerate(limited_evidence):
            logger.info("Evidence %d: %s (score: %.4f)", i + 1, ev["url"], ev["relevance_score"])

        return limited_evidence

    async def _calculate_relevance_fallback(self, queries: list[str], content: str) -> float:
        """Fallback relevance calculation method (improved version of the original)."""
        if not queries or not content:
            return 0.0

        # Normalize content
        content_lower = content.lower()

        total_score = 0.0
        content_words = [word for word in content_lower.split() if word not in self.stop_words and len(word) > 2]

        for query in queries:
            query_lower = query.lower()
            query_words = [word for word in query_lower.split() if word not in self.stop_words and len(word) > 2]

            if not query_words:
                continue

            # 1. Exact phrase matching (highest weight)
            phrase_score = 0.0
            if query_lower in content_lower:
                phrase_score = 1.0

            # 2. Individual word matching with TF-IDF-like scoring
            word_scores = []
            for word in query_words:
                if word in content_lower:
                    # Count occurrences
                    word_count = content_lower.count(word)
                    # Calculate term frequency
                    tf = word_count / len(content_words) if content_words else 0
                    # Simple inverse document frequency approximation
                    # Rare words get higher scores
                    idf = 1.0 / (1.0 + word_count * 0.1)
                    word_score = tf * idf
                    word_scores.append(word_score)
                else:
                    word_scores.append(0.0)

            # 3. Partial word matching (for related terms)
            partial_scores = []
            for word in query_words:
                partial_matches = sum(
                    1 for content_word in content_words if word in content_word or content_word in word
                )
                partial_score = partial_matches / len(content_words) if content_words else 0
                # Lower weight for partial matches
                partial_scores.append(partial_score * 0.5)

            # 4. Proximity scoring (words appearing close together)
            proximity_score = 0.0
            if len(query_words) > 1:
                content_text = " ".join(content_words)
                for i in range(len(query_words) - 1):
                    word1, word2 = query_words[i], query_words[i + 1]
                    if word1 in content_text and word2 in content_text:
                        # Find positions and calculate distance
                        pos1 = content_text.find(word1)
                        pos2 = content_text.find(word2)
                        if pos1 != -1 and pos2 != -1:
                            distance = abs(pos1 - pos2)
                            # Closer words get higher scores
                            proximity_score += 1.0 / (1.0 + distance * 0.01)

            # Combine scores with weights
            avg_word_score = sum(word_scores) / len(word_scores) if word_scores else 0
            avg_partial_score = sum(partial_scores) / len(partial_scores) if partial_scores else 0

            query_score = (
                phrase_score * 0.4  # Exact phrase: 40%
                + avg_word_score * 0.35  # Word matching: 35%
                + avg_partial_score * 0.15  # Partial matching: 15%
                + proximity_score * 0.1  # Proximity: 10%
            )

            total_score += query_score

        return total_score / len(queries) if queries else 0.0

    async def clear_cache(self):
        """Clear the intelligent cache."""
        await self.cache.clear()
        logger.info("Intelligent cache cleared")

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = await self.cache.get_stats()
        return {
            "cache_levels": stats.get("levels", {}),
            "total_entries": stats.get("total_entries", 0),
            "hit_rate": stats.get("hit_rate", 0.0),
            "memory_usage": stats.get("memory_usage", 0),
            "strategies_active": stats.get("strategies", []),
        }

    async def optimize_cache(self):
        """Optimize cache performance based on usage patterns."""
        await self.cache.optimize()
        logger.info("Cache optimization completed")

    async def close(self):
        """Close the web scraper and clean up resources."""
        if self.web_scraper:
            await self.web_scraper.close()
            logger.info("SourceManager: WebScraper closed")

        # Close relevance manager
        if self.relevance_manager:
            await self.relevance_manager.close()
            logger.info("SourceManager: RelevanceManager closed")

        # Close intelligent cache
        if self.cache:
            await self.cache.close()
            logger.info("SourceManager: IntelligentCache closed")

        logger.info("SourceManager: Resources cleaned up")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Backward compatibility alias
SourceManager = EnhancedSourceManager

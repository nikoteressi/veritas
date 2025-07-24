"""
Source manager for web scraping and source management.

Handles scraping of web sources and caching for efficient verification.
"""

import logging
from typing import Any

from agent.services.web_scraper import WebScraper

logger = logging.getLogger(__name__)


class SourceManager:
    """Manages web sources and scraping operations."""

    def __init__(self):
        """
        Initialize SourceManager with configurable concurrency limit.

        Args:
            max_concurrent_instances: Maximum number of concurrent scraping instances (default: 3)
        """
        self.web_scraper = WebScraper()
        self._scrape_cache: dict[str, str] = {}

    async def scrape_sources_batch(self, urls: list[str]) -> dict[str, str]:
        """Scrape multiple sources in parallel with caching."""
        if not urls:
            return {}

        # Filter out already cached URLs
        urls_to_scrape = [url for url in urls if url not in self._scrape_cache]

        if not urls_to_scrape:
            logger.info("All sources already cached")
            return {
                url: self._scrape_cache[url]
                for url in urls
                if url in self._scrape_cache
            }

        logger.info(f"Scraping {len(urls_to_scrape)} new sources")

        # Use scrape_urls directly for all URLs at once
        try:
            scrape_results = await self.web_scraper.scrape_urls(urls_to_scrape)
        except Exception as e:
            logger.error(f"Error during batch scraping: {e}")
            # Fallback: mark all URLs as failed
            scraped_content = {}
            for url in urls_to_scrape:
                scraped_content[url] = f"Scraping failed: {str(e)}"

            # Add cached content for requested URLs
            for url in urls:
                if url in self._scrape_cache and url not in scraped_content:
                    scraped_content[url] = self._scrape_cache[url]

            return scraped_content

        # Process results and update cache
        scraped_content = {}

        for result in scrape_results:
            url = result["url"]
            if result["status"] == "success" and result["content"]:
                content = result["content"]
                self._scrape_cache[url] = content
                scraped_content[url] = content
            else:
                error_msg = result.get("error_message", "No content extracted")
                scraped_content[url] = f"Failed to scrape: {error_msg}"

        # Add cached content for requested URLs
        for url in urls:
            if url in self._scrape_cache and url not in scraped_content:
                scraped_content[url] = self._scrape_cache[url]

        logger.info(f"Successfully scraped {len(scraped_content)} sources")
        return scraped_content

    def extract_sources_from_search_results(
        self, search_results: list[dict[str, Any]]
    ) -> set[str]:
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

    def filter_evidence_for_cluster(
        self, cluster_queries: list[str], scraped_content: dict[str, str]
    ) -> list[dict[str, Any]]:
        """Filter scraped content to find evidence relevant to cluster queries."""
        evidence = []
        
        logger.info(f"Filtering evidence for {len(cluster_queries)} queries from {len(scraped_content)} sources")
        logger.debug(f"Cluster queries: {cluster_queries}")

        for url, content in scraped_content.items():
            if not content or content.startswith("Failed to scrape"):
                logger.debug(f"Skipping {url}: invalid content")
                continue

            # Check if content is relevant to any cluster query
            relevance_score = self._calculate_relevance(
                cluster_queries, content)
            
            logger.debug(f"URL: {url[:50]}... - Relevance score: {relevance_score:.4f}")

            if relevance_score > 0.05:  # Lowered threshold for better sensitivity
                evidence.append(
                    {
                        "url": url,
                        "content": content[:2000],  # Limit content length
                        "relevance_score": relevance_score,
                        "source_type": "web",
                    }
                )
                logger.info(f"Added evidence from {url} with score {relevance_score:.4f}")
            else:
                logger.debug(f"Filtered out {url} - score {relevance_score:.4f} below threshold")

        # Sort by relevance score
        evidence.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        logger.info(f"Found {len(evidence)} relevant sources before limiting")

        # Limit to top 10 most relevant sources
        limited_evidence = evidence[:10]
        
        if len(evidence) > 10:
            logger.info(f"Limited evidence to top 10 sources (from {len(evidence)})")
        
        logger.info(f"Final evidence count: {len(limited_evidence)}")
        for i, ev in enumerate(limited_evidence):
            logger.info(f"Evidence {i+1}: {ev['url']} (score: {ev['relevance_score']:.4f})")

        return limited_evidence

    def _calculate_relevance(self, queries: list[str], content: str) -> float:
        """Calculate relevance score between queries and content using improved algorithm."""
        if not queries or not content:
            return 0.0

        # Normalize content
        content_lower = content.lower()
        
        # Common stop words to filter out
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        total_score = 0.0
        content_words = [word for word in content_lower.split() if word not in stop_words and len(word) > 2]
        
        for query in queries:
            query_lower = query.lower()
            query_words = [word for word in query_lower.split() if word not in stop_words and len(word) > 2]
            
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
                partial_matches = sum(1 for content_word in content_words if word in content_word or content_word in word)
                partial_score = partial_matches / len(content_words) if content_words else 0
                partial_scores.append(partial_score * 0.5)  # Lower weight for partial matches
            
            # 4. Proximity scoring (words appearing close together)
            proximity_score = 0.0
            if len(query_words) > 1:
                content_text = ' '.join(content_words)
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
                phrase_score * 0.4 +           # Exact phrase: 40%
                avg_word_score * 0.35 +        # Word matching: 35%
                avg_partial_score * 0.15 +     # Partial matching: 15%
                proximity_score * 0.1          # Proximity: 10%
            )
            
            total_score += query_score

        return total_score / len(queries) if queries else 0.0

    async def clear_cache(self):
        """Clear the scraping cache."""
        self._scrape_cache.clear()
        logger.info("Scraping cache cleared")

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return {
            "cached_urls": len(self._scrape_cache),
            "total_cache_size": sum(
                len(content) for content in self._scrape_cache.values()
            ),
        }

    async def close(self):
        """Close the web scraper and clean up resources."""
        if self.web_scraper:
            await self.web_scraper.close()
            logger.info("SourceManager: WebScraper closed")

        # Clear cache to free memory
        self._scrape_cache.clear()
        logger.info("SourceManager: Resources cleaned up")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

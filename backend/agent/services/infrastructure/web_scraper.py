from __future__ import annotations

# agent/services/web_scraper.py
import asyncio
import logging
import re
from datetime import datetime

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.extraction_strategy import NoExtractionStrategy

from app.exceptions import AgentError

logger = logging.getLogger(__name__)


class WebScraper:
    """
    A simple web scraper that uses AsyncWebCrawler to scrape URLs.
    Enhanced with concurrent scraping limits to prevent resource exhaustion.
    """

    def __init__(self, max_concurrent_scrapes: int = 3):
        self.browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            extra_args=["--no-sandbox", "--disable-gpu",
                        "--disable-logging", "--silent"]
        )
        self.crawler = AsyncWebCrawler(
            config=self.browser_config,
            verbose=False
        )
        self._initialized = False
        # Semaphore to limit concurrent scraping operations
        self.semaphore = asyncio.Semaphore(max_concurrent_scrapes)
        self.max_concurrent_scrapes = max_concurrent_scrapes

    def _extract_publication_date(self, html_content: str) -> str | None:
        """
        Extract publication date from HTML content using various strategies.

        Args:
            html_content: Raw HTML content

        Returns:
            ISO formatted date string or None if not found
        """
        if not html_content:
            return None

        # Common meta tag patterns for publication dates
        meta_patterns = [
            # Open Graph
            r'<meta\s+property=["\']article:published_time["\'][^>]*content=["\']([^"\']+)["\']',
            r'<meta\s+property=["\']og:published_time["\'][^>]*content=["\']([^"\']+)["\']',
            # Twitter Cards
            r'<meta\s+name=["\']twitter:published_time["\'][^>]*content=["\']([^"\']+)["\']',
            # Standard meta tags
            r'<meta\s+name=["\']date["\'][^>]*content=["\']([^"\']+)["\']',
            r'<meta\s+name=["\']publish_date["\'][^>]*content=["\']([^"\']+)["\']',
            r'<meta\s+name=["\']publication_date["\'][^>]*content=["\']([^"\']+)["\']',
            r'<meta\s+name=["\']pubdate["\'][^>]*content=["\']([^"\']+)["\']',
            r'<meta\s+name=["\']created["\'][^>]*content=["\']([^"\']+)["\']',
            # Schema.org structured data
            r'"datePublished"\s*:\s*"([^"]+)"',
            r'"dateCreated"\s*:\s*"([^"]+)"',
            # Time tags
            r'<time[^>]*datetime=["\']([^"\']+)["\']',
            r'<time[^>]*pubdate[^>]*datetime=["\']([^"\']+)["\']',
        ]

        for pattern in meta_patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            if matches:
                date_str = matches[0].strip()
                # Try to parse and validate the date
                parsed_date = self._parse_date_string(date_str)
                if parsed_date:
                    return parsed_date

        # Fallback: Look for common date patterns in text
        text_date_patterns = [
            # ISO format variations
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)',
            r'(\d{4}-\d{2}-\d{2})',
            # Common formats
            r'Published:?\s*(\d{1,2}/\d{1,2}/\d{4})',
            r'Date:?\s*(\d{1,2}/\d{1,2}/\d{4})',
            r'(\w+\s+\d{1,2},?\s+\d{4})',  # "January 15, 2024"
        ]

        for pattern in text_date_patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            if matches:
                date_str = matches[0].strip()
                parsed_date = self._parse_date_string(date_str)
                if parsed_date:
                    return parsed_date

        return None

    def _parse_date_string(self, date_str: str) -> str | None:
        """
        Parse various date string formats and return ISO format.

        Args:
            date_str: Date string to parse

        Returns:
            ISO formatted date string or None if parsing fails
        """
        if not date_str:
            return None

        # Common date formats to try
        date_formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y",
        ]

        for fmt in date_formats:
            try:
                parsed = datetime.strptime(date_str, fmt)
                return parsed.isoformat()
            except ValueError:
                continue

        # Try to handle timezone-aware strings
        try:
            # Remove common timezone abbreviations and try again
            cleaned = re.sub(
                r'\s+(UTC|GMT|EST|PST|CST|MST|EDT|PDT|CDT|MDT)\s*$', '', date_str, flags=re.IGNORECASE)
            for fmt in date_formats:
                try:
                    parsed = datetime.strptime(cleaned, fmt)
                    return parsed.isoformat()
                except ValueError:
                    continue
        except Exception:
            pass

        logger.debug("Could not parse date string: %s", date_str)
        return None

    async def _ensure_initialized(self):
        if not self._initialized:
            logger.info("AsyncWebCrawler will be initialized on first use.")
            # In recent versions, explicit init() is not needed.
            # The crawler initializes automatically on the first arun() call.
            self._initialized = True

    async def scrape_urls(self, urls: list[str]) -> list[dict]:
        """
        Scrape a list of URLs with controlled concurrency to prevent resource exhaustion.

        Args:
            urls: List of URLs to scrape

        Returns:
            List of scraping results
        """
        await self._ensure_initialized()

        logger.info(
            "Starting scraping of %d URLs with max %d concurrent operations",
            len(urls), self.max_concurrent_scrapes
        )

        # Create semaphore-controlled tasks
        tasks = [self._scrape_single_url_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred during scraping
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_message = f"Exception during scraping for {urls[i]}: {result}"
                logger.error(error_message)
                processed_results.append(
                    {
                        "url": urls[i],
                        "content": None,
                        "status": "error",
                        "content_length": 0,
                        "publication_date": None,
                        "error_message": str(result),
                    }
                )
            else:
                processed_results.append(result)

        successful_scrapes = len(
            [r for r in processed_results if r.get("status") == "success"])
        logger.info(
            "Completed scraping: %d/%d successful", successful_scrapes, len(urls))

        return processed_results

    async def _scrape_single_url_with_semaphore(self, url: str) -> dict:
        """
        Scrape a single URL with semaphore control to limit concurrent operations.
        """
        async with self.semaphore:
            return await self._scrape_single_url(url)

    async def _scrape_single_url(self, url: str) -> dict:
        """
        Scrape a single URL with a simple configuration and extract publication date.
        """
        try:
            run_config = CrawlerRunConfig(
                extraction_strategy=NoExtractionStrategy(),
                page_timeout=30000,
                verbose=False
            )
            result = await self.crawler.arun(url, config=run_config)

            if result and result.success and result.markdown.raw_markdown:
                # Extract publication date from HTML
                publication_date = None
                if hasattr(result, 'html') and result.html:
                    publication_date = self._extract_publication_date(
                        result.html)
                    logger.debug(
                        "Extracted publication date for %s: %s", url, publication_date)
                elif hasattr(result, 'cleaned_html') and result.cleaned_html:
                    publication_date = self._extract_publication_date(
                        result.cleaned_html)
                    logger.debug(
                        "Extracted publication date from cleaned HTML for %s: %s", url, publication_date)

                return {
                    "url": url,
                    "content": result.markdown.raw_markdown,
                    "status": "success",
                    "content_length": len(result.markdown.raw_markdown),
                    "publication_date": publication_date,
                    "error_message": None,
                }
            else:
                error_message = f"Scraping failed for {url}: No content or success=False."
                logger.error(error_message)
                return {
                    "url": url,
                    "content": None,
                    "status": "error",
                    "content_length": 0,
                    "publication_date": None,
                    "error_message": error_message,
                }
        except Exception as e:
            error_message = f"Exception during scraping for {url}: {e}"
            logger.error(error_message, exc_info=True)
            raise AgentError(f"Web scraping failed for {url}: {str(e)}") from e

    async def close(self):
        """
        Close the crawler and release resources.
        """
        if self._initialized:
            await self.crawler.close()
            self._initialized = False
            logger.info("WebScraper resources released.")

    async def __aenter__(self):
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

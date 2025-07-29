# agent/services/web_scraper.py
import asyncio
import logging

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.extraction_strategy import NoExtractionStrategy

logger = logging.getLogger(__name__)


class WebScraper:
    """
    A simple web scraper that uses AsyncWebCrawler to scrape URLs.
    Enhanced with concurrent scraping limits to prevent resource exhaustion.
    """

    def __init__(self, max_concurrent_scrapes: int = 3):
        self.browser_config = BrowserConfig(
            headless=True, verbose=False, extra_args=["--no-sandbox", "--disable-gpu"]
        )
        self.crawler = AsyncWebCrawler(config=self.browser_config)
        self._initialized = False
        # Semaphore to limit concurrent scraping operations
        self.semaphore = asyncio.Semaphore(max_concurrent_scrapes)
        self.max_concurrent_scrapes = max_concurrent_scrapes

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
            f"Starting scraping of {len(urls)} URLs with max {self.max_concurrent_scrapes} concurrent operations"
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
                        "error_message": str(result),
                    }
                )
            else:
                processed_results.append(result)

        successful_scrapes = len(
            [r for r in processed_results if r.get("status") == "success"]
        )
        logger.info(
            f"Completed scraping: {successful_scrapes}/{len(urls)} successful")

        return processed_results

    async def _scrape_single_url_with_semaphore(self, url: str) -> dict:
        """
        Scrape a single URL with semaphore control to limit concurrent operations.
        """
        async with self.semaphore:
            return await self._scrape_single_url(url)

    async def _scrape_single_url(self, url: str) -> dict:
        """
        Scrape a single URL with a simple configuration.
        """
        try:
            run_config = CrawlerRunConfig(
                extraction_strategy=NoExtractionStrategy(),
                page_timeout=30000,
            )
            result = await self.crawler.arun(url, config=run_config)

            if result and result.success and result.markdown.raw_markdown:
                return {
                    "url": url,
                    "content": result.markdown.raw_markdown,
                    "status": "success",
                    "content_length": len(result.markdown.raw_markdown),
                    "error_message": None,
                }
            else:
                error_message = (
                    f"Scraping failed for {url}: No content or success=False."
                )
                logger.error(error_message)
                return {
                    "url": url,
                    "content": None,
                    "status": "error",
                    "content_length": 0,
                    "error_message": error_message,
                }
        except Exception as e:
            error_message = f"Exception during scraping for {url}: {e}"
            logger.error(error_message, exc_info=True)
            return {
                "url": url,
                "content": None,
                "status": "error",
                "content_length": 0,
                "error_message": str(e),
            }

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
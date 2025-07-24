# agent/services/web_scraper.py
import asyncio
import logging
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.extraction_strategy import NoExtractionStrategy


logger = logging.getLogger(__name__)

class WebScraper:
    """
    A simple web scraper that uses AsyncWebCrawler to scrape URLs.
    """

    def __init__(self):
        self.browser_config = BrowserConfig(
            headless=True,
            verbose=True,
            extra_args=['--no-sandbox', '--disable-gpu']
        )
        self.crawler = AsyncWebCrawler(config=self.browser_config)
        self._initialized = False

    async def _ensure_initialized(self):
        if not self._initialized:
            logger.info("AsyncWebCrawler will be initialized on first use.")
            # In recent versions, explicit init() is not needed.
            # The crawler initializes automatically on the first arun() call.
            self._initialized = True

    async def scrape_urls(self, urls: list[str]) -> list[dict]:
        """
        Scrape a list of URLs concurrently.
        """
        await self._ensure_initialized()
        tasks = [self._scrape_single_url(url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

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
                    'url': url,
                    'content': result.markdown.raw_markdown,
                    'status': 'success',
                    'content_length': len(result.markdown.raw_markdown),
                    'error_message': None
                }
            else:
                error_message = f"Scraping failed for {url}: No content or success=False."
                logger.error(error_message)
                return {
                    'url': url,
                    'content': None,
                    'status': 'error',
                    'content_length': 0,
                    'error_message': error_message
                }
        except Exception as e:
            error_message = f"Exception during scraping for {url}: {e}"
            logger.error(error_message, exc_info=True)
            return {
                'url': url,
                'content': None,
                'status': 'error',
                'content_length': 0,
                'error_message': str(e)
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

import asyncio
import logging
from crawl4ai import AsyncWebCrawler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    """
    A service to scrape web pages using Crawl4AI.
    """
    def __init__(self):
        self.crawler = AsyncWebCrawler()

    async def scrape_urls(self, urls: list[str]) -> list[dict]:
        """
        Scrapes a list of URLs concurrently and returns their content.

        Args:
            urls: A list of URLs to scrape.

        Returns:
            A list of dictionaries, where each dictionary contains the URL,
            the scraped content in Markdown, and a status.
        """
        if not urls:
            return []

        logger.info(f"Starting to scrape {len(urls)} URLs.")

        async def scrape_single_url(url: str) -> dict:
            """Helper function to scrape a single URL and handle errors."""
            try:
                logger.info(f"Scraping URL: {url}")
                # The crawler is configured for single-page scraping by default
                result = await self.crawler.arun(url, output_format="markdown")
                
                if result and result.success and result.markdown and result.markdown.fit_markdown:
                    logger.info(f"Successfully scraped content from {url}.")
                    return {"url": url, "content": result.markdown.fit_markdown, "status": "success"}
                elif result and not result.success:
                    logger.error(f"Crawling error for URL {url}: {result.error_message}")
                    return {"url": url, "content": None, "status": "error", "error_message": result.error_message}
                else:
                    logger.warning(f"No meaningful content found for URL: {url}")
                    return {"url": url, "content": None, "status": "no_content"}
            except Exception as e:
                logger.error(f"An unexpected error occurred while scraping {url}: {e}")
                return {"url": url, "content": None, "status": "error", "error_message": str(e)}

        tasks = [scrape_single_url(url) for url in urls]
        results = await asyncio.gather(*tasks)
        logger.info("Finished scraping all URLs.")
        return results 
import asyncio
import logging
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    """
    A service to scrape web pages with enhanced crawl4ai configuration.
    """
    def __init__(self):
        # Enhanced browser configuration
        self.browser_config = BrowserConfig(
            headless=True,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport_width=1920,
            viewport_height=1080,
            accept_downloads=False,
            java_script_enabled=True,
            ignore_https_errors=True,
            extra_args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
                "--disable-extensions",
                "--disable-gpu"
            ]
        )
        
        # JavaScript code for dynamic content loading
        self.js_code = """
        // Wait for page to be fully loaded
        await new Promise(resolve => {
            if (document.readyState === 'complete') {
                resolve();
            } else {
                window.addEventListener('load', resolve);
            }
        });
        
        // Additional wait for dynamic content
        await new Promise(resolve => setTimeout(resolve, 3000));
        
        // Scroll to trigger lazy loading
        window.scrollTo(0, document.body.scrollHeight);
        await new Promise(resolve => setTimeout(resolve, 1000));
        window.scrollTo(0, 0);
        """
        
        # Primary crawler configuration
        self.primary_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            word_count_threshold=10,
            exclude_external_links=True,
            remove_overlay_elements=True,
            process_iframes=True,
            wait_for="css:body",  # More reliable than networkidle
            magic=True,
            js_code=self.js_code
        )
        
        # Fallback configuration for difficult sites
        self.fallback_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            word_count_threshold=5,
            wait_for="css:body",  # Simple wait for body element
            magic=True
        )
        
        self.crawler = AsyncWebCrawler(config=self.browser_config)

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
            """
            Scrape a single URL with enhanced error handling and fallback strategies.
            """
            try:
                logger.info(f"Starting to scrape URL: {url}")
                
                # Try primary configuration first
                result = await self.crawler.arun(url, config=self.primary_config)
                
                # Detailed diagnostics
                if result:
                    logger.info(f"Crawl result for {url}:")
                    logger.info(f"  - Success: {result.success}")
                    logger.info(f"  - Status code: {getattr(result, 'status_code', 'N/A')}")
                    logger.info(f"  - Has markdown: {hasattr(result, 'markdown') and result.markdown is not None}")
                    
                    # Check if we have usable content
                    content = None
                    if result.success and hasattr(result, 'markdown') and result.markdown:
                        # In crawl4ai 0.7.0, markdown is a StringCompatibleMarkdown object
                        markdown_content = str(result.markdown).strip()
                        
                        # Check if we have meaningful content (not just error pages)
                        if len(markdown_content) > 50 and not ("404" in markdown_content and "not be found" in markdown_content):
                            content = markdown_content
                            logger.info(f"Successfully extracted content for {url} (length: {len(content)})")
                        else:
                            logger.warning(f"Content too short or appears to be error page for {url}: {markdown_content[:100]}...")
                        
                        if content:
                            return {"url": url, "content": content, "status": "success"}
                    
                    # If primary config failed, try fallback
                    logger.warning(f"Primary config failed for {url}, trying fallback configuration")
                    fallback_result = await self.crawler.arun(url, config=self.fallback_config)
                    
                    if fallback_result and fallback_result.success and hasattr(fallback_result, 'markdown') and fallback_result.markdown:
                        fallback_content = str(fallback_result.markdown).strip()
                        
                        # Check if fallback content is meaningful
                        if len(fallback_content) > 30 and not ("404" in fallback_content and "not be found" in fallback_content):
                            logger.info(f"Fallback configuration succeeded for {url} (length: {len(fallback_content)})")
                            return {"url": url, "content": fallback_content, "status": "success"}
                    
                    # If both configs failed, provide detailed error info
                    error_msg = getattr(result, 'error', 'Unknown error')
                    logger.error(f"Both primary and fallback configs failed for {url}: {error_msg}")
                    return {
                        "url": url, 
                        "content": None, 
                        "status": "no_content",
                        "error_message": f"No usable content extracted. Error: {error_msg}"
                    }
                else:
                    logger.error(f"No result returned for {url}")
                    return {
                        "url": url, 
                        "content": None, 
                        "status": "error", 
                        "error_message": "No result returned from crawler"
                    }
                    
            except Exception as e:
                logger.error(f"Exception while scraping {url}: {str(e)}")
                return {
                    "url": url, 
                    "content": None, 
                    "status": "error", 
                    "error_message": f"Exception: {str(e)}"
                }

        tasks = [scrape_single_url(url) for url in urls]
        results = await asyncio.gather(*tasks)
        logger.info("Finished scraping all URLs.")
        return results

    async def close(self):
        """
        Close the crawler and clean up resources.
        """
        try:
            if self.crawler:
                # In crawl4ai 0.7.0, use close() instead of aclose()
                if hasattr(self.crawler, 'close'):
                    await self.crawler.close()
                elif hasattr(self.crawler, 'aclose'):
                    await self.crawler.aclose()
                logger.info("WebScraper resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error closing WebScraper: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        try:
            if self.crawler:
                # In crawl4ai 0.7.0, use close() instead of aclose()
                if hasattr(self.crawler, 'close'):
                    await self.crawler.close()
                elif hasattr(self.crawler, 'aclose'):
                    await self.crawler.aclose()
                logger.info("WebScraper closed successfully")
        except Exception as e:
            logger.error(f"Error closing WebScraper: {e}")
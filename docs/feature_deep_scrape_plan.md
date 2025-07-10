
# Comprehensive Plan: Deep Scraping Integration with Crawl4AI

## 1. Objective

To fundamentally enhance the fact-checking pipeline by integrating a deep scraping capability. Instead of relying solely on brief search engine snippets, the system will fetch and analyze the full content of the most relevant source web pages. This will provide the LLM with a much richer, more reliable context, leading to significantly more accurate and well-substantiated verdicts.

## 2. Core Strategy: Adopt Crawl4AI

We will use `Crawl4AI`, a modern, open-source web crawler designed specifically for AI applications. This choice is deliberate and strategic.

- **AI-First Design**: `Crawl4AI` is built to convert unstructured web content into clean, LLM-friendly formats like Markdown, which is perfect for our analysis pipeline.
- **Dynamic Content Handling**: It leverages a browser engine (`Playwright`) to render JavaScript, ensuring we can scrape modern, dynamic websites where simple HTTP requests would fail.
- **Scalability**: While our initial implementation will be for targeted scraping, `Crawl4AI` provides a solid foundation for future, more extensive crawling tasks.

## 3. Prerequisite: Dependency Management and Environment Setup

This is a critical first step. Failure to manage dependencies correctly will prevent the application from starting.

1.  **Modify `requirements.txt`**: Add `crawl4ai` and `playwright` to the `backend/requirements.txt` file. Explicitly defining both dependencies is crucial for a stable build.
    ```
    # backend/requirements.txt
    ...
    crawl4ai==0.5.1
    playwright==1.40.0 # Or a version compatible with Crawl4AI
    ...
    ```
2.  **Update `Dockerfile`**: The Playwright browsers must be installed within our Docker image. This is a **mandatory** step for the scraper to function.
    - Add the command `RUN playwright install --with-deps` to `backend/Dockerfile` after the `pip install` step. This ensures the necessary browser binaries and their system dependencies are present.
    ```Dockerfile
    # backend/Dockerfile
    ...
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Install Playwright browsers and dependencies
    RUN playwright install --with-deps
    ...
    ```
3.  **Rebuild Docker Container**: After modifying the files, rebuild the Docker container to apply the changes.
    - Run the command `docker-compose build backend`.

## 4. Phase 1: Implement a Dedicated Web Scraping Service

To ensure modularity and separation of concerns, all scraping logic will be encapsulated in a new, dedicated service.

1.  **File Creation**: Create a new file: `backend/agent/services/web_scraper.py`.
2.  **Class Definition**: Inside the new file, define a `WebScraper` class.
3.  **Scraping Method**: Implement a primary asynchronous method, `scrape_urls(urls: list[str]) -> list[dict]`.
    - This method will take a list of URLs and process them concurrently.
    - It will use `crawl4ai.AsyncWebCrawler`.
    - **Configuration**: The crawler will be configured for single-page scraping (not deep crawling) and to return content in Markdown format.
    - **Error Handling**: The implementation will include robust error handling for network timeouts, HTTP errors, and cases where a page returns no meaningful content. Each result dictionary will contain the URL, the scraped content, and a status (e.g., 'success', 'error').
    - **Logging**: Add comprehensive logging for each URL being scraped.

## 5. Phase 2: Intelligent Source Selection via LLM

This is the intelligent core of our pipeline. We will leverage an LLM to identify the most credible and relevant sources.

1.  **Target File**: `backend/agent/fact_checkers/financial_checker.py`.
2.  **New LLM-Powered Selection Method**: Implement a new private method, e.g., `_select_credible_sources(search_results, claim_context)`.
    - **Input**: This method takes the full list of search results and the context of the claim being checked (including the `primary_thesis`).
    - **LLM Prompt**: It will construct a specialized prompt for an LLM to act as an expert analyst and select the top 2-3 most authoritative sources.
    - **Output & Reliability**: The method will parse the LLM's JSON response. **In adherence with the fail-fast principle, if the LLM response is not a valid JSON or is empty, the method will raise an exception.** This prevents the pipeline from proceeding with low-quality or incomplete data. No fallback to simple heuristics will be implemented.

## 6. Phase 3: Targeted Scraping and Final Verdict Generation

This phase combines the vetted sources with scraping to generate the final, high-quality verdict.

1.  **Integration**: In the `check_claim` method of `financial_checker.py`, call `_select_credible_sources`. Then, pass the returned list of vetted URLs to our `web_scraper.scrape_urls()` service.
2.  **Final Prompt Construction**: The final prompt for verdict generation will be built exclusively from the full, clean Markdown content of the successfully scraped, LLM-vetted pages.
3.  **Final Verdict**: Send the enriched prompt to the LLM to get the final verdict.

## 7. Phase 4: Legal and Attribution

To comply with open-source licensing, we must provide attribution.

1.  **Update `README.md`**: Add a "Powered by" notice to the project's main `README.md` file, acknowledging the use of `Crawl4AI`.

## 8. Phase 5: Testing and Validation

A rigorous testing strategy is non-negotiable and must be updated to cover the new logic.

1.  **Unit Tests**:
    - Create `backend/tests/unit/services/test_web_scraper.py` to test the `WebScraper`.
    - In `backend/tests/unit/fact_checkers/test_financial_checker.py`, add a test for `_select_credible_sources` to ensure it correctly parses a valid LLM response and, crucially, **raises an exception for invalid responses**.
2.  **Integration Tests**:
    - Update `backend/tests/integration/fact_checkers/test_financial_checker.py`.
    - The test will mock the necessary calls (LLM for source selection, web scraper, LLM for verdict) to verify the entire pipeline is wired together correctly and handles both success and failure scenarios as designed.
3.  **End-to-End (E2E) Test**:
    - Perform manual E2E tests, including one where the LLM for source selection is forced to fail (e.g., by returning malformed JSON) to confirm that the entire operation stops gracefully with a clear error, as per the fail-fast design. 
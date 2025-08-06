"""
Custom tools for the LangChain agent.
"""

from __future__ import annotations

import json
import logging

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from agent.models.search_models import (
    SearchError,
    SearchResponse,
    SearchResult,
    SearchResultWrapper,
)
from app.config import settings
from app.exceptions import ToolError

logger = logging.getLogger(__name__)


class SearxNGSearchInput(BaseModel):
    """Input schema for SearxNG search tool."""

    query: str = Field(description="Search query to execute")
    category: str | None = Field(default="general", description="Search category (general, news, science, etc.)")
    engines: str | None = Field(default=None, description="Specific search engines to use")
    language: str | None = Field(default="en", description="Search language")
    max_results: int | None = Field(default=10, description="Maximum number of results to return (default: 10)")


class SearxNGSearchTool(BaseTool):
    """Tool for searching the web using SearxNG."""

    name: str = "searxng_search"
    description: str = """
    Search the web for information using SearxNG search engine.
    Use this tool to find current information, verify facts, and gather evidence.
    Provide a clear, specific search query for best results.
    """
    args_schema: type = SearxNGSearchInput

    def _run(
        self,
        query: str,
        category: str = "general",
        engines: str = None,
        language: str = "en",
        max_results: int = 10,
    ) -> str:
        """Execute the search synchronously."""
        try:
            # Prepare search parameters
            params = {
                "q": query,
                "format": "json",
                "category": category,
                "language": language,
            }

            if engines:
                params["engines"] = engines

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            }

            # Make request to SearxNG using httpx
            with httpx.Client(timeout=30.0) as client:
                response = client.get(f"{settings.searxng_url}/search", params=params, headers=headers)
                response.raise_for_status()

                data = response.json()
                results = data.get("results", [])

                if not results:
                    # Return error wrapper for no results
                    error_wrapper = SearchResultWrapper(
                        success=False,
                        error=SearchError(
                            error_type="NO_RESULTS",
                            message=f"No search results found for query: {query}",
                            query=query,
                        ),
                    )
                    return error_wrapper.model_dump_json(indent=2)

                # Convert to Pydantic models
                search_results = []
                for result in results[:max_results]:  # Use configurable limit
                    try:
                        search_result = SearchResult(
                            title=result.get("title", ""),
                            url=result.get("url", ""),
                            content=result.get("content", ""),
                            engine=result.get("engine", ""),
                        )
                        search_results.append(search_result)
                    except (ValueError, TypeError, KeyError) as e:
                        logger.warning("Failed to parse search result: %s", e)
                        continue

                # Create structured response
                search_response = SearchResponse(
                    query=query,
                    results=search_results,
                    total_results=len(search_results),
                    search_info={
                        "category": category,
                        "language": language,
                        "engines": engines,
                    },
                )

                result_wrapper = SearchResultWrapper(success=True, data=search_response)

                logger.info(
                    "SearxNG search completed: %d results for '%s'",
                    len(search_results),
                    query,
                )
                return result_wrapper.model_dump_json(indent=2)

        except Exception as e:
            error_msg = f"SearxNG search failed: {e}"
            logger.error(error_msg)
            raise ToolError(error_msg) from e

    async def _arun(
        self,
        query: str,
        category: str = "general",
        engines: str = None,
        language: str = "en",
        max_results: int = 10,
    ) -> str:
        """Execute the search asynchronously."""
        try:
            # Prepare search parameters
            params = {
                "q": query,
                "format": "json",
                "category": category,
                "language": language,
            }

            if engines:
                params["engines"] = engines

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            }

            # Make async request to SearxNG
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{settings.searxng_url}/search", params=params, headers=headers)
                response.raise_for_status()

                data = response.json()
                results = data.get("results", [])

                if not results:
                    # Return error wrapper for no results
                    error_wrapper = SearchResultWrapper(
                        success=False,
                        error=SearchError(
                            error_type="NO_RESULTS",
                            message=f"No search results found for query: {query}",
                            query=query,
                        ),
                    )
                    return error_wrapper.model_dump_json(indent=2)

                # Convert to Pydantic models
                search_results = []
                for result in results[:max_results]:  # Use configurable limit
                    try:
                        search_result = SearchResult(
                            title=result.get("title", ""),
                            url=result.get("url", ""),
                            content=result.get("content", ""),
                            engine=result.get("engine", ""),
                        )
                        search_results.append(search_result)
                    except Exception as e:
                        logger.warning("Failed to parse search result: %s", e)
                        continue

                # Create structured response
                search_response = SearchResponse(
                    query=query,
                    results=search_results,
                    total_results=len(search_results),
                    search_info={
                        "category": category,
                        "language": language,
                        "engines": engines,
                    },
                )

                result_wrapper = SearchResultWrapper(success=True, data=search_response)

                logger.info(
                    "SearxNG search completed: %d results for '%s'",
                    len(search_results),
                    query,
                )
                return result_wrapper.model_dump_json(indent=2)

        except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError) as e:
            error_msg = f"SearxNG search failed: {e}"
            logger.error(error_msg)
            raise ToolError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error in SearxNG search: {e}"
            logger.error(error_msg)
            raise ToolError(error_msg) from e


# Initialize tools
searxng_tool = SearxNGSearchTool()

# Export tools list
AVAILABLE_TOOLS = [
    searxng_tool,
]

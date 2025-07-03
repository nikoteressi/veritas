"""
Custom tools for the LangChain agent.
"""
import json
import logging
from typing import Optional

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)


class SearxNGSearchInput(BaseModel):
    """Input schema for SearxNG search tool."""
    query: str = Field(description="Search query to execute")
    category: Optional[str] = Field(default="general", description="Search category (general, news, science, etc.)")
    engines: Optional[str] = Field(default=None, description="Specific search engines to use")
    language: Optional[str] = Field(default="en", description="Search language")


class SearxNGSearchTool(BaseTool):
    """Tool for searching the web using SearxNG."""

    name: str = "searxng_search"
    description: str = """
    Search the web for information using SearxNG search engine.
    Use this tool to find current information, verify facts, and gather evidence.
    Provide a clear, specific search query for best results.
    """
    args_schema: type = SearxNGSearchInput
    
    def _run(self, query: str, category: str = "general", engines: str = None, language: str = "en") -> str:
        """Execute the search synchronously."""
        try:
            # Prepare search parameters
            params = {
                "q": query,
                "format": "json",
                "category": category,
                "language": language
            }
            
            if engines:
                params["engines"] = engines
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            }
            
            # Make request to SearxNG
            response = requests.get(
                f"{settings.searxng_url}/search",
                params=params,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            if not results:
                return f"No search results found for query: {query}"
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results[:5]):  # Limit to top 5 results
                formatted_result = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "engine": result.get("engine", "")
                }
                formatted_results.append(formatted_result)
            
            logger.info(f"SearxNG search completed: {len(formatted_results)} results for '{query}'")
            return json.dumps(formatted_results, indent=2)
            
        except Exception as e:
            error_msg = f"SearxNG search failed: {e}"
            logger.error(error_msg)
            return error_msg
    
    async def _arun(self, query: str, category: str = "general", engines: str = None, language: str = "en") -> str:
        """Execute the search asynchronously."""
        # For now, use the sync version
        # In production, you might want to use aiohttp for async requests
        return self._run(query, category, engines, language)


# Initialize tools
searxng_tool = SearxNGSearchTool()

# Export tools list
AVAILABLE_TOOLS = [
    searxng_tool,
]

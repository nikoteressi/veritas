"""
Custom tools for the LangChain agent.
"""
import json
import logging
from typing import Dict, Any, Optional, List

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import AsyncSessionLocal
from app.crud import UserCRUD

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
            
            # Make request to SearxNG
            response = requests.get(
                f"{settings.searxng_url}/search",
                params=params,
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


class DatabaseReputationInput(BaseModel):
    """Input schema for database reputation tool."""
    nickname: str = Field(description="User nickname to look up or update")
    action: str = Field(description="Action to perform: 'get' or 'update'")
    verdict: Optional[str] = Field(default=None, description="Verdict for update: true, partially_true, false, ironic")


class DatabaseReputationTool(BaseTool):
    """Tool for interacting with user reputation database."""

    name: str = "database_reputation"
    description: str = """
    Get or update user reputation information in the database.
    Use 'get' action to retrieve current reputation data.
    Use 'update' action to update reputation after verification (requires verdict).
    """
    args_schema: type = DatabaseReputationInput
    
    def _run(self, nickname: str, action: str, verdict: str = None) -> str:
        """Execute database operation synchronously."""
        # This is a simplified sync version
        # In practice, you'd want to handle this differently
        return f"Database operation: {action} for {nickname} (verdict: {verdict})"
    
    async def _arun(self, nickname: str, action: str, verdict: str = None) -> str:
        """Execute database operation asynchronously."""
        try:
            async with AsyncSessionLocal() as db:
                if action == "get":
                    user = await UserCRUD.get_user_by_nickname(db, nickname)
                    if user:
                        reputation_data = {
                            "nickname": user.nickname,
                            "true_count": user.true_count,
                            "partially_true_count": user.partially_true_count,
                            "false_count": user.false_count,
                            "ironic_count": user.ironic_count,
                            "total_posts_checked": user.total_posts_checked,
                            "warning_issued": user.warning_issued,
                            "notification_issued": user.notification_issued
                        }
                        logger.info(f"Retrieved reputation for {nickname}")
                        return json.dumps(reputation_data, indent=2)
                    else:
                        return f"No reputation data found for user: {nickname}"
                
                elif action == "update" and verdict:
                    user = await UserCRUD.update_user_reputation(db, nickname, verdict)
                    result = {
                        "nickname": user.nickname,
                        "verdict": verdict,
                        "new_total": user.total_posts_checked,
                        "warning_issued": user.warning_issued,
                        "notification_issued": user.notification_issued
                    }
                    logger.info(f"Updated reputation for {nickname}: {verdict}")
                    return json.dumps(result, indent=2)
                
                else:
                    return f"Invalid action or missing verdict: {action}"
                    
        except Exception as e:
            error_msg = f"Database operation failed: {e}"
            logger.error(error_msg)
            return error_msg


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
            
            # Make request to SearxNG
            response = requests.get(
                f"{settings.searxng_url}/search",
                params=params,
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
database_tool = DatabaseReputationTool()

# Export tools list
AVAILABLE_TOOLS = [
    searxng_tool,
    database_tool
]

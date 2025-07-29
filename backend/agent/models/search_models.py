"""
from __future__ import annotations

Pydantic models for search results and related data structures.
"""

from typing import Any

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Model for individual search result from SearxNG."""

    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the search result")
    content: str = Field(description="Content/snippet of the search result")
    engine: str = Field(description="Search engine that provided this result")
    score: float | None = Field(default=None, description="Relevance score if available")

    def validate_url(self, v):
        """Ensure URL is not empty."""
        if not v or not v.strip():
            raise ValueError("URL cannot be empty")
        return v.strip()


class SearchResponse(BaseModel):
    """Model for complete search response from SearxNG."""

    query: str = Field(description="Original search query")
    results: list[SearchResult] = Field(description="List of search results")
    total_results: int = Field(description="Total number of results found")
    search_info: dict[str, Any] = Field(default_factory=dict, description="Additional search metadata")

    def validate_results(self, v):
        """Ensure results list is valid."""
        return v or []

    def get_urls(self) -> list[str]:
        """Extract all URLs from search results."""
        return [result.url for result in self.results if result.url]

    def get_top_results(self, limit: int = 5) -> list[SearchResult]:
        """Get top N results."""
        return self.results[:limit]


class SearchError(BaseModel):
    """Model for search error responses."""

    error_type: str = Field(description="Type of error")
    message: str = Field(description="Error message")
    query: str | None = Field(default=None, description="Original query that caused the error")


class SearchResultWrapper(BaseModel):
    """Wrapper for search results that can handle both success and error cases."""

    success: bool = Field(description="Whether the search was successful")
    data: SearchResponse | None = Field(default=None, description="Search response data if successful")
    error: SearchError | None = Field(default=None, description="Error information if failed")

    def validate_data_error_consistency(self, v, values):
        """Ensure data and error are consistent with success flag."""
        success = values.get("success", False)
        if success and not values.get("data"):
            raise ValueError("Success=True requires data to be present")
        if not success and not values.get("error"):
            raise ValueError("Success=False requires error to be present")
        return v

    def get_urls(self) -> list[str]:
        """Extract URLs from successful search results."""
        if self.success and self.data:
            return self.data.get_urls
        return []

    def get_results(self) -> list[SearchResult]:
        """Get search results if available."""
        if self.success and self.data:
            return self.data.results
        return []

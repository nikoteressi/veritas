"""
Pydantic models for extracted information.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExtractedInfo(BaseModel):
    """Extracted information (replaces extracted_info)."""

    username: str | None = Field(None, description="Username")
    post_date: str | None = Field(None, description="Post date")
    mentioned_dates: list[str] = Field(default_factory=list, description="Mentioned dates")
    extracted_text: str | None = Field(None, description="Extracted text")

    # Additional metadata
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

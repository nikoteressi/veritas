"""
Pydantic models for extracted information.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ExtractedInfo(BaseModel):
    """Extracted information (replaces extracted_info)."""
    username: Optional[str] = Field(None, description="Username")
    post_date: Optional[str] = Field(None, description="Post date")
    mentioned_dates: List[str] = Field(default_factory=list, description="Mentioned dates")
    extracted_text: Optional[str] = Field(None, description="Extracted text")
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
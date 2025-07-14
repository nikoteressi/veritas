"""
Pydantic models for hierarchical fact structures.
"""
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class Fact(BaseModel):
    """Represents a single, atomic, verifiable fact that supports the primary thesis."""
    description: str = Field(description="A clear, concise statement of the fact for verification.")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured data extracted for this fact (e.g., amounts, dates, entities) to aid in targeted verification."
    )


class FactHierarchy(BaseModel):
    """Represents the structured, hierarchical understanding of the claims made in the source."""
    primary_thesis: str = Field(description="The single, overarching claim or main point of the source. This summarizes the entire message.")
    supporting_facts: List[Fact] = Field(description="A list of atomic, verifiable facts that support the primary thesis.") 
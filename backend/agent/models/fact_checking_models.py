from __future__ import annotations

from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    """Represents a single search query with its type."""

    type: str = Field(
        ...,
        description="The category of the search query (e.g., 'Direct Verification', 'Contextual Analysis').",
    )
    query: str = Field(..., description="The search query string.")


class QueryGenerationReasoning(BaseModel):
    """Captures the reasoning steps of the LLM during query generation."""

    claim_analysis: dict = Field(
        ...,
        description="Step-by-step analysis of the claim's components (entities, actions, etc.).",
    )
    domain_determination: str = Field(..., description="The determined domain for the claim (e.g., Finance, Science).")
    query_formulation_plan: str = Field(..., description="The plan or reasoning for formulating the list of queries.")


class QueryGenerationOutput(BaseModel):
    """
    A structured output model for the query generation process.
    The LLM is expected to populate this model and return it as a single JSON object.
    """

    reasoning: QueryGenerationReasoning = Field(
        ..., description="The detailed reasoning process behind the query generation."
    )
    queries: list[SearchQuery] = Field(..., description="The list of generated search queries for fact-checking.")

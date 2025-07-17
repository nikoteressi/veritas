from pydantic import BaseModel, Field
from .fact import FactHierarchy


class PostAnalysisResult(BaseModel):
    fact_hierarchy: FactHierarchy = Field(
        ..., description="The hierarchy of facts extracted from the post.")
    primary_topic: str = Field(
        ..., description="The primary topic or domain of the post (e.g., financial, political).")

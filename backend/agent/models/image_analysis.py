"""
Pydantic model for structured image analysis results.
"""
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from .fact import FactHierarchy


class ImageAnalysisResult(BaseModel):
    """Pydantic model for structured image analysis results."""
    username: str = Field(
        description="The COMPLETE username of the person who posted the content. Extract the full username without truncation. Should be 'unknown' if not identifiable."
    )
    post_date: Optional[str] = Field(
        description="The timestamp of the post, e.g., 'May 21, 2024' or '15 hours ago' or '2 days ago'. Extract exactly as seen in the image. Return null if not available."
    )
    mentioned_dates: List[str] = Field(
        default_factory=list,
        description="A list of any other dates or time references found in the text."
    )
    extracted_text: str = Field(
        description="A verbatim transcription of all visible text from the image."
    )
    fact_hierarchy: FactHierarchy = Field(
        description="A structured representation of the claims and their relationships."
    )
    primary_topic: str = Field(
        default="general",
        description="The primary topic, chosen from: financial, medical, political, scientific, technology, entertainment, general, humorous/ironic."
    )
    irony_assessment: str = Field(
        default="not_ironic",
        description="Assessment of irony, chosen from: not_ironic, potentially_ironic, clearly_ironic."
    )
    visual_elements_summary: str = Field(
        description="A summary of key visual elements like charts, graphs, or UI components. If no visual elements are found, return 'No visual elements found'."
    )
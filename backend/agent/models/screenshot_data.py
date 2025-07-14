"""
Pydantic models for structured data extracted from social media screenshots.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class PostContent(BaseModel):
    author: Optional[str] = Field(None, description="The username of the post's author.")
    timestamp: Optional[str] = Field(None, description="The relative or absolute timestamp of the post (e.g., '15h ago', 'July 10').")
    text_body: Optional[str] = Field(None, description="The full, verbatim text of the main post.")
    hashtags: List[str] = Field(default_factory=list, description="A list of all hashtags found in the post text, without the '#' symbol.")


class PostVisuals(BaseModel):
    description: Optional[str] = Field(None, description="A neutral, factual description of any image or video content within the post.")
    text_identified_in_visuals: List[str] = Field(default_factory=list, description="A list of any text found embedded directly within the image/video itself.")


class EngagementStats(BaseModel):
    likes: Optional[str] = Field(None, description="The number of likes/hearts/reactions.")
    comments: Optional[str] = Field(None, description="The number of comments.")
    shares: Optional[str] = Field(None, description="The number of shares/reposts/retweets.")
    other: Optional[str] = Field(None, description="Any other visible numeric metric.")


class VisibleComment(BaseModel):
    author: Optional[str] = Field(None, description="The username of the commenter.")
    timestamp: Optional[str] = Field(None, description="The timestamp of the comment, if visible.")
    text: Optional[str] = Field(None, description="The verbatim text of the comment, including emojis.")


class PostInteractions(BaseModel):
    engagement_stats: EngagementStats = Field(default_factory=EngagementStats)
    visible_comments: List[VisibleComment] = Field(default_factory=list)


class ScreenshotData(BaseModel):
    """
    Root model for data extracted from a social media post screenshot.
    """
    post_content: PostContent = Field(default_factory=PostContent)
    post_visuals: PostVisuals = Field(default_factory=PostVisuals)
    post_interactions: PostInteractions = Field(default_factory=PostInteractions) 
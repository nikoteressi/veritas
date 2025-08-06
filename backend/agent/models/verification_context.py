"""
Verification context model for the verification pipeline.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from agent.models.fact import FactHierarchy
from agent.models.graph import FactGraph
from agent.models.post_analysis_result import PostAnalysisResult
from agent.services.infrastructure.event_emission import EventEmissionService
from agent.services.output.result_compiler import ResultCompiler

from .extracted_info import ExtractedInfo
from .image_analysis import ImageAnalysisResult
from .internal import FactCheckResult, VerdictResult
from .motives_analysis import MotivesAnalysisResult
from .screenshot_data import ScreenshotData
from .summarization_result import SummarizationResult
from .temporal_analysis import TemporalAnalysisResult


class VerificationContext(BaseModel):
    """
    Verification context.

    This model holds all data that flows through the verification pipeline,
    providing type safety and validation for all steps.
    """

    # Input data
    image_bytes: bytes = Field(..., description="Image content to analyze")
    user_prompt: str = Field(..., description="User's question/prompt")
    session_id: str = Field(..., description="Session identifier")
    filename: str | None = Field(
        None, description="Optional filename for display")

    # Database and services (excluded from serialization)
    db: AsyncSession | None = Field(
        None, exclude=True, description="Database session")
    event_service: EventEmissionService | None = Field(
        None, exclude=True, description="Event emission service")
    result_compiler: ResultCompiler | None = Field(
        None, exclude=True, description="Result compiler service")
    progress_manager: Any | None = Field(
        None, exclude=True, description="Progress manager service")

    # Step-specific data
    validated_data: dict[str, Any] | None = None
    screenshot_data: ScreenshotData | None = None
    primary_topic: str | None = Field(
        None,
        description="The primary topic or domain of the post (e.g., financial, political).",
    )
    fact_hierarchy: FactHierarchy | None = None
    fact_graph: FactGraph | None = Field(
        None, description="Graph representation of facts and their relationships")
    summary: str | None = None
    claims: list[str] = Field(
        default_factory=list,
        description="A simple list of claims, for components that need it. Will be derived from fact_hierarchy.",
    )
    user_reputation: Any | None = None
    updated_reputation: Any | None = Field(
        None, description="Reputation object after update")
    warnings: list[str] = Field(
        default_factory=list,
        description="A list of warnings generated during the verification process.",
    )
    additional_context: str | None = Field(
        None, description="Additional context for verification processes")

    # Typed analysis results
    temporal_analysis_result: TemporalAnalysisResult | None = Field(
        None, description="Temporal analysis")
    motives_analysis_result: MotivesAnalysisResult | None = Field(
        None, description="Motives analysis")
    extracted_info_typed: ExtractedInfo | None = Field(
        None, description="Extracted information")

    # Analysis results
    analysis_result: ImageAnalysisResult | None = Field(
        None, description="[DEPRECATED] Old image analysis result")
    post_analysis_result: PostAnalysisResult | None = Field(
        None, description="Post analysis result")

    # Fact checking results
    fact_check_result: FactCheckResult | None = Field(
        None, description="Fact checking result")

    # Summarization result
    summarization_result: SummarizationResult | None = Field(
        None, description="Summarization result")

    # Final verdict
    verdict_result: VerdictResult | None = Field(
        None, description="Final verdict result")

    # Storage results
    verification_record: Any | None = Field(
        None, description="Verification record from database")
    verification_id: str | None = Field(
        None, description="ID of the verification record")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        json_encoders = {
            bytes: lambda v: v.decode("utf-8", errors="ignore"),
            datetime: lambda v: v.isoformat(),
        }

    # Methods for working with typed fields
    def set_temporal_analysis(self, result: TemporalAnalysisResult) -> None:
        """Set typed temporal analysis result."""
        self.temporal_analysis_result = result

    def get_temporal_analysis(self) -> TemporalAnalysisResult | None:
        """Get typed temporal analysis result."""
        return self.temporal_analysis_result

    def set_motives_analysis(self, result: MotivesAnalysisResult) -> None:
        """Set typed motives analysis result."""
        self.motives_analysis_result = result

    def get_motives_analysis(self) -> MotivesAnalysisResult | None:
        """Get typed motives analysis result."""
        return self.motives_analysis_result

    def set_extracted_info(self, info: ExtractedInfo) -> None:
        """Set typed extracted info."""
        self.extracted_info_typed = info

    def get_extracted_info(self) -> ExtractedInfo | None:
        """Get typed extracted info."""
        return self.extracted_info_typed

    def set_summarization_result(self, result: SummarizationResult) -> None:
        """Set typed summarization result."""
        self.summarization_result = result

    def get_summarization_result(self) -> SummarizationResult | None:
        """Get typed summarization result."""
        return self.summarization_result

    def set_fact_hierarchy(self, fact_hierarchy: FactHierarchy):
        """Sets the fact hierarchy and derives the simple claims list."""
        self.fact_hierarchy = fact_hierarchy
        if fact_hierarchy and fact_hierarchy.supporting_facts:
            self.claims = [
                fact.description for fact in fact_hierarchy.supporting_facts]
        else:
            self.claims = []

    def set_fact_graph(self, fact_graph: FactGraph) -> None:
        """Set the fact graph."""
        self.fact_graph = fact_graph

    def get_fact_graph(self) -> FactGraph | None:
        """Get the fact graph."""
        return self.fact_graph

    def get_event_service(self) -> EventEmissionService | None:
        """Get the event service."""
        return self.event_service

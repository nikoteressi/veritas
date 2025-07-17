"""
Verification context model for the verification pipeline.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from .image_analysis import ImageAnalysisResult
from .screenshot_data import ScreenshotData
from .internal import FactCheckResult, VerdictResult
from .temporal_analysis import TemporalAnalysisResult
from .motives_analysis import MotivesAnalysisResult
from .extracted_info import ExtractedInfo
from .summarization_result import SummarizationResult
from agent.models.fact import Fact, FactHierarchy
from agent.services.event_emission import EventEmissionService
from agent.services.result_compiler import ResultCompiler
from agent.models.post_analysis_result import PostAnalysisResult


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
    filename: Optional[str] = Field(None, description="Optional filename for display")
    
    # Database and services (excluded from serialization)
    db: Optional[AsyncSession] = Field(None, exclude=True, description="Database session")
    event_service: Optional[EventEmissionService] = Field(None, exclude=True, description="Event emission service")
    result_compiler: Optional[ResultCompiler] = Field(None, exclude=True, description="Result compiler service")
    
    # Step-specific data
    validated_data: Optional[Dict[str, Any]] = None
    screenshot_data: Optional[ScreenshotData] = None
    primary_topic: Optional[str] = Field(None, description="The primary topic or domain of the post (e.g., financial, political).")
    fact_hierarchy: Optional[FactHierarchy] = None
    summary: Optional[str] = None
    claims: List[str] = Field(default_factory=list, description="A simple list of claims, for components that need it. Will be derived from fact_hierarchy.")
    user_reputation: Optional[Any] = None
    updated_reputation: Optional[Any] = Field(None, description="Reputation object after update")
    warnings: List[str] = Field(default_factory=list, description="A list of warnings generated during the verification process.")
    
    # Typed analysis results
    temporal_analysis_result: Optional[TemporalAnalysisResult] = Field(None, description="Temporal analysis")
    motives_analysis_result: Optional[MotivesAnalysisResult] = Field(None, description="Motives analysis")
    extracted_info_typed: Optional[ExtractedInfo] = Field(None, description="Extracted information")
    
    # Analysis results
    analysis_result: Optional[ImageAnalysisResult] = Field(None, description="[DEPRECATED] Old image analysis result")
    post_analysis_result: PostAnalysisResult = Field(None, description="Post analysis result")

    # Fact checking results
    fact_check_result: Optional[FactCheckResult] = Field(None, description="Fact checking result")
    
    # Summarization result
    summarization_result: Optional[SummarizationResult] = Field(None, description="Summarization result")
    
    # Final verdict
    verdict_result: Optional[VerdictResult] = Field(None, description="Final verdict result")
    
    # Storage results
    verification_record: Optional[Any] = Field(None, description="Verification record from database")
    verification_id: Optional[str] = Field(None, description="ID of the verification record")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_encoders = {
            bytes: lambda v: v.decode('utf-8', errors='ignore'),
            datetime: lambda v: v.isoformat(),
        }
    
    # Methods for working with typed fields
    def set_temporal_analysis(self, result: TemporalAnalysisResult) -> None:
        """Set typed temporal analysis result."""
        self.temporal_analysis_result = result
    
    def get_temporal_analysis(self) -> Optional[TemporalAnalysisResult]:
        """Get typed temporal analysis result."""
        return self.temporal_analysis_result
    
    def set_motives_analysis(self, result: MotivesAnalysisResult) -> None:
        """Set typed motives analysis result."""
        self.motives_analysis_result = result
    
    def get_motives_analysis(self) -> Optional[MotivesAnalysisResult]:
        """Get typed motives analysis result."""
        return self.motives_analysis_result
    
    def set_extracted_info(self, info: ExtractedInfo) -> None:
        """Set typed extracted info."""
        self.extracted_info_typed = info
    
    def get_extracted_info(self) -> Optional[ExtractedInfo]:
        """Get typed extracted info."""
        return self.extracted_info_typed
    
    def set_summarization_result(self, result: SummarizationResult) -> None:
        """Set typed summarization result."""
        self.summarization_result = result
    
    def get_summarization_result(self) -> Optional[SummarizationResult]:
        """Get typed summarization result."""
        return self.summarization_result
    
    def set_fact_hierarchy(self, fact_hierarchy: FactHierarchy):
        """Sets the fact hierarchy and derives the simple claims list."""
        self.fact_hierarchy = fact_hierarchy
        if fact_hierarchy and fact_hierarchy.supporting_facts:
            self.claims = [fact.description for fact in fact_hierarchy.supporting_facts]
        else:
            self.claims = []
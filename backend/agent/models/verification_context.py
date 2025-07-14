"""
Verification context model for the verification pipeline.
"""
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from .image_analysis import ImageAnalysisResult
from .screenshot_data import ScreenshotData
from .internal import FactCheckResult, VerdictResult
from agent.models.fact import Fact, FactHierarchy
from agent.services.event_emission import EventEmissionService
from agent.services.result_compiler import ResultCompiler


class VerificationContext(BaseModel):
    """
    Context model for the verification pipeline.
    
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
    fact_hierarchy: Optional[FactHierarchy] = None
    claims: List[str] = Field(default_factory=list, description="A simple list of claims, for components that need it. Will be derived from fact_hierarchy.")
    user_reputation: Optional[Any] = None
    updated_reputation: Optional[Any] = Field(None, description="Reputation object after update")
    temporal_analysis: Optional[Dict[str, Any]] = None
    motives_analysis: Optional[Dict[str, Any]] = None
    
    # Extracted information - maintained for compatibility
    extracted_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Generic dictionary for extracted info, used for compatibility.")
    
    # Analysis results
    analysis_result: Optional[ImageAnalysisResult] = Field(None, description="[DEPRECATED] Old image analysis result")

    # Fact checking results
    fact_check_result: Optional[FactCheckResult] = Field(None, description="Fact checking result")
    
    # Final verdict
    verdict_result: Optional[VerdictResult] = Field(None, description="Final verdict result")
    
    # Storage results
    verification_record: Optional[Any] = Field(None, description="Verification record from database")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_encoders = {
            bytes: lambda v: v.decode('utf-8', errors='ignore'),
            datetime: lambda v: v.isoformat(),
        }
    
    def get_extracted_info(self) -> Dict[str, Any]:
        """Get the generic extracted info dictionary."""
        return self.extracted_info or {}

    def set_extracted_info(self, key: str, value: Any):
        """Set a value in the generic extracted info dictionary."""
        if self.extracted_info is None:
            self.extracted_info = {}
        self.extracted_info[key] = value
    
    def set_temporal_analysis(self, analysis: Dict[str, Any]) -> None:
        """Set temporal analysis in both direct field and extracted_info."""
        self.temporal_analysis = analysis
        extracted_info = self.get_extracted_info()
        extracted_info["temporal_analysis"] = analysis
        self.extracted_info = extracted_info
    
    def set_motives_analysis(self, analysis: Dict[str, Any]) -> None:
        """Set motives analysis in both direct field and extracted_info."""
        self.motives_analysis = analysis
        extracted_info = self.get_extracted_info()
        extracted_info["motives_analysis"] = analysis
        self.extracted_info = extracted_info
    
    def set_fact_hierarchy(self, fact_hierarchy: FactHierarchy):
        """Sets the fact hierarchy and derives the simple claims list."""
        self.fact_hierarchy = fact_hierarchy
        if fact_hierarchy and fact_hierarchy.supporting_facts:
            self.claims = [fact.description for fact in fact_hierarchy.supporting_facts]
        else:
            self.claims = []

    def get_temporal_analysis(self) -> Dict[str, Any]:
        """Get the temporal analysis result."""
        return self.temporal_analysis or {}
    
    def get_motives_analysis(self) -> Dict[str, Any]:
        """Get motives analysis from either direct field or extracted_info."""
        if self.motives_analysis is not None:
            return self.motives_analysis
        return self.get_extracted_info().get("motives_analysis", {})
    
    def update_extracted_info(self, key: str, value: Any) -> None:
        """Update a specific key in extracted_info."""
        extracted_info = self.get_extracted_info()
        extracted_info[key] = value
        self.extracted_info = extracted_info 
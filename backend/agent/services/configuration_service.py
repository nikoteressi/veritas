"""
Configuration service for centralized application settings.
"""
import os
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass


class VerificationSteps(Enum):
    """Enumeration of verification pipeline steps."""
    VALIDATION = "validation"
    IMAGE_ANALYSIS = "image_analysis"
    REPUTATION_RETRIEVAL = "reputation_retrieval"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    MOTIVES_ANALYSIS = "motives_analysis"
    FACT_CHECKING = "fact_checking"
    VERDICT_GENERATION = "verdict_generation"
    REPUTATION_UPDATE = "reputation_update"
    RESULT_STORAGE = "result_storage"


class VerdictTypes(Enum):
    """Enumeration of possible verdict types."""
    TRUE = "true"
    PARTIALLY_TRUE = "partially_true"
    FALSE = "false"
    IRONIC = "ironic"
    INCONCLUSIVE = "inconclusive"


@dataclass
class ValidationConfig:
    """Configuration for validation rules."""
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    min_file_size: int = 100  # 100 bytes
    allowed_image_types: list = None
    min_prompt_length: int = 10
    max_prompt_length: int = 2000
    
    def __post_init__(self):
        if self.allowed_image_types is None:
            self.allowed_image_types = [
                'image/jpeg', 'image/jpg', 'image/png', 
                'image/gif', 'image/webp'
            ]


@dataclass
class ProcessingConfig:
    """Configuration for processing settings."""
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 300  # 5 minutes
    fact_checking_timeout_seconds: int = 120  # 2 minutes
    max_search_queries: int = 5
    max_claims_per_request: int = 10
    
    # Vector store settings
    vector_store_collection_name: str = "veritas_verification_results"
    vector_store_embedding_dimension: int = 1536
    
    # LLM settings
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4000


@dataclass
class ReputationConfig:
    """Configuration for reputation system."""
    initial_reputation_score: float = 0.0
    warning_threshold: float = -10.0
    notification_threshold: float = -20.0
    
    # Score adjustments
    true_verdict_score: float = 1.0
    partially_true_verdict_score: float = 0.5
    false_verdict_score: float = -2.0
    ironic_verdict_score: float = -1.0


@dataclass
class SecurityConfig:
    """Configuration for security settings."""
    suspicious_patterns: list = None
    max_session_duration_minutes: int = 60
    rate_limit_requests_per_minute: int = 30
    
    def __post_init__(self):
        if self.suspicious_patterns is None:
            self.suspicious_patterns = [
                'javascript:', '<script', 'eval(', 'exec(',
                'import os', 'subprocess', '__import__'
            ]


class ConfigurationService:
    """
    Centralized configuration service for the application.
    
    This service provides access to all configuration settings
    used throughout the application, ensuring consistency and
    easy maintenance.
    """
    
    def __init__(self):
        self._validation_config = ValidationConfig()
        self._processing_config = ProcessingConfig()
        self._reputation_config = ReputationConfig()
        self._security_config = SecurityConfig()
        
        # Load environment-specific overrides
        self._load_from_environment()
    
    @property
    def validation(self) -> ValidationConfig:
        """Get validation configuration."""
        return self._validation_config
    
    @property
    def processing(self) -> ProcessingConfig:
        """Get processing configuration."""
        return self._processing_config
    
    @property
    def reputation(self) -> ReputationConfig:
        """Get reputation configuration."""
        return self._reputation_config
    
    @property
    def security(self) -> SecurityConfig:
        """Get security configuration."""
        return self._security_config
    
    def get_pipeline_steps(self) -> list:
        """Get the ordered list of pipeline steps."""
        return [step.value for step in VerificationSteps]
    
    def get_verdict_types(self) -> list:
        """Get the list of possible verdict types."""
        return [verdict.value for verdict in VerdictTypes]
    
    def is_valid_verdict(self, verdict: str) -> bool:
        """Check if a verdict is valid."""
        return verdict in [v.value for v in VerdictTypes]
    
    def get_step_display_name(self, step: str) -> str:
        """Get display name for a pipeline step."""
        display_names = {
            VerificationSteps.VALIDATION.value: "Validation",
            VerificationSteps.IMAGE_ANALYSIS.value: "Image Analysis",
            VerificationSteps.REPUTATION_RETRIEVAL.value: "Reputation Retrieval",
            VerificationSteps.TEMPORAL_ANALYSIS.value: "Temporal Analysis",
            VerificationSteps.MOTIVES_ANALYSIS.value: "Motives Analysis",
            VerificationSteps.FACT_CHECKING.value: "Fact Checking",
            VerificationSteps.VERDICT_GENERATION.value: "Verdict Generation",
            VerificationSteps.REPUTATION_UPDATE.value: "Reputation Update",
            VerificationSteps.RESULT_STORAGE.value: "Result Storage"
        }
        return display_names.get(step, step.replace('_', ' ').title())
    
    def get_progress_message(self, step: str) -> str:
        """Get progress message for a pipeline step."""
        progress_messages = {
            VerificationSteps.VALIDATION.value: "Validating request...",
            VerificationSteps.IMAGE_ANALYSIS.value: "Analyzing image content...",
            VerificationSteps.REPUTATION_RETRIEVAL.value: "Retrieving user reputation...",
            VerificationSteps.TEMPORAL_ANALYSIS.value: "Analyzing temporal context...",
            VerificationSteps.MOTIVES_ANALYSIS.value: "Analyzing user motives...",
            VerificationSteps.FACT_CHECKING.value: "Fact-checking claims...",
            VerificationSteps.VERDICT_GENERATION.value: "Generating verdict...",
            VerificationSteps.REPUTATION_UPDATE.value: "Updating reputation...",
            VerificationSteps.RESULT_STORAGE.value: "Saving results..."
        }
        return progress_messages.get(step, f"Processing {step}...")
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as a dictionary."""
        return {
            'validation': self._validation_config.__dict__,
            'processing': self._processing_config.__dict__,
            'reputation': self._reputation_config.__dict__,
            'security': self._security_config.__dict__
        }
    
    def _load_from_environment(self):
        """Load configuration overrides from environment variables."""
        # Validation config
        if os.getenv('VERITAS_MAX_FILE_SIZE'):
            self._validation_config.max_file_size = int(os.getenv('VERITAS_MAX_FILE_SIZE'))
        
        if os.getenv('VERITAS_MIN_PROMPT_LENGTH'):
            self._validation_config.min_prompt_length = int(os.getenv('VERITAS_MIN_PROMPT_LENGTH'))
        
        if os.getenv('VERITAS_MAX_PROMPT_LENGTH'):
            self._validation_config.max_prompt_length = int(os.getenv('VERITAS_MAX_PROMPT_LENGTH'))
        
        # Processing config
        if os.getenv('VERITAS_MAX_CONCURRENT_REQUESTS'):
            self._processing_config.max_concurrent_requests = int(os.getenv('VERITAS_MAX_CONCURRENT_REQUESTS'))
        
        if os.getenv('VERITAS_REQUEST_TIMEOUT'):
            self._processing_config.request_timeout_seconds = int(os.getenv('VERITAS_REQUEST_TIMEOUT'))
        
        if os.getenv('VERITAS_FACT_CHECK_TIMEOUT'):
            self._processing_config.fact_checking_timeout_seconds = int(os.getenv('VERITAS_FACT_CHECK_TIMEOUT'))
        
        # Reputation config
        if os.getenv('VERITAS_WARNING_THRESHOLD'):
            self._reputation_config.warning_threshold = float(os.getenv('VERITAS_WARNING_THRESHOLD'))
        
        if os.getenv('VERITAS_NOTIFICATION_THRESHOLD'):
            self._reputation_config.notification_threshold = float(os.getenv('VERITAS_NOTIFICATION_THRESHOLD'))
        
        # Security config
        if os.getenv('VERITAS_RATE_LIMIT_RPM'):
            self._security_config.rate_limit_requests_per_minute = int(os.getenv('VERITAS_RATE_LIMIT_RPM'))
        
        if os.getenv('VERITAS_MAX_SESSION_DURATION'):
            self._security_config.max_session_duration_minutes = int(os.getenv('VERITAS_MAX_SESSION_DURATION'))
    
    def validate_configuration(self) -> Optional[str]:
        """Validate configuration for consistency and correctness."""
        # Check file size limits
        if self._validation_config.max_file_size <= self._validation_config.min_file_size:
            return "Max file size must be greater than min file size"
        
        # Check prompt length limits
        if self._validation_config.max_prompt_length <= self._validation_config.min_prompt_length:
            return "Max prompt length must be greater than min prompt length"
        
        # Check reputation thresholds
        if self._reputation_config.warning_threshold >= self._reputation_config.notification_threshold:
            return "Warning threshold must be less than notification threshold"
        
        # Check processing limits
        if self._processing_config.max_concurrent_requests <= 0:
            return "Max concurrent requests must be greater than 0"
        
        # Check timeouts
        if self._processing_config.request_timeout_seconds <= 0:
            return "Request timeout must be greater than 0"
        
        return None  # Configuration is valid


# Singleton instance
configuration_service = ConfigurationService() 
"""
Configuration settings for the Veritas application.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings


class VerificationSteps(Enum):
    """Enumeration of verification pipeline steps."""

    VALIDATION = "validation"
    SCREENSHOT_PARSING = "screenshot_parsing"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    POST_ANALYSIS = "post_analysis"
    REPUTATION_RETRIEVAL = "reputation_retrieval"
    FACT_CHECKING = "fact_checking"
    SUMMARIZATION = "summarization"
    MOTIVES_ANALYSIS = "motives_analysis"
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


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database settings
    db_host: str = Field(default="localhost", env="DB_HOST")
    db_port: int = Field(default=5432, env="DB_PORT")
    db_name: str = Field(default="veritas_db", env="DB_NAME")
    db_user: str = Field(default="veritas_user", env="DB_USER")
    db_password: str = Field(default="", env="DB_PASSWORD")

    # Ollama settings
    ollama_base_url: str = Field(
        default="http://localhost:11434", env="OLLAMA_BASE_URL"
    )
    vision_model_name: str = Field(
        default="llava:latest", env="VISION_MODEL_NAME")
    reasoning_model_name: str = Field(
        default="qwen:7b", env="REASONING_MODEL_NAME")
    embedding_model_name: str = Field(
        default="nomic-embed-text:latest", env="EMBEDDING_MODEL_NAME"
    )

    # Web scraping settings
    use_remote_llm_for_extraction: bool = Field(
        default=True, env="USE_REMOTE_LLM_FOR_EXTRACTION"
    )
    scraping_extraction_model: str = Field(
        default="cyberuser42/DeepSeek-R1-Distill-Llama-8B:latest",
        env="SCRAPING_EXTRACTION_MODEL",
    )
    page_timeout: int = Field(
        default=60000, env="PAGE_TIMEOUT")  # 60 seconds in ms
    delay_before_return_html: int = Field(
        default=5000, env="DELAY_BEFORE_RETURN_HTML"
    )  # 5 seconds in ms

    # SearxNG settings
    searxng_url: str = Field(
        default="http://localhost:8888", env="SEARXNG_URL")

    # Application settings
    app_host: str = Field(default="0.0.0.0", env="APP_HOST")
    app_port: int = Field(default=8000, env="APP_PORT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    logging_config_file: str = Field(
        default="logging.conf", env="LOGGING_CONFIG_FILE")

    # Security settings
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173", env="CORS_ORIGINS"
    )

    # ChromaDB settings
    chroma_persist_directory: str = Field(
        default="./data/chroma_db", env="CHROMA_PERSIST_DIRECTORY"
    )
    chroma_host: str = Field(default="localhost", env="CHROMA_HOST")
    chroma_port: int = Field(default=8002, env="CHROMA_PORT")

    # Redis configuration
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")

    # Neo4j configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(
        default="veritas_password", env="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", env="NEO4J_DATABASE")

    # Validation settings
    max_file_size: int = Field(
        default=10 * 1024 * 1024, env="VERITAS_MAX_FILE_SIZE"
    )  # 10MB
    min_file_size: int = Field(
        default=100, env="VERITAS_MIN_FILE_SIZE")  # 100 bytes
    allowed_image_types: list[str] = Field(
        default=["image/jpeg", "image/jpg",
                 "image/png", "image/gif", "image/webp"],
        env="VERITAS_ALLOWED_IMAGE_TYPES",
    )
    min_prompt_length: int = Field(default=10, env="VERITAS_MIN_PROMPT_LENGTH")
    max_prompt_length: int = Field(
        default=2000, env="VERITAS_MAX_PROMPT_LENGTH")

    # Processing settings
    max_concurrent_requests: int = Field(
        default=10, env="VERITAS_MAX_CONCURRENT_REQUESTS"
    )
    request_timeout_seconds: int = Field(
        default=300, env="VERITAS_REQUEST_TIMEOUT"
    )  # 5 minutes
    fact_checking_timeout_seconds: int = Field(
        default=120, env="VERITAS_FACT_CHECK_TIMEOUT"
    )  # 2 minutes
    max_search_queries: int = Field(
        default=5, env="VERITAS_MAX_SEARCH_QUERIES")
    max_claims_per_request: int = Field(
        default=10, env="VERITAS_MAX_CLAIMS_PER_REQUEST"
    )

    # Vector store settings
    vector_store_collection_name: str = Field(
        default="veritas_verification_results", env="VERITAS_VECTOR_COLLECTION"
    )
    vector_store_embedding_dimension: int = Field(
        default=1536, env="VERITAS_EMBEDDING_DIMENSION"
    )

    # LLM settings
    llm_temperature: float = Field(default=0.1, env="VERITAS_LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=4000, env="VERITAS_LLM_MAX_TOKENS")

    # Reputation system settings
    initial_reputation_score: float = Field(
        default=0.0, env="VERITAS_INITIAL_REPUTATION"
    )
    warning_threshold: float = Field(
        default=-20.0, env="VERITAS_WARNING_THRESHOLD")
    notification_threshold: float = Field(
        default=-10.0, env="VERITAS_NOTIFICATION_THRESHOLD"
    )

    # Reputation score adjustments
    true_verdict_score: float = Field(
        default=1.0, env="VERITAS_TRUE_VERDICT_SCORE")
    partially_true_verdict_score: float = Field(
        default=0.5, env="VERITAS_PARTIALLY_TRUE_VERDICT_SCORE"
    )
    false_verdict_score: float = Field(
        default=-2.0, env="VERITAS_FALSE_VERDICT_SCORE")
    ironic_verdict_score: float = Field(
        default=-1.0, env="VERITAS_IRONIC_VERDICT_SCORE"
    )

    # Security settings
    suspicious_patterns: list[str] = Field(
        default=[
            "javascript:",
            "<script",
            "eval(",
            "exec(",
            "import os",
            "subprocess",
            "__import__",
        ],
        env="VERITAS_SUSPICIOUS_PATTERNS",
    )
    max_session_duration_minutes: int = Field(
        default=60, env="VERITAS_MAX_SESSION_DURATION"
    )
    rate_limit_requests_per_minute: int = Field(
        default=30, env="VERITAS_RATE_LIMIT_RPM"
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        if isinstance(self.cors_origins, str):
            return [
                origin.strip()
                for origin in str(self.cors_origins).split(",")
                if origin.strip()
            ]
        return []

    @property
    def database_url(self) -> str:
        """Construct the database URL from individual components."""
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    # Configuration helper methods
    def get_pipeline_steps(self) -> list[str]:
        """Get the ordered list of pipeline steps."""
        return [step.value for step in VerificationSteps]

    def get_verdict_types(self) -> list[str]:
        """Get the list of possible verdict types."""
        return [verdict.value for verdict in VerdictTypes]

    def is_valid_verdict(self, verdict: str) -> bool:
        """Check if a verdict is valid."""
        return verdict in [v.value for v in VerdictTypes]

    def get_step_display_name(self, step: str) -> str:
        """Get display name for a pipeline step."""
        display_names = {
            VerificationSteps.VALIDATION.value: "Validation",
            VerificationSteps.SCREENSHOT_PARSING.value: "Image Analysis",
            VerificationSteps.REPUTATION_RETRIEVAL.value: "Reputation Retrieval",
            VerificationSteps.TEMPORAL_ANALYSIS.value: "Temporal Analysis",
            VerificationSteps.FACT_CHECKING.value: "Fact Checking",
            VerificationSteps.VERDICT_GENERATION.value: "Verdict Generation",
            VerificationSteps.MOTIVES_ANALYSIS.value: "Motives Analysis",
            VerificationSteps.REPUTATION_UPDATE.value: "Reputation Update",
            VerificationSteps.RESULT_STORAGE.value: "Result Storage",
        }
        return display_names.get(step, step.replace("_", " ").title())

    def get_progress_message(self, step: str) -> str:
        """Get progress message for a pipeline step."""
        progress_messages = {
            VerificationSteps.VALIDATION.value: "Validating request...",
            VerificationSteps.SCREENSHOT_PARSING.value: "Analyzing image content...",
            VerificationSteps.REPUTATION_RETRIEVAL.value: "Retrieving user reputation...",
            VerificationSteps.TEMPORAL_ANALYSIS.value: "Analyzing temporal context...",
            VerificationSteps.FACT_CHECKING.value: "Fact-checking claims...",
            VerificationSteps.VERDICT_GENERATION.value: "Generating verdict...",
            VerificationSteps.MOTIVES_ANALYSIS.value: "Analyzing user motives...",
            VerificationSteps.REPUTATION_UPDATE.value: "Updating reputation...",
            VerificationSteps.RESULT_STORAGE.value: "Saving results...",
        }
        return progress_messages.get(step, f"Processing {step}...")

    def validate_configuration(self) -> str | None:
        """Validate configuration for consistency and correctness."""
        # Check file size limits
        if self.max_file_size <= self.min_file_size:
            return "Max file size must be greater than min file size"

        # Check prompt length limits
        if self.max_prompt_length <= self.min_prompt_length:
            return "Max prompt length must be greater than min prompt length"

        # Check reputation thresholds
        if self.warning_threshold >= self.notification_threshold:
            return "Warning threshold must be less than notification threshold"

        # Check processing limits
        if self.max_concurrent_requests <= 0:
            return "Max concurrent requests must be greater than 0"

        # Check timeouts
        if self.request_timeout_seconds <= 0:
            return "Request timeout must be greater than 0"

        return None  # Configuration is valid

    class Config:
        """Configuration class for Pydantic settings behavior."""

        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

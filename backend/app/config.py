"""
Configuration settings for the Veritas application.
"""
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database settings
    db_host: str = Field(default="localhost", env="DB_HOST")
    db_port: int = Field(default=5432, env="DB_PORT")
    db_name: str = Field(default="veritas_db", env="DB_NAME")
    db_user: str = Field(default="veritas_user", env="DB_USER")
    db_password: str = Field(default="", env="DB_PASSWORD")
    
    # Ollama settings
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llava:latest", env="OLLAMA_MODEL")
    
    # SearxNG settings
    searxng_url: str = Field(default="http://localhost:8888", env="SEARXNG_URL")
    
    # Application settings
    app_host: str = Field(default="0.0.0.0", env="APP_HOST")
    app_port: int = Field(default=8000, env="APP_PORT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Security settings
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        env="CORS_ORIGINS"
    )

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(',') if origin.strip()]

    # ChromaDB settings
    chroma_persist_directory: str = Field(
        default="./data/chroma_db", 
        env="CHROMA_PERSIST_DIRECTORY"
    )
    
    # Reputation system settings
    warning_threshold: int = Field(default=3, env="WARNING_THRESHOLD")
    notification_threshold: int = Field(default=5, env="NOTIFICATION_THRESHOLD")
    
    @property
    def database_url(self) -> str:
        """Construct the database URL from individual components."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

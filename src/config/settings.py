"""
Application configuration management
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import List, Optional
import os
import re

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Environment
    environment: str = Field("development", env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    port: int = Field(8000, env="PORT")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # GitLab Configuration
    gitlab_url: str = Field(..., env="GITLAB_URL")
    gitlab_token: str = Field(..., env="GITLAB_TOKEN")
    gitlab_webhook_secret: Optional[str] = Field(None, env="GITLAB_WEBHOOK_SECRET")
    gitlab_trigger_tag: str = Field("ai-review", env="GITLAB_TRIGGER_TAG")
    
    @field_validator('gitlab_url')
    @classmethod
    def validate_gitlab_url(cls, v: str) -> str:
        """Validate GitLab URL format"""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('GitLab URL must include protocol (http:// or https://)')
        # Remove trailing slash for consistency
        return v.rstrip('/')
        
    @field_validator('gitlab_token')
    @classmethod
    def validate_gitlab_token(cls, v: str) -> str:
        """Validate GitLab token format"""
        if not v or len(v) < 20:
            raise ValueError('GitLab token must be at least 20 characters long')
        return v
    
    # AI Model Configuration
    ai_model: str = Field("openai:gpt-4o", env="AI_MODEL")
    ai_temperature: float = Field(0.3, env="AI_TEMPERATURE")
    ai_max_tokens: int = Field(4000, env="AI_MAX_TOKENS")
    ai_retries: int = Field(3, env="AI_RETRIES")
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_model_name: str = Field("gpt-4o", env="OPENAI_MODEL_NAME")
    
    # Anthropic Configuration
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    anthropic_model_name: str = Field("claude-3-5-sonnet-latest", env="ANTHROPIC_MODEL_NAME")
    
    # Google Configuration
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")
    gemini_model_name: str = Field("gemini-1.5-pro", env="GEMINI_MODEL_NAME")
    
    # Security
    allowed_origins: List[str] = Field(
        default_factory=lambda: [],  # Empty by default, requires explicit configuration
        env="ALLOWED_ORIGINS"
    )
    api_key: Optional[str] = Field(None, env="API_KEY")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(True, env="RATE_LIMIT_ENABLED")
    webhook_rate_limit: str = Field("10/minute", env="WEBHOOK_RATE_LIMIT")
    
    # Request limits
    max_request_size: int = Field(10 * 1024 * 1024, env="MAX_REQUEST_SIZE")  # 10MB default
    
    @field_validator('allowed_origins')
    @classmethod
    def validate_origins(cls, v: List[str]) -> List[str]:
        """Set secure defaults for CORS origins based on environment"""
        if not v:  # If empty list provided
            # In production, default to empty (no CORS) - must be explicitly configured
            # In development, allow localhost
            return ["http://localhost:3000", "http://localhost:8000"] if os.getenv("ENVIRONMENT", "development") != "production" else []
        return v
        
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"
        
    def __repr__(self) -> str:
        """Secure representation that doesn't expose secrets"""
        return f"<{self.__class__.__name__} gitlab_url={self.gitlab_url} environment={self.environment}>"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Create global settings instance
settings = Settings()
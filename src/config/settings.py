"""
Application configuration management
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os

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
        ["*"],
        env="ALLOWED_ORIGINS"
    )
    api_key: Optional[str] = Field(None, env="API_KEY")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Create global settings instance
settings = Settings()
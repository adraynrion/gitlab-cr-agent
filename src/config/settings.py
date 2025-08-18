"""
Application configuration management
"""

import os
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    model_config = {
        "env_file": ".env" if os.getenv("TESTING") != "true" else None,
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    # Environment
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    # GitLab Configuration
    gitlab_url: str = Field(
        default="https://gitlab.example.com" if os.getenv("TESTING") == "true" else "",
        min_length=1,
    )
    gitlab_token: str = Field(
        default="glpat-test-token-1234567890" if os.getenv("TESTING") == "true" else "",
        min_length=1,
    )
    gitlab_webhook_secret: Optional[str] = Field(default=None)
    gitlab_trigger_tag: str = Field(default="ai-review")

    @field_validator("gitlab_url")
    @classmethod
    def validate_gitlab_url(cls, v: str) -> str:
        """Validate GitLab URL format"""
        if not v.startswith(("http://", "https://")):
            raise ValueError("GitLab URL must include protocol (http:// or https://)")
        # Remove trailing slash for consistency
        return v.rstrip("/")

    @field_validator("gitlab_token")
    @classmethod
    def validate_gitlab_token(cls, v: str) -> str:
        """Validate GitLab token format"""
        if not v or len(v) < 20:
            raise ValueError("GitLab token must be at least 20 characters long")
        return v

    # AI Model Configuration
    ai_model: str = Field(default="openai:gpt-4o")
    ai_temperature: float = Field(default=0.3)
    ai_max_tokens: int = Field(default=4000)
    ai_retries: int = Field(default=3)

    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None)
    openai_model_name: str = Field(default="gpt-4o")
    openai_base_url: Optional[str] = Field(default=None)

    # Anthropic Configuration
    anthropic_api_key: Optional[str] = Field(default=None)
    anthropic_model_name: str = Field(default="claude-3-5-sonnet-latest")
    anthropic_base_url: Optional[str] = Field(default=None)

    # Google Configuration
    google_api_key: Optional[str] = Field(default=None)
    gemini_model_name: str = Field(default="gemini-2.5-pro")
    google_base_url: Optional[str] = Field(default=None)

    # Security
    allowed_origins: List[str] = Field(
        default_factory=lambda: []  # Empty by default, requires explicit configuration
    )
    api_key: Optional[str] = Field(default=None)

    # Rate limiting
    rate_limit_enabled: bool = Field(default=True)
    webhook_rate_limit: str = Field(default="10/minute")
    global_rate_limit: str = Field(
        default="100/minute"
    )  # Global rate limit for all requests

    # Request limits
    max_request_size: int = Field(default=10 * 1024 * 1024)  # 10MB default
    max_diff_size: int = Field(default=1 * 1024 * 1024)  # 1MB default for diff content

    # HTTP Client Configuration
    request_timeout: float = Field(default=30.0)  # Request timeout in seconds
    max_connections: int = Field(default=100)  # Maximum concurrent connections
    max_keepalive_connections: int = Field(default=20)  # Maximum keepalive connections
    keepalive_expiry: float = Field(
        default=30.0
    )  # Keepalive connection expiry in seconds

    # Circuit Breaker Configuration
    circuit_breaker_failure_threshold: int = Field(
        default=5
    )  # Number of failures before opening circuit
    circuit_breaker_timeout: int = Field(
        default=60
    )  # Seconds to wait before attempting recovery
    circuit_breaker_expected_exception: str = Field(
        default="httpx.HTTPStatusError,httpx.RequestError"
    )

    # Context7 MCP Configuration (Documentation Validation via Model Context Protocol)
    context7_enabled: bool = Field(default=True)  # Enable Context7 MCP integration

    @field_validator("allowed_origins")
    @classmethod
    def validate_origins(cls, v: List[str]) -> List[str]:
        """Set secure defaults for CORS origins based on environment"""
        if not v:  # If empty list provided
            # In production, default to empty (no CORS) - must be explicitly configured
            # In development, allow localhost
            return (
                ["http://localhost:3000", "http://localhost:8000"]
                if os.getenv("ENVIRONMENT", "development") != "production"
                else []
            )
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"

    def __repr__(self) -> str:
        """Secure representation that doesn't expose secrets"""
        return f"<{self.__class__.__name__} gitlab_url={self.gitlab_url} environment={self.environment}>"


# Create global settings instance (skip during testing)
settings = None if os.getenv("TESTING") == "true" else Settings()


def get_settings() -> Settings:
    """Get settings instance (for testing compatibility)"""
    global settings
    if settings is None:
        settings = Settings()
    return settings

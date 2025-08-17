"""
Tests for src/config/settings.py
"""

import os
from unittest.mock import patch

from src.config.settings import Settings, get_settings


class TestSettings:
    """Test Settings class configuration"""

    def test_settings_creation_defaults(self):
        """Test basic settings creation with defaults"""
        settings = Settings()
        assert settings.environment == "development"
        assert settings.log_level == "INFO"
        assert settings.port == 8000
        assert settings.host == "0.0.0.0"
        assert settings.debug is False

    def test_settings_with_environment_variables(self):
        """Test settings with environment variables"""
        with patch.dict(
            os.environ,
            {
                "ENVIRONMENT": "production",
                "PORT": "9000",
                "LOG_LEVEL": "DEBUG",
                "HOST": "127.0.0.1",
                "DEBUG": "true",
            },
        ):
            settings = Settings()
            assert settings.environment == "production"
            assert settings.port == 9000
            assert settings.log_level == "DEBUG"
            assert settings.host == "127.0.0.1"
            assert settings.debug is True

    def test_settings_production_mode(self):
        """Test settings in production mode"""
        with patch.dict(
            os.environ,
            {"ENVIRONMENT": "production", "DEBUG": "false", "LOG_LEVEL": "WARNING"},
        ):
            settings = Settings()
            assert settings.environment == "production"
            assert settings.debug is False
            assert settings.log_level == "WARNING"

    def test_settings_context7_configuration(self):
        """Test settings with Context7 options enabled"""
        with patch.dict(
            os.environ,
            {
                "CONTEXT7_ENABLED": "true",
                "CONTEXT7_MAX_TOKENS": "10000",
                "CONTEXT7_CACHE_TTL": "7200",
            },
        ):
            settings = Settings()
            assert settings.context7_enabled is True
            assert settings.context7_max_tokens == 10000
            assert settings.context7_cache_ttl == 7200

    def test_settings_rate_limiting_configuration(self):
        """Test settings with rate limiting configuration"""
        with patch.dict(
            os.environ,
            {
                "RATE_LIMIT_ENABLED": "true",
                "WEBHOOK_RATE_LIMIT": "20/minute",
                "MAX_REQUEST_SIZE": "20971520",
                "MAX_DIFF_SIZE": "2097152",
            },
        ):
            settings = Settings()
            assert settings.rate_limit_enabled is True
            assert settings.webhook_rate_limit == "20/minute"
            assert settings.max_request_size == 20971520
            assert settings.max_diff_size == 2097152

    def test_settings_circuit_breaker_configuration(self):
        """Test settings with circuit breaker configuration"""
        with patch.dict(
            os.environ,
            {
                "CIRCUIT_BREAKER_FAILURE_THRESHOLD": "10",
                "CIRCUIT_BREAKER_TIMEOUT": "120",
                "REQUEST_TIMEOUT": "60.0",
                "MAX_CONNECTIONS": "200",
                "MAX_KEEPALIVE_CONNECTIONS": "40",
                "KEEPALIVE_EXPIRY": "60.0",
            },
        ):
            settings = Settings()
            assert settings.circuit_breaker_failure_threshold == 10
            assert settings.circuit_breaker_timeout == 120
            assert settings.request_timeout == 60.0

    def test_settings_gitlab_configuration(self):
        """Test GitLab-specific settings"""
        with patch.dict(
            os.environ,
            {
                "GITLAB_URL": "https://custom-gitlab.com",
                "GITLAB_TOKEN": "custom-token-12345",
                "GITLAB_TRIGGER_TAG": "@custom-review",
            },
        ):
            settings = Settings()
            assert settings.gitlab_url == "https://custom-gitlab.com"
            assert settings.gitlab_token == "custom-token-12345"
            assert settings.gitlab_trigger_tag == "@custom-review"

    def test_settings_ai_model_configuration(self):
        """Test AI model configuration"""
        with patch.dict(
            os.environ,
            {
                "AI_MODEL": "openai:gpt-4-turbo",
                "AI_RETRIES": "5",
                "OPENAI_API_KEY": "test-key-123",
            },
        ):
            settings = Settings()
            assert settings.ai_model == "openai:gpt-4-turbo"
            assert settings.ai_retries == 5

    def test_settings_validation(self):
        """Test settings validation and required attributes"""
        settings = Settings()

        # Test that required attributes exist
        assert hasattr(settings, "gitlab_url")
        assert hasattr(settings, "ai_model")
        assert hasattr(settings, "context7_enabled")
        assert hasattr(settings, "environment")
        assert hasattr(settings, "port")


class TestGetSettings:
    """Test get_settings function"""

    def test_get_settings_caching(self):
        """Test settings caching mechanism"""
        # First call creates settings
        settings1 = get_settings()
        # Second call should return cached instance
        settings2 = get_settings()
        assert settings1 is settings2

    def test_get_settings_returns_settings_instance(self):
        """Test get_settings returns Settings instance"""
        settings = get_settings()
        assert isinstance(settings, Settings)

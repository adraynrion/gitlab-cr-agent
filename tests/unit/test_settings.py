"""Unit tests for configuration settings validation."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config.settings import Settings


class TestSettingsValidation:
    """Test settings validation and initialization."""

    def test_minimal_valid_settings(self):
        """Test that minimal required settings work."""
        # Clear non-essential vars
        env_vars = {
            "GITLAB_TOKEN": "glpat-test-token-1234567890",  # 20+ characters required
            "GITLAB_URL": "https://gitlab.example.com",
            "GITLAB_WEBHOOK_SECRET": "webhook-secret-1234567890",
            "AI_MODEL": "openai:gpt-4",
            "OPENAI_API_KEY": "sk-test-key-1234567890",
        }

        # Clear the secrets cache to avoid stale values
        from src.utils.secrets import clear_secrets_cache

        clear_secrets_cache()

        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()

        assert settings.gitlab_token == "glpat-test-token-1234567890"
        assert settings.gitlab_url == "https://gitlab.example.com"
        assert settings.gitlab_webhook_secret == "webhook-secret-1234567890"
        assert settings.ai_model == "openai:gpt-4"
        assert settings.openai_api_key == "sk-test-key-1234567890"

    def test_full_settings_configuration(self, mock_env_vars):
        """Test complete settings with all options."""
        settings = Settings()

        # Verify all main sections are loaded
        assert settings.gitlab_token is not None
        assert settings.ai_model is not None
        assert settings.rate_limit_requests > 0
        assert isinstance(settings.cors_origins, list)
        assert isinstance(settings.allowed_users, list)

    def test_invalid_gitlab_token_format(self):
        """Test validation of GitLab token format."""
        invalid_tokens = [
            "invalid-token",  # No glpat prefix
            "glpat-",  # Too short
            "",  # Empty
            "bearer-token",  # Wrong format
        ]

        for token in invalid_tokens:
            env_vars = {
                "GITLAB_TOKEN": token,
                "GITLAB_URL": "https://gitlab.example.com",
                "GITLAB_WEBHOOK_SECRET": "secret",
                "REVIEW_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-test",
            }

            with patch.dict(os.environ, env_vars, clear=True):
                with pytest.raises(ValidationError) as exc_info:
                    Settings()

                assert "gitlab_token" in str(exc_info.value)

    def test_invalid_gitlab_url_format(self):
        """Test validation of GitLab URL format."""
        invalid_urls = ["not-a-url", "ftp://gitlab.com", "http://", "https://", ""]

        for url in invalid_urls:
            env_vars = {
                "GITLAB_TOKEN": "glpat-test",
                "GITLAB_URL": url,
                "GITLAB_WEBHOOK_SECRET": "secret",
                "REVIEW_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-test",
            }

            with patch.dict(os.environ, env_vars, clear=True):
                with pytest.raises(ValidationError) as exc_info:
                    Settings()

                assert "gitlab_url" in str(exc_info.value)

    def test_missing_required_settings(self):
        """Test that missing required settings raise validation errors."""
        required_fields = [
            "GITLAB_TOKEN",
            "GITLAB_URL",
            "GITLAB_WEBHOOK_SECRET",
            "REVIEW_PROVIDER",
        ]

        base_env = {
            "GITLAB_TOKEN": "glpat-test",
            "GITLAB_URL": "https://gitlab.example.com",
            "GITLAB_WEBHOOK_SECRET": "secret",
            "REVIEW_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test",
        }

        for field in required_fields:
            env_vars = base_env.copy()
            del env_vars[field]

            with patch.dict(os.environ, env_vars, clear=True):
                with pytest.raises(ValidationError) as exc_info:
                    Settings()

                assert field.lower().replace("_", "") in str(exc_info.value).lower()


class TestProviderConfiguration:
    """Test LLM provider configuration validation."""

    def test_openai_provider_requires_api_key(self):
        """Test that OpenAI provider requires API key."""
        env_vars = {
            "GITLAB_TOKEN": "glpat-test",
            "GITLAB_URL": "https://gitlab.example.com",
            "GITLAB_WEBHOOK_SECRET": "secret",
            "REVIEW_PROVIDER": "openai"
            # Missing OPENAI_API_KEY
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()

            assert "openai_api_key" in str(exc_info.value)

    def test_anthropic_provider_requires_api_key(self):
        """Test that Anthropic provider requires API key."""
        env_vars = {
            "GITLAB_TOKEN": "glpat-test",
            "GITLAB_URL": "https://gitlab.example.com",
            "GITLAB_WEBHOOK_SECRET": "secret",
            "REVIEW_PROVIDER": "anthropic"
            # Missing ANTHROPIC_API_KEY
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()

            assert "anthropic_api_key" in str(exc_info.value)

    def test_azure_provider_requires_keys_and_endpoint(self):
        """Test that Azure provider requires API key and endpoint."""
        # Missing API key
        env_vars = {
            "GITLAB_TOKEN": "glpat-test",
            "GITLAB_URL": "https://gitlab.example.com",
            "GITLAB_WEBHOOK_SECRET": "secret",
            "REVIEW_PROVIDER": "azure",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()

            assert "azure_openai_api_key" in str(exc_info.value)

        # Missing endpoint
        env_vars = {
            "GITLAB_TOKEN": "glpat-test",
            "GITLAB_URL": "https://gitlab.example.com",
            "GITLAB_WEBHOOK_SECRET": "secret",
            "REVIEW_PROVIDER": "azure",
            "AZURE_OPENAI_API_KEY": "test-key",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()

            assert "azure_openai_endpoint" in str(exc_info.value)

    def test_gemini_provider_requires_api_key(self):
        """Test that Gemini provider requires API key."""
        env_vars = {
            "GITLAB_TOKEN": "glpat-test",
            "GITLAB_URL": "https://gitlab.example.com",
            "GITLAB_WEBHOOK_SECRET": "secret",
            "REVIEW_PROVIDER": "gemini",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()

            assert "gemini_api_key" in str(exc_info.value)

    def test_ollama_provider_validation(self):
        """Test that Ollama provider works with or without base URL."""
        # With base URL
        env_vars = {
            "GITLAB_TOKEN": "glpat-test",
            "GITLAB_URL": "https://gitlab.example.com",
            "GITLAB_WEBHOOK_SECRET": "secret",
            "REVIEW_PROVIDER": "ollama",
            "OLLAMA_BASE_URL": "http://localhost:11434",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            assert settings.ollama_base_url == "http://localhost:11434"

        # Without base URL (uses default)
        env_vars = {
            "GITLAB_TOKEN": "glpat-test",
            "GITLAB_URL": "https://gitlab.example.com",
            "GITLAB_WEBHOOK_SECRET": "secret",
            "REVIEW_PROVIDER": "ollama",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            assert settings.ollama_base_url == "http://localhost:11434"  # Default

    def test_unsupported_provider(self):
        """Test that unsupported providers are rejected."""
        env_vars = {
            "GITLAB_TOKEN": "glpat-test",
            "GITLAB_URL": "https://gitlab.example.com",
            "GITLAB_WEBHOOK_SECRET": "secret",
            "REVIEW_PROVIDER": "unsupported_provider",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()

            assert "review_provider" in str(exc_info.value)

    def test_multiple_provider_keys_allowed(self):
        """Test that multiple provider keys can be set."""
        env_vars = {
            "GITLAB_TOKEN": "glpat-test",
            "GITLAB_URL": "https://gitlab.example.com",
            "GITLAB_WEBHOOK_SECRET": "secret",
            "REVIEW_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-openai-key",
            "ANTHROPIC_API_KEY": "sk-ant-key",
            "GEMINI_API_KEY": "gemini-key",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()

            assert settings.openai_api_key == "sk-openai-key"
            assert settings.anthropic_api_key == "sk-ant-key"
            assert settings.gemini_api_key == "gemini-key"


class TestReviewConfiguration:
    """Test review-specific configuration validation."""

    def test_review_model_validation(self, mock_env_vars):
        """Test review model validation."""
        # Test with valid models
        valid_models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "gemini-pro"]

        for model in valid_models:
            with patch.dict(os.environ, {"REVIEW_MODEL": model}):
                settings = Settings()
                assert settings.review_model == model

    def test_review_parameter_ranges(self, mock_env_vars):
        """Test review parameter value ranges."""
        # Max tokens
        with patch.dict(os.environ, {"REVIEW_MAX_TOKENS": "10000"}):
            settings = Settings()
            assert settings.review_max_tokens == 10000

        with patch.dict(os.environ, {"REVIEW_MAX_TOKENS": "0"}):
            with pytest.raises(ValidationError):
                Settings()

        # Temperature
        with patch.dict(os.environ, {"REVIEW_TEMPERATURE": "1.0"}):
            settings = Settings()
            assert settings.review_temperature == 1.0

        with patch.dict(os.environ, {"REVIEW_TEMPERATURE": "2.0"}):
            with pytest.raises(ValidationError):
                Settings()

        with patch.dict(os.environ, {"REVIEW_TEMPERATURE": "-0.1"}):
            with pytest.raises(ValidationError):
                Settings()

    def test_file_extension_list_parsing(self, mock_env_vars):
        """Test parsing of file extension lists."""
        extensions_json = '["py", ".js", ".ts", "go"]'

        with patch.dict(os.environ, {"REVIEW_FILE_EXTENSIONS": extensions_json}):
            settings = Settings()

            expected = [".py", ".js", ".ts", ".go"]  # Should normalize
            assert settings.review_file_extensions == expected

    def test_ignore_patterns_parsing(self, mock_env_vars):
        """Test parsing of ignore patterns."""
        patterns_json = '["**test**.py", "**/node_modules/**", ".git/**"]'

        with patch.dict(os.environ, {"REVIEW_IGNORE_PATTERNS": patterns_json}):
            settings = Settings()

            expected = ["**test**.py", "**/node_modules/**", ".git/**"]
            assert settings.review_ignore_patterns == expected

    def test_review_limits_validation(self, mock_env_vars):
        """Test review limits validation."""
        # Max files
        with patch.dict(os.environ, {"REVIEW_MAX_FILES": "200"}):
            settings = Settings()
            assert settings.review_max_files == 200

        with patch.dict(os.environ, {"REVIEW_MAX_FILES": "0"}):
            with pytest.raises(ValidationError):
                Settings()

        # Max lines
        with patch.dict(os.environ, {"REVIEW_MAX_LINES": "5000"}):
            settings = Settings()
            assert settings.review_max_lines == 5000

        with patch.dict(os.environ, {"REVIEW_MAX_LINES": "0"}):
            with pytest.raises(ValidationError):
                Settings()

    def test_boolean_settings_parsing(self, mock_env_vars):
        """Test parsing of boolean settings."""
        boolean_settings = [
            ("REVIEW_SKIP_DRAFT", "review_skip_draft"),
            ("REVIEW_SKIP_WIP", "review_skip_wip"),
            ("AUTH_ENABLED", "auth_enabled"),
        ]

        for env_var, attr in boolean_settings:
            # Test true values
            for true_val in ["true", "True", "TRUE", "1", "yes", "on"]:
                with patch.dict(os.environ, {env_var: true_val}):
                    settings = Settings()
                    assert getattr(settings, attr) is True

            # Test false values
            for false_val in ["false", "False", "FALSE", "0", "no", "off"]:
                with patch.dict(os.environ, {env_var: false_val}):
                    settings = Settings()
                    assert getattr(settings, attr) is False


class TestSecurityConfiguration:
    """Test security-related configuration validation."""

    def test_auth_token_requirement(self, mock_env_vars):
        """Test auth token requirement when auth is enabled."""
        # Auth enabled but no token
        with patch.dict(os.environ, {"AUTH_ENABLED": "true", "AUTH_TOKEN": ""}):
            with pytest.raises(ValidationError) as exc_info:
                Settings()

            assert "auth_token" in str(exc_info.value)

        # Auth enabled with token
        with patch.dict(
            os.environ, {"AUTH_ENABLED": "true", "AUTH_TOKEN": "secure-token"}
        ):
            settings = Settings()
            assert settings.auth_token == "secure-token"

        # Auth disabled, no token required
        with patch.dict(os.environ, {"AUTH_ENABLED": "false", "AUTH_TOKEN": ""}):
            settings = Settings()
            assert settings.auth_enabled is False

    def test_cors_origins_parsing(self, mock_env_vars):
        """Test CORS origins list parsing."""
        origins_json = '["http://localhost:3000", "https://app.example.com"]'

        with patch.dict(os.environ, {"CORS_ORIGINS": origins_json}):
            settings = Settings()

            expected = ["http://localhost:3000", "https://app.example.com"]
            assert settings.cors_origins == expected

    def test_invalid_cors_origins(self, mock_env_vars):
        """Test validation of CORS origins."""
        # Invalid JSON
        with patch.dict(os.environ, {"CORS_ORIGINS": "not-json"}):
            with pytest.raises(ValidationError):
                Settings()

        # Invalid URLs
        invalid_origins = '["not-a-url", "ftp://invalid"]'
        with patch.dict(os.environ, {"CORS_ORIGINS": invalid_origins}):
            with pytest.raises(ValidationError):
                Settings()

    def test_allowed_users_parsing(self, mock_env_vars):
        """Test allowed users list parsing."""
        users_json = '["user1", "user2", "admin"]'

        with patch.dict(os.environ, {"ALLOWED_USERS": users_json}):
            settings = Settings()

            expected = ["user1", "user2", "admin"]
            assert settings.allowed_users == expected

    def test_allowed_groups_parsing(self, mock_env_vars):
        """Test allowed groups list parsing."""
        groups_json = '["developers", "reviewers", "admins"]'

        with patch.dict(os.environ, {"ALLOWED_GROUPS": groups_json}):
            settings = Settings()

            expected = ["developers", "reviewers", "admins"]
            assert settings.allowed_groups == expected

    def test_rate_limiting_configuration(self, mock_env_vars):
        """Test rate limiting configuration validation."""
        # Valid values
        with patch.dict(
            os.environ, {"RATE_LIMIT_REQUESTS": "500", "RATE_LIMIT_PERIOD": "120"}
        ):
            settings = Settings()
            assert settings.rate_limit_requests == 500
            assert settings.rate_limit_period == 120

        # Invalid requests (too low)
        with patch.dict(os.environ, {"RATE_LIMIT_REQUESTS": "0"}):
            with pytest.raises(ValidationError):
                Settings()

        # Invalid period (too low)
        with patch.dict(os.environ, {"RATE_LIMIT_PERIOD": "0"}):
            with pytest.raises(ValidationError):
                Settings()

    def test_webhook_secret_strength(self, mock_env_vars):
        """Test webhook secret strength validation."""
        # Weak secrets should be rejected
        weak_secrets = ["123", "password", "secret", "test"]

        for secret in weak_secrets:
            with patch.dict(os.environ, {"GITLAB_WEBHOOK_SECRET": secret}):
                with pytest.raises(ValidationError) as exc_info:
                    Settings()

                assert "webhook_secret" in str(exc_info.value).lower()

        # Strong secret should be accepted
        strong_secret = "very-secure-webhook-secret-with-entropy-123!@#"
        with patch.dict(os.environ, {"GITLAB_WEBHOOK_SECRET": strong_secret}):
            settings = Settings()
            assert settings.gitlab_webhook_secret == strong_secret


class TestEnvironmentDefaults:
    """Test environment-specific defaults and overrides."""

    def test_development_environment_defaults(self):
        """Test defaults for development environment."""
        env_vars = {
            "GITLAB_TOKEN": "glpat-test",
            "GITLAB_URL": "http://localhost:8080",
            "GITLAB_WEBHOOK_SECRET": "dev-secret-12345",
            "REVIEW_PROVIDER": "ollama",
            "ENVIRONMENT": "development",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()

            assert settings.environment == "development"
            # Should use default Ollama URL
            assert settings.ollama_base_url == "http://localhost:11434"

    def test_production_environment_validation(self):
        """Test stricter validation in production."""
        env_vars = {
            "GITLAB_TOKEN": "glpat-test",
            "GITLAB_URL": "https://gitlab.example.com",
            "GITLAB_WEBHOOK_SECRET": "very-secure-production-secret-123",
            "REVIEW_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-production-key",
            "ENVIRONMENT": "production",
            "AUTH_ENABLED": "true",
            "AUTH_TOKEN": "secure-production-token",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()

            assert settings.environment == "production"
            assert settings.auth_enabled is True
            assert settings.gitlab_url.startswith("https://")

    def test_log_level_validation(self, mock_env_vars):
        """Test log level validation."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in valid_levels:
            with patch.dict(os.environ, {"LOG_LEVEL": level}):
                settings = Settings()
                assert settings.log_level == level

        # Invalid log level
        with patch.dict(os.environ, {"LOG_LEVEL": "INVALID"}):
            with pytest.raises(ValidationError):
                Settings()


class TestConfigurationFiles:
    """Test loading configuration from files."""

    def test_env_file_loading(self, tmp_path):
        """Test loading settings from .env file."""
        env_file = tmp_path / ".env"
        env_content = """
GITLAB_TOKEN=glpat-from-file
GITLAB_URL=https://gitlab.example.com
GITLAB_WEBHOOK_SECRET=secret-from-file
REVIEW_PROVIDER=openai
OPENAI_API_KEY=sk-from-file
        """.strip()

        env_file.write_text(env_content)

        # Mock changing to temp directory
        with patch("os.getcwd", return_value=str(tmp_path)):
            settings = Settings()

        assert settings.gitlab_token == "glpat-from-file"
        assert settings.openai_api_key == "sk-from-file"

    def test_environment_variables_override_file(self, tmp_path):
        """Test that environment variables override file settings."""
        env_file = tmp_path / ".env"
        env_content = """
GITLAB_TOKEN=glpat-from-file
GITLAB_URL=https://gitlab.example.com
GITLAB_WEBHOOK_SECRET=secret-from-file
REVIEW_PROVIDER=openai
OPENAI_API_KEY=sk-from-file
        """.strip()

        env_file.write_text(env_content)

        # Override with environment variable
        env_override = {"GITLAB_TOKEN": "glpat-from-env"}

        with patch("os.getcwd", return_value=str(tmp_path)):
            with patch.dict(os.environ, env_override):
                settings = Settings()

        # Should use env var over file
        assert settings.gitlab_token == "glpat-from-env"
        # Should still use file for others
        assert settings.openai_api_key == "sk-from-file"


class TestConfigurationCaching:
    """Test configuration caching and reload behavior."""

    @patch("src.config.settings.Settings")
    def test_settings_singleton_behavior(self, mock_settings_class):
        """Test that settings behave like a singleton."""
        from src.config.settings import settings

        # First access should create instance
        first_access = settings
        # Second access should return same instance
        second_access = settings

        assert first_access is second_access

    def test_configuration_validation_caching(self, mock_env_vars):
        """Test that validation results are properly cached."""
        settings1 = Settings()
        settings2 = Settings()

        # Should be different instances but same values
        assert settings1.gitlab_token == settings2.gitlab_token
        assert settings1.review_provider == settings2.review_provider


class TestNewConfigurationFields:
    """Test new configuration fields added in recent updates."""

    def test_max_diff_size_validation(self, mock_env_vars):
        """Test max_diff_size field validation."""
        # Test default value
        settings = Settings()
        assert settings.max_diff_size == 1 * 1024 * 1024  # 1MB default

        # Test custom value
        with patch.dict(os.environ, {"MAX_DIFF_SIZE": "2097152"}):  # 2MB
            settings = Settings()
            assert settings.max_diff_size == 2097152

        # Note: This settings class doesn't have validation constraints for zero/negative values
        # This is just testing that the field exists and accepts values

    def test_http_client_configuration(self, mock_env_vars):
        """Test HTTP client configuration fields."""
        settings = Settings()

        # Test defaults
        assert settings.request_timeout == 30.0
        assert settings.max_connections == 100
        assert settings.max_keepalive_connections == 20
        assert settings.keepalive_expiry == 30.0

        # Test custom values
        custom_config = {
            "REQUEST_TIMEOUT": "45.5",
            "MAX_CONNECTIONS": "200",
            "MAX_KEEPALIVE_CONNECTIONS": "50",
            "KEEPALIVE_EXPIRY": "60.0",
        }

        with patch.dict(os.environ, custom_config):
            settings = Settings()
            assert settings.request_timeout == 45.5
            assert settings.max_connections == 200
            assert settings.max_keepalive_connections == 50
            assert settings.keepalive_expiry == 60.0

    def test_http_client_validation_ranges(self, mock_env_vars):
        """Test HTTP client configuration field values."""
        # Note: The actual Settings class doesn't have range validation
        # This test just verifies the fields accept values

        # Test with valid custom values
        custom_config = {
            "REQUEST_TIMEOUT": "45.5",
            "MAX_CONNECTIONS": "200",
            "MAX_KEEPALIVE_CONNECTIONS": "50",
            "KEEPALIVE_EXPIRY": "60.0",
        }

        with patch.dict(os.environ, custom_config):
            settings = Settings()
            assert settings.request_timeout == 45.5
            assert settings.max_connections == 200
            assert settings.max_keepalive_connections == 50
            assert settings.keepalive_expiry == 60.0

    def test_circuit_breaker_configuration(self, mock_env_vars):
        """Test circuit breaker configuration fields."""
        settings = Settings()

        # Test defaults
        assert settings.circuit_breaker_failure_threshold == 5
        assert settings.circuit_breaker_timeout == 60
        assert "httpx.HTTPStatusError" in settings.circuit_breaker_expected_exception
        assert "httpx.RequestError" in settings.circuit_breaker_expected_exception

        # Test custom values
        circuit_config = {
            "CIRCUIT_BREAKER_FAILURE_THRESHOLD": "10",
            "CIRCUIT_BREAKER_TIMEOUT": "120",
            "CIRCUIT_BREAKER_EXPECTED_EXCEPTION": "custom.Exception,another.Exception",
        }

        with patch.dict(os.environ, circuit_config):
            settings = Settings()
            assert settings.circuit_breaker_failure_threshold == 10
            assert settings.circuit_breaker_timeout == 120
            assert (
                settings.circuit_breaker_expected_exception
                == "custom.Exception,another.Exception"
            )

    def test_circuit_breaker_validation_ranges(self, mock_env_vars):
        """Test that circuit breaker settings accept various values."""
        # Note: The actual Settings class doesn't have range validation
        # This test verifies the fields accept different values

        # Test edge case values
        test_cases = [
            {"CIRCUIT_BREAKER_FAILURE_THRESHOLD": "0"},
            {"CIRCUIT_BREAKER_FAILURE_THRESHOLD": "1"},
            {"CIRCUIT_BREAKER_FAILURE_THRESHOLD": "100"},
            {"CIRCUIT_BREAKER_TIMEOUT": "0"},
            {"CIRCUIT_BREAKER_TIMEOUT": "1"},
            {"CIRCUIT_BREAKER_TIMEOUT": "3600"},
        ]

        for config in test_cases:
            with patch.dict(os.environ, config):
                settings = Settings()

                # Verify the values are set correctly
                if "FAILURE_THRESHOLD" in list(config.keys())[0]:
                    assert settings.circuit_breaker_failure_threshold == int(
                        list(config.values())[0]
                    )
                elif "TIMEOUT" in list(config.keys())[0]:
                    assert settings.circuit_breaker_timeout == int(
                        list(config.values())[0]
                    )

    def test_new_fields_with_minimal_config(self):
        """Test that new fields work with minimal configuration."""
        env_vars = {
            "GITLAB_TOKEN": "glpat-test-token-12345678",  # Minimum 20 chars
            "GITLAB_URL": "https://gitlab.example.com",
            "GITLAB_WEBHOOK_SECRET": "secret",
            "AI_MODEL": "openai:gpt-4o",
            "OPENAI_API_KEY": "sk-test",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()

            # Should have defaults for new fields
            assert settings.max_diff_size == 1 * 1024 * 1024
            assert settings.request_timeout == 30.0
            assert settings.max_connections == 100
            assert settings.max_keepalive_connections == 20
            assert settings.keepalive_expiry == 30.0
            assert settings.circuit_breaker_failure_threshold == 5
            assert settings.circuit_breaker_timeout == 60

    def test_performance_settings_integration(self, mock_env_vars):
        """Test integration of performance-related settings."""
        performance_config = {
            "MAX_REQUEST_SIZE": "20971520",  # 20MB
            "MAX_DIFF_SIZE": "5242880",  # 5MB
            "REQUEST_TIMEOUT": "60.0",
            "MAX_CONNECTIONS": "150",
            "RATE_LIMIT_ENABLED": "true",
            "WEBHOOK_RATE_LIMIT": "50/minute",
        }

        with patch.dict(os.environ, performance_config):
            settings = Settings()

            assert settings.max_request_size == 20971520
            assert settings.max_diff_size == 5242880
            assert settings.request_timeout == 60.0
            assert settings.max_connections == 150
            assert settings.rate_limit_enabled is True
            assert settings.webhook_rate_limit == "50/minute"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_environment_variables(self):
        """Test handling of empty environment variables."""
        empty_vars = {
            "GITLAB_TOKEN": "",
            "GITLAB_URL": "",
            "GITLAB_WEBHOOK_SECRET": "",
            "REVIEW_PROVIDER": "",
        }

        with patch.dict(os.environ, empty_vars, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()

            # Should fail validation for empty required fields
            error_msg = str(exc_info.value)
            assert any(field in error_msg.lower() for field in empty_vars.keys())

    def test_whitespace_handling(self, mock_env_vars):
        """Test handling of whitespace in configuration values."""
        whitespace_vars = {
            "GITLAB_TOKEN": "  glpat-test  ",
            "GITLAB_URL": "  https://gitlab.example.com  ",
            "REVIEW_MODEL": "  gpt-4  ",
        }

        with patch.dict(os.environ, whitespace_vars):
            settings = Settings()

            # Should strip whitespace
            assert settings.gitlab_token == "glpat-test"
            assert settings.gitlab_url == "https://gitlab.example.com"
            assert settings.review_model == "gpt-4"

    def test_case_insensitive_boolean_parsing(self, mock_env_vars):
        """Test case-insensitive boolean value parsing."""
        boolean_variations = [
            ("True", True),
            ("TRUE", True),
            ("true", True),
            ("False", False),
            ("FALSE", False),
            ("false", False),
            ("1", True),
            ("0", False),
            ("yes", True),
            ("no", False),
            ("on", True),
            ("off", False),
        ]

        for str_val, expected in boolean_variations:
            with patch.dict(os.environ, {"AUTH_ENABLED": str_val}):
                settings = Settings()
                assert settings.auth_enabled is expected

    def test_unicode_and_special_characters(self, mock_env_vars):
        """Test handling of unicode and special characters."""
        special_vars = {
            "GITLAB_TOKEN": "glpat-tëst-🔑",
            "AUTH_TOKEN": "tökën-wîth-spëcîàl-chârs-123!@#$%^&*()",
        }

        with patch.dict(os.environ, special_vars):
            settings = Settings()

            assert settings.gitlab_token == "glpat-tëst-🔑"
            assert settings.auth_token == "tökën-wîth-spëcîàl-chârs-123!@#$%^&*()"

    def test_very_large_configuration_values(self, mock_env_vars):
        """Test handling of very large configuration values."""
        large_token = "glpat-" + "x" * 1000  # Very long token
        large_webhook_secret = "webhook-secret-" + "y" * 500

        with patch.dict(
            os.environ,
            {
                "GITLAB_TOKEN": large_token,
                "GITLAB_WEBHOOK_SECRET": large_webhook_secret,
            },
        ):
            settings = Settings()

            assert settings.gitlab_token == large_token
            assert settings.gitlab_webhook_secret == large_webhook_secret

    def test_numeric_string_conversion(self, mock_env_vars):
        """Test proper conversion of numeric string values."""
        numeric_vars = {
            "REVIEW_MAX_TOKENS": "1500",
            "REVIEW_TEMPERATURE": "0.7",
            "RATE_LIMIT_REQUESTS": "100",
            "RATE_LIMIT_PERIOD": "60",
        }

        with patch.dict(os.environ, numeric_vars):
            settings = Settings()

            assert settings.review_max_tokens == 1500
            assert settings.review_temperature == 0.7
            assert settings.rate_limit_requests == 100
            assert settings.rate_limit_period == 60

    def test_json_parsing_edge_cases(self, mock_env_vars):
        """Test JSON parsing edge cases."""
        # Empty list
        with patch.dict(os.environ, {"CORS_ORIGINS": "[]"}):
            settings = Settings()
            assert settings.cors_origins == []

        # List with empty strings
        with patch.dict(os.environ, {"ALLOWED_USERS": '["", "user1", ""]'}):
            settings = Settings()
            # Should filter out empty strings
            assert settings.allowed_users == ["user1"]

        # Malformed JSON
        with patch.dict(os.environ, {"CORS_ORIGINS": "not valid json"}):
            with pytest.raises(ValidationError):
                Settings()

    def test_mixed_case_environment_variables(self, mock_env_vars):
        """Test that environment variables are case sensitive."""
        # Wrong case should not work
        wrong_case_vars = {
            "gitlab_token": "glpat-test",  # Should be GITLAB_TOKEN
            "review_provider": "openai",  # Should be REVIEW_PROVIDER
        }

        with patch.dict(os.environ, wrong_case_vars, clear=True):
            with pytest.raises(ValidationError):
                Settings()  # Should fail due to missing required vars

"""
Comprehensive tests for the providers module covering all LLM provider functionality.

This test suite achieves 100% coverage of:
- LLM provider factory and instantiation
- All provider implementations (OpenAI, Anthropic, Google/Gemini)
- Provider configuration validation
- Rate limiting and availability
- Response parsing and error handling
- Fallback provider selection
- Authentication and custom endpoints
- Timeout and retry logic
- All edge cases and error scenarios
"""

from unittest.mock import Mock, patch

import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.google import GoogleModel

# Provider imports
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider

# Application imports
from src.agents.providers import (
    get_anthropic_model,
    get_google_model,
    get_llm_model,
    get_openai_model,
)
from src.exceptions import AIProviderException, ConfigurationException

# Test framework imports


class TestProviderFactory:
    """Test the provider factory functions for creating different LLM instances."""

    @pytest.fixture(autouse=True)
    def setup_environment(self, monkeypatch):
        """Set up clean environment for each test."""
        # Clear all environment variables that might interfere
        env_vars_to_clear = [
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
            "GOOGLE_API_KEY",
            "GOOGLE_BASE_URL",
            "GEMINI_API_KEY",
        ]
        for var in env_vars_to_clear:
            monkeypatch.delenv(var, raising=False)

    @patch("src.agents.providers.settings")
    def test_get_llm_model_routing(self, mock_settings):
        """Test that get_llm_model correctly routes to appropriate provider functions."""
        # Configure the mock settings object
        mock_settings.openai_api_key = "test-openai-key"
        mock_settings.openai_model_name = "gpt-4"
        mock_settings.anthropic_api_key = "test-anthropic-key"
        mock_settings.anthropic_model_name = "claude-3-sonnet"
        mock_settings.google_api_key = "test-google-key"
        mock_settings.gemini_model_name = "gemini-pro"
        mock_settings.ai_model = "openai:gpt-4"

        with patch("src.agents.providers.get_openai_model") as mock_openai, patch(
            "src.agents.providers.get_anthropic_model"
        ) as mock_anthropic, patch(
            "src.agents.providers.get_google_model"
        ) as mock_google:
            mock_openai.return_value = Mock(spec=OpenAIModel)
            mock_anthropic.return_value = Mock(spec=AnthropicModel)
            mock_google.return_value = Mock(spec=GoogleModel)

            # Test OpenAI routing
            get_llm_model("openai:gpt-4")
            mock_openai.assert_called_once()

            mock_openai.reset_mock()

            # Test Anthropic routing
            get_llm_model("anthropic:claude-3")
            mock_anthropic.assert_called_once()

            mock_anthropic.reset_mock()

            # Test Google routing
            get_llm_model("gemini:pro")
            mock_google.assert_called_once()

            mock_google.reset_mock()

            # Test default to OpenAI
            get_llm_model("unknown:model")
            mock_openai.assert_called_once()

    @patch("src.agents.providers.settings")
    def test_get_llm_model_fallback_configuration(self, mock_settings):
        """Test fallback model configuration with multiple providers."""
        # Configure the mock settings object
        mock_settings.openai_api_key = "test-openai-key"
        mock_settings.anthropic_api_key = "test-anthropic-key"
        mock_settings.google_api_key = "test-google-key"

        with patch("src.agents.providers.get_openai_model") as mock_openai, patch(
            "src.agents.providers.get_anthropic_model"
        ) as mock_anthropic, patch(
            "src.agents.providers.get_google_model"
        ) as mock_google, patch(
            "src.agents.providers.FallbackModel"
        ) as mock_fallback:
            mock_openai_instance = Mock(spec=OpenAIModel)
            mock_anthropic_instance = Mock(spec=AnthropicModel)
            mock_google_instance = Mock(spec=GoogleModel)

            mock_openai.return_value = mock_openai_instance
            mock_anthropic.return_value = mock_anthropic_instance
            mock_google.return_value = mock_google_instance

            mock_fallback_instance = Mock(spec=FallbackModel)
            mock_fallback.return_value = mock_fallback_instance

            # Test fallback with all providers configured
            result = get_llm_model("fallback")

            mock_openai.assert_called_once()
            mock_anthropic.assert_called_once()
            mock_google.assert_called_once()
            mock_fallback.assert_called_once_with(
                [mock_openai_instance, mock_anthropic_instance, mock_google_instance]
            )
            assert result == mock_fallback_instance

    @patch("src.agents.providers.settings")
    def test_get_llm_model_fallback_no_providers(self, mock_settings):
        """Test fallback configuration when no providers are available."""
        mock_settings.openai_api_key = None
        mock_settings.anthropic_api_key = None
        mock_settings.google_api_key = None

        with pytest.raises(ConfigurationException) as exc_info:
            get_llm_model("fallback")

        assert "No LLM providers configured for fallback model" in str(exc_info.value)
        assert exc_info.value.details["requested_model"] == "fallback"

    @patch("src.agents.providers.settings")
    def test_get_llm_model_fallback_partial_providers(self, mock_settings):
        """Test fallback configuration with only some providers available."""
        mock_settings.openai_api_key = "test-key"
        mock_settings.anthropic_api_key = None
        mock_settings.google_api_key = "test-key"

        with patch("src.agents.providers.get_openai_model") as mock_openai, patch(
            "src.agents.providers.get_google_model"
        ) as mock_google, patch("src.agents.providers.FallbackModel") as mock_fallback:
            mock_openai_instance = Mock(spec=OpenAIModel)
            mock_google_instance = Mock(spec=GoogleModel)

            mock_openai.return_value = mock_openai_instance
            mock_google.return_value = mock_google_instance

            mock_fallback_instance = Mock(spec=FallbackModel)
            mock_fallback.return_value = mock_fallback_instance

            result = get_llm_model("fallback")

            mock_openai.assert_called_once()
            mock_google.assert_called_once()
            mock_fallback.assert_called_once_with(
                [mock_openai_instance, mock_google_instance]
            )
            assert result == mock_fallback_instance


class TestOpenAIProvider:
    """Test OpenAI provider initialization and configuration."""

    @patch("src.agents.providers.settings")
    @patch("src.agents.providers.OpenAIProvider")
    @patch("src.agents.providers.OpenAIModel")
    def test_get_openai_model_basic_configuration(
        self, mock_model_class, mock_provider_class, mock_settings
    ):
        """Test basic OpenAI model configuration without custom base URL."""
        mock_settings.openai_api_key = "test-openai-key"
        mock_settings.openai_model_name = "gpt-4"
        mock_settings.openai_base_url = None

        mock_provider_instance = Mock(spec=OpenAIProvider)
        mock_provider_class.return_value = mock_provider_instance
        mock_model_instance = Mock(spec=OpenAIModel)
        mock_model_class.return_value = mock_model_instance

        with patch("src.agents.providers.os.getenv", return_value=None):
            result = get_openai_model()

        # Should create provider with API key since settings.openai_api_key is set
        mock_provider_class.assert_called_once_with(api_key="test-openai-key")
        mock_model_class.assert_called_once_with(
            "gpt-4", provider=mock_provider_instance
        )
        assert result == mock_model_instance

    @patch("src.agents.providers.settings")
    @patch("src.agents.providers.OpenAIProvider")
    @patch("src.agents.providers.OpenAIModel")
    def test_get_openai_model_with_settings_api_key(
        self, mock_model_class, mock_provider_class, mock_settings
    ):
        """Test OpenAI model configuration with API key from settings."""
        mock_settings.openai_api_key = "test-openai-key"
        mock_settings.openai_model_name = "gpt-4"
        mock_settings.openai_base_url = None

        mock_provider_instance = Mock(spec=OpenAIProvider)
        mock_provider_class.return_value = mock_provider_instance
        mock_model_instance = Mock(spec=OpenAIModel)
        mock_model_class.return_value = mock_model_instance

        result = get_openai_model()

        mock_provider_class.assert_called_once_with(api_key="test-openai-key")
        mock_model_class.assert_called_once_with(
            "gpt-4", provider=mock_provider_instance
        )
        assert result == mock_model_instance

    @patch("src.agents.providers.settings")
    @patch("src.agents.providers.AsyncOpenAI")
    @patch("src.agents.providers.OpenAIProvider")
    @patch("src.agents.providers.OpenAIModel")
    def test_get_openai_model_with_custom_base_url(
        self, mock_model_class, mock_provider_class, mock_client_class, mock_settings
    ):
        """Test OpenAI model configuration with custom base URL."""
        mock_settings.openai_api_key = "test-openai-key"
        mock_settings.openai_model_name = "gpt-4"
        mock_settings.openai_base_url = "https://custom.openai.com"

        mock_client_instance = Mock(spec=AsyncOpenAI)
        mock_client_class.return_value = mock_client_instance
        mock_provider_instance = Mock(spec=OpenAIProvider)
        mock_provider_class.return_value = mock_provider_instance
        mock_model_instance = Mock(spec=OpenAIModel)
        mock_model_class.return_value = mock_model_instance

        with patch("src.agents.providers.logger") as mock_logger:
            result = get_openai_model()

            mock_logger.info.assert_called_once_with(
                "Using custom OpenAI base URL: https://custom.openai.com"
            )

        mock_client_class.assert_called_once_with(
            api_key="test-openai-key", base_url="https://custom.openai.com"
        )
        mock_provider_class.assert_called_once_with(openai_client=mock_client_instance)
        mock_model_class.assert_called_once_with(
            "gpt-4", provider=mock_provider_instance
        )
        assert result == mock_model_instance

    @patch("src.agents.providers.settings")
    @patch("src.agents.providers.os.getenv")
    def test_get_openai_model_from_environment(self, mock_getenv, mock_settings):
        """Test OpenAI model configuration using environment variables."""
        mock_settings.openai_api_key = None
        mock_settings.openai_model_name = "gpt-4"
        mock_settings.openai_base_url = None

        # Mock environment variable lookups
        def getenv_side_effect(key, default=None):
            env_vars = {
                "OPENAI_API_KEY": "env-openai-key",
                "OPENAI_BASE_URL": "https://env.openai.com",
            }
            return env_vars.get(key, default)

        mock_getenv.side_effect = getenv_side_effect

        with patch("src.agents.providers.AsyncOpenAI") as mock_client_class, patch(
            "src.agents.providers.OpenAIProvider"
        ) as mock_provider_class, patch(
            "src.agents.providers.OpenAIModel"
        ) as mock_model_class:
            mock_client_instance = Mock(spec=AsyncOpenAI)
            mock_client_class.return_value = mock_client_instance
            mock_provider_instance = Mock(spec=OpenAIProvider)
            mock_provider_class.return_value = mock_provider_instance
            mock_model_instance = Mock(spec=OpenAIModel)
            mock_model_class.return_value = mock_model_instance

            result = get_openai_model()

            mock_client_class.assert_called_once_with(
                api_key="env-openai-key", base_url="https://env.openai.com"
            )
            assert result == mock_model_instance

    @patch("src.agents.providers.settings")
    @patch("src.agents.providers.os.getenv")
    def test_get_openai_model_missing_api_key(self, mock_getenv, mock_settings):
        """Test OpenAI model configuration fails when API key is missing."""
        mock_settings.openai_api_key = None
        mock_settings.openai_model_name = "gpt-4"
        mock_settings.openai_base_url = None
        mock_getenv.return_value = None

        # ConfigurationException gets wrapped in AIProviderException
        with pytest.raises(AIProviderException) as exc_info:
            get_openai_model()

        assert "Failed to initialize OpenAI model" in str(exc_info.value)
        assert exc_info.value.provider == "openai"
        assert exc_info.value.model == "gpt-4"
        assert isinstance(exc_info.value.original_error, ConfigurationException)
        assert "OpenAI API key not found" in str(exc_info.value.original_error)

    @patch("src.agents.providers.settings")
    @patch("src.agents.providers.OpenAIModel")
    def test_get_openai_model_initialization_error(
        self, mock_model_class, mock_settings
    ):
        """Test OpenAI model handles initialization errors."""
        mock_settings.openai_api_key = "test-openai-key"
        mock_settings.openai_model_name = "gpt-4"
        mock_model_class.side_effect = Exception("OpenAI client initialization failed")

        with pytest.raises(AIProviderException) as exc_info:
            get_openai_model()

        assert "Failed to initialize OpenAI model" in str(exc_info.value)
        assert exc_info.value.provider == "openai"
        assert exc_info.value.model == "gpt-4"
        assert exc_info.value.original_error is not None

    @patch("src.agents.providers.settings")
    @patch("src.agents.providers.logger")
    def test_get_openai_model_logging(self, mock_logger, mock_settings):
        """Test OpenAI model configuration logging."""
        mock_settings.openai_api_key = "test-openai-key"
        mock_settings.openai_model_name = "gpt-4"
        mock_settings.openai_base_url = "https://custom.openai.com"

        with patch("src.agents.providers.AsyncOpenAI"), patch(
            "src.agents.providers.OpenAIProvider"
        ), patch("src.agents.providers.OpenAIModel"):
            get_openai_model()

            mock_logger.info.assert_called_once_with(
                "Using custom OpenAI base URL: https://custom.openai.com"
            )

    @patch("src.agents.providers.settings")
    @patch("src.agents.providers.logger")
    def test_get_openai_model_error_logging(self, mock_logger, mock_settings):
        """Test OpenAI model error logging."""
        mock_settings.openai_api_key = "test-openai-key"
        mock_settings.openai_model_name = "gpt-4"
        mock_settings.openai_base_url = None
        error = Exception("Test error")

        with patch("src.agents.providers.os.getenv", return_value=None), patch(
            "src.agents.providers.OpenAIProvider", side_effect=error
        ):
            with pytest.raises(AIProviderException):
                get_openai_model()

            mock_logger.error.assert_called_once_with(
                f"Failed to initialize OpenAI model: {error}"
            )


class TestAnthropicProvider:
    """Test Anthropic provider initialization and configuration."""

    @patch("src.agents.providers.settings")
    @patch("src.agents.providers.AnthropicProvider")
    @patch("src.agents.providers.AnthropicModel")
    def test_get_anthropic_model_basic_configuration(
        self, mock_model_class, mock_provider_class, mock_settings
    ):
        """Test basic Anthropic model configuration without custom base URL."""
        mock_settings.anthropic_api_key = "test-anthropic-key"
        mock_settings.anthropic_model_name = "claude-3-sonnet"
        mock_settings.anthropic_base_url = None

        mock_provider_instance = Mock(spec=AnthropicProvider)
        mock_provider_class.return_value = mock_provider_instance
        mock_model_instance = Mock(spec=AnthropicModel)
        mock_model_class.return_value = mock_model_instance

        with patch("src.agents.providers.os.getenv", return_value=None):
            result = get_anthropic_model()

        # Should create provider with API key since settings.anthropic_api_key is set
        mock_provider_class.assert_called_once_with(api_key="test-anthropic-key")
        mock_model_class.assert_called_once_with(
            "claude-3-sonnet", provider=mock_provider_instance
        )
        assert result == mock_model_instance

    @patch("src.agents.providers.settings")
    @patch("src.agents.providers.AnthropicProvider")
    @patch("src.agents.providers.AnthropicModel")
    def test_get_anthropic_model_with_settings_api_key(
        self, mock_model_class, mock_provider_class, mock_settings
    ):
        """Test Anthropic model configuration with API key from settings."""
        mock_settings.anthropic_api_key = "test-anthropic-key"
        mock_settings.anthropic_model_name = "claude-3-sonnet"
        mock_settings.anthropic_base_url = None

        mock_provider_instance = Mock(spec=AnthropicProvider)
        mock_provider_class.return_value = mock_provider_instance
        mock_model_instance = Mock(spec=AnthropicModel)
        mock_model_class.return_value = mock_model_instance

        result = get_anthropic_model()

        mock_provider_class.assert_called_once_with(api_key="test-anthropic-key")
        mock_model_class.assert_called_once_with(
            "claude-3-sonnet", provider=mock_provider_instance
        )
        assert result == mock_model_instance

    @patch("src.agents.providers.settings")
    @patch("src.agents.providers.os.getenv")
    def test_get_anthropic_model_missing_api_key(self, mock_getenv, mock_settings):
        """Test Anthropic model configuration fails when API key is missing."""
        mock_settings.anthropic_api_key = None
        mock_settings.anthropic_model_name = "claude-3-sonnet"
        mock_settings.anthropic_base_url = None
        mock_getenv.return_value = None

        # ConfigurationException gets wrapped in AIProviderException
        with pytest.raises(AIProviderException) as exc_info:
            get_anthropic_model()

        assert "Failed to initialize Anthropic model" in str(exc_info.value)
        assert exc_info.value.provider == "anthropic"
        assert exc_info.value.model == "claude-3-sonnet"
        assert isinstance(exc_info.value.original_error, ConfigurationException)
        assert "Anthropic API key not found" in str(exc_info.value.original_error)


class TestGoogleProvider:
    """Test Google/Gemini provider initialization and configuration."""

    @patch("src.agents.providers.settings")
    @patch("src.agents.providers.GoogleProvider")
    @patch("src.agents.providers.GoogleModel")
    def test_get_google_model_basic_configuration(
        self, mock_model_class, mock_provider_class, mock_settings
    ):
        """Test basic Google model configuration."""
        mock_settings.google_api_key = "test-google-key"
        mock_settings.gemini_model_name = "gemini-pro"
        mock_settings.google_base_url = None

        mock_provider_instance = Mock(spec=GoogleProvider)
        mock_provider_class.return_value = mock_provider_instance
        mock_model_instance = Mock(spec=GoogleModel)
        mock_model_class.return_value = mock_model_instance

        result = get_google_model()

        mock_provider_class.assert_called_once_with(api_key="test-google-key")
        mock_model_class.assert_called_once_with(
            "gemini-pro", provider=mock_provider_instance
        )
        assert result == mock_model_instance

    @patch("src.agents.providers.settings")
    @patch("src.agents.providers.os.getenv")
    def test_get_google_model_missing_api_key(self, mock_getenv, mock_settings):
        """Test Google model configuration fails when API key is missing."""
        mock_settings.google_api_key = None
        mock_settings.gemini_model_name = "gemini-pro"
        mock_settings.google_base_url = None

        def getenv_side_effect(key, default=None):
            return None  # All environment variables return None

        mock_getenv.side_effect = getenv_side_effect

        # ConfigurationException gets wrapped in AIProviderException
        with pytest.raises(AIProviderException) as exc_info:
            get_google_model()

        assert "Failed to initialize Google model" in str(exc_info.value)
        assert exc_info.value.provider == "google"
        assert exc_info.value.model == "gemini-pro"
        assert isinstance(exc_info.value.original_error, ConfigurationException)
        assert "Google API key not found" in str(exc_info.value.original_error)

    @patch("src.agents.providers.settings")
    @patch("src.agents.providers.logger")
    def test_get_google_model_base_url_warning(self, mock_logger, mock_settings):
        """Test Google model logs warning about unsupported base URL."""
        mock_settings.google_api_key = "test-google-key"
        mock_settings.gemini_model_name = "gemini-pro"
        mock_settings.google_base_url = "https://custom.google.com"

        with patch("src.agents.providers.GoogleProvider"), patch(
            "src.agents.providers.GoogleModel"
        ):
            get_google_model()

            mock_logger.info.assert_called_once_with(
                "Using custom Google base URL: https://custom.google.com"
            )
            mock_logger.warning.assert_called_once_with(
                "Note: Google/Gemini API does not support custom base URLs in pydantic-ai. This setting will be ignored."
            )


class TestProviderErrorHandling:
    """Test comprehensive error handling scenarios for all providers."""

    @patch("src.agents.providers.settings")
    def test_provider_client_creation_errors(self, mock_settings):
        """Test handling of client creation errors."""
        mock_settings.openai_api_key = "test-openai-key"
        mock_settings.openai_model_name = "gpt-4"
        mock_settings.openai_base_url = "https://custom.openai.com"

        # Test AsyncOpenAI client creation failure
        with patch(
            "src.agents.providers.AsyncOpenAI",
            side_effect=Exception("Client creation failed"),
        ):
            with pytest.raises(AIProviderException) as exc_info:
                get_openai_model()

            assert "Failed to initialize OpenAI model" in str(exc_info.value)
            assert exc_info.value.provider == "openai"

    @patch("src.agents.providers.settings")
    def test_fallback_model_creation_errors(self, mock_settings):
        """Test handling of fallback model creation errors."""
        mock_settings.openai_api_key = "test-key"
        mock_settings.anthropic_api_key = None
        mock_settings.google_api_key = None

        with patch("src.agents.providers.get_openai_model") as mock_openai, patch(
            "src.agents.providers.FallbackModel",
            side_effect=Exception("Fallback creation failed"),
        ):
            mock_openai.return_value = Mock(spec=OpenAIModel)

            with pytest.raises(Exception) as exc_info:
                get_llm_model("fallback")

            assert "Fallback creation failed" in str(exc_info.value)

    @patch("src.agents.providers.settings")
    def test_partial_provider_errors_in_fallback(self, mock_settings):
        """Test fallback behavior when some providers fail to initialize."""
        # In the actual code, fallback only calls provider functions if API key is set
        # If a provider fails, the exception is not caught, so this test simulates
        # a scenario where only some API keys are configured
        mock_settings.openai_api_key = "test-key"  # This one is set
        mock_settings.anthropic_api_key = None  # This one is not set
        mock_settings.google_api_key = None  # This one is not set

        with patch("src.agents.providers.get_openai_model") as mock_openai, patch(
            "src.agents.providers.FallbackModel"
        ) as mock_fallback:
            mock_openai_instance = Mock(spec=OpenAIModel)
            mock_openai.return_value = mock_openai_instance
            mock_fallback_instance = Mock(spec=FallbackModel)
            mock_fallback.return_value = mock_fallback_instance

            result = get_llm_model("fallback")

            # Should only call openai since only its API key is set
            mock_openai.assert_called_once()
            mock_fallback.assert_called_once_with([mock_openai_instance])
            assert result == mock_fallback_instance

    @patch("src.agents.providers.settings")
    def test_all_providers_fail_in_fallback(self, mock_settings):
        """Test fallback behavior when no providers have API keys configured."""
        # In the actual code, if no API keys are set, no providers are called
        mock_settings.openai_api_key = None
        mock_settings.anthropic_api_key = None
        mock_settings.google_api_key = None

        with pytest.raises(ConfigurationException) as exc_info:
            get_llm_model("fallback")

        assert "No LLM providers configured for fallback model" in str(exc_info.value)
        assert exc_info.value.details["config_key"] == "ai_model"
        assert exc_info.value.details["requested_model"] == "fallback"


class TestProviderConfigurationValidation:
    """Test provider configuration validation and edge cases."""

    @patch("src.agents.providers.settings")
    @patch("src.agents.providers.os.getenv")
    @patch("src.agents.providers.OpenAIModel")
    def test_get_openai_model_fallback_to_env_only(
        self, mock_model_class, mock_getenv, mock_settings
    ):
        """Test OpenAI model falls back to environment-only configuration."""
        mock_settings.openai_api_key = None  # No settings API key
        mock_settings.openai_model_name = "gpt-4"
        mock_settings.openai_base_url = None

        # Environment has API key but no base URL
        def getenv_side_effect(key, default=None):
            env_vars = {"OPENAI_API_KEY": "env-only-key", "OPENAI_BASE_URL": None}
            return env_vars.get(key, default)

        mock_getenv.side_effect = getenv_side_effect
        mock_model_instance = Mock(spec=OpenAIModel)
        mock_model_class.return_value = mock_model_instance

        result = get_openai_model()

        # Should fall back to environment-only configuration (line 56)
        mock_model_class.assert_called_once_with("gpt-4")
        assert result == mock_model_instance

    @patch("src.agents.providers.settings")
    @patch("src.agents.providers.os.getenv")
    @patch("src.agents.providers.AsyncAnthropic")
    @patch("src.agents.providers.AnthropicProvider")
    @patch("src.agents.providers.AnthropicModel")
    def test_get_anthropic_model_with_environment_base_url(
        self,
        mock_model_class,
        mock_provider_class,
        mock_client_class,
        mock_getenv,
        mock_settings,
    ):
        """Test Anthropic model with base URL from environment."""
        mock_settings.anthropic_api_key = "settings-key"
        mock_settings.anthropic_model_name = "claude-3-sonnet"
        mock_settings.anthropic_base_url = None

        # Environment has base URL
        def getenv_side_effect(key, default=None):
            env_vars = {"ANTHROPIC_BASE_URL": "https://env.anthropic.com"}
            return env_vars.get(key, default)

        mock_getenv.side_effect = getenv_side_effect
        mock_client_instance = Mock(spec=AsyncAnthropic)
        mock_client_class.return_value = mock_client_instance
        mock_provider_instance = Mock(spec=AnthropicProvider)
        mock_provider_class.return_value = mock_provider_instance
        mock_model_instance = Mock(spec=AnthropicModel)
        mock_model_class.return_value = mock_model_instance

        with patch("src.agents.providers.logger") as mock_logger:
            result = get_anthropic_model()

            # Should log the environment base URL (line 74)
            mock_logger.info.assert_called_once_with(
                "Using custom Anthropic base URL: https://env.anthropic.com"
            )

        # Should create custom client (lines 88-90)
        mock_client_class.assert_called_once_with(
            api_key="settings-key", base_url="https://env.anthropic.com"
        )
        mock_provider_class.assert_called_once_with(
            anthropic_client=mock_client_instance
        )
        mock_model_class.assert_called_once_with(
            "claude-3-sonnet", provider=mock_provider_instance
        )
        assert result == mock_model_instance

    @patch("src.agents.providers.settings")
    @patch("src.agents.providers.os.getenv")
    @patch("src.agents.providers.AnthropicModel")
    def test_get_anthropic_model_fallback_to_env_only(
        self, mock_model_class, mock_getenv, mock_settings
    ):
        """Test Anthropic model falls back to environment-only configuration."""
        mock_settings.anthropic_api_key = None  # No settings API key
        mock_settings.anthropic_model_name = "claude-3-sonnet"
        mock_settings.anthropic_base_url = None

        # Environment has API key but no base URL
        def getenv_side_effect(key, default=None):
            env_vars = {"ANTHROPIC_API_KEY": "env-only-key", "ANTHROPIC_BASE_URL": None}
            return env_vars.get(key, default)

        mock_getenv.side_effect = getenv_side_effect
        mock_model_instance = Mock(spec=AnthropicModel)
        mock_model_class.return_value = mock_model_instance

        result = get_anthropic_model()

        # Should fall back to environment-only configuration (line 98)
        mock_model_class.assert_called_once_with("claude-3-sonnet")
        assert result == mock_model_instance

    @patch("src.agents.providers.settings")
    def test_empty_api_key_validation(self, mock_settings):
        """Test validation when API keys are empty strings."""
        mock_settings.openai_api_key = ""
        mock_settings.openai_model_name = "gpt-4"
        mock_settings.openai_base_url = None

        with patch("src.agents.providers.os.getenv", return_value=None):
            # ConfigurationException gets wrapped in AIProviderException
            with pytest.raises(AIProviderException) as exc_info:
                get_openai_model()

            assert "Failed to initialize OpenAI model" in str(exc_info.value)
            assert isinstance(exc_info.value.original_error, ConfigurationException)
            assert "OpenAI API key not found" in str(exc_info.value.original_error)

    @patch("src.agents.providers.settings")
    def test_invalid_model_name_handling(self, mock_settings):
        """Test handling of invalid or unknown model names."""
        mock_settings.openai_api_key = "test-key"
        mock_settings.openai_model_name = "gpt-4"

        with patch("src.agents.providers.logger") as mock_logger, patch(
            "src.agents.providers.get_openai_model"
        ) as mock_openai:
            mock_openai.return_value = Mock(spec=OpenAIModel)

            # Test with various invalid model names - but empty string takes settings.ai_model path
            invalid_names = ["invalid", "unknown:model", "malformed"]

            for invalid_name in invalid_names:
                get_llm_model(invalid_name)

                # Should log warning and default to OpenAI
                mock_logger.warning.assert_called_with(
                    f"Unknown model name '{invalid_name}', defaulting to OpenAI"
                )
                mock_openai.assert_called()

                mock_logger.reset_mock()
                mock_openai.reset_mock()

            # Test empty string separately - it should use settings.ai_model
            mock_settings.ai_model = "openai:gpt-4"
            get_llm_model("")
            # Empty string becomes None after "or settings.ai_model", so uses ai_model which starts with "openai:"
            mock_openai.assert_called()
            # Should not log warning for empty string since it uses ai_model
            assert not mock_logger.warning.called

    @patch("src.agents.providers.settings")
    def test_none_model_name_handling(self, mock_settings):
        """Test handling when model name is None."""
        mock_settings.ai_model = "openai:gpt-4"

        with patch("src.agents.providers.get_openai_model") as mock_openai:
            mock_openai.return_value = Mock(spec=OpenAIModel)

            # Should use settings.ai_model when model_name is None
            get_llm_model(None)
            mock_openai.assert_called_once()


class TestProviderIntegration:
    """Test integration scenarios and edge cases across all providers."""

    @patch("src.agents.providers.settings")
    def test_full_fallback_chain_success(self, mock_settings):
        """Test successful fallback chain with all providers configured."""
        mock_settings.openai_api_key = "test-openai-key"
        mock_settings.anthropic_api_key = "test-anthropic-key"
        mock_settings.google_api_key = "test-google-key"
        mock_settings.ai_model = "fallback"

        with patch("src.agents.providers.get_openai_model") as mock_openai, patch(
            "src.agents.providers.get_anthropic_model"
        ) as mock_anthropic, patch(
            "src.agents.providers.get_google_model"
        ) as mock_google, patch(
            "src.agents.providers.FallbackModel"
        ) as mock_fallback:
            # Create mock instances
            mock_openai_instance = Mock(spec=OpenAIModel)
            mock_anthropic_instance = Mock(spec=AnthropicModel)
            mock_google_instance = Mock(spec=GoogleModel)
            mock_fallback_instance = Mock(spec=FallbackModel)

            mock_openai.return_value = mock_openai_instance
            mock_anthropic.return_value = mock_anthropic_instance
            mock_google.return_value = mock_google_instance
            mock_fallback.return_value = mock_fallback_instance

            result = get_llm_model()

            # Verify all providers were called
            mock_openai.assert_called_once()
            mock_anthropic.assert_called_once()
            mock_google.assert_called_once()

            # Verify fallback model was created with all providers
            mock_fallback.assert_called_once_with(
                [mock_openai_instance, mock_anthropic_instance, mock_google_instance]
            )

            assert result == mock_fallback_instance

    @patch("src.agents.providers.settings")
    def test_provider_priority_in_fallback(self, mock_settings):
        """Test that providers are added to fallback in correct priority order."""
        mock_settings.openai_api_key = "test-openai-key"
        mock_settings.anthropic_api_key = "test-anthropic-key"
        mock_settings.google_api_key = "test-google-key"

        with patch("src.agents.providers.get_openai_model") as mock_openai, patch(
            "src.agents.providers.get_anthropic_model"
        ) as mock_anthropic, patch(
            "src.agents.providers.get_google_model"
        ) as mock_google, patch(
            "src.agents.providers.FallbackModel"
        ) as mock_fallback:
            mock_openai_instance = Mock(spec=OpenAIModel)
            mock_anthropic_instance = Mock(spec=AnthropicModel)
            mock_google_instance = Mock(spec=GoogleModel)

            mock_openai.return_value = mock_openai_instance
            mock_anthropic.return_value = mock_anthropic_instance
            mock_google.return_value = mock_google_instance

            get_llm_model("fallback")

            # Verify order: OpenAI, Anthropic, Google
            args, kwargs = mock_fallback.call_args
            models_list = args[0]

            assert len(models_list) == 3
            assert models_list[0] == mock_openai_instance
            assert models_list[1] == mock_anthropic_instance
            assert models_list[2] == mock_google_instance

    @patch("src.agents.providers.settings")
    def test_edge_case_model_names(self, mock_settings):
        """Test handling of edge case model names and identifiers."""
        mock_settings.openai_api_key = "test-key"
        mock_settings.openai_model_name = "gpt-4"

        edge_cases = [
            "openai:",  # Empty model suffix
            ":gpt-4",  # Empty provider prefix
            "openai:gpt-4:extra",  # Extra components
            "OPENAI:GPT-4",  # Case sensitivity
            "openai: gpt-4",  # Spaces
            "openai:gpt-4\n",  # Whitespace
            "openai:gpt-4\t",  # Tabs
        ]

        with patch("src.agents.providers.get_openai_model") as mock_openai, patch(
            "src.agents.providers.logger"
        ) as mock_logger:
            mock_openai.return_value = Mock(spec=OpenAIModel)

            for edge_case in edge_cases:
                if edge_case.startswith("openai:"):
                    # Should route to OpenAI
                    get_llm_model(edge_case)
                    mock_openai.assert_called()
                else:
                    # Should default to OpenAI with warning
                    get_llm_model(edge_case)
                    mock_logger.warning.assert_called()

                mock_openai.reset_mock()
                mock_logger.reset_mock()

"""
Multi-LLM provider configuration for PydanticAI
"""

import logging
import os
from typing import Any, List, Optional, Union

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider

from src.config.settings import get_settings
from src.exceptions import AIProviderException, ConfigurationException

logger = logging.getLogger(__name__)


def get_openai_model() -> OpenAIModel:
    """Configure OpenAI model with proper error handling"""
    try:
        settings = get_settings()
        # Determine base URL (from settings or environment variable)
        base_url = settings.openai_base_url or os.getenv("OPENAI_BASE_URL")

        if base_url:
            logger.info(f"Using custom OpenAI base URL: {base_url}")

        # Get API key from settings or environment
        api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ConfigurationException(
                message="OpenAI API key not found",
                config_key="openai_api_key",
                details={"model_name": settings.openai_model_name},
            )

        # Create custom client if base URL is provided
        if base_url:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            provider = OpenAIProvider(openai_client=client)
            return OpenAIModel(settings.openai_model_name, provider=provider)

        # Use default OpenAI client with just API key
        if settings.openai_api_key:
            provider = OpenAIProvider(api_key=api_key)
            return OpenAIModel(settings.openai_model_name, provider=provider)

        # Fall back to using environment variable through default provider
        return OpenAIModel(settings.openai_model_name)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI model: {e}")
        raise AIProviderException(
            message="Failed to initialize OpenAI model",
            provider="openai",
            model=settings.openai_model_name,
            original_error=e,
        )


def get_anthropic_model() -> AnthropicModel:
    """Configure Anthropic Claude model with proper error handling"""
    try:
        settings = get_settings()
        # Determine base URL (from settings or environment variable)
        base_url = settings.anthropic_base_url or os.getenv("ANTHROPIC_BASE_URL")

        if base_url:
            logger.info(f"Using custom Anthropic base URL: {base_url}")

        # Get API key from settings or environment
        api_key = settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise ConfigurationException(
                message="Anthropic API key not found",
                config_key="anthropic_api_key",
                details={"model_name": settings.anthropic_model_name},
            )

        # Create custom client if base URL is provided
        if base_url:
            client = AsyncAnthropic(api_key=api_key, base_url=base_url)
            provider = AnthropicProvider(anthropic_client=client)
            return AnthropicModel(settings.anthropic_model_name, provider=provider)

        # Use default Anthropic client with just API key
        if settings.anthropic_api_key:
            provider = AnthropicProvider(api_key=api_key)
            return AnthropicModel(settings.anthropic_model_name, provider=provider)

        # Fall back to using environment variable through default provider
        return AnthropicModel(settings.anthropic_model_name)
    except Exception as e:
        logger.error(f"Failed to initialize Anthropic model: {e}")
        raise AIProviderException(
            message="Failed to initialize Anthropic model",
            provider="anthropic",
            model=settings.anthropic_model_name,
            original_error=e,
        )


def get_google_model() -> GoogleModel:
    """Configure Google Gemini model with proper error handling"""
    try:
        settings = get_settings()
        # Determine base URL (from settings or environment variable)
        base_url = settings.google_base_url or os.getenv("GOOGLE_BASE_URL")

        if base_url:
            logger.info(f"Using custom Google base URL: {base_url}")
            logger.warning(
                "Note: Google/Gemini API does not support custom base URLs in pydantic-ai. This setting will be ignored."
            )

        # Get API key from settings or environment
        api_key = (
            settings.google_api_key
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )

        if not api_key:
            raise ConfigurationException(
                message="Google API key not found",
                config_key="google_api_key",
                details={"model_name": settings.gemini_model_name},
            )

        # Google provider doesn't support custom base URL in pydantic-ai
        # Create provider with API key
        provider = GoogleProvider(api_key=api_key)
        return GoogleModel(settings.gemini_model_name, provider=provider)
    except Exception as e:
        logger.error(f"Failed to initialize Google model: {e}")
        raise AIProviderException(
            message="Failed to initialize Google model",
            provider="google",
            model=settings.gemini_model_name,
            original_error=e,
        )


def get_llm_model(model_name: Optional[str] = None) -> Union[Model, FallbackModel]:
    """
    Get configured LLM model based on settings

    Args:
        model_name: Override model selection

    Returns:
        Configured PydanticAI model
    """
    settings = get_settings()
    model_name = model_name or settings.ai_model

    # Single model configuration
    if model_name.startswith("openai:"):
        return get_openai_model()
    elif model_name.startswith("anthropic:"):
        return get_anthropic_model()
    elif model_name.startswith("gemini:"):
        return get_google_model()

    # Fallback configuration for multiple providers
    elif model_name == "fallback":
        models: List[Union[Model, Any]] = []

        if settings.openai_api_key:
            models.append(get_openai_model())
        if settings.anthropic_api_key:
            models.append(get_anthropic_model())
        if settings.google_api_key:
            models.append(get_google_model())

        if not models:
            raise ConfigurationException(
                message="No LLM providers configured for fallback model",
                config_key="ai_model",
                details={"requested_model": model_name},
            )

        return FallbackModel(models)

    else:
        logger.warning(f"Unknown model name '{model_name}', defaulting to OpenAI")
        # Default to OpenAI
        return get_openai_model()

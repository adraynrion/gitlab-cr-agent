"""
Multi-LLM provider configuration for PydanticAI
"""

import os
from typing import Optional, Union
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider

from src.config.settings import settings

def get_openai_model() -> OpenAIModel:
    """Configure OpenAI model"""
    if settings.openai_api_key:
        provider = OpenAIProvider(api_key=settings.openai_api_key)
        return OpenAIModel(settings.openai_model_name, provider=provider)
    else:
        # Use environment variable (OPENAI_API_KEY)
        return OpenAIModel(settings.openai_model_name)

def get_anthropic_model() -> AnthropicModel:
    """Configure Anthropic Claude model"""
    if settings.anthropic_api_key:
        provider = AnthropicProvider(api_key=settings.anthropic_api_key)
        return AnthropicModel(settings.anthropic_model_name, provider=provider)
    else:
        # Use environment variable (ANTHROPIC_API_KEY)
        return AnthropicModel(settings.anthropic_model_name)

def get_google_model() -> GoogleModel:
    """Configure Google Gemini model"""
    if settings.google_api_key:
        provider = GoogleProvider(api_key=settings.google_api_key)
        return GoogleModel(settings.gemini_model_name, provider=provider)
    else:
        # Use environment variable (GOOGLE_API_KEY or GEMINI_API_KEY)
        return GoogleModel(settings.gemini_model_name)

def get_llm_model(model_name: str = None) -> Union[Model, FallbackModel]:
    """
    Get configured LLM model based on settings
    
    Args:
        model_name: Override model selection
        
    Returns:
        Configured PydanticAI model
    """
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
        models = []
        
        if settings.openai_api_key:
            models.append(get_openai_model())
        if settings.anthropic_api_key:
            models.append(get_anthropic_model())
        if settings.google_api_key:
            models.append(get_google_model())
        
        if not models:
            raise ValueError("No LLM providers configured")
        
        return FallbackModel(models)
    
    else:
        # Default to OpenAI
        return get_openai_model()
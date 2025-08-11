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

from src.config.settings import settings

def get_openai_model() -> OpenAIModel:
    """Configure OpenAI model"""
    return OpenAIModel(
        settings.openai_model_name,
        api_key=settings.openai_api_key,
        temperature=settings.ai_temperature,
        max_tokens=settings.ai_max_tokens
    )

def get_anthropic_model() -> AnthropicModel:
    """Configure Anthropic Claude model"""
    return AnthropicModel(
        settings.anthropic_model_name,
        api_key=settings.anthropic_api_key,
        temperature=settings.ai_temperature,
        max_tokens=settings.ai_max_tokens
    )

def get_google_model() -> GoogleModel:
    """Configure Google Gemini model"""
    return GoogleModel(
        settings.gemini_model_name,
        api_key=settings.google_api_key,
        temperature=settings.ai_temperature,
        max_tokens=settings.ai_max_tokens
    )

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
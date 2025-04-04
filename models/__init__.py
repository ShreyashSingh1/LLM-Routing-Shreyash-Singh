"""Models package for different LLM provider implementations."""

from typing import Dict, Any

from models.base import LLMProvider
from models.openai import OpenAIProvider
from models.anthropic import AnthropicProvider
from models.mistral import MistralProvider
from models.google import GoogleProvider
from config import MODEL_CONFIGS

__all__ = [
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "MistralProvider",
    "GoogleProvider",
    "create_provider",
]


def create_provider(provider_name: str) -> LLMProvider:
    """Create and initialize a provider instance.
    
    Args:
        provider_name: Name of the provider to create
        
    Returns:
        Initialized provider instance
        
    Raises:
        ValueError: If the provider name is not recognized
    """
    # Get provider configuration
    if provider_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown provider: {provider_name}")
    
    config = MODEL_CONFIGS[provider_name]
    
    # Create the appropriate provider instance
    if provider_name == "openai":
        return OpenAIProvider(config)
    elif provider_name == "anthropic":
        return AnthropicProvider(config)
    elif provider_name == "mistral":
        return MistralProvider(config)
    elif provider_name == "google":
        return GoogleProvider(config)
    else:
        raise ValueError(f"No implementation for provider: {provider_name}")
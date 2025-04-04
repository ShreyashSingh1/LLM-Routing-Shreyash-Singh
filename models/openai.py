"""OpenAI LLM provider implementation."""

from typing import Dict, Any, Optional

from models.base import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI implementation of the LLM provider."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the OpenAI provider.
        
        Args:
            config: Configuration dictionary for the provider
        """
        super().__init__(config)
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a response using LangChain's ChatGroq model."""
        return super().generate(prompt, **kwargs)
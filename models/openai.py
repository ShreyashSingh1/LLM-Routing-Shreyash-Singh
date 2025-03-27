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
        """Generate a response using OpenAI's API (mock implementation).
        
        Args:
            prompt: The input prompt/query
            **kwargs: Additional parameters for the generation
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Use the mock implementation from the base class
        return self.mock_generate(prompt, **kwargs)